# train.py
import pytorch_lightning as pl
import logging
import torch
import torch.nn as nn  # For nn.Module, nn.Linear etc.
import torch.nn.functional as F
import contextlib
import os
import re
from jarc_reactor.utils.loop_schedule import compute_scheduled_b
from jarc_reactor.utils.metrics import compute_copy_metrics_on_batch
from types import SimpleNamespace
import torch.distributed as dist

from omegaconf import OmegaConf
from omegaconf import open_dict

from pathlib import Path # Re-added for Path objects
import hydra 

import jarc_reactor.models.transformer_model as transformer_model
# Alias to support monkeypatching via `jarc_reactor.utils.train.TransformerModel`
# in fast unit tests that avoid importing the full model stack.
TransformerModel = transformer_model.TransformerModel
# Record the original class at import time so we can detect which target was monkeypatched later.
_ORIGINAL_TRANSFORMER_CLS = transformer_model.TransformerModel
from jarc_reactor.models.peft.nora import NoRAActivation  # PEFT adapters (NoRA)
from jarc_reactor.models.peft.lora_linear import LoRALinear  # PEFT adapters (LoRA)
from jarc_reactor.models.peft.ponder_phi import PonderPhiNoRAActivation  # Ponder-phi adaptive activation


class NonFiniteLogitsError(RuntimeError):
    """Raised when model logits contain NaN or Inf during training."""
    pass

from ..utils.regularizers import gauge_loss, cebr_loss
from ..losses.contrastive import info_nce
from ..utils.logging_config import RankZeroFilter
from jarc_reactor.calibration.temperature_scaling import (
    compute_ece_from_logits_binned,
    compute_mce_from_logits_binned,
    compute_ece_from_probs_binned,
    compute_mce_from_probs,
    plot_reliability_pre_post_from_logits,
)
from jarc_reactor.utils.rdrop import masked_symmetric_kl
from jarc_reactor.calibration import make_calibrator
from jarc_reactor.optimizers.sam import SAM
from jarc_reactor.losses.expected_cost_loss import (
    build_slot_costs,
    apply_risk_scaling,
    expected_cost_loss as _expected_cost_loss,
    risk_from_entropy,
)
from jarc_reactor.utils.swa import SWAController
from jarc_reactor.utils.swa_diagnostics import run_swa_diagnostics_from_trainer, state_from_model, load_state_into_model

# Initialize logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TransformerTrainer(pl.LightningModule):
    def __init__(self, config):
        """
        Initialize the TransformerTrainer from a configuration, preparing model, losses, logging, and regularisation state.
        
        This constructor:
        - Ensures `config` is an OmegaConf mapping and saves a sanitized copy for hyperparameter logging.
        - Applies safe defaults for missing model-level fields required by downstream code.
        - Configures a per-run file logger (train.log) unless the config indicates a test environment.
        - Instantiates the model architecture selected by `config.model.model_name` and exposes `self.core_model` and `self.dropout`.
        - Prepares Quantization-Aware Training when enabled.
        - Configures loss-related state: pad token id, optional class-balanced loss placeholder, CrossEntropy criterion, and optional Dice loss settings.
        - Reads and stores training hyperparameters (learning rate, calibration config, R-Drop settings) and initializes R-Drop telemetry counters.
        - Constructs regulariser-related parameters (gauge/CEBR params, lambda penalties), head-level CEBR structures when enabled, optional Functional Risk Minimisation (FRM), and the RegulariserHyperGrad wrapper when configured.
        - Applies any initial overrides for regulariser parameters from configuration.
        
        Parameters:
            config (dict or OmegaConf): Configuration describing model, training, logging, regularisation, and other runtime options. The constructor will coerce plain dicts into an OmegaConf object and may mutate `self.config` to populate missing structured defaults.
        """
        super().__init__()
        # Ensure config is an OmegaConf object for consistent attribute access
        if not isinstance(config, OmegaConf):
            config = OmegaConf.create(config)

        # Sanitize config for hyperparameter logging
        # Convert to a primitive container (dict) and remove the schema, which is not a hyperparameter
        hparams_for_logging = OmegaConf.to_container(config, resolve=True)
        if 'schema' in hparams_for_logging:
            del hparams_for_logging['schema']

        self.save_hyperparameters(hparams_for_logging)
        # Store the original OmegaConf config object for use within the module
        self.config = config
        # Back-compat: provide read/write alias property `cfg` that proxies to `config`
        # (no duplicate reference to reduce memory footprint)
        # Ensure structured defaults for new schema fields when tests provide plain dicts
        try:
            from jarc_reactor.config_schema import ModelConfigSchema  # noqa: F401
        except Exception:
            pass
        # Provide defaults for missing keys to satisfy structured access in models
        if OmegaConf.select(self.config, "model.context_effect_gain", default=None) is None:
            try:
                with open_dict(self.config.model):
                    self.config.model.context_effect_gain = 4.0
            except Exception:
                pass
        if OmegaConf.select(self.config, "model.context_scaling_factor", default=None) is None:
            try:
                with open_dict(self.config.model):
                    self.config.model.context_scaling_factor = 2.0
            except Exception:
                pass

        logger.info("Initializing TransformerTrainer")
        # Ensure conflict guard is present in trainer (runtime will add)
        logger.propagate = True  # Ensure logger propagates to root logger

        # Initialize logging if necessary
        # Resolve log_dir relative to Hydra's output directory
        # Safely fetch from config; some tests/eval paths may omit this key
        log_dir_name = OmegaConf.select(self.config, "logging.log_dir", default=None)
        if not log_dir_name or str(log_dir_name).strip() == "":
            log_dir_name = "logs/app"  # mirrors default in `conf/logging/default.yaml`
        try:
            hydra_run_dir = Path(hydra.core.hydra_config.HydraConfig.get().run.dir)
            resolved_log_dir = hydra_run_dir / log_dir_name
        except Exception as e:
            # Fallback if not in a Hydra run context (e.g. direct instantiation for tests)
            # This might not be ideal for production but prevents crashes in other contexts.
            logger.warning(f"Could not get Hydra run directory (error: {e}). Using log_dir relative to CWD.")
            resolved_log_dir = Path(log_dir_name)

        # Only configure file logging if not in a test environment to prevent resource leaks.
        self._file_handler = None  # track per-instance handler for cleanup
        if not self.config.get('is_test_environment', False):
            resolved_log_dir.mkdir(parents=True, exist_ok=True)
            log_file_path = resolved_log_dir / "train.log"
            # Avoid duplicating handlers for the same file on this logger
            same_file_exists = any(
                isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', None) == str(log_file_path)
                for h in logger.handlers
            )
            if not same_file_exists:
                fh = logging.FileHandler(log_file_path)
                fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
                # Ensure only rank-0 emits to file in DDP
                try:
                    fh.addFilter(RankZeroFilter())
                except Exception:
                    pass
                logger.addHandler(fh)
                self._file_handler = fh
        # Initialize LoRA modules
        #self.lora_A = nn.Linear(config.model.lora_in_features, config.model.lora_out_features)
        #self.lora_B = nn.Linear(config.model.lora_in_features, config.model.lora_out_features)
        # Test-time adaptation adapter (initialized lazily)
        self._sar_adapter = None

        # ------------------------------------------------------------------
        # Instantiate model architecture according to config.model.model_name
        # ------------------------------------------------------------------
        model_name = getattr(config.model, "model_name", "transformer")

        # If user requested H-Net, honour the master enabled switch
        if model_name == "hnet2d1d_t3_cs":
            hnet_cfg_raw = getattr(config.model, "hnet", {}) if hasattr(config.model, "hnet") else {}
            enabled = True
            if isinstance(hnet_cfg_raw, dict):
                enabled = hnet_cfg_raw.get("enabled", True)
            else:
                enabled = getattr(hnet_cfg_raw, "enabled", True)
            if not enabled:
                logger.info("H-Net disabled via config; using default Transformer model instead.")
                model_name = "transformer"

        # Instantiate the chosen model
        if model_name in {"transformer", "default"}:
            # Safely fetch optional DSLA config; some eval/test overlays omit it
            dsla_cfg = OmegaConf.select(config, "dsla", default=None)
            # Support both monkeypatch styles used by tests:
            #  - train.TransformerModel (patched on this module)
            #  - jarc_reactor.models.transformer_model.TransformerModel (patched on the model module)
            alias_cls = globals().get("TransformerModel", None)
            module_cls = transformer_model.TransformerModel
            # Prefer an explicitly patched alias only if it differs from the original import-time class;
            # otherwise prefer the (possibly patched) class on the model module.
            if alias_cls is not None and alias_cls is not _ORIGINAL_TRANSFORMER_CLS:
                ModelClass = alias_cls
            else:
                ModelClass = module_cls
            self.model = ModelClass(config=config.model, dsla_config=dsla_cfg)
            # Alias for uniform callback access
            self.core_model = self.model
            # Log DEQ activation state and encoder type
            try:
                deq_cfg = OmegaConf.select(config, "model.deq", default=None)
                deq_enabled = bool(getattr(deq_cfg, "enabled", False)) if deq_cfg is not None else False
                enc_type = type(getattr(self.model, "encoder", None)).__name__
                if deq_enabled:
                    logger.info(f"DEQ requested (enabled=true). Encoder type: {enc_type}")
                else:
                    logger.info(f"DEQ disabled. Encoder type: {enc_type}")
            except Exception:
                pass
            # FRM features capture: hook the input to the final projection layer as features
            self._frm_last_hidden = None
            try:
                def _frm_hook(module, inputs, output):
                    """
                    Store the module's input hidden state on self for Functional Risk Minimisation (FRM) use.
                    
                    Attempts to set self._frm_last_hidden to the first element of the hook `inputs` tuple (expected shape [B, L, D]); if extraction fails, sets self._frm_last_hidden to None.
                    
                    Parameters:
                        module: The module the hook was registered on (ignored).
                        inputs: Hook inputs tuple; the first element is expected to be the hidden tensor [B, L, D].
                        output: The module output (ignored).
                    """
                    try:
                        # inputs is a tuple; first element is hidden tensor [B,L,D]
                        self._frm_last_hidden = inputs[0]
                    except Exception:
                        self._frm_last_hidden = None
                if hasattr(self.model, 'output_fc') and isinstance(self.model.output_fc, nn.Module):
                    self.model.output_fc.register_forward_hook(_frm_hook)
            except Exception:
                self._frm_last_hidden = None
        elif model_name == "hnet2d1d_t3_cs":
            from jarc_reactor.models.hnet2d.hnet2d1d_t3_cs import HN2D1D_T3_CS
            self.model = HN2D1D_T3_CS(config.model.hnet)
            # Alias for uniform callback access
            self.core_model = self.model
        elif model_name == "HN2D1D":
            from jarc_reactor.models.hnet2d.hnet2d1d import HN2D1D
            # Merge the dedicated hnet sub-config onto the main model config for convenience
            hnet_cfg_fallback = OmegaConf.create({})
            hnet_cfg_fallback = OmegaConf.create({})
            # Merge the configs as OmegaConf objects to preserve structure
            # To handle optimizer-injected keys not present in the H-Net schema,
            # we temporarily disable the strict schema validation (struct mode)
            # on the config copy, perform the merge, and then re-enable it.
            hnet_cfg = config.model.copy()
            OmegaConf.set_struct(hnet_cfg, False)
            OmegaConf.merge(hnet_cfg, config.model.hnet or hnet_cfg_fallback)
            OmegaConf.set_struct(hnet_cfg, True)
            self.model = HN2D1D(hnet_cfg)
        else:
            raise ValueError(f"Unknown model_name={model_name}")

        self.dropout = self.model.dropout  # Expose dropout attribute

        # Enable QAT if configured
        if hasattr(config.training, 'use_qat') and config.training.use_qat:
            self.model.train()  # Ensure model is in training mode for QAT
            self.model = torch.quantization.prepare_qat(self.model)
            logger.info("Model prepared for Quantization-Aware Training")

        # Modified criterion to handle padding separately
        self.pad_token_id = int(getattr(config.model, "pad_token_id", 10))
        # Optional colour-balanced loss
        self.use_class_balanced_loss = bool(getattr(config.training, "class_balanced_loss", False))
        self.class_weights = None  # Filled in setup() if enabled
        # Criterion initialised without weights; may be replaced in setup()
        # IMPORTANT: Do not ignore PAD. PAD must be learned as a valid class.
        self.criterion = nn.CrossEntropyLoss()
        
        # ------------------------- Dice loss configuration -------------------------
        self.use_dice_loss = bool(getattr(config.training, "dice_loss_enabled", False))
        self.dice_weight = float(getattr(config.training, "dice_loss_weight", 1.0))
        if self.use_dice_loss and self.dice_weight <= 0:
            logger.warning("dice_loss_weight <= 0 while dice_loss_enabled is True; disabling Dice loss.")
            self.use_dice_loss = False
        
        self.learning_rate = config.training.learning_rate  # Access learning_rate from config.training
        # Optimizer reference (set in configure_optimizers) for LR logging
        self._optimizer_ref = None

        # ------------- Calibration (Guo et al., 2017) -------------
        self.calibration_cfg = OmegaConf.select(self.config, "calibration", default=None)
        self._calibrator = None

        # ------------------------- R-Drop configuration -------------------------
        self.rdrop_enabled = getattr(config, "rdrop", None) is not None and bool(config.rdrop.enabled)
        if self.rdrop_enabled:
            # Store alpha as a tensor to support `.item()` in tests while avoiding parameter pollution
            try:
                self.rdrop_alpha = torch.tensor(float(config.rdrop.alpha))
            except Exception:
                self.rdrop_alpha = torch.tensor(0.0)
            self.rdrop_share_mask = bool(getattr(config.rdrop, "share_dropout_mask", False))
            # Optional: warn about unused rdrop.p to avoid confusion
            try:
                p_val = float(getattr(config.rdrop, "p", 0.0))
                if p_val not in (0.0,):
                    logger.warning("R-Drop config 'p' is currently unused and ignored; model dropout rates are used.")
            except Exception:
                pass
        else:
            self.rdrop_alpha = None
            self.rdrop_share_mask = False

        # R-Drop telemetry counters (aggregated over training)
        self._rdrop_stats = SimpleNamespace(
            sanitize_count=0,   # number of times non-finite logits were detected pre-KL
            clip_tokens=0,      # number of per-token KL values clipped
            clip_events=0,      # number of batches where any clipping occurred
            fp32_kl_steps=0,    # number of KL computations forced to fp32
        )

        # ------------------------------------------------------------
        #  Hyper-gradient descent on regulariser weights
        # ------------------------------------------------------------
        def _get_param(key, default_val):
            val = OmegaConf.select(config.model, key, default=default_val)
            return nn.Parameter(torch.tensor(float(val), dtype=torch.float32), requires_grad=False)

        self.gauge_alpha = _get_param('gauge_alpha', 0.0)
        self.cebr_alpha = _get_param('cebr_alpha', 0.0)
        self.cebr_beta = _get_param('cebr_beta', 0.0)
        self.cebr_gamma = _get_param('cebr_gamma', 0.0)

        # Optionally override initial regulariser parameters from regulariser_hypergrad.init
        reg_cfg_raw = getattr(config, "regulariser_hypergrad", {})
        # Support both dict and OmegaConf.DictConfig
        init_overrides = None
        if isinstance(reg_cfg_raw, dict):
            init_overrides = reg_cfg_raw.get("init", {}) or {}
        else:
            init_overrides = getattr(reg_cfg_raw, "init", {}) or {}
            # Convert DictConfig to plain dict
            if init_overrides:
                try:
                    init_overrides = OmegaConf.to_container(init_overrides, resolve=True)
                except Exception:
                    pass
        if init_overrides:
            with torch.no_grad():
                for name in ["gauge_alpha", "cebr_alpha", "cebr_beta", "cebr_gamma"]:
                    if name in init_overrides and getattr(self, name, None) is not None:
                        val = float(init_overrides[name])
                        getattr(self, name).copy_(torch.tensor(val, dtype=torch.float32))

        # ------------------------------------------------------------
        #  Regularisation penalties (confidence entropy, PAD coverage, histogram KL)
        # ------------------------------------------------------------
        # Lambda hyperparameters controlling each penalty term
        self.lambda_conf = float(getattr(config.training, "lambda_conf", 0.0))  # Confidence entropy
        self.lambda_cov = float(getattr(config.training, "lambda_cov", 0.0))   # PAD token coverage
        self.lambda_kl  = float(getattr(config.training, "lambda_kl", 0.0))    # Histogram KL divergence

        # Placeholder for dataset prior histogram (computed during setup when datamodule is available)
        self.dataset_prior = None

        # ------------------------------------------------------------
        #  Head-level VE/ORC CEBR regulariser
        # ------------------------------------------------------------
        from jarc_reactor.regularisers.cebr_head import HeadLevelCEBR
        self.cebr_head_cfg = self.config.model.cebr_head if hasattr(self.config.model, "cebr_head") else None
        if self.cebr_head_cfg and self.cebr_head_cfg.enabled:
            n_head = self.config.model.n_head
            self.head_cebr = HeadLevelCEBR(
                n_head,
                tau_init=self.cebr_head_cfg.tau_global,
                tau_eps=self.cebr_head_cfg.tau_eps,
                topk_orc=self.cebr_head_cfg.topk_orc,
                power_iters=self.cebr_head_cfg.power_iters,
                p_min=self.cebr_head_cfg.p_min,
            )
            # Pre-compute per-head weights (static tensors)
            thirds = [n_head // 3, n_head // 3, n_head - 2 * (n_head // 3)]
            ve_weights = [
                self.cebr_head_cfg.ve_alpha_micro,
                self.cebr_head_cfg.ve_alpha_meso,
                self.cebr_head_cfg.ve_alpha_macro,
            ]
            orc_weights = [
                self.cebr_head_cfg.orc_beta_micro,
                self.cebr_head_cfg.orc_beta_meso,
                self.cebr_head_cfg.orc_beta_macro,
            ]
            ve_alpha_h = []
            orc_beta_h = []
            for i, cnt in enumerate(thirds):
                ve_alpha_h.extend([ve_weights[i]] * cnt)
                orc_beta_h.extend([orc_weights[i]] * cnt)
            # Store as non-trainable parameters so hypergrad can update them
            self.ve_alpha_h = nn.Parameter(torch.tensor(ve_alpha_h, dtype=torch.float32), requires_grad=False)
            self.orc_beta_h = nn.Parameter(torch.tensor(orc_beta_h, dtype=torch.float32), requires_grad=False)
            # Backwards-compat attribute names
            self._ve_alpha_h = self.ve_alpha_h
            self._orc_beta_h = self.orc_beta_h
        else:
            self.head_cebr = None
        
        # ------------ Functional Risk Minimisation -------------
        frm_cfg = OmegaConf.select(config, "model.frm")
        self.frm_enabled = bool(frm_cfg.get("enabled", False)) if frm_cfg is not None else False
        if self.frm_enabled:
            from jarc_reactor.models.modules.frm import FunctionalRisk
            self.frm = FunctionalRisk(frm_cfg)
            self.lambda_frm = self.frm.lambda_frm
        else:
            self.frm = None
            self.lambda_frm = None

        self._reg_params = {
            "gauge_alpha": self.gauge_alpha,
            "cebr_alpha": self.cebr_alpha,
            "cebr_beta": self.cebr_beta,
            "cebr_gamma": self.cebr_gamma,
             **({"lambda_frm": self.lambda_frm} if self.frm_enabled else {}),
            # Head-level τ_ deltas participate in hypergrad if enabled
             **({
                 "tau_delta": self.head_cebr.tau_delta,
                 "ve_alpha_h": getattr(self, "ve_alpha_h", None),
                 "orc_beta_h": getattr(self, "orc_beta_h", None),
             } if self.head_cebr is not None and self.cebr_head_cfg.learnable else {}),
        }

        from jarc_reactor.optimizers.regulariser_hypergrad import RegulariserHyperGrad
        reg_cfg = getattr(config, "regulariser_hypergrad", {})
        if reg_cfg.get("enabled", False):
            self.reg_hg = RegulariserHyperGrad(self._reg_params, reg_cfg)
        else:
            self.reg_hg = None

        # Optional override of initial values from regulariser_hypergrad.init
        init_dict = reg_cfg.get("init", {}) if isinstance(reg_cfg, dict) else {}
        for k, v in init_dict.items():
            if k in self._reg_params:
                self._reg_params[k].data.fill_(float(v))

        # ------------------------------------------------------------
        #  Stochastic Weight Averaging (SWA) init
        # ------------------------------------------------------------
        self.swa_cfg = OmegaConf.select(self.config, "training.swa", default=None)
        if self.swa_cfg is None:
            self.swa_enabled = False
            self._swa_start_epoch = None
            self.swa_ctrl = None
        else:
            self.swa_enabled = bool(getattr(self.swa_cfg, "enabled", False))
            max_epochs = int(OmegaConf.select(self.config, "training.max_epochs", default=100))
            se = getattr(self.swa_cfg, "start_epoch", None)
            self._swa_start_epoch = int(se) if se is not None else max(0, int(0.75 * max_epochs))
            self.swa_ctrl = SWAController(self.swa_cfg)

    # ------------------------------------------------------------------
    #  Class-balanced loss helpers
    # ------------------------------------------------------------------
    def setup(self, stage: str) -> None:  # Lightning hook
        super().setup(stage)
        if stage == "fit" and self.use_class_balanced_loss and self.class_weights is None:
            self._compute_class_weights()

    def _compute_class_weights(self) -> None:
        """Infer inverse-frequency colour weights from the training dataloader and set criterion weights.

        Additional options:
          * ``class_balanced_smoothing_eps`` adds a small ε to every class to avoid
            extreme weights.
          * ``class_balanced_min_weight`` / ``class_balanced_max_weight`` clamp the
            final normalised weights to a reasonable range.
        """
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is None:
            logger.warning("Class-balanced loss enabled but datamodule unavailable; skipping weight computation.")
            return
        dl = datamodule.train_dataloader()
        vocab_size = int(getattr(self.config.model, "vocab_size", 11))
        # Accumulate colour frequencies on CPU to avoid device mismatches (e.g., MPS vs CPU tensors)
        freq = torch.zeros(vocab_size, dtype=torch.long)
        with torch.no_grad():
            for batch in dl:
                tgt = batch[1]  # (src, tgt, ...)
                freq += torch.bincount(tgt.flatten().cpu().clamp_max(vocab_size - 1), minlength=vocab_size)

        # ---------------- Dataset prior histogram (for KL regulariser) ----------------
        # Compute normalised prior distribution excluding PAD token
        prior = freq.float()
        prior[self.pad_token_id] = 0.0
        prior_sum = prior.sum().item()
        if prior_sum > 0:
            prior = prior / prior_sum
        else:
            # Fallback to uniform over non-pad classes to avoid NaNs
            non_pad = torch.ones_like(prior)
            non_pad[self.pad_token_id] = 0.0
            prior = non_pad / non_pad.sum()
        # Store on CPU; will be moved to correct device on first use
        self.dataset_prior = prior
        total = freq.sum().item() if freq.sum() > 0 else 1

        # Base inverse-frequency weights
        weights = total / (freq.float() + 1e-6)

        # --- Smoothing -------------------------------------------------------
        smooth_eps = float(getattr(self.config.training, "class_balanced_smoothing_eps", 0.0))
        if smooth_eps > 0:
            weights = weights + smooth_eps

        # --- Normalise & clamp ---------------------------------------------
        weights = weights / weights.mean().clamp_min(1e-6)

        w_min = getattr(self.config.training, "class_balanced_min_weight", None)
        w_max = getattr(self.config.training, "class_balanced_max_weight", None)
        if w_min is not None or w_max is not None:
            weights = weights.clamp(
                min=float(w_min) if w_min is not None else None,
                max=float(w_max) if w_max is not None else None,
            )

        self.class_weights = weights.to(self.device)
        # IMPORTANT: Do not ignore PAD. Provide weights including PAD.
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        logger.info("Class-balanced CE enabled. Weights: %s", self.class_weights.cpu().tolist())

    def debug_batch(self, batch, name=""):
        """Helper function to debug batch data"""
        src, tgt, ctx_input, ctx_output, task_ids = batch
        #logger.info(f"\nDEBUG - Batch {name}:")
        #logger.info(f"Source shape: {src.shape}, dtype: {src.dtype}")
        #logger.info(f"Target shape: {tgt.shape}, dtype: {tgt.dtype}")
        #logger.info(f"Context input shape: {ctx_input.shape if ctx_input is not None else None}")
        #logger.info(f"Context output shape: {ctx_output.shape if ctx_output is not None else None}")
        #logger.info(f"Task IDs: {task_ids}")
        return batch

    def forward(self, src, tgt, ctx_input=None, ctx_output=None, aug_grid=None,
                loop_b_override=None, loop_return_all=None, loop_truncate_T=None):
        # Relay optional kwargs to the underlying model only if it's supported by its signature
        """
        Forward inputs through the underlying model, forwarding optional arguments only if the model's signature accepts them.

        Parameters:
            src: Source/input tensor(s) for the model.
            tgt: Target tensor(s) for the model.
            ctx_input: Optional context input provided to the model.
            ctx_output: Optional context output provided to the model.
            aug_grid: Optional augmentation/grid information; passed only if the model accepts `aug_grid`.
            loop_b_override: Optional loop override value; passed only if the model accepts `loop_b_override`.
            loop_return_all: Optional flag to request all loop outputs; passed only if the model accepts `loop_return_all`.
            loop_truncate_T: Optional truncation length for looped models; passed only if the model accepts `loop_truncate_T`.

        Returns:
            The underlying model's output — typically logits and any auxiliary outputs produced by that model.
        """
        import inspect
        sig = inspect.signature(self.model.forward)
        extra = {}
        if 'aug_grid' in sig.parameters and aug_grid is not None:
            extra['aug_grid'] = aug_grid
        if 'loop_b_override' in sig.parameters and loop_b_override is not None:
            extra['loop_b_override'] = loop_b_override
        if 'loop_return_all' in sig.parameters and loop_return_all is not None:
            extra['loop_return_all'] = loop_return_all
        if 'loop_truncate_T' in sig.parameters and loop_truncate_T is not None:
            extra['loop_truncate_T'] = loop_truncate_T

        return self.model(src, tgt, ctx_input, ctx_output, **extra)

    def _compute_loss(self, y_hat, tgt, src=None):
        """
        Compute the cross-entropy loss over a 2D grid while preserving grid semantics and ignoring padded/invalid cells.
        
        This flattens predictions and targets from shape [B, H, W, C] and [B, H, W] into per-token tensors, sanitizes non-finite logits, replaces out-of-range target indices with the padding id so they are ignored, and computes cross-entropy only over valid (non-pad) positions. If no valid targets remain, returns a scalar 0.0 on the same device/dtype as the logits. When a source grid is provided and the training configuration enables change-weighted CE (change_weight > 1.0), tokens that differ between source and target may be upweighted for the loss. Class-balanced weights from self.class_weights (if present) are applied to the per-class CE calculation.
        
        Parameters:
            y_hat (torch.Tensor): Logits with shape [B, H, W, num_classes].
            tgt (torch.Tensor): Integer targets with shape [B, H, W].
            src (torch.Tensor | None): Optional source/grid tensor with the same spatial shape as tgt; used to compute change-weighted CE when enabled in config.
        
        Returns:
            torch.Tensor: Scalar loss tensor computed over non-padded grid cells.
        """
        # ---------------- Default grid-aware CE path ----------------
        # Apply Phase 3 insertion gating if enabled (logit adjustments)
        y_hat = self._apply_insertion_gating(y_hat, tgt)
        y_hat_flat = y_hat.reshape(-1, y_hat.size(-1))  # [B*H*W, C]
        tgt_flat = tgt.reshape(-1).long()

        # Sanitise non-finite logits to keep loss finite
        if not torch.isfinite(y_hat_flat).all():
            with torch.no_grad():
                n_nan = torch.isnan(y_hat_flat).sum().item()
                n_inf = torch.isinf(y_hat_flat).sum().item()
                # Compute min/max over finite values only to avoid dependency on torch.nanmin/torch.nanmax
                finite_mask = torch.isfinite(y_hat_flat)
                if finite_mask.any():
                    finite_vals = y_hat_flat[finite_mask]
                    min_val = finite_vals.min().item()
                    max_val = finite_vals.max().item()
                else:
                    # No finite values; use sentinel values for logging
                    min_val = float('nan')
                    max_val = float('nan')
                logger.warning(
                    f"Non-finite logits detected in _compute_loss: NaN={n_nan}, Inf={n_inf}, min={min_val:.3e}, max={max_val:.3e}. Sanitising and continuing."
                )
            # Replace NaNs/Infs with safe finite sentinels before further processing
            y_hat_flat = torch.nan_to_num(y_hat_flat, nan=0.0, posinf=0.0, neginf=0.0)

        # Clamp logits to improve numerical stability
        y_hat_flat = torch.clamp(y_hat_flat, min=-10.0, max=10.0)

        # Final sanitization pass (handles any infinities introduced by clamp’s bounds)
        y_hat_flat = torch.nan_to_num(y_hat_flat, nan=0.0, posinf=10.0, neginf=-10.0)

        # Replace invalid target indices with pad so they are ignored
        vocab_size = y_hat_flat.size(-1)
        invalid_tgt = (tgt_flat != self.pad_token_id) & ((tgt_flat < 0) | (tgt_flat >= vocab_size))
        if invalid_tgt.any():
            with torch.no_grad():
                min_idx = tgt_flat.min().item()
                max_idx = tgt_flat.max().item()
                bad = invalid_tgt.sum().item()
                logger.warning(
                    f"Found {bad} invalid target indices (min={min_idx}, max={max_idx}, vocab={vocab_size}); replacing with pad_token_id={self.pad_token_id}"
                )
            tgt_flat = tgt_flat.masked_fill(invalid_tgt, self.pad_token_id)

        # Compute loss over all positions, including PAD
        # Optionally apply change-weighted CE during training when src is provided
        change_weight = float(getattr(self.config.training, "change_weight", 1.0)) if hasattr(self, "config") else 1.0
        class_w = getattr(self, "class_weights", None)
        if self.training and (src is not None) and (change_weight is not None) and (float(change_weight) > 1.0):
            # Flatten src to align with tgt_flat
            try:
                src_flat = src.reshape(-1).long()
            except Exception:
                src_flat = None
            ce_vec = F.cross_entropy(y_hat_flat, tgt_flat, weight=class_w, reduction='none')
            if src_flat is not None and src_flat.numel() == tgt_flat.numel():
                pad_id = int(getattr(self.config.model, "pad_token_id", y_hat_flat.size(-1) - 1))
                change_mask = (tgt_flat != src_flat) & (tgt_flat != pad_id)
                weights = torch.ones_like(ce_vec, dtype=ce_vec.dtype)
                weights = torch.where(change_mask, weights * float(change_weight), weights)
                return (ce_vec * weights).mean()
            return ce_vec.mean()
        else:
            return F.cross_entropy(y_hat_flat, tgt_flat, weight=class_w, reduction='mean')

    # --------------------------- Phase 3: Insertion gating ---------------------------
    def _apply_insertion_gating(self, logits: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Apply soft gating to per-token logits to discourage non-EOS tokens for slots considered complete.
        
        When enabled in configuration, slots determined to be "complete" (for example, when pad_marks_complete is true and the target equals the pad token) will have a fixed logit shift subtracted from all tokens except the EOS token. This is a soft (additive) gating that preserves gradients and leaves logits unchanged if gating is disabled, the model output is not augmented for insertion gating, no slots are complete, or an internal error occurs.
        
        Parameters:
            logits (torch.Tensor): Logits tensor of shape (B, H, W, K) where K is the vocabulary (possibly augmented).
            tgt (torch.Tensor): Target token ids tensor with shape broadcastable to (B, H, W) used to determine completed slots.
        
        Returns:
            torch.Tensor: Logits tensor with the same shape as `logits`, with non-EOS logits for completed slots reduced by the configured gating shift when gating is applied.
        """
        try:
            ins_cfg = getattr(self.config.model, "insertion", None)
            if ins_cfg is None or not bool(getattr(ins_cfg, "enabled", False)):
                return logits
            apply_now = (self.training and bool(getattr(ins_cfg, "train_gating", True))) or ((not self.training) and bool(getattr(ins_cfg, "eval_gating", False)))
            if not apply_now:
                return logits

            V = int(getattr(self.config.model, "vocab_size", logits.size(-1)))
            K = int(logits.size(-1))
            # Augmented head required: base V + hedged V + U + EOS
            EOS = V + V + 1
            if K <= EOS:
                # Not augmented; nothing to gate
                return logits

            shift = float(getattr(ins_cfg, "gating_logit_shift", 8.0))
            pad_id = int(getattr(self.config.model, "pad_token_id", V - 1))
            pad_marks_complete = bool(getattr(ins_cfg, "pad_marks_complete", False))

            B, H, W, _ = logits.shape
            L = H * W
            logits_flat = logits.view(B * L, K)
            tgt_flat = tgt.view(B * L)

            # Completion mask per slot
            if pad_marks_complete:
                complete = (tgt_flat == pad_id)
            else:
                complete = torch.zeros_like(tgt_flat, dtype=torch.bool)

            if not complete.any():
                return logits

            # Disallow all tokens except EOS for complete slots
            disallow = torch.zeros_like(logits_flat, dtype=torch.bool)
            disallow[complete, :] = True
            disallow[complete, EOS] = False

            # Apply logit shift
            logits_flat = logits_flat - shift * disallow.to(logits_flat.dtype)
            return logits_flat.view(B, H, W, K)
        except Exception:
            return logits

    def _compute_accuracy(self, y_hat, tgt):
        """
        Compute per-cell and per-grid accuracy metrics for predicted grids.
        
        Parameters:
            y_hat (Tensor): Logits or class scores with shape [B, H, W, C].
            tgt (Tensor): Integer target classes with shape [B, H, W]; padding positions use self.pad_token_id.
        
        Returns:
            dict: Metrics including:
                - cell_accuracy: Fraction of non-padding cells predicted correctly.
                - grid_accuracy: Fraction of samples where all non-padding cells are correct.
                - row_all_correct_rate: Fraction of rows (across batch and height) fully correct ignoring padding.
                - col_all_correct_rate: Fraction of columns (across batch and width) fully correct ignoring padding.
                - grid_tol_<thr>: For each configured tolerance threshold (e.g., 0.90 -> grid_tol_0p90), fraction of samples whose per-sample cell accuracy is >= threshold.
        """
        with torch.no_grad():
            # Get predicted classes
            predictions = torch.argmax(y_hat, dim=-1)  # [batch, H, W]
            
            # Debug predictions
            #logger.info(f"\nDEBUG - Accuracy computation:")
            #logger.info(f"Predictions shape: {predictions.shape}, dtype: {predictions.dtype}")
            #logger.info(f"Target shape: {tgt.shape}, dtype: {tgt.dtype}")
            
            # Create mask for non-padding elements
            valid_mask = (tgt != self.pad_token_id)
            
            # Cell-wise accuracy
            correct_cells = (predictions == tgt) & valid_mask
            # Add epsilon to prevent division by zero if valid_mask is all False
            cell_accuracy = correct_cells.float().sum() / (valid_mask.float().sum() + 1e-6)

            # Full grid accuracy (only count grid as correct if all non-padded cells match)
            corr_or_pad = (correct_cells | ~valid_mask)
            # Support [B,H,W] or [B,T] by flattening non-batch dims
            if corr_or_pad.dim() >= 2:
                grid_matches = corr_or_pad.view(corr_or_pad.size(0), -1).all(dim=1)
            else:
                grid_matches = corr_or_pad.bool()
            grid_accuracy = grid_matches.float().mean()

            # ---------------- Additional diagnostics ----------------
            # Per-sample cell accuracy for tolerant grid metrics
            B, H, W = predictions.shape
            per_sample_valid = valid_mask.reshape(B, -1).float().sum(dim=1)
            per_sample_correct = correct_cells.reshape(B, -1).float().sum(dim=1)
            per_sample_cell_acc = per_sample_correct / (per_sample_valid + 1e-6)

            # Threshold list from config (e.g., metrics.grid_tol_thresholds: [0.90, 0.95])
            try:
                tol_thresholds = OmegaConf.select(self.config, 'metrics.grid_tol_thresholds', default=[0.90, 0.95])
                # Ensure it's a list of floats
                if tol_thresholds is None:
                    tol_thresholds = [0.90, 0.95]
                tol_thresholds = [float(t) for t in tol_thresholds]
            except Exception:
                tol_thresholds = [0.90, 0.95]

            # Build extra metrics dict
            extra = {}
            for thr in tol_thresholds:
                # fraction of samples whose per-grid cell-accuracy >= thr
                meets = (per_sample_cell_acc >= float(thr)).float().mean()
                thr_name = (f"{thr:.2f}").replace('.', 'p')  # e.g., 0.90 -> '0p90'
                extra[f'grid_tol_{thr_name}'] = meets

            # Fraction of rows/columns that are entirely correct (padded cells ignored)
            correct_or_pad = correct_cells | ~valid_mask
            row_full = correct_or_pad.all(dim=2)  # [B, H]
            col_full = correct_or_pad.all(dim=1)  # [B, W]
            row_all_correct_rate = row_full.float().mean()
            col_all_correct_rate = col_full.float().mean()

            # Debug accuracy metrics
            #logger.info(f"Cell accuracy: {cell_accuracy.item():.4f}")
            #logger.info(f"Grid accuracy: {grid_accuracy.item():.4f}")

            return {
                'cell_accuracy': cell_accuracy,
                'grid_accuracy': grid_accuracy,
                'row_all_correct_rate': row_all_correct_rate,
                'col_all_correct_rate': col_all_correct_rate,
                **extra,
            }

    def _unpack_for_training(self, mo):
        """
        Normalize model outputs into a consistent 4-tuple used by training routines.
        
        Parameters:
            mo: The raw model output. May be a single tensor (y_hat) or a tuple with various shapes:
                - A 3-tuple (y_hat, psi_list, attn_probs) is unpacked into those three values.
                - A 2-tuple where the second element is a dict is returned as (y_hat, None, None, (y_hat, dict)).
                - Any other tuple is treated as (first_element_as_y_hat, None, None, original_tuple).
                - A non-tuple value is treated as y_hat with the remaining values set to None and packed_out equal to mo.
        
        Returns:
            A 4-tuple (y_hat, psi_list, attn_probs, packed_out) where:
            - y_hat: primary model outputs (logits/predictions).
            - psi_list: auxiliary outputs (may be None if not provided).
            - attn_probs: attention probabilities (may be None if not provided).
            - packed_out: the original model output structure or a constructed tuple preserving the original content.
        """
        if isinstance(mo, tuple):
            if len(mo) == 3:
                return mo[0], mo[1], mo[2], mo
            elif len(mo) == 2 and isinstance(mo[1], dict):
                return mo[0], None, None, (mo[0], mo[1])
            else:
                return mo[0], None, None, mo
        else:
            return mo, None, None, mo

    def _unpack_batch(self, batch):
        """Unpack batch for transformer or H-Net models and store self.batch for callbacks."""
        model_name = getattr(self.config.model, "model_name", "transformer")
        aug_grid = None
        if model_name == "hnet2d1d_t3_cs":
            if len(batch) == 3:
                src, tgt, aug_grid = batch
                ctx_input, ctx_output, task_ids = None, None, None
            elif len(batch) == 5:
                src, tgt, ctx_input, ctx_output, task_ids = batch
                aug_grid = None
            elif len(batch) == 6:
                src, tgt, ctx_input, ctx_output, task_ids, aug_grid = batch
            else:
                raise ValueError(f"hnet2d1d_t3_cs expects a batch of length 3, 5 or 6, but got {len(batch)}")
            # Preserve the original batch for callbacks
            self.batch = batch
        else:
            if len(batch) == 6:
                src, tgt, ctx_input, ctx_output, task_ids, aug_grid = batch
            elif len(batch) == 5:
                src, tgt, ctx_input, ctx_output, task_ids = batch
                aug_grid = None
            else:
                raise ValueError(f"Transformer model expects a batch of length 5 or 6, but got {len(batch)}")

            # Squeeze possible extra batch dimension of size 1
            if src.dim() > 3 and src.shape[0] == 1:
                src, tgt, ctx_input, ctx_output = [t.squeeze(0) for t in [src, tgt, ctx_input, ctx_output]]
                # Keep shapes aligned for downstream callbacks/optimisers
                try:
                    if torch.is_tensor(task_ids) and task_ids.dim() > 0 and task_ids.shape[0] == 1:
                        task_ids = task_ids.squeeze(0)
                except Exception:
                    pass
                try:
                    if torch.is_tensor(aug_grid) and aug_grid.dim() > 0 and aug_grid.shape[0] == 1:
                        aug_grid = aug_grid.squeeze(0)
                except Exception:
                    pass

            # Store batch for callbacks after potential squeezing
            if aug_grid is not None:
                self.batch = (src, tgt, ctx_input, ctx_output, task_ids, aug_grid)
            else:
                self.batch = (src, tgt, ctx_input, ctx_output, task_ids)

        return src, tgt, ctx_input, ctx_output, task_ids, aug_grid

    def _forward_with_rdrop(self, src, tgt, ctx_input, ctx_output, aug_grid, b_override=None):
        """
        Perform two stochastic forward passes (optionally using a shared dropout mask) and compute the R-Drop training loss.
        
        This runs two model forwards with the same inputs, computes the average cross-entropy loss across the two outputs, optionally adds Dice loss (computed on the first pass), computes a masked symmetric KL divergence between the two output distributions, and returns the first pass outputs along with the combined loss scaled by the configured R-Drop weight.
        
        Returns:
            y_hat1 (Tensor): Logits from the first forward pass.
            psi_list1 (Any): Auxiliary model outputs (e.g., per-layer features) unpacked from the first pass.
            attn_probs1 (Any): Attention probability tensors unpacked from the first pass.
            loss (Tensor): Combined training loss = 0.5*(CE1 + CE2) + rdrop_alpha * KL, with Dice loss added if enabled.
        """
        # Prepare deterministic seed across ranks when share-mask is requested
        if self.rdrop_share_mask:
            try:
                import torch.distributed as dist
                ddp_ready = dist.is_available() and dist.is_initialized()
            except Exception:
                ddp_ready = False
            if ddp_ready:
                # Rank-0 samples and broadcasts without touching global torch RNG
                seed_tensor = torch.empty((), dtype=torch.int64, device=tgt.device)
                if dist.get_rank() == 0:
                    raw = int.from_bytes(os.urandom(8), byteorder="little") % (2**31)
                    seed_tensor.fill_(raw)
                dist.broadcast(seed_tensor, src=0)
                shared_seed = int(seed_tensor.item())
            else:
                shared_seed = int(int.from_bytes(os.urandom(8), byteorder="little") % (2**31))
        else:
            shared_seed = None

        # First pass (optionally under forked RNG scope)
        if self.rdrop_share_mask:
            with torch.random.fork_rng():
                torch.manual_seed(shared_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(shared_seed)
                model_out1 = self(src, tgt, ctx_input, ctx_output, aug_grid=aug_grid, loop_b_override=b_override)
        else:
            model_out1 = self(src, tgt, ctx_input, ctx_output, aug_grid=aug_grid, loop_b_override=b_override)

        # Second pass (reuse the same seed when share-mask)
        if self.rdrop_share_mask:
            with torch.random.fork_rng():
                torch.manual_seed(shared_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(shared_seed)
                model_out2 = self(src, tgt, ctx_input, ctx_output, aug_grid=aug_grid, loop_b_override=b_override)
        else:
            model_out2 = self(src, tgt, ctx_input, ctx_output, aug_grid=aug_grid, loop_b_override=b_override)

        y_hat1, psi_list1, attn_probs1, _ = self._unpack_for_training(model_out1)
        y_hat2, _, _, _ = self._unpack_for_training(model_out2)

        # Cross-entropy losses (averaged)
        ce_loss1 = self._compute_loss(y_hat1, tgt, src)
        ce_loss2 = self._compute_loss(y_hat2, tgt, src)
        ce_loss = 0.5 * (ce_loss1 + ce_loss2)

        # Optional Dice loss (computed on first pass for efficiency)
        dice_loss_val = None
        if self.use_dice_loss:
            from jarc_reactor.losses.dice import dice_loss as _dice
            dice_loss_val = _dice(y_hat1.detach(), tgt, pad_token_id=self.pad_token_id)
            ce_loss = ce_loss + self.dice_weight * dice_loss_val

        # Masked symmetric KL divergence between the two output distributions
        valid_mask = (tgt.reshape(-1).long() != self.pad_token_id)
        # Device-aware autocast to avoid CUDA warnings on MPS/CPU
        amp_enabled = bool(getattr(self.config.model, "amp_enabled", False))
        dev_type = src.device.type if hasattr(src, "device") else (
            "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        autocast_ctx = (
            torch.amp.autocast(dev_type)
            if amp_enabled and dev_type in ("cuda", "mps")
            else contextlib.nullcontext()
        )
        with autocast_ctx:
            kl_loss = self._compute_rdrop_kl(y_hat1, y_hat2, tgt)

        loss = ce_loss + float(self.rdrop_alpha) * kl_loss
        self.log('train_rdrop_kl', kl_loss.detach(), prog_bar=True)
        if dice_loss_val is not None:
            self.log('train_dice_loss', dice_loss_val.detach(), prog_bar=True)
        return y_hat1, psi_list1, attn_probs1, loss

    def _forward_with_rdrop_concat(self, src, tgt, ctx_input, ctx_output, aug_grid, b_override=None):
        """Single forward on concatenated batch [x; x] then split to compute RD losses."""
        B = src.size(0)
        # Duplicate along batch dimension
        src2 = torch.cat([src, src], dim=0)
        tgt2 = torch.cat([tgt, tgt], dim=0)
        ctx_in2 = torch.cat([ctx_input, ctx_input], dim=0) if ctx_input is not None else None
        ctx_out2 = torch.cat([ctx_output, ctx_output], dim=0) if ctx_output is not None else None
        aug2 = torch.cat([aug_grid, aug_grid], dim=0) if aug_grid is not None and torch.is_tensor(aug_grid) else aug_grid

        model_out = self(src2, tgt2, ctx_in2, ctx_out2, aug_grid=aug2, loop_b_override=b_override)
        y_hat_all, psi_list_all, attn_probs_all, _ = self._unpack_for_training(model_out)

        # Split into two views
        y_hat1 = y_hat_all[:B]
        y_hat2 = y_hat_all[B:]

        # Attempt to split aux as well if present
        psi_list1 = None
        if isinstance(psi_list_all, list):
            psi_list1 = [t[:B] for t in psi_list_all]
        attn_probs1 = None
        if isinstance(attn_probs_all, torch.Tensor):
            attn_probs1 = attn_probs_all[:B]

        # CE averaged
        ce_loss1 = self._compute_loss(y_hat1, tgt)
        ce_loss2 = self._compute_loss(y_hat2, tgt)
        ce_loss = 0.5 * (ce_loss1 + ce_loss2)

        # Optional Dice on first half
        dice_loss_val = None
        if self.use_dice_loss:
            from jarc_reactor.losses.dice import dice_loss as _dice
            dice_loss_val = _dice(y_hat1.detach(), tgt, pad_token_id=self.pad_token_id)
            ce_loss = ce_loss + self.dice_weight * dice_loss_val

        # Masked symmetric KL
        valid_mask = (tgt.reshape(-1).long() != self.pad_token_id)
        # Device-aware autocast to avoid CUDA warnings on MPS/CPU
        amp_enabled = bool(getattr(self.config.model, "amp_enabled", False))
        dev_type = src.device.type if hasattr(src, "device") else (
            "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        autocast_ctx = (
            torch.amp.autocast(dev_type)
            if amp_enabled and dev_type in ("cuda", "mps")
            else contextlib.nullcontext()
        )
        with autocast_ctx:
            kl_loss = self._compute_rdrop_kl(y_hat1, y_hat2, tgt)

        loss = ce_loss + float(self.rdrop_alpha) * kl_loss
        self.log('train_rdrop_kl', kl_loss.detach(), prog_bar=True)
        if dice_loss_val is not None:
            self.log('train_dice_loss', dice_loss_val.detach(), prog_bar=True)
        return y_hat1, psi_list1, attn_probs1, loss

    def _forward_standard_and_aux(self, src, tgt, ctx_input, ctx_output, aug_grid, task_ids=None, loop_b_override=None):
        """
        Run a standard forward pass, compute the primary loss (cross-entropy or expected-cost) and aggregate model-provided auxiliary and regularization penalties.
        
        Parameters:
        	src: Input source tensor (model-specific).
        	tgt: Target tensor used for loss computation.
        	ctx_input: Optional context input tensor supplied to the model.
        	ctx_output: Optional context output tensor supplied to the model.
        	aug_grid: Optional augmentation grid passed through to the model and some loss terms.
        	task_ids (optional): Tensor of per-sample task identifiers used for task-specific diagnostics.
        	loop_b_override (optional): Override value for looped/iterative models' internal loop parameter.
        
        Returns:
        	y_hat: Model logits tensor for the primary prediction head.
        	psi_list: List or structure of intermediate features/auxiliary outputs produced by the model (may be None).
        	attn_probs: Attention probability tensors produced by the model (may be None).
        	loss: Scalar tensor representing the aggregated training loss (primary loss plus any auxiliary and regularization penalties).
        """
        model_out = self(src, tgt, ctx_input, ctx_output, aug_grid=aug_grid, loop_b_override=loop_b_override)
        y_hat, psi_list, attn_probs, packed_out = self._unpack_for_training(model_out)

        # ---------------- Primary loss: CE (default) or Expected-Cost (Phase 2) ----------------
        loss_mode = str(getattr(self.config.model, "loss_mode", "ce"))
        ec_cfg = OmegaConf.select(self.config.model, "loss.expected_cost", default=None)
        use_expected_cost = (loss_mode == "expected_cost") and (ec_cfg is not None) and bool(getattr(ec_cfg, "enabled", True))
        if use_expected_cost:
            loss = self._expected_cost_scalar(y_hat, tgt, aug_grid)
            self.log("train_expected_cost", loss.detach(), prog_bar=False)
        else:
            # Scheduled sampling (two-pass surrogate) when enabled
            try:
                ss_enabled = bool(OmegaConf.select(self.config, 'training.scheduled_sampling.enabled', default=False))
            except Exception:
                ss_enabled = False
            y_hat_for_loss = y_hat
            if ss_enabled:
                # Check warmup period: use 100% teacher forcing if still in warmup epochs
                try:
                    warmup_epochs = int(OmegaConf.select(self.config, 'training.scheduled_sampling.teacher_forcing_warmup_epochs', default=0))
                except Exception:
                    warmup_epochs = 0
                
                if self.current_epoch < warmup_epochs:
                    # Still in warmup: force p=0 (100% teacher forcing, no scheduled sampling)
                    p = 0.0
                else:
                    # Post-warmup: compute probability p from schedule string
                    try:
                        prob_str = str(OmegaConf.select(self.config, 'training.scheduled_sampling.probability', default='const(0.0)'))
                    except Exception:
                        prob_str = 'const(0.0)'
                    p = self._ss_prob(prob_str)
                if p > 0.0:
                    with torch.no_grad():
                        pred_teacher = torch.argmax(y_hat, dim=-1)  # [B,H,W]
                        mask = torch.rand_like(tgt.float()) < float(p)
                        mask = mask & (tgt != self.pad_token_id)
                        tgt_ss = torch.where(mask, pred_teacher, tgt)
                    # Second forward with mixed inputs
                    mo_ss = self(src, tgt_ss, ctx_input, ctx_output, aug_grid=aug_grid, loop_b_override=loop_b_override)
                    y_hat_ss, _, _, _ = self._unpack_for_training(mo_ss)
                    y_hat_for_loss = y_hat_ss
            loss = self._compute_loss(y_hat_for_loss, tgt, src)

        # ---------------- Optional refinement auxiliary losses (iterative models) ----------------
        base_w = float(getattr(self.config.training, 'refinement_aux_weight', 0.0))
        logits_hist = None
        if isinstance(packed_out, tuple) and len(packed_out) == 2 and isinstance(packed_out[1], dict):
            logits_hist = packed_out[1].get('logits_history', None)

        # Per-round auxiliary CE removed (was specific to a deprecated model path)

        # -- (2) Optional knowledge-distillation across rounds (student = early, teacher = final) --
        kd_w = float(getattr(self.config.training, 'kd_weight', 0.0))
        if kd_w > 0 and logits_hist and isinstance(logits_hist, list):
            tau = float(getattr(self.config.training, 'kd_tau', 1.0))
            kd_loss = 0.0
            teacher_flat_full = y_hat.reshape(-1, y_hat.size(-1)).detach()
            tgt_flat = tgt.reshape(-1)
            teacher_flat = teacher_flat_full
            for i, l in enumerate(logits_hist[:-1]):
                w = (0.5 ** i)  # geometric decay mirrors CE aux default
                student_flat = l.reshape(-1, l.size(-1))
                kd_term = F.kl_div(
                    F.log_softmax(student_flat / tau, dim=-1),
                    F.softmax(teacher_flat / tau, dim=-1),
                    reduction='batchmean',
                ) * (tau * tau)
                kd_loss = kd_loss + w * kd_term
            if isinstance(kd_loss, torch.Tensor) and kd_loss.requires_grad:
                scaled_kd = kd_w * kd_loss
                loss = loss + scaled_kd
                self.log('train_kd_loss', kd_loss.detach(), prog_bar=True)

        # Cleanup logits_history to avoid leaking large lists further downstream
        if isinstance(packed_out, tuple) and len(packed_out) == 2 and isinstance(packed_out[1], dict):
            packed_out[1].pop('logits_history', None)

        # Aggregate model-provided auxiliary losses
        if isinstance(packed_out, tuple) and len(packed_out) == 2 and isinstance(packed_out[1], dict):
            aux_dict = packed_out[1]
            aux_cfg = getattr(self.config.model, "hnet", {}) if hasattr(self.config.model, "hnet") else {}

            # Shape contrastive requires special handling
            if "shape_contrastive_tensors" in aux_dict and aux_dict["shape_contrastive_tensors"] is not None:
                z1, z2 = aux_dict.pop("shape_contrastive_tensors")
                mode = getattr(aux_cfg, "info_nce_mode", "symmetric")
                weight = float(getattr(aux_cfg, "alpha_shape_ctr", 0.0))
                if weight > 0:
                    temp = float(getattr(aux_cfg, "shape_ctr_tau", 0.07))
                    contrastive_loss = info_nce(z1, z2, temperature=temp, mode=mode)
                    loss = loss + weight * contrastive_loss
                    self.log("train_shape_contrastive_loss", contrastive_loss.detach(), prog_bar=True)

            for name, val in aux_dict.items():
                weight = float(getattr(aux_cfg, f"alpha_{name}", 1.0)) if isinstance(aux_cfg, dict) else float(getattr(aux_cfg, f"alpha_{name}", 1.0))
                if weight > 0 and val is not None:
                    # Skip non-numeric auxiliary entries (e.g., lists)
                    if not isinstance(val, (torch.Tensor, int, float)):
                        continue
                    if isinstance(val, torch.Tensor) and not torch.isfinite(val).all():
                        with torch.no_grad():
                            n_nan = torch.isnan(val).sum().item()
                            n_inf = torch.isinf(val).sum().item()
                            logger.warning(f"Non-finite aux '{name}' in training: NaN={n_nan}, Inf={n_inf}. Replacing with 0.")
                        val = torch.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
                    loss = loss + weight * val
                    self.log(f"train_{name}_loss", (val.detach() if isinstance(val, torch.Tensor) else torch.tensor(val)), prog_bar=True)

        # ---------------- Ponder-phi per-site aux losses (Phase 1 placeholder) ----------------
        # Sum any aux_loss() exposed by PonderPhiNoRAActivation modules. In Phase 1 this returns 0.0.
        try:
            pphi_aux = None
            for m in self.model.modules():
                if isinstance(m, PonderPhiNoRAActivation) and hasattr(m, 'aux_loss'):
                    val = m.aux_loss()
                    if isinstance(val, (int, float)):
                        val = torch.tensor(float(val), device=loss.device)
                    if isinstance(val, torch.Tensor):
                        pphi_aux = (val if pphi_aux is None else (pphi_aux + val))
            if isinstance(pphi_aux, torch.Tensor):
                if not torch.isfinite(pphi_aux).all():
                    pphi_aux = torch.nan_to_num(pphi_aux, nan=0.0, posinf=0.0, neginf=0.0)
                loss = loss + pphi_aux
                # Lightweight log (will be zero in Phase 1)
                self.log('train_ponder_phi_aux', pphi_aux.detach(), prog_bar=False)
        except Exception:
            pass

        # ---------------- Rule-consistency auxiliary loss ----------------
        lambda_rule = float(getattr(self.config.training, 'lambda_rule', 0.0))
        rule_vec = getattr(self.model, 'last_rule_vec', None)
        if lambda_rule > 0.0 and rule_vec is not None:
            centred = rule_vec - rule_vec.mean(0, keepdim=True)
            rule_loss_val = centred.pow(2).mean()
            loss = loss + lambda_rule * rule_loss_val
            self.log('train_rule_loss', rule_loss_val.detach(), prog_bar=False)
            # Rule-vector similarity metrics
            if task_ids is not None:
                B = rule_vec.size(0)
                cos = F.cosine_similarity(rule_vec.unsqueeze(1), rule_vec.unsqueeze(0), dim=-1)  # [B,B]
                same = task_ids.unsqueeze(1).eq(task_ids.unsqueeze(0))
                eye = torch.eye(B, dtype=torch.bool, device=rule_vec.device)
                same_offdiag = same & ~eye
                diff = ~same
                if same_offdiag.any():
                    self.log('rule_sim_same', cos[same_offdiag].mean().detach(), prog_bar=False)
                if diff.any():
                    self.log('rule_sim_diff', cos[diff].mean().detach(), prog_bar=True)

        # Attention entropy over output-grid attention weights
        attn_w = getattr(self.model, 'last_extra', {}).get('grid_attn') if hasattr(self.model, 'last_extra') else None
        if attn_w is not None:
            ent = -(attn_w * (attn_w + 1e-9).log()).sum(-1).mean()
            self.log('attn_entropy', ent.detach(), prog_bar=False)

        # ---------------- Regularisation penalties ----------------
        if any(p > 0.0 for p in [self.lambda_conf, self.lambda_cov, self.lambda_kl]):
            probs = F.softmax(y_hat, dim=-1)  # [B, H, W, C]

            # ---- Confidence entropy penalty ----
            if self.lambda_conf > 0.0:
                entropy = -(probs * (probs + 1e-9).log()).sum(dim=-1)  # [B, H, W]
                conf_pen = -entropy.mean()  # Negative entropy (higher when predictions are confident)
                loss = loss + self.lambda_conf * conf_pen
                # Renamed for clarity: negative entropy of prediction distribution
                self.log("train_neg_entropy", conf_pen.detach(), prog_bar=True)

            # ---- PAD token coverage penalty ----
            if self.lambda_cov > 0.0:
                pad_probs = probs[..., self.pad_token_id]  # [B, H, W]
                cov_pen = pad_probs.mean()
                loss = loss + self.lambda_cov * cov_pen
                self.log("train_padcov_pen", cov_pen.detach(), prog_bar=False)

            # ---- Histogram KL divergence penalty ----
            if self.lambda_kl > 0.0 and self.dataset_prior is not None:
                # Move prior to current device lazily
                prior = self.dataset_prior.to(probs.device)
                probs_flat = probs.reshape(-1, probs.size(-1))  # [N, C]
                hist_pred = probs_flat.mean(dim=0)  # [C]
                # Exclude PAD token and renormalise
                hist_pred[self.pad_token_id] = 0.0
                hist_sum = hist_pred.sum() + 1e-9
                hist_pred = hist_pred / hist_sum
                # Small epsilon to avoid log(0)
                eps = 1e-9
                kl_vec = hist_pred * ((hist_pred + eps) / (prior + eps)).log()
                kl_pen = kl_vec.sum()
                loss = loss + self.lambda_kl * kl_pen
                self.log("train_histkl_pen", kl_pen.detach(), prog_bar=True)

        return y_hat, psi_list, attn_probs, loss

    def _ss_prob(self, schedule: str) -> float:
        """
        Parse a scheduled-sampling probability from a schedule string.
        
        Supported formats:
        - "linear(start=START, end=END, steps=STEPS)" — linearly interpolates between START and END based on trainer.global_step (clamped to [0,1]).
        - "const(V)" — returns the constant value V.
        - A plain numeric string (e.g., "0.1").
        
        Uses self.trainer.global_step when present; returns 0.0 on parse errors.
        
        Parameters:
            schedule (str): Schedule string specifying how the probability should be produced.
        
        Returns:
            float: Parsed probability value (or 0.0 if the schedule cannot be parsed).
        """
        try:
            s = schedule.strip()
            gs = 0
            try:
                gs = int(getattr(getattr(self, 'trainer', None), 'global_step', 0) or 0)
            except Exception:
                gs = 0
            m = re.match(r"linear\(\s*start=([0-9eE+\-.]+)\s*,\s*end=([0-9eE+\-.]+)\s*,\s*steps=([0-9eE+\-.]+)\s*\)", s)
            if m:
                start = float(m.group(1)); end = float(m.group(2)); steps = max(1.0, float(m.group(3)))
                frac = max(0.0, min(1.0, gs / steps))
                return float(start + (end - start) * frac)
            m = re.match(r"const\(\s*([0-9eE+\-.]+)\s*\)", s)
            if m:
                return float(m.group(1))
            # Fallback numeric
            try:
                return float(s)
            except Exception:
                return 0.0
        except Exception:
            return 0.0

    def _apply_regularisers_and_hypergrad(self, loss, y_hat, psi_list, attn_probs):
        """
        Apply configured regularisers to the provided loss, optionally perform a hyper-gradient update, and emit related training logs.
        
        This applies Functional Risk Minimisation (FRM), head-level CEBR, gauge loss on psi embeddings, and legacy token-level CEBR according to trainer configuration. If a regulariser hyper-gradient controller (reg_hg) is enabled and any regulariser parameters require gradients, a hyper-gradient step may be executed which can update learnable regulariser parameters.
        
        Returns:
        	updated_loss (torch.Tensor): The loss tensor augmented with the configured regularisation penalties.
        """
        unweighted_ve, unweighted_orc = None, None

        # Functional Risk Minimisation
        if self.frm_enabled:
            feats = getattr(self, '_frm_last_hidden', None)
            loss, frm_reg = self.frm(loss, logits=y_hat, features=feats, model=self.model)
            self.log("train_frm_reg", frm_reg.detach(), prog_bar=True)
            # Log effective lambda after schedule
            try:
                eff = getattr(self.frm, "_last_lambda_effective", None)
                if eff is not None:
                    self.log("train_frm_lambda", eff, prog_bar=False)
            except Exception:
                pass

        # Head-level CEBR
        if self.head_cebr is not None and attn_probs is not None:
            cebr_head_loss, head_logs = self.head_cebr(attn_probs, self._ve_alpha_h, self._orc_beta_h)
            loss = loss + cebr_head_loss
            self.log("cebr_head_loss", cebr_head_loss, prog_bar=True)
            unweighted_ve = head_logs.get("ve_unweighted")
            unweighted_orc = head_logs.get("orc_unweighted")

        # Gauge loss on ψ embeddings
        if psi_list is not None and self.gauge_alpha.item() > 0:
            g_loss = gauge_loss(psi_list)
            loss = loss + self.gauge_alpha * g_loss
            self.log("gauge_loss", g_loss, prog_bar=True)

        # Legacy token-level CEBR
        if attn_probs is not None and self.cebr_alpha.item() > 0:
            loss = cebr_loss(loss, attn_probs, alpha=self.cebr_alpha, beta=self.cebr_beta, gamma=self.cebr_gamma)

        # Hyper-gradient step for learnable regularisers
        learnable_regs = [getattr(self, n, None) for n in ["gauge_alpha", "cebr_alpha", "cebr_beta", "cebr_gamma"]]
        if self.reg_hg is not None and any(getattr(p, "requires_grad", False) for p in learnable_regs):
            reg_vals = {}
            if psi_list is not None:
                reg_vals["gauge_alpha"] = gauge_loss(psi_list).detach()
            if attn_probs is not None:
                if unweighted_ve is not None:
                    reg_vals["ve_alpha_h"] = unweighted_ve.detach()
                if unweighted_orc is not None:
                    reg_vals["orc_beta_h"] = unweighted_orc.detach()

                attn_var = attn_probs.var(dim=-1).mean(dim=-1)
                ve_loss_scalar = F.mse_loss(attn_var, attn_var.new_full(attn_var.shape, 1.0))
                off_diag = attn_probs - torch.diagonal(attn_probs, dim1=-2, dim2=-1).unsqueeze(-1)
                orc_loss_scalar = off_diag.abs().mean()
                entropy = -(attn_probs * (attn_probs + 1e-9).log()).sum(dim=-1).mean()
                reg_vals.update({
                    "cebr_alpha": ve_loss_scalar.detach(),
                    "cebr_beta": orc_loss_scalar.detach(),
                    "cebr_gamma": entropy.detach(),
                })

            reg_vals = {k: v for k, v in reg_vals.items() if v is not None}
            if reg_vals and self.reg_hg is not None:
                self.reg_hg.step(unweighted_ve, unweighted_orc)

        return loss

    def _compute_rdrop_kl(self, y_hat1: torch.Tensor, y_hat2: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Compute R-Drop symmetric KL with stability and semantics options.

        - Sanitizes logits (nan_to_num + clamp) to avoid NaNs/Infs.
        - Optionally applies insertion-gating to align CE/KL semantics.
        - Optionally computes in fp32 regardless of outer autocast.
        - Optionally clips per-token KL before reduction.
        """
        # Defensive: log if non-finite prior to sanitization
        try:
            if (not torch.isfinite(y_hat1).all()) or (not torch.isfinite(y_hat2).all()):
                logger.warning("Non-finite logits detected before R-Drop KL; applying sanitization.")
                try:
                    self._rdrop_stats.sanitize_count += 1
                except Exception:
                    pass
        except Exception:
            pass

        # Sanitize logits
        y1 = torch.nan_to_num(y_hat1, nan=0.0, posinf=1e4, neginf=-1e4)
        y2 = torch.nan_to_num(y_hat2, nan=0.0, posinf=1e4, neginf=-1e4)
        y1 = torch.clamp(y1, min=-10.0, max=10.0)
        y2 = torch.clamp(y2, min=-10.0, max=10.0)

        # Optional: align semantics by applying insertion gating
        try:
            if bool(OmegaConf.select(self.config, "rdrop.apply_insertion_gating_for_kl", default=False)):
                y1 = self._apply_insertion_gating(y1, tgt)
                y2 = self._apply_insertion_gating(y2, tgt)
        except Exception:
            pass

        # Optional: force fp32 compute inside an autocast-disabled region
        dev_type = y1.device.type if hasattr(y1, "device") else (
            "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        force_fp32 = bool(OmegaConf.select(self.config, "rdrop.force_fp32_kl", default=True))
        autocast_off = torch.amp.autocast(dev_type, enabled=False) if force_fp32 and dev_type in ("cuda", "mps") else contextlib.nullcontext()

        with autocast_off:
            y1f = y1.float() if force_fp32 else y1
            y2f = y2.float() if force_fp32 else y2

            # Track fp32 usage
            try:
                if force_fp32:
                    self._rdrop_stats.fp32_kl_steps += 1
            except Exception:
                pass

            clip_max = float(OmegaConf.select(self.config, "rdrop.kl_clip_max", default=0.0))
            if clip_max and clip_max > 0.0:
                # Compute per-token KL, count clips, clamp, then mean
                kl_tokens_raw = masked_symmetric_kl(y1f, y2f, valid_mask=None, reduction="none")
                try:
                    clipped_mask = kl_tokens_raw > clip_max
                    n_clipped = int(clipped_mask.sum().item())
                    if n_clipped > 0:
                        self._rdrop_stats.clip_tokens += n_clipped
                        self._rdrop_stats.clip_events += 1
                except Exception:
                    pass
                kl_tokens = torch.clamp(kl_tokens_raw, max=clip_max)
                kl_loss = kl_tokens.mean()
            else:
                # Default: include PAD positions in KL consistency (PAD must be learned)
                kl_loss = masked_symmetric_kl(y1f, y2f, valid_mask=None, reduction="mean")

        return kl_loss

    def on_train_end(self):  # Lightning hook
        # Log R-Drop telemetry (if any)
        """
        Perform end-of-training cleanup and record R-Drop telemetry.
        
        Logs aggregated R-Drop statistics (sanitize events, clipped tokens/events, and fp32 KL steps) if R-Drop is enabled and telemetry is present, then closes the trainer's per-instance file handler.
        """
        try:
            stats = getattr(self, "_rdrop_stats", None)
            if stats is not None and self.rdrop_enabled:
                self.log('rdrop_kl_sanitize_events', torch.tensor(float(stats.sanitize_count)))
                self.log('rdrop_kl_clip_tokens', torch.tensor(float(stats.clip_tokens)))
                self.log('rdrop_kl_clip_events', torch.tensor(float(stats.clip_events)))
                self.log('rdrop_kl_fp32_steps', torch.tensor(float(stats.fp32_kl_steps)))
        except Exception:
            pass
        # Existing cleanup
        self._cleanup_file_handler()

    # --------------------------- Checkpoint I/O for SWA ---------------------------
    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """
        Save SWA controller state into the checkpoint if SWA is enabled.
        
        If the trainer has SWA enabled and a configured SWA controller, stores its
        state dict into checkpoint['swa_state']. Any exceptions raised during this
        operation are suppressed to avoid interfering with checkpoint saving.
        
        Parameters:
            checkpoint (dict): Mutable checkpoint dictionary to be updated in-place.
        """
        try:
            if getattr(self, 'swa_enabled', False) and self.swa_ctrl is not None:
                checkpoint['swa_state'] = self.swa_ctrl.state_dict()
        except Exception:
            pass

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """
        Load SWA controller state from a checkpoint if SWA is enabled for this trainer.
        
        If the checkpoint contains an 'swa_state' key and this instance has swa_enabled True and a non-None swa_ctrl,
        the controller's state is loaded via swa_ctrl.load_state_dict(state). Any exceptions raised during loading
        are caught and ignored.
         
        Parameters:
            checkpoint (dict): Checkpoint dictionary potentially containing an 'swa_state' entry.
        """
        try:
            state = checkpoint.get('swa_state', None)
            if state is not None and getattr(self, 'swa_enabled', False) and self.swa_ctrl is not None:
                self.swa_ctrl.load_state_dict(state)
        except Exception:
            pass

    def configure_optimizers(self):
        """
        Construct the training optimizer (with optional PEFT-aware parameter groups, SAM or HyperGradient wrapping) and an optional learning-rate scheduler.
        
        Builds parameter groups for base parameters and PEFT variants (NoRA, LoRA, Ponder-phi) applying per-group LR/weight-decay overrides when configured. Supports SAM (with perturb/exclude partitioning and PEFT integration), standard optimizers ('adam', 'adamw', 'sgd'), and HyperGradientWrapper-backed optimizers (names ending with '_hd' when available). If scheduling is enabled, returns either a cosine-annealing-with-restarts scheduler (resume-safe) or a ReduceLROnPlateau fallback.
        
        Returns:
            dict or torch.optim.Optimizer: If scheduling is enabled, returns a dict of the form
              {
                "optimizer": optimizer,
                "lr_scheduler": {
                  "scheduler": scheduler,
                  "interval": "epoch" or as required,
                  "frequency": 1,
                  "monitor": <monitor name>,
                  "name": <scheduler_name>
                }
              }
            Otherwise returns either an optimizer instance or a dict {"optimizer": optimizer} consistent with Lightning's expectations.
        
        Raises:
            ValueError: if an unsupported optimizer name or unsupported SAM base optimizer is requested.
        """
        # Build optimiser (possibly hypergradient wrapper)
        try:
            from jarc_reactor.optimizers.hypergrad import HyperGradientWrapper
        except Exception:
            HyperGradientWrapper = None  # type: ignore

        opt_cfg = getattr(self.config, 'optimizer', None)
        if opt_cfg is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=float(getattr(self, 'learning_rate', 1e-3)))
            self._optimizer_ref = optimizer
            return optimizer

        name = str(getattr(opt_cfg, 'name', 'adam'))

        # -----------------------------
        # Phase 2: PEFT param grouping
        # -----------------------------
        # Collect NoRA/LoRA params for per-group LR/WD overrides
        def _collect_peft_params(model: nn.Module):
            """
            Collect trainable PEFT adapter parameters from a model, grouped by adapter type.
            
            Scans all submodules and gathers parameters with requires_grad for:
            - NoRAActivation modules (returned first),
            - LoRALinear modules (returned second),
            - PonderPhiNoRAActivation slice parameters (returned third),
            - PonderPhiNoRAActivation halting-head parameters (returned fourth).
            
            Parameters:
                model (nn.Module): The module to scan for PEFT adapter submodules.
            
            Returns:
                tuple[list[torch.nn.Parameter], list[torch.nn.Parameter], list[torch.nn.Parameter], list[torch.nn.Parameter]]:
                    A 4-tuple of lists: (nora_params, lora_params, ponder_slice_params, ponder_halting_params).
                    Each list contains the trainable `torch.nn.Parameter` objects found for that adapter group.
            """
            nora_params: list[torch.nn.Parameter] = []
            lora_params: list[torch.nn.Parameter] = []
            ponder_slice_params: list[torch.nn.Parameter] = []
            ponder_halting_params: list[torch.nn.Parameter] = []
            for m in model.modules():
                if isinstance(m, NoRAActivation):
                    for p in m.parameters(recurse=True):
                        if p.requires_grad:
                            nora_params.append(p)
                if isinstance(m, LoRALinear):
                    for p in m.parameters(recurse=True):
                        if p.requires_grad:
                            lora_params.append(p)
                if isinstance(m, PonderPhiNoRAActivation):
                    # Separate groups for slices vs halting head
                    for p in m.slice_parameters():
                        if p is not None and getattr(p, 'requires_grad', False):
                            ponder_slice_params.append(p)
                    for p in m.halting_parameters():
                        if p is not None and getattr(p, 'requires_grad', False):
                            ponder_halting_params.append(p)
            return nora_params, lora_params, ponder_slice_params, ponder_halting_params

        nora_cfg = getattr(getattr(self.config, 'model', None), 'nora', None)
        try:
            nora_enabled = bool(getattr(nora_cfg, 'enabled', False)) if nora_cfg is not None else False
        except Exception:
            nora_enabled = False
        try:
            lora_enabled = bool(getattr(nora_cfg, 'nora_plus_lora', False)) if nora_cfg is not None else False
        except Exception:
            lora_enabled = False

        # Gather trainable params for PEFT
        nora_params, lora_params, ponder_slice_params, ponder_halting_params = _collect_peft_params(self.model)
        nora_ids = {id(p) for p in nora_params} if (nora_enabled and nora_params) else set()
        lora_ids = {id(p) for p in lora_params} if (lora_enabled and lora_params) else set()
        pphi_slice_ids = {id(p) for p in ponder_slice_params} if ponder_slice_params else set()
        pphi_halting_ids = {id(p) for p in ponder_halting_params} if ponder_halting_params else set()
        # Base params exclude PEFT params
        base_params = [
            p for p in self.parameters()
            if p.requires_grad and id(p) not in nora_ids and id(p) not in lora_ids and id(p) not in pphi_slice_ids and id(p) not in pphi_halting_ids
        ]

        # Per-group LR/WD
        base_lr = float(getattr(opt_cfg, 'lr', getattr(self, 'learning_rate', 1e-3)))
        # NoRA: tolerate None and non-scalars in per_layer_lr/weight_decay
        if nora_cfg is not None:
            try:
                _nora_lr_raw = getattr(nora_cfg, 'per_layer_lr', None)
                nora_lr = float(base_lr if _nora_lr_raw is None else _nora_lr_raw)
            except Exception:
                nora_lr = base_lr
            try:
                _nora_wd_raw = getattr(nora_cfg, 'weight_decay', 0.0)
                nora_wd = float(0.0 if _nora_wd_raw is None else _nora_wd_raw)
            except Exception:
                nora_wd = 0.0
        else:
            nora_lr = base_lr
            nora_wd = 0.0
        lora_cfg = getattr(nora_cfg, 'lora', {}) if (nora_cfg is not None) else {}
        try:
            lora_lr = float(getattr(lora_cfg, 'lr', base_lr))
        except Exception:
            # lora.lr might be absent or non-scalar in configs; fallback to base lr
            lora_lr = base_lr
        try:
            lora_wd = float(getattr(lora_cfg, 'weight_decay', 0.0))
        except Exception:
            lora_wd = 0.0
        # Master toggle to enable/disable SAM globally (support legacy model.sam_enabled and new optimizer.sam_enabled)
        _sam_raw = OmegaConf.select(self.config, 'optimizer.sam_enabled', default=None)
        if _sam_raw is None:
            sam_enabled = bool(OmegaConf.select(self.config, 'model.sam_enabled', default=True))
        else:
            sam_enabled = bool(_sam_raw)
        if name == 'sam' and sam_enabled:
            # Map base optimizer
            base_name = str(getattr(opt_cfg, 'base', 'adamw')).lower()
            if base_name == 'adamw':
                base_cls = torch.optim.AdamW
                base_kwargs = dict(
                    lr=float(opt_cfg.get('lr', getattr(self, 'learning_rate', 1e-3))),
                    betas=tuple(getattr(opt_cfg, 'base_betas', (0.9, 0.999))),
                    eps=float(getattr(opt_cfg, 'base_eps', 1e-8)),
                    weight_decay=float(getattr(opt_cfg, 'weight_decay', 0.01)),
                )
            elif base_name == 'adam':
                base_cls = torch.optim.Adam
                base_kwargs = dict(
                    lr=float(opt_cfg.get('lr', getattr(self, 'learning_rate', 1e-3))),
                    betas=tuple(getattr(opt_cfg, 'base_betas', (0.9, 0.999))),
                    eps=float(getattr(opt_cfg, 'base_eps', 1e-8)),
                    weight_decay=float(getattr(opt_cfg, 'weight_decay', 0.0)),
                )
            elif base_name == 'sgd':
                base_cls = torch.optim.SGD
                base_kwargs = dict(
                    lr=float(opt_cfg.get('lr', getattr(self, 'learning_rate', 1e-2))),
                    momentum=float(getattr(opt_cfg, 'momentum', 0.9)),
                    weight_decay=float(getattr(opt_cfg, 'weight_decay', 0.0)),
                )
            else:
                raise ValueError(f"SAM base optimizer '{base_name}' not supported")

            # Build param groups with perturb flags (exclude norm/bias optionally)
            exclude_norm_bias = bool(getattr(opt_cfg, 'exclude_norm_bias', True))
            perturb_params = []
            exclude_params = []
            for m in self.modules():
                is_norm = isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d))
                for n, p in m.named_parameters(recurse=False):
                    if not p.requires_grad:
                        continue
                    is_bias = (n == 'bias')
                    if exclude_norm_bias and (is_norm or is_bias):
                        exclude_params.append(p)
                    else:
                        perturb_params.append(p)
            # Fallback: if module traversal missed some, include remaining model parameters
            seen = set([id(p) for p in perturb_params] + [id(p) for p in exclude_params])
            for p in self.parameters():
                if p.requires_grad and id(p) not in seen:
                    perturb_params.append(p)

            rho = float(getattr(opt_cfg, 'rho', 0.05))
            p_norm = getattr(opt_cfg, 'p', 2)
            adaptive = bool(getattr(opt_cfg, 'adaptive', False))
            eps = float(getattr(opt_cfg, 'eps', 1e-12))
            linf_sign_variant = bool(getattr(opt_cfg, 'linf_sign_variant', False))
            adaptive_floor = float(getattr(opt_cfg, 'adaptive_floor', 0.0))

            # Optional: extend exclusion by name substrings (reassign from perturb to exclude)
            name_substrings = getattr(opt_cfg, 'exclude_name_substrings', []) or []
            if name_substrings:
                # Build sets for fast membership and mutation
                perturb_set = {id(p): p for p in perturb_params}
                exclude_set = {id(p): p for p in exclude_params}
                for n, p in self.named_parameters():
                    if not p.requires_grad:
                        continue
                    if any(s in n for s in name_substrings):
                        pid = id(p)
                        if pid in perturb_set:
                            # move from perturb to exclude
                            exclude_set[pid] = p
                            perturb_set.pop(pid, None)
                        else:
                            exclude_set[pid] = p
                perturb_params = list(perturb_set.values())
                exclude_params = list(exclude_set.values())

            # Remove PEFT params from default groups to avoid duplication
            if nora_ids:
                perturb_params = [p for p in perturb_params if id(p) not in nora_ids]
                exclude_params = [p for p in exclude_params if id(p) not in nora_ids]
            if lora_ids:
                perturb_params = [p for p in perturb_params if id(p) not in lora_ids]
                exclude_params = [p for p in exclude_params if id(p) not in lora_ids]
            if pphi_slice_ids:
                perturb_params = [p for p in perturb_params if id(p) not in pphi_slice_ids]
                exclude_params = [p for p in exclude_params if id(p) not in pphi_slice_ids]
            if pphi_halting_ids:
                perturb_params = [p for p in perturb_params if id(p) not in pphi_halting_ids]
                exclude_params = [p for p in exclude_params if id(p) not in pphi_halting_ids]

            param_groups = []
            if perturb_params:
                param_groups.append({'params': perturb_params, 'perturb': True, 'rho': rho, 'p': p_norm, 'adaptive': adaptive, 'eps': eps, 'linf_sign_variant': linf_sign_variant, 'adaptive_floor': adaptive_floor})
            if exclude_params:
                param_groups.append({'params': exclude_params, 'perturb': False, 'rho': rho, 'p': p_norm, 'adaptive': adaptive, 'eps': eps, 'linf_sign_variant': linf_sign_variant, 'adaptive_floor': adaptive_floor})
            # Add PEFT groups with LR/WD overrides (perturbable)
            if nora_enabled and nora_params:
                param_groups.append({'params': nora_params, 'perturb': True, 'rho': rho, 'p': p_norm, 'adaptive': adaptive, 'eps': eps, 'linf_sign_variant': linf_sign_variant, 'adaptive_floor': adaptive_floor, 'lr': nora_lr, 'weight_decay': nora_wd})
            if lora_enabled and lora_params:
                param_groups.append({'params': lora_params, 'perturb': True, 'rho': rho, 'p': p_norm, 'adaptive': adaptive, 'eps': eps, 'linf_sign_variant': linf_sign_variant, 'adaptive_floor': adaptive_floor, 'lr': lora_lr, 'weight_decay': lora_wd})
            # Ponder-phi: slices and halting head groups (use NoRA LR, zero WD; halting head smaller LR)
            if ponder_slice_params:
                param_groups.append({'params': ponder_slice_params, 'perturb': True, 'rho': rho, 'p': p_norm, 'adaptive': adaptive, 'eps': eps, 'linf_sign_variant': linf_sign_variant, 'adaptive_floor': adaptive_floor, 'lr': nora_lr, 'weight_decay': 0.0})
            if ponder_halting_params:
                param_groups.append({'params': ponder_halting_params, 'perturb': True, 'rho': rho, 'p': p_norm, 'adaptive': adaptive, 'eps': eps, 'linf_sign_variant': linf_sign_variant, 'adaptive_floor': adaptive_floor, 'lr': 0.5 * nora_lr, 'weight_decay': 0.0})

            optimizer = SAM(param_groups, base_cls, rho=rho, adaptive=adaptive, p=p_norm, eps=eps, linf_sign_variant=linf_sign_variant, adaptive_floor=adaptive_floor, **base_kwargs)
        elif name == 'sam' and not sam_enabled:
            # SAM requested but disabled via model.sam_enabled -> use the requested base optimizer directly
            base_name = str(getattr(opt_cfg, 'base', 'adamw')).lower()
            # Build param group list to support PEFT LR/WD overrides
            pg = []
            if base_params:
                pg.append({'params': base_params})
            if nora_enabled and nora_params:
                pg.append({'params': nora_params, 'lr': nora_lr, 'weight_decay': nora_wd})
            if lora_enabled and lora_params:
                pg.append({'params': lora_params, 'lr': lora_lr, 'weight_decay': lora_wd})
            if ponder_slice_params:
                pg.append({'params': ponder_slice_params, 'lr': nora_lr, 'weight_decay': 0.0})
            if ponder_halting_params:
                pg.append({'params': ponder_halting_params, 'lr': 0.5 * nora_lr, 'weight_decay': 0.0})
            if base_name == 'adamw':
                optimizer = torch.optim.AdamW(
                    pg,
                    lr=float(opt_cfg.get('lr', getattr(self, 'learning_rate', 1e-3))),
                    betas=tuple(getattr(opt_cfg, 'base_betas', (0.9, 0.999))),
                    eps=float(getattr(opt_cfg, 'base_eps', 1e-8)),
                    weight_decay=float(getattr(opt_cfg, 'weight_decay', 0.01)),
                )
            elif base_name == 'adam':
                optimizer = torch.optim.Adam(
                    pg,
                    lr=float(opt_cfg.get('lr', getattr(self, 'learning_rate', 1e-3))),
                    betas=tuple(getattr(opt_cfg, 'base_betas', (0.9, 0.999))),
                    eps=float(getattr(opt_cfg, 'base_eps', 1e-8)),
                    weight_decay=float(getattr(opt_cfg, 'weight_decay', 0.0)),
                )
            elif base_name == 'sgd':
                optimizer = torch.optim.SGD(
                    pg,
                    lr=float(opt_cfg.get('lr', getattr(self, 'learning_rate', 1e-2))),
                    momentum=float(getattr(opt_cfg, 'momentum', 0.9)),
                    weight_decay=float(getattr(opt_cfg, 'weight_decay', 0.0)),
                )
            else:
                raise ValueError(f"SAM base optimizer '{base_name}' not supported")
        elif name.endswith('_hd') and HyperGradientWrapper is not None:
            optimizer = HyperGradientWrapper.from_config(opt_cfg, self.parameters())
        else:
            # Fallback to simple construction with PEFT param groups
            pg = []
            if base_params:
                pg.append({'params': base_params})
            if nora_enabled and nora_params:
                pg.append({'params': nora_params, 'lr': nora_lr, 'weight_decay': nora_wd})
            if lora_enabled and lora_params:
                pg.append({'params': lora_params, 'lr': lora_lr, 'weight_decay': lora_wd})
            if name == 'adam':
                optimizer = torch.optim.Adam(
                    pg,
                    lr=opt_cfg.get('lr', 1e-3),
                    betas=getattr(opt_cfg, 'betas', (0.9, 0.999)),
                    eps=getattr(opt_cfg, 'eps', 1e-8),
                    weight_decay=getattr(opt_cfg, 'weight_decay', 0.0),
                )
            elif name == 'adamw':
                optimizer = torch.optim.AdamW(
                    pg,
                    lr=opt_cfg.get('lr', 1e-3),
                    betas=getattr(opt_cfg, 'betas', (0.9, 0.999)),
                    eps=getattr(opt_cfg, 'eps', 1e-8),
                    weight_decay=getattr(opt_cfg, 'weight_decay', 0.01),
                )
            elif name == 'sgd':
                optimizer = torch.optim.SGD(
                    pg,
                    lr=opt_cfg.get('lr', 1e-2),
                    momentum=getattr(opt_cfg, 'momentum', 0.0),
                )
            else:
                raise ValueError(f"Unknown optimiser {name}")

        # Keep a reference for LR logging
        self._optimizer_ref = optimizer

        # Optionally disable scheduler: always return unified dict format
        if not self.config.training.get('use_scheduler', True):
            return {"optimizer": optimizer}

        # Validate and normalize scheduler config
        sch = self._validate_and_normalize_scheduler_cfg()

        if sch['use_cosine_annealing']:
            # Resume-safe last_epoch handling (tests expect direct mapping)
            resume_epoch = getattr(self.config.training, 'resume_from_epoch', None)
            if resume_epoch is not None:
                last_epoch = max(int(resume_epoch) - 1, -1)
            else:
                last_epoch = int(sch.get('last_epoch', -1))
            # Ensure initial_lr keys exist when resuming
            if last_epoch != -1:
                try:
                    for group in optimizer.param_groups:
                        if 'initial_lr' not in group:
                            group['initial_lr'] = group.get('lr', 0.0)
                except Exception:
                    pass
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=sch['T_0'],
                T_mult=sch['T_mult'],
                eta_min=sch['eta_min'],
                last_epoch=last_epoch,
            )
            lr_dict = {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
                'monitor': sch['monitor'],  # ignored by this scheduler
                'name': 'cosine_annealing',
            }
            return {
                'optimizer': optimizer,
                'lr_scheduler': lr_dict,
            }

        # Fallback: ReduceLROnPlateau (requires monitor)
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': plateau,
                'monitor': sch['monitor'],
                'name': 'reduce_on_plateau',
            },
        }

    def _log_periodic_metrics(self, y_hat, tgt, batch_idx):
        """
        Log periodic training diagnostics and accuracy metrics every configured number of steps.
        
        Collects and logs cell- and grid-level accuracies, optional SAM optimizer diagnostics, calibration metrics (ECE/MCE and NLL) before and after an available calibrator, and any lightweight BAM diagnostics for the current training batch when the batch index falls on the configured logging interval.
        
        Parameters:
            y_hat (torch.Tensor): Model logits for the current batch (may be augmented-aware).
            tgt (torch.Tensor): Target token indices for the current batch.
            batch_idx (int): Current training batch index used to determine logging cadence.
        """
        log_interval = max(int(getattr(self.config.training, "log_every_n_steps", 50)), 1)
        if (batch_idx % log_interval) == 0 and y_hat is not None:
            # Apply insertion gating for metrics if configured for this phase
            yh_metrics = self._apply_insertion_gating(y_hat, tgt)
            # Compute accuracy with augmented-awareness (EOS-on-PAD correctness)
            accuracies = self._compute_accuracy_baseaware(yh_metrics.detach(), tgt)
            # Aggregate augmented logits to base for calibration/NLL
            yh_base = self._aggregated_base_logits(yh_metrics)
            self.log('train_cell_accuracy', accuracies['cell_accuracy'], prog_bar=True)
            self.log('train_grid_accuracy', accuracies['grid_accuracy'], prog_bar=True)
            # Surface additional accuracy diagnostics without crowding the progress bar
            for k, v in accuracies.items():
                if k in ('cell_accuracy', 'grid_accuracy'):
                    continue
                try:
                    self.log(f'train_{k}', v, prog_bar=False)
                except Exception:
                    pass
            # Optional: SAM diagnostics (grad norm, rho, etc.) when SAM optimizer is active
            try:
                opt = getattr(self, '_optimizer_ref', None)
                if isinstance(opt, SAM):
                    gn = getattr(opt, '_last_grad_norm', None)
                    if gn is not None:
                        self.log('train_sam_grad_norm', torch.tensor(float(gn), device=yh_metrics.device), prog_bar=False)
                    try:
                        rho_val = float(opt.param_groups[0].get('rho', 0.0))
                        self.log('train_sam_rho', torch.tensor(rho_val, device=yh_metrics.device), prog_bar=False)
                        try:
                            adaptive_flag = bool(opt.param_groups[0].get('adaptive', False))
                            p_val = opt.param_groups[0].get('p', 2)
                            try:
                                p_num = float(p_val)
                            except Exception:
                                p_num = float('inf') if str(p_val).lower() in ('inf', 'infinity') else 2.0
                            self.log('train_sam_adaptive', torch.tensor(1.0 if adaptive_flag else 0.0, device=yh_metrics.device), prog_bar=False)
                            self.log('train_sam_pnorm', torch.tensor(p_num, device=yh_metrics.device), prog_bar=False)
                        except Exception:
                            pass
                    except Exception:
                        pass
            except Exception:
                pass
            # Training-time calibration metrics (pre/post) on current batch
            if bool(getattr(self.config.training, 'calibration_on_train', True)):
                try:
                    num_bins = int(OmegaConf.select(self.calibration_cfg, 'num_bins', default=15)) if self.calibration_cfg is not None else 15
                    binning = str(OmegaConf.select(self.calibration_cfg, 'binning', default='equal_width')) if self.calibration_cfg is not None else 'equal_width'
                    log_mce = bool(OmegaConf.select(self.calibration_cfg, 'log_mce', default=False)) if self.calibration_cfg is not None else False
                    with torch.no_grad():
                        # ECE/MCE pre
                        probs_base = F.softmax(yh_base, dim=-1)
                        ece_pre, _ = compute_ece_from_probs_binned(probs_base, tgt, num_bins=num_bins, ignore_index=self.pad_token_id, binning=binning)
                        self.log('train_ece_pre', ece_pre, prog_bar=False)
                        if log_mce:
                            mce_pre = compute_mce_from_probs(probs_base, tgt, num_bins=num_bins, ignore_index=self.pad_token_id, binning=binning)
                            self.log('train_mce_pre', mce_pre, prog_bar=True)
                        # NLL pre (masked CE on raw logits)
                        y_flat = yh_base.reshape(-1, yh_base.size(-1))
                        t_flat = tgt.reshape(-1).long()
                        mask = (t_flat != self.pad_token_id)
                        if mask.any():
                            nll_pre = F.cross_entropy(y_flat[mask], t_flat[mask])
                            self.log('train_nll_pre', nll_pre, prog_bar=False)
                        # Post metrics only if a calibrator is available (e.g., loaded state)
                        if self._calibrator is not None:
                            y_cal = self._calibrator(yh_base)
                            probs_post = F.softmax(y_cal, dim=-1)
                            ece_post, _ = compute_ece_from_probs_binned(probs_post, tgt, num_bins=num_bins, ignore_index=self.pad_token_id, binning=binning)
                            self.log('train_ece_post', ece_post, prog_bar=False)
                            if log_mce:
                                mce_post = compute_mce_from_probs(probs_post, tgt, num_bins=num_bins, ignore_index=self.pad_token_id, binning=binning)
                                self.log('train_mce_post', mce_post, prog_bar=True)
                            y_cal_flat = y_cal.reshape(-1, y_cal.size(-1))
                            if mask.any():
                                nll_post = F.cross_entropy(y_cal_flat[mask], t_flat[mask])
                                self.log('train_nll_post', nll_post, prog_bar=False)
                except Exception as e:
                    logger.debug(f"Skipped train calibration metrics: {e}")
            # BAM-specific lightweight diagnostics (if available)
            try:
                self._log_bam_metrics()
            except Exception:
                pass

    # ----------------------- BAM diagnostics helpers -----------------------
    def _iter_bam_modules(self):
        """
        Iterate the model's encoder and decoder and yield BAMMultiheadAttention modules when present.
        
        Yields:
            BAMMultiheadAttention: Attention modules found in encoder/decoder layers. Only modules whose `bias` attribute is not None and whose `bias` exposes a `compute_bias` attribute are yielded.
        """
        model = getattr(self, 'model', None)
        if model is None:
            return
        # Encoder: either torch.nn.TransformerEncoder with .layers, or custom encoder_layers
        enc = getattr(model, 'encoder', None)
        if enc is not None and hasattr(enc, 'layers'):
            for layer in getattr(enc, 'layers', []):
                attn = getattr(layer, 'self_attn', None)
                if attn is not None and hasattr(attn, 'bias') and getattr(attn, 'bias') is not None and hasattr(attn.bias, 'compute_bias'):
                    yield attn
        enc_layers = getattr(model, 'encoder_layers', None)
        if enc_layers is not None:
            for layer in enc_layers:
                attn = getattr(layer, 'self_attn', None)
                if attn is not None and hasattr(attn, 'bias') and getattr(attn, 'bias') is not None and hasattr(attn.bias, 'compute_bias'):
                    yield attn
        # Decoder
        dec = getattr(model, 'decoder', None)
        if dec is not None and hasattr(dec, 'layers'):
            for layer in getattr(dec, 'layers', []):
                attn = getattr(layer, 'self_attn', None)
                if attn is not None and hasattr(attn, 'bias') and getattr(attn, 'bias') is not None and hasattr(attn.bias, 'compute_bias'):
                    yield attn

    @torch.no_grad()
    def _log_bam_metrics(self) -> None:
        """Compute and log lightweight MHA metrics from BAM diagnostics if enabled.

        Logs:
          - train_bam_attn_mean_dist: E[|j-i|] averaged over (B,H,L)
          - train_bam_bias_std: mean per-head std of B over (L,L)
        """
        # Take the first BAM module that has diagnostics stored
        last_attn = None
        last_bias = None
        for attn_mod in self._iter_bam_modules() or []:
            try:
                la = getattr(attn_mod, 'last_attn', None)
                lb = getattr(attn_mod, 'last_bias', None)
                if (la is not None and isinstance(la, torch.Tensor)) or (lb is not None and isinstance(lb, torch.Tensor)):
                    last_attn = la if la is not None else last_attn
                    last_bias = lb if lb is not None else last_bias
                    # Prefer a module that has both
                    if last_attn is not None and last_bias is not None:
                        break
            except Exception:
                continue

        # Attention mean distance
        if isinstance(last_attn, torch.Tensor) and last_attn.numel() > 0:
            try:
                attn = last_attn.to(torch.float32)  # [B,H,L,L]
                B, H, L, _ = attn.shape
                idx = torch.arange(L, device=attn.device, dtype=attn.dtype)
                r = (idx[None, :] - idx[:, None]).abs()  # [L,L]
                exp_dist = (attn * r).sum(dim=-1)  # [B,H,L]
                mean_dist = exp_dist.mean()
                self.log('train_bam_attn_mean_dist', mean_dist, prog_bar=True)
            except Exception:
                pass

        # Bias std per head
        if isinstance(last_bias, torch.Tensor) and last_bias.numel() > 0:
            try:
                Bmat = last_bias.to(torch.float32)  # [H,L,L]
                H = Bmat.size(0)
                std_per_head = Bmat.view(H, -1).std(dim=-1)
                mean_std = std_per_head.mean()
                self.log('train_bam_bias_std', mean_std, prog_bar=True)
            except Exception:
                pass

    def on_train_epoch_end(self) -> None:
        """
        Enforce BAM negative-beta constraints and run per-epoch SWA capture, logging SWA model count when applicable.
        
        Performs two maintenance tasks at the end of a training epoch:
        - Invokes bias.constrain_negative_beta() on any BAM attention module biases that expose that method.
        - If SWA is enabled and the epoch is past the SWA start epoch, captures the model into the SWA controller and logs the current SWA model count under the key "swa/n_models".
        
        All errors raised by these diagnostics or SWA bookkeeping are suppressed to avoid interrupting training.
        """
        try:
            for attn_mod in self._iter_bam_modules() or []:
                bias = getattr(attn_mod, 'bias', None)
                if bias is not None and hasattr(bias, 'constrain_negative_beta'):
                    bias.constrain_negative_beta()
        except Exception:
            # Never fail the training loop due to diagnostics
            pass
        # SWA per-epoch capture (constant schedule)
        try:
            if getattr(self, 'swa_enabled', False) and self.swa_ctrl is not None:
                if int(self.current_epoch) >= int(getattr(self, '_swa_start_epoch', 1 << 30)):
                    if self.swa_ctrl.capture_epoch_end(self.model, int(self.current_epoch), int(self._swa_start_epoch)):
                        try:
                            self.log('swa/n_models', torch.tensor(int(self.swa_ctrl.n_models), device=self.device), prog_bar=False)
                        except Exception:
                            pass
        except Exception:
            pass

    def training_step(self, batch, batch_idx):
        # Unpack batch and ensure callbacks receive the appropriate view
        """
        Perform a single training step: unpack the batch, run the appropriate forward path (standard, R-Drop, or looped/ponder variants), compute and regularize the loss, log diagnostics, and return the scalar training loss.
        
        This method:
        - Unpacks model inputs from the dataloader batch and prepares targets.
        - Selects a forward/loss computation path based on model configuration (looped transformer with optional ponder aggregation, loop-T window averaging, R‑Drop two-pass or concatenated strategies, or the standard single forward), including expected-cost or cross-entropy loss modes where configured.
        - Applies configured regularizers and hyper-gradient updates.
        - Emits periodic training diagnostics (accuracy, DEQ telemetry, ponder metrics, R‑Drop/KL, calibration/SWA diagnostics, and learning rate) via the Lightning logger.
        - Returns the scalar loss used by the optimizer.
        
        Returns:
            torch.Tensor: Scalar training loss for the current batch (used for backpropagation).
        """
        src, tgt, ctx_input, ctx_output, task_ids, aug_grid = self._unpack_batch(batch)

        # Convert target to long type
        tgt = tgt.long()

        # Inform model of current epoch for schedules (if supported)
        if hasattr(self.core_model, "use_soft_inter_round"):
            setattr(self.core_model, "_epoch", int(self.current_epoch))

        # Forward and base loss (with optional Looped T-window loss aggregation)
        # If looped encoder is enabled and T > 0, request per-iteration logits history and average CE over last T.
        looped_enabled = bool(getattr(getattr(self.config.model, 'looped', None), 'enabled', False)) if hasattr(self.config, 'model') else False
        loop_T = int(getattr(getattr(self.config.model, 'looped', SimpleNamespace()), 'T', 0)) if looped_enabled else 0
        # Resolve scheduled loop depth override (b_eff), if any
        b_eff = None
        if looped_enabled:
            try:
                loop_cfg = getattr(self.config.model, 'looped', None)
                epoch = int(getattr(self, 'current_epoch', 0))
                step = int(getattr(getattr(self, 'trainer', None), 'global_step', 0) or 0)
                b_eff = compute_scheduled_b(loop_cfg, epoch, step)
            except Exception:
                b_eff = None

        # ---------------- Looped Ponder expected-loss aggregation ----------------
        # If Looped Ponder is enabled and aggregation='expected', compute Σ_t q_t·L_t and add KL/ponder-cost.
        ponder_enabled = bool(OmegaConf.select(self.config, 'model.looped.ponder.enabled', default=False))
        agg_mode = str(OmegaConf.select(self.config, 'model.looped.ponder.loss.aggregation', default='single'))
        use_expected_task = bool(OmegaConf.select(self.config, 'model.looped.ponder.loss.use_expected_task', default=True))

        if looped_enabled and ponder_enabled and use_expected_task and (agg_mode.lower() == 'expected') and not self.rdrop_enabled:
            # Request per-loop logits for all steps; do not truncate here to let PMF length match logits_history.
            if b_eff is not None:
                mo = self(src, tgt, ctx_input, ctx_output, aug_grid=aug_grid, loop_return_all=True, loop_b_override=b_eff)
            else:
                mo = self(src, tgt, ctx_input, ctx_output, aug_grid=aug_grid, loop_return_all=True)
            if isinstance(mo, tuple) and len(mo) == 2 and isinstance(mo[1], dict):
                y_hat = mo[0]
                extras = mo[1]
                logits_hist = extras.get('logits_history', None)
                lp = extras.get('loop_ponder', None)
                if isinstance(logits_hist, list) and isinstance(lp, dict) and ('pmf' in lp):
                    # Step losses
                    loss_mode = str(getattr(self.config.model, "loss_mode", "ce"))
                    ec_cfg = OmegaConf.select(self.config.model, "loss.expected_cost", default=None)
                    use_expected_cost = (loss_mode == "expected_cost") and (ec_cfg is not None) and bool(getattr(ec_cfg, "enabled", True))
                    step_losses = [
                        (self._expected_cost_scalar(logits_t, tgt, aug_grid) if use_expected_cost else self._compute_loss(logits_t, tgt))
                        for logits_t in logits_hist
                    ]
                    if len(step_losses) == 0:
                        loss = y_hat.new_tensor(0.0)
                    else:
                        L = torch.stack(step_losses)                    # [T]
                        q = lp['pmf']                                   # [B, T]
                        # Weight by batch-mean pmf to avoid per-sample CE; keeps it simple and stable
                        qw = q.mean(dim=0).to(L.dtype)                  # [T]
                        loss = (L * qw).sum()
                    # Regularisers (averaged across batch)
                    reg_total = lp.get('reg_total', None)
                    if isinstance(reg_total, torch.Tensor):
                        loss = loss + reg_total.mean()
                        try:
                            self.log('train_loop_reg', reg_total.mean().detach(), prog_bar=False)
                        except Exception:
                            pass
                    # Log core ponder diagnostics
                    try:
                        est = lp.get('expected_steps', None)
                        if isinstance(est, torch.Tensor):
                            self.log('train_loop_Esteps', est.mean().detach(), prog_bar=True)
                        kl = lp.get('kl', None)
                        if isinstance(kl, torch.Tensor):
                            self.log('train_loop_kl', kl.mean().detach(), prog_bar=False)
                        pc = lp.get('ponder_cost', None)
                        if isinstance(pc, torch.Tensor):
                            self.log('train_loop_ponder_cost', pc.mean().detach(), prog_bar=False)
                        tb = lp.get('tbptt_detach_step', None)
                        if isinstance(tb, torch.Tensor):
                            self.log('train_loop_tbptt_detach_step', tb.detach(), prog_bar=False)
                    except Exception:
                        pass
                    psi_list = None
                    attn_probs = None
                else:
                    # Fallback to standard path if extras missing
                    y_hat, psi_list, attn_probs, loss = self._forward_standard_and_aux(src, tgt, ctx_input, ctx_output, aug_grid, task_ids, loop_b_override=b_eff)
            else:
                # Fallback to standard path if return format unexpected
                y_hat, psi_list, attn_probs, loss = self._forward_standard_and_aux(src, tgt, ctx_input, ctx_output, aug_grid, task_ids, loop_b_override=b_eff)

        # ---------------- Looped T-window CE/EC aggregation (legacy) ----------------
        elif looped_enabled and loop_T > 0 and not self.rdrop_enabled:
            # Request logits_history from the model by asking it to return_all and truncate gradients to T.
            if b_eff is not None:
                mo = self(src, tgt, ctx_input, ctx_output, aug_grid=aug_grid, loop_return_all=True, loop_truncate_T=loop_T, loop_b_override=b_eff)
            else:
                mo = self(src, tgt, ctx_input, ctx_output, aug_grid=aug_grid, loop_return_all=True, loop_truncate_T=loop_T)
            if isinstance(mo, tuple) and len(mo) == 2 and isinstance(mo[1], dict) and 'logits_history' in mo[1]:
                y_hat = mo[0]
                logits_hist = mo[1]['logits_history']
                # Compute loss for last T iterations, average them (include PAD)
                loss_mode = str(getattr(self.config.model, "loss_mode", "ce"))
                ec_cfg = OmegaConf.select(self.config.model, "loss.expected_cost", default=None)
                use_expected_cost = (loss_mode == "expected_cost") and (ec_cfg is not None) and bool(getattr(ec_cfg, "enabled", True))
                if use_expected_cost:
                    losses = [self._expected_cost_scalar(logits_t, tgt, aug_grid) for logits_t in logits_hist[-loop_T:]]
                else:
                    losses = [self._compute_loss(logits_t, tgt, src) for logits_t in logits_hist[-loop_T:]]
                if len(losses) == 0:
                    loss = y_hat.new_tensor(0.0)
                else:
                    loss = torch.stack(losses).mean()
                psi_list = None
                attn_probs = None
            else:
                # Fallback to standard path if history missing
                y_hat, psi_list, attn_probs, loss = self._forward_standard_and_aux(src, tgt, ctx_input, ctx_output, aug_grid, task_ids)
        elif self.rdrop_enabled:
            k = int(OmegaConf.select(self.config, "rdrop.k", default=1))
            # Prefer trainer.global_step when available; allow test override via _rdrop_test_step
            try:
                step_idx = int(getattr(getattr(self, 'trainer', None), 'global_step', 0))
            except Exception:
                step_idx = 0
            step_idx = int(getattr(self, '_rdrop_test_step', step_idx))
            perform_rdrop = (k <= 1) or ((step_idx % k) == 0)
            if not perform_rdrop:
                y_hat, psi_list, attn_probs, loss = self._forward_standard_and_aux(src, tgt, ctx_input, ctx_output, aug_grid, task_ids, loop_b_override=b_eff)
            else:
                # Default to two-pass behavior unless explicitly overridden
                use_concat = bool(OmegaConf.select(self.config, "rdrop.use_concat", default=False))
                # Fallback to two-pass if share-mask requested (concat cannot enforce identical masks per-sample halves)
                if self.rdrop_share_mask and use_concat:
                    use_concat = False
                if use_concat:
                    y_hat, psi_list, attn_probs, loss = self._forward_with_rdrop_concat(src, tgt, ctx_input, ctx_output, aug_grid, b_override=b_eff)
                else:
                    y_hat, psi_list, attn_probs, loss = self._forward_with_rdrop(src, tgt, ctx_input, ctx_output, aug_grid, b_override=b_eff)
        else:
            y_hat, psi_list, attn_probs, loss = self._forward_standard_and_aux(src, tgt, ctx_input, ctx_output, aug_grid, task_ids, loop_b_override=b_eff)

        # Regularisers, CEBR, and optional hyper-grad
        loss = self._apply_regularisers_and_hypergrad(loss, y_hat, psi_list, attn_probs)

        # ---------------- DEQ telemetry logging (Phase 2) ----------------
        try:
            if bool(OmegaConf.select(self.config, 'model.deq.log_metrics', default=False)):
                enc = getattr(self.model, 'encoder', None)
                info = getattr(enc, 'last_info', None)
                if isinstance(info, dict) and info:
                    def _log(name, key, prog=False):
                        """
                        Log a tensor from the enclosing `info` mapping to the trainer/logger if present.
                        
                        If `info` contains `key` and the value is a torch.Tensor, detaches it and calls `self.log` with `name` and `prog_bar=prog`.
                        
                        Parameters:
                            name (str): Metric name to use when logging.
                            key (hashable): Key to look up the tensor value in the surrounding `info` mapping.
                            prog (bool): If True, forward the metric to the progress bar via `prog_bar=True`.
                        """
                        v = info.get(key, None)
                        if isinstance(v, torch.Tensor):
                            self.log(name, v.detach(), prog_bar=prog)
                    _log('deq/iters', 'iters', prog=False)
                    _log('deq/final_residual', 'final_residual', prog=False)
                    _log('deq/converged_fraction', 'converged_fraction', prog=False)
                    _log('deq/used_fallback', 'used_fallback', prog=False)
                    _log('deq/adapted_tol', 'adapted_tol', prog=False)
        except Exception:
            pass

        # Periodic metrics
        self._log_periodic_metrics(y_hat, tgt, batch_idx)

        # Copy-bias diagnostics (training)
        try:
            with torch.no_grad():
                preds = torch.argmax(y_hat, dim=-1)
                metrics = compute_copy_metrics_on_batch(
                    src.detach(),
                    tgt.detach(),
                    preds.detach(),
                )

            copy_rate = metrics.get('copy_rate')
            if copy_rate is not None:
                self.log(
                    'train_copy_rate',
                    copy_rate,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    batch_size=src.size(0),
                )

            change_recall = metrics.get('change_recall')
            if change_recall is not None:
                self.log(
                    'train_change_recall',
                    change_recall,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    batch_size=src.size(0),
                )

            cell_accuracy = metrics.get('cell_accuracy')
            change_precision = metrics.get('change_precision')
            transformation_f1 = metrics.get('transformation_f1')
            
            if change_precision is not None:
                self.log('train_change_precision', change_precision, on_step=True, on_epoch=False, prog_bar=False)
            if transformation_f1 is not None:
                self.log('train_transformation_f1', transformation_f1, on_step=True, on_epoch=False, prog_bar=False)
            
            if copy_rate is not None and cell_accuracy is not None:
                # Legacy transformation score
                transformation_score = (1.0 - copy_rate) * cell_accuracy
                self.log(
                    'train_transformation_score',
                    transformation_score,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    batch_size=src.size(0),
                )
            
            if transformation_f1 is not None and cell_accuracy is not None:
                # F1-based transformation quality score
                transformation_quality_score = transformation_f1 * cell_accuracy
                self.log(
                    'train_transformation_quality_score',
                    transformation_quality_score,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    batch_size=src.size(0),
                )
        except Exception as exc:  # pragma: no cover - diagnostics must never break training
            logger.debug(f"Skipping train copy metrics logging due to: {exc}")

        # Learning rate logging (if optimizer is available)
        self._log_learning_rate()

        # Log total loss and return
        self.log('train_loss', loss, on_step=True, prog_bar=True)
        return loss

    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        """Centralized, DDP-safe gradient clipping using Lightning's API.

        Lightning calls this hook with values from the Trainer. If they are None,
        we fall back to `self.config.training.gradient_clip_val` (default 1.0).
        """
        # Fallback to config when Trainer doesn't provide a value
        if gradient_clip_val is None:
            gradient_clip_val = float(getattr(self.config.training, "gradient_clip_val", 1.0))
        if gradient_clip_val is None or float(gradient_clip_val) <= 0:
            return
        gradient_clip_algorithm = gradient_clip_algorithm or "norm"
        # Delegate to Lightning's internal, strategy-aware implementation
        self.clip_gradients(
            optimizer,
            gradient_clip_val=float(gradient_clip_val),
            gradient_clip_algorithm=gradient_clip_algorithm,
        )

    # --------------------------- Back-compat config alias ---------------------------
    @property
    def cfg(self):
        """Alias for self.config for backward compatibility; no extra reference stored."""
        return self.config

    @cfg.setter
    def cfg(self, value):
        self.config = value

    # --------------------------- Logging cleanup ---------------------------
    def _cleanup_file_handler(self):
        """Close and detach the per-instance file handler to prevent FD leaks."""
        try:
            if getattr(self, "_file_handler", None) is not None:
                try:
                    logger.removeHandler(self._file_handler)
                except Exception:
                    pass
                try:
                    self._file_handler.close()
                except Exception:
                    pass
                self._file_handler = None
        except Exception:
            # Best-effort cleanup; never raise during teardown
            pass

    def on_fit_end(self):  # Lightning hook
        # SWA: finalize BN and optionally swap to averaged weights and save checkpoint
        """
        Perform end-of-fit finalization tasks including SWA finalization, calibration fitting/loading, and cleanup.
        
        If SWA is enabled and models have been accumulated, finalize batch-norm statistics (optionally using a data loader), optionally swap averaged weights into the model for evaluation, run optional SWA diagnostics, and optionally save a dedicated SWA checkpoint. If calibration is enabled, attempt to load a saved calibrator state or fit a temperature-style calibrator on the validation set, optionally save the calibrator state, and optionally generate and save a reliability diagram. All file I/O, diagnostics, and optional steps are controlled by configuration and any errors during these operations are caught and logged; the method always performs instance file-handler cleanup at the end.
        """
        try:
            if getattr(self, 'swa_enabled', False) and self.swa_ctrl is not None and int(self.swa_ctrl.n_models) > 0:
                bn_cfg = getattr(self, 'swa_cfg', None)
                # Capture base (SGD) state BEFORE applying SWA for diagnostics
                try:
                    base_state = state_from_model(self.model)
                except Exception:
                    base_state = None
                if bn_cfg is not None and bool(getattr(bn_cfg, 'bn_recompute', True)):
                    dm = getattr(self.trainer, 'datamodule', None)
                    loader_sel = str(getattr(bn_cfg, 'bn_recompute_loader', 'train'))
                    max_batches = getattr(bn_cfg, 'bn_max_batches', None)
                    ddp_broadcast = bool(getattr(bn_cfg, 'ddp_sync_bn_update', True))
                    ddp_bn_mode = str(getattr(bn_cfg, 'ddp_bn_mode', 'broadcast'))
                    dl = None
                    try:
                        if dm is not None:
                            if loader_sel == 'train' and hasattr(dm, 'train_dataloader'):
                                dl = dm.train_dataloader()
                            elif loader_sel == 'val' and hasattr(dm, 'val_dataloader'):
                                dl = dm.val_dataloader()
                    except Exception:
                        dl = None
                    if dl is not None:
                        self.swa_ctrl.finalize_bn(
                            self.model, dl, device=self.device,
                            max_batches=max_batches,
                            ddp_broadcast=ddp_broadcast,
                            ddp_mode=ddp_bn_mode,
                        )
                # Use SWA weights for subsequent eval if requested; otherwise restore base weights
                if bool(getattr(bn_cfg, 'eval_with_swa', True)):
                    self.swa_ctrl.apply_to_model(self.model)
                else:
                    try:
                        if base_state is not None:
                            load_state_into_model(self.model, base_state)
                    except Exception:
                        logger.debug("SWA: failed to restore base weights; proceeding with current weights.")
                # Diagnostics: run after SWA weights are applied
                try:
                    diag_cfg = getattr(bn_cfg, 'diagnostics', None)
                    if diag_cfg is not None and bool(getattr(diag_cfg, 'enabled', False)):
                        dm = getattr(self.trainer, 'datamodule', None)
                        dl_diag = None
                        eval_loader = str(getattr(diag_cfg, 'eval_loader', 'val'))
                        if dm is not None:
                            if eval_loader == 'val' and hasattr(dm, 'val_dataloader'):
                                dl_diag = dm.val_dataloader()
                            elif eval_loader == 'train' and hasattr(dm, 'train_dataloader'):
                                dl_diag = dm.train_dataloader()
                        if dl_diag is not None:
                            # Resolve output directory
                            out_dir = getattr(diag_cfg, 'output_dir', None)
                            if out_dir is None:
                                try:
                                    hydra_run_dir = Path(hydra.core.hydra_config.HydraConfig.get().run.dir)
                                    base_dir = hydra_run_dir
                                except Exception:
                                    base_dir = Path('.')
                                out_dir = str(base_dir / 'swa_diagnostics')
                            # SWA state from current model; base state captured earlier
                            try:
                                swa_state = state_from_model(self.model)
                            except Exception:
                                swa_state = None
                            if base_state is not None and swa_state is not None:
                                run_swa_diagnostics_from_trainer(
                                    self, self.model, dl_diag, bn_cfg,
                                    base_state=base_state,
                                    swa_state=swa_state,
                                    out_dir=out_dir,
                                )
                                logger.info(f"SWA diagnostics written to {out_dir}")
                except Exception as e:
                    logger.warning(f"SWA diagnostics failed: {e}")
                # Optionally save a dedicated SWA checkpoint
                try:
                    if bool(getattr(bn_cfg, 'save_swa_checkpoint', True)) and hasattr(self.trainer, 'save_checkpoint'):
                        # Resolve output path from Hydra run dir
                        try:
                            hydra_run_dir = Path(hydra.core.hydra_config.HydraConfig.get().run.dir)
                            base_dir = hydra_run_dir
                        except Exception:
                            base_dir = Path('.')
                        base_dir.mkdir(parents=True, exist_ok=True)
                        out_ckpt = base_dir / 'swa_final.ckpt'
                        self.trainer.save_checkpoint(str(out_ckpt))
                        logger.info(f"Saved SWA checkpoint to {out_ckpt}")
                except Exception as e:
                    logger.warning(f"Failed to save SWA checkpoint: {e}")
        except Exception as e:
            logger.debug(f"SWA finalize skipped: {e}")
        # Fit temperature scaler on validation set if enabled
        """
        Run final calibration steps after training completes.
        
        If calibration is enabled in the object's configuration, attempt to load a saved calibrator state or fit a temperature calibrator on the validation dataloader, optionally save the calibrator state, and optionally generate and save a reliability diagram for the validation set. All file path resolution and saving/loading are configurable; any errors during loading, fitting, saving, or plotting are caught and logged without raising. Finally, perform instance file-handler cleanup.
        """
        try:
            if self.calibration_cfg is not None and bool(OmegaConf.select(self.calibration_cfg, 'enabled', default=False)):
                dm = getattr(self.trainer, 'datamodule', None)
                if dm is not None and hasattr(dm, 'val_dataloader'):
                    # Optional: load previously saved calibrator state
                    load_state = bool(OmegaConf.select(self.calibration_cfg, 'load_state', default=False))
                    if load_state:
                        try:
                            method = str(OmegaConf.select(self.calibration_cfg, 'method', default='temperature'))
                            params = {
                                'num_bins': int(OmegaConf.select(self.calibration_cfg, 'num_bins', default=15)),
                                'binning': str(OmegaConf.select(self.calibration_cfg, 'binning', default='equal_width')),
                                'multi_class': str(OmegaConf.select(self.calibration_cfg, 'multi_class', default='top1')),
                                'a': float(OmegaConf.select(self.calibration_cfg, 'a', default=1.0)),
                                'b': float(OmegaConf.select(self.calibration_cfg, 'b', default=1.0)),
                            }
                            from jarc_reactor.calibration import make_calibrator as _mk
                            cal = _mk(method, **params)
                            load_path = OmegaConf.select(self.calibration_cfg, 'load_path', default=None)
                            if load_path is None:
                                # Resolve default state path from state_dir/filename
                                state_dir = OmegaConf.select(self.calibration_cfg, 'state_dir', default=None)
                                state_filename = OmegaConf.select(self.calibration_cfg, 'state_filename', default='calibrator.pt')
                                try:
                                    hydra_run_dir = Path(hydra.core.hydra_config.HydraConfig.get().run.dir)
                                    base_dir = hydra_run_dir / OmegaConf.select(self.config, 'training.training_log_dir', default='logs/training')
                                except Exception:
                                    base_dir = Path('logs/training')
                                if state_dir is not None:
                                    base_dir = Path(str(state_dir))
                                load_path = str(base_dir / str(state_filename))
                            sd = torch.load(load_path, map_location='cpu')
                            cal.load_state_dict(sd)
                            self._calibrator = cal
                            logger.info(f"Loaded calibrator state from {load_path}")
                        except Exception as e:
                            logger.warning(f"Failed to load calibrator state, falling back to fit: {e}")
                            self._fit_temperature_scaler(dm.val_dataloader())
                    else:
                        # Fit on validation set
                        self._fit_temperature_scaler(dm.val_dataloader())

                    # Optionally save calibrator state
                    try:
                        if bool(OmegaConf.select(self.calibration_cfg, 'save_state', default=False)) and self._calibrator is not None:
                            state_dir = OmegaConf.select(self.calibration_cfg, 'state_dir', default=None)
                            state_filename = OmegaConf.select(self.calibration_cfg, 'state_filename', default='calibrator.pt')
                            if state_dir is None:
                                try:
                                    hydra_run_dir = Path(hydra.core.hydra_config.HydraConfig.get().run.dir)
                                    base_dir = hydra_run_dir / OmegaConf.select(self.config, 'training.training_log_dir', default='logs/training')
                                except Exception:
                                    base_dir = Path('logs/training')
                            else:
                                base_dir = Path(str(state_dir))
                            base_dir.mkdir(parents=True, exist_ok=True)
                            out_state = base_dir / str(state_filename)
                            torch.save(self._calibrator.state_dict(), str(out_state))
                            logger.info(f"Saved calibrator state to {out_state}")
                    except Exception as e:
                        logger.warning(f"Failed to save calibrator state: {e}")
                    # Optionally save reliability diagram
                    diagrams_cfg = OmegaConf.select(self.calibration_cfg, 'diagrams', default=None)
                    if diagrams_cfg is not None and bool(OmegaConf.select(diagrams_cfg, 'generate_on_fit_end', default=True)):
                        try:
                            logits, targets = self._gather_logits_targets(dm.val_dataloader())
                            save_dir = OmegaConf.select(diagrams_cfg, 'save_dir', default=None)
                            if save_dir is None:
                                # Try hydra run dir + training_log_dir fallback
                                try:
                                    hydra_run_dir = Path(hydra.core.hydra_config.HydraConfig.get().run.dir)
                                    base_dir = hydra_run_dir / OmegaConf.select(self.config, 'training.training_log_dir', default='logs/training')
                                except Exception:
                                    base_dir = Path('logs/training')
                            else:
                                base_dir = Path(str(save_dir))
                            base_dir.mkdir(parents=True, exist_ok=True)
                            out_path = base_dir / 'reliability_val.png'
                            plot_reliability_pre_post_from_logits(
                                logits, targets, scaler=self._calibrator,
                                num_bins=int(OmegaConf.select(self.calibration_cfg, 'num_bins', default=15)),
                                ignore_index=self.pad_token_id,
                                binning=str(OmegaConf.select(self.calibration_cfg, 'binning', default='equal_width')),
                                show_hist=bool(OmegaConf.select(diagrams_cfg, 'histograms', default=False)),
                                save_path=str(out_path)
                            )
                            logger.info(f"Saved reliability diagram to {out_path}")
                        except Exception as e:
                            logger.warning(f"Failed to save reliability diagram on_fit_end: {e}")
        except Exception as e:
            logger.warning(f"Calibration fit failed: {e}")
        self._cleanup_file_handler()

    def on_validation_epoch_start(self):  # Lightning hook
        """
        Reset per-skill validation accumulators and advance FRM epoch tracking.
        
        This clears internal counters used to accumulate per-skill totals and correct counts
        for the upcoming validation epoch, and, if Functional Risk Minimisation (FRM) is enabled,
        updates the FRM instance with the current epoch for any epoch-dependent scheduling.
        """
        try:
            self._val_skill_totals = {0: 0, 1: 0, 2: 0, 3: 0}
            self._val_skill_correct = {0: 0, 1: 0, 2: 0, 3: 0}
        except Exception:
            pass
        # Update FRM epoch for schedule tracking (if enabled)
        try:
            if self.frm_enabled and hasattr(self, 'frm') and self.frm is not None:
                self.frm.set_epoch(self.current_epoch)
        except Exception:
            pass

    def on_train_epoch_start(self) -> None:  # Lightning hook
        """
        Notify the Functional Risk Minimizer (FRM) of the current training epoch.
        
        Called at the start of each training epoch (Lightning hook). If FRM is enabled and available on the trainer, calls its set_epoch method with the trainer's current_epoch; any exceptions raised during this call are ignored.
        """
        try:
            if self.frm_enabled and hasattr(self, 'frm') and self.frm is not None:
                self.frm.set_epoch(self.current_epoch)
        except Exception:
            pass

    def on_train_batch_start(self, batch, batch_idx: int) -> None:  # Lightning hook
        """
        Override optimizer learning rate per the SWA controller at the start of each training batch once SWA has started.
        
        When SWA is enabled and the current epoch is at or past the configured SWA start epoch, this hook:
        - Ensures a per-run SWA start global step is recorded.
        - Computes the SWA learning rate for the current global step via the SWA controller and applies it to the trainer's optimizer.
        - Logs a one-time informational message if schedulers are effectively paused and emits a per-step metric `swa/lr` when possible.
        
        All errors are caught and suppressed to avoid interrupting training.
        """
        try:
            if not getattr(self, 'swa_enabled', False) or self.swa_ctrl is None:
                return
            if int(self.current_epoch) < int(getattr(self, '_swa_start_epoch', 1 << 30)):
                return
            opt = getattr(self, '_optimizer_ref', None)
            if opt is None:
                return
            # Optional: log once that schedulers are effectively paused during SWA
            try:
                if bool(getattr(getattr(self, 'swa_cfg', None), 'pause_schedulers', True)) and not getattr(self, '_swa_logged_scheduler_pause', False):
                    self._swa_logged_scheduler_pause = True
                    logger.info("SWA: pause_schedulers=True; overriding LR per batch. Any LR schedulers are effectively paused.")
            except Exception:
                pass
            # Lazily set SWA start global step at the first batch in SWA phase
            if getattr(self.swa_ctrl, '_start_global_step', None) is None:
                try:
                    self.swa_ctrl._start_global_step = int(getattr(getattr(self, 'trainer', None), 'global_step', 0))
                except Exception:
                    self.swa_ctrl._start_global_step = 0
            gs = int(getattr(getattr(self, 'trainer', None), 'global_step', 0))
            lr = float(self.swa_ctrl.lr_for_step(gs, int(self.swa_ctrl._start_global_step)))
            self.swa_ctrl.set_optimizer_lr(opt, lr)
            # Optional LR trace logging
            try:
                self.log('swa/lr', torch.tensor(lr, device=self.device), on_step=True, prog_bar=False, logger=True)
            except Exception:
                pass
        except Exception:
            # Never interrupt training due to SWA scheduling
            pass

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:  # Lightning hook
        """
        Capture an SWA snapshot at configured policy steps after each training batch.
        
        When SWA is enabled and the trainer has passed the configured SWA start epoch, this hook queries the SWA controller to possibly capture the current model state (based on global step, epoch and SWA policy). If a snapshot is captured, logs the number of SWA models and a cycle index metric for cyclical SWA. All errors during capture or logging are suppressed to avoid interrupting training.
        """
        try:
            if not getattr(self, 'swa_enabled', False) or self.swa_ctrl is None:
                return
            if int(self.current_epoch) < int(getattr(self, '_swa_start_epoch', 1 << 30)):
                return
            gs = int(getattr(getattr(self, 'trainer', None), 'global_step', 0))
            started = int(self.swa_ctrl._start_global_step or 0)
            captured = self.swa_ctrl.maybe_capture(self.model, gs, int(self.current_epoch), int(self._swa_start_epoch), started)
            if captured:
                try:
                    self.log('swa/n_models', torch.tensor(int(self.swa_ctrl.n_models), device=self.device), on_step=True, prog_bar=False, logger=True)
                    # If cyclical, also log cycle index at capture
                    c = int(getattr(self.swa_cfg, 'cycle_len_steps', 1) or 1)
                    i = max(1, (gs - started) + 1)
                    self.log('swa/cycle_idx', torch.tensor(i // max(1, c), device=self.device), on_step=True, prog_bar=False, logger=True)
                except Exception:
                    pass
        except Exception:
            pass

    def on_validation_end(self):  # Lightning hook
        # Optionally save per-epoch reliability diagram
        """
        Save a per-epoch reliability diagram for calibration if configured, and clean up the trainer's file handler.
        
        When calibration is enabled and diagram generation on validation end is enabled, this method:
        - Gathers logits and targets from the datamodule's validation dataloader.
        - Determines an output directory (explicit `diagrams.save_dir` if set; otherwise the Hydra run directory combined with `training.training_log_dir`, falling back to `logs/training`).
        - Creates the directory, generates a pre/post reliability diagram using the current calibrator and calibration settings (number of bins, binning strategy, histogram option, and `ignore_index` set to the trainer's pad token id), and saves the image as `reliability_val_epoch_{current_epoch}.png`.
        - Logs success or a warning on failure.
        
        This method swallows exceptions raised during diagram generation and always calls `_cleanup_file_handler()` to release any per-instance file handler resources.
        """
        try:
            if self.calibration_cfg is not None and bool(OmegaConf.select(self.calibration_cfg, 'enabled', default=False)):
                diagrams_cfg = OmegaConf.select(self.calibration_cfg, 'diagrams', default=None)
                if diagrams_cfg is not None and bool(OmegaConf.select(diagrams_cfg, 'generate_on_validation_epoch_end', default=False)):
                    dm = getattr(self.trainer, 'datamodule', None)
                    if dm is not None and hasattr(dm, 'val_dataloader'):
                        logits, targets = self._gather_logits_targets(dm.val_dataloader())
                        save_dir = OmegaConf.select(diagrams_cfg, 'save_dir', default=None)
                        if save_dir is None:
                            try:
                                hydra_run_dir = Path(hydra.core.hydra_config.HydraConfig.get().run.dir)
                                base_dir = hydra_run_dir / OmegaConf.select(self.config, 'training.training_log_dir', default='logs/training')
                            except Exception:
                                base_dir = Path('logs/training')
                        else:
                            base_dir = Path(str(save_dir))
                        base_dir.mkdir(parents=True, exist_ok=True)
                        out_path = base_dir / f'reliability_val_epoch_{self.current_epoch}.png'
                        plot_reliability_pre_post_from_logits(
                            logits, targets, scaler=self._calibrator,
                            num_bins=int(OmegaConf.select(self.calibration_cfg, 'num_bins', default=15)),
                            ignore_index=self.pad_token_id,
                            binning=str(OmegaConf.select(self.calibration_cfg, 'binning', default='equal_width')),
                            show_hist=bool(OmegaConf.select(diagrams_cfg, 'histograms', default=False)),
                            save_path=str(out_path)
                        )
                        logger.info(f"Saved epoch reliability diagram to {out_path}")
        except Exception as e:
            logger.warning(f"Failed to save reliability diagram on_validation_end: {e}")
        self._cleanup_file_handler()

    def on_validation_epoch_end(self):  # Lightning hook
        """
        Log per-skill accuracy metrics at the end of a validation epoch for external monitoring.
        
        Checks for internal dictionaries `_val_skill_totals` and `_val_skill_correct` and, if present, computes accuracies for up to four observed skills and logs them under `acc_A`/`acc_B`/`acc_C`/`acc_D` and duplicated as `val_acc_A`/`val_acc_B`/`val_acc_C`/`val_acc_D`. If the required attributes are missing or empty, the method no-ops. Exceptions during logging are suppressed.
        """
        try:
            totals = getattr(self, '_val_skill_totals', None)
            corrects = getattr(self, '_val_skill_correct', None)
            if not isinstance(totals, dict) or not isinstance(corrects, dict):
                return
            def ratio(num, den):
                """
                Compute the ratio of `num` to `den` and return it as a torch tensor placed on `self.device`.
                
                Parameters:
                    num: Numerator value (numeric).
                    den: Denominator value (numeric or None). If `den` is `None` or less than or equal to 0, the function treats it as zero and yields 0.0.
                
                Returns:
                    torch.Tensor: A scalar floating-point tensor on `self.device` equal to `num / den`, or `0.0` if `den` is `None` or <= 0.
                """
                den = float(den) if den is not None else 0.0
                return torch.tensor(0.0, device=self.device) if den <= 0 else torch.tensor(float(num) / den, device=self.device)
            observed_skills = sorted(totals.keys())
            if not observed_skills:
                return

            letters = ['A', 'B', 'C', 'D']
            for idx, skill_id in enumerate(observed_skills[: len(letters)]):
                value = ratio(corrects.get(skill_id, 0), totals.get(skill_id, 0))
                tag = letters[idx]
                self.log(f'acc_{tag}', value, prog_bar=False)
                self.log(f'val_acc_{tag}', value, prog_bar=False)
        except Exception:
            pass

    def test_step(self, batch, batch_idx):
        # Unpack
        """
        Run a single test step: perform a model forward, optionally run the SAR test-time adapter, apply insertion gating, compute augmented-aware accuracies, and log calibration and diagnostic metrics.
        
        Returns:
            torch.Tensor: Scalar grid accuracy for the batch (logged as 'test_grid_accuracy').
        """
        src, tgt, ctx_input, ctx_output, task_ids = batch
        tgt = tgt.long()
        model_out = self(src, tgt, ctx_input, ctx_output)
        y_hat = model_out[0] if isinstance(model_out, tuple) else model_out
        # ---------------- SAR (Phase 1) adapter: run before gating ----------------
        try:
            sar_cfg = getattr(getattr(self.config, 'model', None), 'sar', None)
            if sar_cfg is not None and bool(getattr(sar_cfg, 'enabled', False)):
                # Lazily initialize adapter
                if self._sar_adapter is None:
                    try:
                        from jarc_reactor.tta.adapter import SARAdapter
                        self._sar_adapter = SARAdapter(self)
                    except Exception as e:
                        logger.warning(f"Failed to initialize SARAdapter: {e}")
                        self._sar_adapter = None
                # Compute adaptation step using pre-gating logits
                if self._sar_adapter is not None:
                    # Respect ignore_gating_for_entropy by always passing pre-gating logits
                    self._sar_adapter.step(batch_idx, batch, logits_pre_gating=y_hat, tgt=tgt)
        except Exception as e:
            logger.debug(f"SAR adapter step skipped due to error: {e}")
        # Apply insertion gating for eval if enabled
        y_hat = self._apply_insertion_gating(y_hat, tgt)
        # Accuracies (augmented-aware)
        accuracies = self._compute_accuracy_baseaware(y_hat, tgt)
        self.log('test_cell_accuracy', accuracies['cell_accuracy'], prog_bar=True)
        self.log('test_grid_accuracy', accuracies['grid_accuracy'], prog_bar=True)
        # Log additional diagnostics; surface the first tolerance on progress bar
        t_surfaced = False
        for k, v in accuracies.items():
            if k in ('cell_accuracy', 'grid_accuracy'):
                continue
            try:
                if (not t_surfaced) and k.startswith('grid_tol_'):
                    self.log(f'test_{k}', v, prog_bar=True)
                    t_surfaced = True
                else:
                    self.log(f'test_{k}', v, prog_bar=False)
            except Exception:
                pass
        # Calibration metrics if requested
        try:
            apply_on = str(OmegaConf.select(self.calibration_cfg, 'apply_on', default='eval')) if self.calibration_cfg is not None else 'eval'
            log_mce = bool(OmegaConf.select(self.calibration_cfg, 'log_mce', default=False)) if self.calibration_cfg is not None else False
            num_bins = int(OmegaConf.select(self.calibration_cfg, 'num_bins', default=15)) if self.calibration_cfg is not None else 15
            binning = str(OmegaConf.select(self.calibration_cfg, 'binning', default='equal_width')) if self.calibration_cfg is not None else 'equal_width'
            with torch.no_grad():
                # Pre metrics on aggregated base
                y_base = self._aggregated_base_logits(y_hat)
                probs_base = F.softmax(y_base, dim=-1)
                ece_pre, _ = compute_ece_from_probs_binned(probs_base, tgt, num_bins=num_bins, ignore_index=self.pad_token_id, binning=binning)
                self.log('test_ece_pre', ece_pre, prog_bar=False)
                if log_mce:
                    mce_pre = compute_mce_from_probs(probs_base, tgt, num_bins=num_bins, ignore_index=self.pad_token_id, binning=binning)
                    self.log('test_mce_pre', mce_pre, prog_bar=True)
                y_flat = y_base.reshape(-1, y_base.size(-1))
                t_flat = tgt.reshape(-1).long()
                mask = (t_flat != self.pad_token_id)
                nll_pre = F.cross_entropy(y_flat[mask], t_flat[mask]) if mask.any() else torch.tensor(0.0, device=y_hat.device)
                self.log('test_nll_pre', nll_pre.detach(), prog_bar=False)
                # Post metrics only if calibrator available and allowed
                if self._calibrator is not None and apply_on in ('test', 'both'):
                    y_hat_cal = self._calibrator(y_base)
                    probs_post = F.softmax(y_hat_cal, dim=-1)
                    ece_post, _ = compute_ece_from_probs_binned(probs_post, tgt, num_bins=num_bins, ignore_index=self.pad_token_id, binning=binning)
                    self.log('test_ece_post', ece_post, prog_bar=False)
                    if log_mce:
                        mce_post = compute_mce_from_probs(probs_post, tgt, num_bins=num_bins, ignore_index=self.pad_token_id, binning=binning)
                        self.log('test_mce_post', mce_post, prog_bar=True)
                    ycal_flat = y_hat_cal.reshape(-1, y_hat_cal.size(-1))
                    nll_post = F.cross_entropy(ycal_flat[mask], t_flat[mask]) if mask.any() else torch.tensor(0.0, device=y_hat.device)
                    self.log('test_nll_post', nll_post.detach(), prog_bar=False)
        except Exception as e:
            logger.debug(f"Calibration test metrics skipped: {e}")
        # Return grid accuracy as a scalar for test aggregation
        return accuracies['grid_accuracy']

    def on_test_end(self):  # Lightning hook
        self._cleanup_file_handler()

    def on_predict_end(self):  # Lightning hook
        self._cleanup_file_handler()

    def predict_step(self, batch, batch_idx):
        """Lightning predict hook: runs insertion inference and returns final grids.

        Batch formats supported:
          - (src,): only input grid; init grid defaults to PAD.
          - (src, init_grid): provide an explicit starting grid.
          - (src, tgt, ...): falls back to (src,) semantics for inference.
        """
        try:
            from jarc_reactor.decoding import insertion_infer
        except Exception as e:
            raise RuntimeError(f"insertion_infer unavailable: {e}")

        if isinstance(batch, (list, tuple)):
            src = batch[0]
            init_grid = None
            if len(batch) >= 2 and isinstance(batch[1], torch.Tensor) and batch[1].dim() == 3:
                init_grid = batch[1]
        else:
            src = batch
            init_grid = None

        # Config defaults
        try:
            ins_cfg = getattr(self.config.model, 'insertion', None)
            dec_cfg = getattr(ins_cfg, 'decode', SimpleNamespace()) if ins_cfg is not None else SimpleNamespace()
            selection_k = int(getattr(dec_cfg, 'selection_k', 1))
            max_steps = getattr(dec_cfg, 'max_steps', None)
            strategy = str(getattr(dec_cfg, 'strategy', 'topk'))
            min_conf = getattr(dec_cfg, 'min_conf', None)
            random_seed = getattr(dec_cfg, 'random_seed', None)
            # Use config slot_state.predicted threshold if available
            sc = getattr(ins_cfg, 'slot_state', None) if ins_cfg is not None else None
            pred_cfg = getattr(sc, 'predicted', None) if sc is not None else None
            eos_threshold = getattr(pred_cfg, 'eos_conf_threshold', None) if pred_cfg is not None else None
        except Exception:
            selection_k, max_steps, strategy, min_conf, random_seed, eos_threshold = 1, None, 'topk', None, None, None

        final, _ = insertion_infer(
            self,
            src.long(),
            init_grid.long() if init_grid is not None else None,
            max_steps=max_steps,
            selection_k=selection_k,
            eos_threshold=eos_threshold,
            strategy=strategy,
            min_conf=min_conf,
            random_seed=random_seed,
        )
        return final

    # ------------------------------------------------------------------
    # Calibration helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _gather_logits_targets(self, dl) -> tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        device = self.device
        logits_list = []
        targets_list = []
        max_samples = int(OmegaConf.select(self.calibration_cfg, 'max_samples', default=500_000)) if self.calibration_cfg else 500_000
        collected = 0
        for batch in dl:
            try:
                # Expect batch like (src, tgt, ...)
                src, tgt = batch[0].to(device), batch[1].to(device)
                model_out = self(src, tgt, None, None)
                y_hat = model_out[0] if isinstance(model_out, tuple) else model_out
                # Apply eval-time gating if configured, then aggregate to base
                y_hat = self._apply_insertion_gating(y_hat, tgt)
                y_base = self._aggregated_base_logits(y_hat)
                logits_list.append(y_base.detach().cpu())
                targets_list.append(tgt.detach().cpu())
                collected += tgt.numel()
                if collected >= max_samples:
                    break
            except Exception as e:
                logger.debug(f"Skipping batch during calibration gather: {e}")
                continue
        if not logits_list:
            raise RuntimeError("No logits collected for calibration fit.")
        logits = torch.cat(logits_list, dim=0)
        targets = torch.cat(targets_list, dim=0)
        return logits, targets

    # --------------------------- Metrics helpers (Phase 1) ---------------------------
    def _aggregated_base_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Aggregate augmented head logits into base-vocab logits via log-sum-exp over {base_c, hedged M_c}.
        Returns logits of shape [B, H, W, V]. If head is not augmented, returns logits[..., :V].
        """
        V = int(getattr(self.config.model, "vocab_size", logits.size(-1)))
        K = int(logits.size(-1))
        if K < V + V + 2:
            return logits[..., :V]
        base = logits[..., :V]
        hedged = logits[..., V + 1: V + 1 + V]
        stacked = torch.stack([base, hedged], dim=-1)  # [..., V, 2]
        agg = torch.logsumexp(stacked, dim=-1)         # [..., V]
        return agg

    def _aggregated_base_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Convenience: softmax over aggregated base logits to get [B,H,W,V] probabilities."""
        agg = self._aggregated_base_logits(logits)
        return F.softmax(agg, dim=-1)

    # --------------------------- Accuracy helper (augmented-aware) ---------------------------
    def _compute_accuracy_baseaware(self, y_hat_full: torch.Tensor, tgt: torch.Tensor) -> dict:
        """Compute accuracy treating EOS-on-PAD as correct when pad_marks_complete is True.

        - Predictions are made by aggregating augmented logits to base via log-sum-exp over {base_c, hedged M_c}.
        - If head is augmented and ins.cfg.pad_marks_complete is True, and EOS is the top logit for a slot
          whose target is PAD, we set the prediction to PAD for that slot.
        """
        with torch.no_grad():
            V = int(getattr(self.config.model, "vocab_size", y_hat_full.size(-1)))
            pad_id = int(getattr(self.config.model, "pad_token_id", V - 1))
            K = int(y_hat_full.size(-1))
            yh_base = self._aggregated_base_logits(y_hat_full)
            preds = torch.argmax(yh_base, dim=-1)
            # EOS-on-PAD correctness mapping
            try:
                ins_cfg = getattr(self.config.model, "insertion", None)
                if ins_cfg is not None and bool(getattr(ins_cfg, "pad_marks_complete", False)) and K >= V + V + 2:
                    EOS = V + V + 1
                    tgt_pad = (tgt == pad_id)
                    # mark where EOS is max over full head
                    eos_is_max = (y_hat_full.argmax(dim=-1) == EOS)
                    preds = torch.where(tgt_pad & eos_is_max, torch.full_like(preds, pad_id), preds)
            except Exception:
                pass

            # Now reuse existing accuracy computation by comparing preds to tgt
            valid_mask = (tgt != pad_id)
            correct_cells = (preds == tgt) & valid_mask
            cell_accuracy = correct_cells.float().sum() / (valid_mask.float().sum() + 1e-6)
            corr_or_pad = (correct_cells | ~valid_mask)
            # Support [B,H,W] or [B,T] by flattening non-batch dims
            if corr_or_pad.dim() >= 2:
                grid_matches = corr_or_pad.view(corr_or_pad.size(0), -1).all(dim=1)
            else:
                grid_matches = corr_or_pad.bool()
            grid_accuracy = grid_matches.float().mean()
            return {"cell_accuracy": cell_accuracy, "grid_accuracy": grid_accuracy}

    # --------------------------- Slot-state + EC helpers (Phase 2) ---------------------------
    def _derive_slot_state(self, tgt: torch.Tensor, logits: torch.Tensor | None = None, aug_grid: torch.Tensor | None = None) -> torch.Tensor | None:
        """
        Derive slot_state (True=incomplete, False=complete) from multiple sources:
          - PAD-based completion when insertion.pad_marks_complete is True.
          - Dataloader-provided mask via aug_grid: True=complete.
          - Predicted completion from logits when enabled: EOS confidence threshold.
        Returns a flattened [B*L] bool tensor or None if no signals are enabled.
        """
        try:
            B, H, W = tgt.shape
            L = H * W
            V = int(getattr(self.config.model, "vocab_size", logits.size(-1) if logits is not None else 0))
            K = int(logits.size(-1)) if logits is not None else int(getattr(self.config.model, "head_size", V))
            pad_id = int(getattr(self.config.model, "pad_token_id", V - 1))
            ins_cfg = getattr(self.config.model, "insertion", None)

            # Start with all incomplete if nothing specified
            slot_state = torch.ones(B, L, dtype=torch.bool, device=tgt.device)

            # (a) PAD-based completion
            pad_marks_complete = bool(getattr(ins_cfg, "pad_marks_complete", False)) if ins_cfg is not None else False
            if pad_marks_complete:
                complete_pad = (tgt.view(B * L) == pad_id).view(B, L)
                slot_state = slot_state & (~complete_pad)

            # (b) Dataloader-provided completion mask via aug_grid (True=complete)
            if aug_grid is not None and torch.is_tensor(aug_grid):
                try:
                    if aug_grid.dtype == torch.bool:
                        complete_dl = aug_grid.view(B, L)
                    else:
                        complete_dl = (aug_grid.view(B, L).to(torch.int64) > 0)
                    slot_state = slot_state & (~complete_dl)
                except Exception:
                    pass

            # (c) Predicted completion from logits (EOS confidence)
            try:
                slot_cfg = OmegaConf.select(ins_cfg, 'slot_state', default=None) if ins_cfg is not None else None
                pred_enabled = bool(OmegaConf.select(slot_cfg, 'predicted.enabled', default=False)) if slot_cfg is not None else False
                if pred_enabled and logits is not None and K >= V + V + 2:
                    eos_thr = float(OmegaConf.select(slot_cfg, 'predicted.eos_conf_threshold', default=0.5))
                    EOS = V + V + 1
                    probs = F.softmax(logits, dim=-1)
                    p_eos = probs[..., EOS]  # [B,H,W]
                    complete_pred = (p_eos >= eos_thr).view(B, L)
                    slot_state = slot_state & (~complete_pred)
            except Exception:
                pass

            return slot_state.view(B * L)
        except Exception:
            return None

    def _expected_cost_scalar(self, logits: torch.Tensor, tgt: torch.Tensor, aug_grid: torch.Tensor | None = None) -> torch.Tensor:
        """Compute Expected-Cost scalar loss for given logits and target grid with Phase 3 gating and anchors.
        Applies risk scaling from entropy and uses dynamic slot_state (PAD, dataloader mask, predicted completion).
        """
        # Apply insertion gating before building EC loss
        y_hat = self._apply_insertion_gating(logits, tgt)
        # Flatten logits and targets to [B, L, K] and [B, L]
        B, H, W, K = y_hat.shape
        logits_flat = y_hat.view(B, H * W, K)
        tgt_flat = tgt.view(B, H * W)

        # Base vocab from config; augmented head K should be V + V + 2
        V = int(getattr(self.config.model, "vocab_size", K))
        if K < V + V + 2:
            # Fallback to CE if head size is not augmented
            return self._compute_loss(y_hat, tgt)

        pad_id = int(getattr(self.config.model, "pad_token_id", V - 1))
        ec_cfg = OmegaConf.select(self.config.model, "loss.expected_cost", default=None)

        # Build per-slot costs; dynamic slot_state combining PAD, dataloader, and predicted completion
        slot_state = self._derive_slot_state(tgt, logits=y_hat, aug_grid=aug_grid)  # flattened [B*L] or None
        if slot_state is not None:
            slot_state = slot_state.view(B, H * W)
        costs = build_slot_costs(
            y_star=tgt_flat.clamp(min=0, max=V - 1),
            slot_state=slot_state,
            base_vocab=V,
            a=float(getattr(ec_cfg, "a", 0.15)) if ec_cfg is not None else 0.15,
            b=float(getattr(ec_cfg, "b", 0.25)) if ec_cfg is not None else 0.25,
            c=float(getattr(ec_cfg, "c", 0.60)) if ec_cfg is not None else 0.60,
            d=float(getattr(ec_cfg, "d", 1.00)) if ec_cfg is not None else 1.00,
            d_finish=float(getattr(ec_cfg, "d_finish", 1.00)) if ec_cfg is not None else 1.00,
            K=K,
        )

        # Risk scaling from entropy (epoch-based progress)
        r_cfg = getattr(ec_cfg, "risk", SimpleNamespace()) if ec_cfg is not None else SimpleNamespace()
        try:
            try:
                max_epochs = float(getattr(self.config.training, "max_epochs", 1))
            except Exception:
                max_epochs = 1.0
            try:
                progress = float(self.current_epoch) / max(max_epochs, 1.0)
            except Exception:
                progress = 1.0
            t = risk_from_entropy(
                logits_flat,
                t_min=float(getattr(r_cfg, "t_min", 0.55)),
                t_max=float(getattr(r_cfg, "t_max", 0.90)),
                alpha=float(getattr(r_cfg, "alpha", 1.0)),
                warmup_frac=float(getattr(r_cfg, "warmup_frac", 0.0)),
                progress_frac=progress,
            )
        except Exception:
            t = torch.full((B, H * W), 0.55, device=y_hat.device, dtype=torch.float32)
        costs = apply_risk_scaling(costs, t, mask_zero=True)

        # Expected cost with optional anchors (anchor only on incomplete slots)
        anchor_mask = slot_state  # True=incomplete
        loss = _expected_cost_loss(
            logits_flat,
            costs,
            valid_mask=None,  # include PAD in expected-cost loss
            anchor_ce_alpha=float(getattr(ec_cfg, "anchor_ce_alpha", 0.0)) if ec_cfg is not None else 0.0,
            entropy_floor_beta=float(getattr(ec_cfg, "entropy_floor_beta", 0.0)) if ec_cfg is not None else 0.0,
            pad_token_id=pad_id,
            base_vocab=V,
            anchor_mask=anchor_mask,
        )
        return loss

    def _fit_temperature_scaler(self, dl) -> None:
        # Collect logits/targets on CPU
        logits, targets = self._gather_logits_targets(dl)
        method = str(OmegaConf.select(self.calibration_cfg, 'method', default='temperature')) if self.calibration_cfg is not None else 'temperature'
        # Build calibrator via factory
        params = {
            'num_bins': int(OmegaConf.select(self.calibration_cfg, 'num_bins', default=15)),
            'binning': str(OmegaConf.select(self.calibration_cfg, 'binning', default='equal_width')),
            'multi_class': str(OmegaConf.select(self.calibration_cfg, 'multi_class', default='top1')),
            'a': float(OmegaConf.select(self.calibration_cfg, 'a', default=1.0)),
            'b': float(OmegaConf.select(self.calibration_cfg, 'b', default=1.0)),
        }
        calibrator = make_calibrator(method, **params)
        stats = calibrator.fit(
            logits, targets, ignore_index=self.pad_token_id,
            optimizer=str(OmegaConf.select(self.calibration_cfg, 'optimizer', default='lbfgs')),
            max_iter=int(OmegaConf.select(self.calibration_cfg, 'max_iter', default=100)),
            lr=float(OmegaConf.select(self.calibration_cfg, 'lr', default=0.01)),
            tolerance=float(OmegaConf.select(self.calibration_cfg, 'tolerance', default=1e-6)),
        )
        # Move to device for inference
        self._calibrator = calibrator  # stateless in device terms; apply returns tensors on same device as inputs
        try:
            if 'T' in stats:
                self.log('calibration/T', torch.tensor(stats.get('T')))
            self.log('calibration/nll_before', torch.tensor(stats.get('nll_before', 0.0)))
            self.log('calibration/nll_after', torch.tensor(stats.get('nll_after', 0.0)))
        except Exception:
            pass

    def teardown(self, stage: str) -> None:  # Lightning hook
        self._cleanup_file_handler()
        return super().teardown(stage)

    def __del__(self):  # Fallback if hooks aren't invoked (e.g., unit tests)
        self._cleanup_file_handler()

    def compute_loss_for_callback(self, batch):
        """Computes loss without logging, for internal use by callbacks."""
        # Unpack batch, allowing for optional augmented grid
        if len(batch) == 6:
            src, tgt, ctx_input, ctx_output, task_ids, aug_grid = batch
        else:
            src, tgt, ctx_input, ctx_output, task_ids = batch
            aug_grid = None

        # Convert target to long type
        tgt = tgt.long()

        # Perform a forward pass
        model_out = self(src, tgt, ctx_input, ctx_output, aug_grid=aug_grid)

        # Unpack model output to get logits (y_hat)
        y_hat = model_out[0] if isinstance(model_out, tuple) else model_out

        # Convert target to long type
        tgt = tgt.long()

        # Compute and return loss
        return self._compute_loss(y_hat, tgt)

    def validation_step(self, batch, batch_idx):
        #batch = self.debug_batch(batch, "validation")
        """
        Run a validation forward pass for a batch, log diagnostics and metrics, and return the computed validation loss.
        
        Performs a forward pass through the model, computes the primary cross-entropy validation loss, accumulates per-skill exact-grid accuracy tallies (if enabled), optionally computes R-Drop KL diagnostics, looped-transformer diagnostics, aggregates and logs auxiliary losses provided by the model, computes a suite of accuracy metrics (cell/grid and tolerance-based), computes a composite dense objective, logs calibration metrics (pre/post) if a calibrator is available, and emits many per-step validation metrics to the logger.
        
        Parameters:
            batch (tuple): Validation batch expected as (src, tgt, ctx_input, ctx_output, task_ids).
                - src: model input tokens.
                - tgt: target tokens (padding index expected in self.pad_token_id).
                - ctx_input / ctx_output: optional contextual input/output used by the model.
                - task_ids: per-sample skill/task identifiers used for per-skill yardstick accumulation.
            batch_idx (int): Index of the batch within the validation epoch (unused except for logging contexts).
        
        Returns:
            torch.Tensor: Scalar validation loss (primary cross-entropy plus weighted auxiliary losses) for the provided batch.
        """
        src, tgt, ctx_input, ctx_output, task_ids = batch
        tgt = tgt.long()
        model_out = self(src, tgt, ctx_input, ctx_output)
        y_hat = model_out[0] if isinstance(model_out, tuple) else model_out
        # Loss
        val_loss = self._compute_loss(y_hat, tgt)
        # STRATEGIC LOGGING: Debug high loss mystery (removed for torch.compile compatibility)
        # The tensor→Python conversions cause graph breaks in compiled code
        # logger.critical(f"[VALSTEP_DEBUG] batch_idx={batch_idx}, batch_size={tgt.size(0)}, "
        #                f"val_loss={float(val_loss.detach().cpu()):.6f}, "
        #                f"y_hat_shape={y_hat.shape}, tgt_shape={tgt.shape}, "
        #                f"y_hat_mean={float(y_hat.mean().detach().cpu()):.4f}, "
        #                f"y_hat_std={float(y_hat.std().detach().cpu()):.4f}, "
        #                f"task_ids_sample={task_ids[:3].tolist() if hasattr(task_ids, '__getitem__') else [task_ids.item()]}")
        # Yardstick per-skill accumulators (exact-grid accuracy per skill id)
        # Yardstick per-skill accumulators (exact-grid accuracy per skill id)
        dynamo_is_compiling = False
        if hasattr(torch, "compiler") and hasattr(torch.compiler, "is_compiling"):
            dynamo_is_compiling = torch.compiler.is_compiling()
        elif hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "is_compiling"):
            dynamo_is_compiling = torch._dynamo.is_compiling()

        if not dynamo_is_compiling:
            try:
                if hasattr(self, '_val_skill_totals') and hasattr(self, '_val_skill_correct'):
                    pred = torch.argmax(y_hat, dim=-1)
                    B = pred.size(0)
                    if hasattr(task_ids, 'size'):
                        sid_tensor = task_ids.view(-1).detach().cpu()
                    else:
                        sid_tensor = torch.tensor([task_ids], dtype=torch.long)
                    for i in range(B):
                        sid = int(sid_tensor[i].item()) if i < sid_tensor.numel() else int(sid_tensor[-1].item())
                        self._val_skill_totals[sid] = self._val_skill_totals.get(sid, 0) + 1
                        self._val_skill_correct[sid] = self._val_skill_correct.get(sid, 0) + int(torch.equal(pred[i], tgt[i]))
            except Exception:
                pass
        # Optional: validation-time KL diagnostic (uses consecutive dropout passes if enabled)
        try:
            if self.rdrop_enabled and bool(OmegaConf.select(self.config, "rdrop.log_val_kl", default=False)):
                # Perform two extra forwards without affecting training state
                mo1 = self(src, tgt, ctx_input, ctx_output)
                mo2 = self(src, tgt, ctx_input, ctx_output)
                yh1 = mo1[0] if isinstance(mo1, tuple) else mo1
                yh2 = mo2[0] if isinstance(mo2, tuple) else mo2
                vm = (tgt.reshape(-1) != self.pad_token_id)
                should_amp = bool(getattr(self.config.model, "amp_enabled", False) and torch.cuda.is_available())
                ctx = torch.amp.autocast("cuda") if should_amp else contextlib.nullcontext()
                with ctx:
                    vkl = masked_symmetric_kl(yh1, yh2, valid_mask=vm, reduction="mean")
                self.log('val_rdrop_kl', vkl.detach(), prog_bar=False)
        except Exception as e:
            logger.debug(f"Skipping val_rdrop_kl logging: {e}")
        # Optional: Looped transformer fixed-point diagnostics
        try:
            looped_enabled = bool(getattr(getattr(self.config.model, 'looped', None), 'enabled', False)) if hasattr(self.config, 'model') else False
            if looped_enabled:
                log_deltas = bool(OmegaConf.select(self.config, 'model.looped.diagnostics.log_deltas', default=False))
                log_per_iter_loss = bool(OmegaConf.select(self.config, 'model.looped.diagnostics.log_per_iter_loss', default=False))
                if log_deltas or log_per_iter_loss:
                    # Use scheduled b if present
                    loop_cfg = getattr(self.config.model, 'looped', None)
                    epoch = int(getattr(self, 'current_epoch', 0))
                    step = int(getattr(getattr(self, 'trainer', None), 'global_step', 0) or 0)
                    b_eff = compute_scheduled_b(loop_cfg, epoch, step)
                    mo = self(src, tgt, ctx_input, ctx_output, loop_return_all=True, loop_b_override=b_eff)
                    diag = mo[1] if (isinstance(mo, tuple) and len(mo) == 2 and isinstance(mo[1], dict)) else {}
                    # Δ-norms across loop iterations from memory_history
                    if log_deltas and isinstance(diag, dict) and 'memory_history' in diag and diag['memory_history']:
                        mems = diag['memory_history']
                        deltas = []
                        for i in range(len(mems) - 1):
                            d = (mems[i+1] - mems[i]).pow(2).sum(dim=-1).sqrt().mean()  # mean over B and L
                            deltas.append(d)
                        if deltas:
                            delta_mean = torch.stack(deltas).mean()
                            delta_last = deltas[-1]
                            self.log('val_loop_delta_mean', delta_mean.detach(), prog_bar=False)
                            self.log('val_loop_delta_last', delta_last.detach(), prog_bar=False)
                    # Per-iteration CE summaries from logits_history
                    if log_per_iter_loss and isinstance(diag, dict) and 'logits_history' in diag and diag['logits_history']:
                        lh = diag['logits_history']
                        t_flat = tgt.reshape(-1).long()
                        valid_mask = (t_flat != self.pad_token_id)
                        ces = []
                        for l in lh:
                            y_flat = l.reshape(-1, l.size(-1))
                            if valid_mask.any():
                                ces.append(F.cross_entropy(y_flat[valid_mask], t_flat[valid_mask]))
                        if ces:
                            ce_mean = torch.stack(ces).mean()
                            ce_last = ces[-1]
                            self.log('val_loop_ce_mean', ce_mean.detach(), prog_bar=False)
                            self.log('val_loop_ce_last', ce_last.detach(), prog_bar=False)
        except Exception as e:
            logger.debug(f"Skipping looped diagnostics: {e}")
        # Aggregate aux validation losses
        if isinstance(model_out, tuple) and len(model_out) == 2 and isinstance(model_out[1], dict):
            aux_dict = model_out[1]
            aux_cfg = getattr(self.config.model, "hnet", {}) if hasattr(self.config.model, "hnet") else {}
            for name, val in aux_dict.items():
                weight = float(aux_cfg.get(f"alpha_{name}", 1.0)) if isinstance(aux_cfg, dict) else float(getattr(aux_cfg, f"alpha_{name}", 1.0))
                if not isinstance(val, (torch.Tensor, int, float)):
                    continue
                if isinstance(val, torch.Tensor) and not torch.isfinite(val).all():
                    with torch.no_grad():
                        n_nan = torch.isnan(val).sum().item()
                        n_inf = torch.isinf(val).sum().item()
                        logger.warning(f"Non-finite aux '{name}' in validation: NaN={n_nan}, Inf={n_inf}. Replacing with 0.")
                    val = torch.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
                val_loss = val_loss + weight * (val if isinstance(val, torch.Tensor) else torch.tensor(val, device=y_hat.device, dtype=torch.float32))
                self.log(f"val_{name}", (val.detach() if isinstance(val, torch.Tensor) else torch.tensor(val)), prog_bar=False, batch_size=tgt.size(0))
        # Accuracies
        accuracies = self._compute_accuracy(y_hat, tgt)
        self.log('val_cell_accuracy', accuracies['cell_accuracy'], prog_bar=True)
        self.log('val_grid_accuracy', accuracies['grid_accuracy'], prog_bar=True)
        # Log tolerance grid accuracy metrics (needed for dense_grid_objective with new weights)
        for k, v in accuracies.items():
            if isinstance(k, str) and k.startswith('grid_tol_'):
                self.log(f'val_{k}', v, prog_bar=False)
        # Compute behavioral copy metrics for validation (HPO optimization targets)
        try:
            from jarc_reactor.utils.metrics import compute_copy_metrics_on_batch
            with torch.no_grad():
                pred = torch.argmax(y_hat, dim=-1)  # [B, L] or [B, H, W]
                # Note: src, tgt, pred should all be on the same device and shape-compatible
                copy_metrics = compute_copy_metrics_on_batch(src, tgt, pred)
                
                if copy_metrics['copy_rate'] is not None:
                    self.log('val_copy_rate', copy_metrics['copy_rate'], 
                            on_epoch=True, prog_bar=False, sync_dist=True, batch_size=tgt.size(0))
                if copy_metrics['change_recall'] is not None:
                    self.log('val_change_recall', copy_metrics['change_recall'], 
                            on_epoch=True, prog_bar=False, sync_dist=True, batch_size=tgt.size(0))
                if copy_metrics['change_precision'] is not None:
                    self.log('val_change_precision', copy_metrics['change_precision'], 
                            on_epoch=True, prog_bar=False, sync_dist=True, batch_size=tgt.size(0))
                if copy_metrics['transformation_f1'] is not None:
                    self.log('val_transformation_f1', copy_metrics['transformation_f1'], 
                            on_epoch=True, prog_bar=False, sync_dist=True, batch_size=tgt.size(0))
                if copy_metrics['cell_accuracy'] is not None and copy_metrics['copy_rate'] is not None:
                    # Legacy transformation score: (1 - copy_rate) * accuracy
                    transform_score_legacy = (1.0 - copy_metrics['copy_rate']) * copy_metrics['cell_accuracy']
                    self.log('val_transformation_score', transform_score_legacy, 
                            on_epoch=True, prog_bar=False, sync_dist=True, batch_size=tgt.size(0))
                if copy_metrics['transformation_f1'] is not None and copy_metrics['cell_accuracy'] is not None:
                    # F1-based transformation quality score: transformation_f1 * accuracy
                    transform_quality_score = copy_metrics['transformation_f1'] * copy_metrics['cell_accuracy']
                    self.log('val_transformation_quality_score', transform_quality_score, 
                            on_epoch=True, prog_bar=False, sync_dist=True, batch_size=tgt.size(0))
                if copy_metrics['pct_changed_target'] is not None:
                    self.log('val_pct_changed_target', copy_metrics['pct_changed_target'], 
                            on_epoch=True, prog_bar=False, sync_dist=True, batch_size=tgt.size(0))
        except Exception as e:
            logger.debug(f"Skipping val copy metrics: {e}")
        # Composite dense objective: blend dense per-cell with tolerant and exact grid signals
        try:
            # Prefer 95% tolerance if present, else first available grid_tol_* metric
            tol_key = None
            if 'grid_tol_0p95' in accuracies:
                tol_key = 'grid_tol_0p95'
            else:
                for k in accuracies.keys():
                    if isinstance(k, str) and k.startswith('grid_tol_'):
                        tol_key = k
                        break
            tol_val = accuracies.get(tol_key, 0.0) if tol_key is not None else 0.0

            # Read weights from config if provided; otherwise use experimentally determined defaults
            # Weights were optimized via CV (Sept 21-22, 2025) for Spearman correlation with final grid accuracy
            # See: docs/test_failures/hpo_vs_golden_mystery_2025-09-21.md lines 839-912
            w_cell = float(OmegaConf.select(self.config, 'metrics.dense_objective_weights.cell', default=0.0))
            w_tol  = float(OmegaConf.select(self.config, 'metrics.dense_objective_weights.tol', default=0.8))
            w_ex   = float(OmegaConf.select(self.config, 'metrics.dense_objective_weights.exact', default=0.2))

            dense_obj = (
                w_cell * accuracies['cell_accuracy']
                + w_tol * (tol_val if isinstance(tol_val, torch.Tensor) else torch.tensor(float(tol_val), device=y_hat.device))
                + w_ex * accuracies['grid_accuracy']
            )
            self.log('val_dense_grid_objective', dense_obj, prog_bar=True)
        except Exception:
            # Never fail validation due to composite objective computation
            pass
        # Log additional diagnostics; only show val_grid_tol_0p95 on progress bar (hide all others)
        for k, v in accuracies.items():
            if k in ('cell_accuracy', 'grid_accuracy'):
                continue
            try:
                # Only val_grid_tol_0p95 gets progress bar surfacing (hide all below 95%)
                if k.startswith('grid_tol_'):
                    show_on_bar = (k == 'grid_tol_0p95')
                    self.log(f'val_{k}', v, prog_bar=show_on_bar)
                else:
                    self.log(f'val_{k}', v, prog_bar=False)
            except Exception:
                pass
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=tgt.size(0))
        try:
            logger.info(f"validation_step: val_loss={float(val_loss.detach().cpu()):.6f}")
        except Exception:
            logger.info("validation_step: val_loss logged (tensor)")
        # Calibration metrics if fitted
        try:
            apply_on = str(OmegaConf.select(self.calibration_cfg, 'apply_on', default='eval')) if self.calibration_cfg is not None else 'eval'
            log_mce = bool(OmegaConf.select(self.calibration_cfg, 'log_mce', default=False)) if self.calibration_cfg is not None else False
            num_bins = int(OmegaConf.select(self.calibration_cfg, 'num_bins', default=15)) if self.calibration_cfg is not None else 15
            binning = str(OmegaConf.select(self.calibration_cfg, 'binning', default='equal_width')) if self.calibration_cfg is not None else 'equal_width'
            with torch.no_grad():
                # Always log pre metrics
                ece_pre = compute_ece_from_logits_binned(y_hat, tgt, num_bins=num_bins, ignore_index=self.pad_token_id, binning=binning)
                self.log('val_ece_pre', ece_pre, prog_bar=False)
                if log_mce:
                    mce_pre = compute_mce_from_logits_binned(y_hat, tgt, num_bins=num_bins, ignore_index=self.pad_token_id, binning=binning)
                    self.log('val_mce_pre', mce_pre, prog_bar=True)
                y_flat = y_hat.reshape(-1, y_hat.size(-1))
                t_flat = tgt.reshape(-1).long()
                mask = (t_flat != self.pad_token_id)
                nll_pre = F.cross_entropy(y_flat[mask], t_flat[mask]) if mask.any() else torch.tensor(0.0, device=y_hat.device)
                self.log('val_nll_pre', nll_pre.detach(), prog_bar=False)
                # Post metrics only if calibrator available and allowed
                if self._calibrator is not None and apply_on in ('eval', 'both'):
                    y_hat_cal = self._calibrator(y_hat)
                    ece_post = compute_ece_from_logits_binned(y_hat_cal, tgt, num_bins=num_bins, ignore_index=self.pad_token_id, binning=binning)
                    self.log('val_ece_post', ece_post, prog_bar=False)
                    if log_mce:
                        mce_post = compute_mce_from_logits_binned(y_hat_cal, tgt, num_bins=num_bins, ignore_index=self.pad_token_id, binning=binning)
                        self.log('val_mce_post', mce_post, prog_bar=True)
                    ycal_flat = y_hat_cal.reshape(-1, y_hat_cal.size(-1))
                    nll_post = F.cross_entropy(ycal_flat[mask], t_flat[mask]) if mask.any() else torch.tensor(0.0, device=y_hat.device)
                    self.log('val_nll_post', nll_post.detach(), prog_bar=False)
        except Exception as e:
            logger.debug(f"Calibration metrics skipped due to: {e}")
        return val_loss

    # --------------------------- LR logging ---------------------------
    def _log_learning_rate(self):
        """Log the current learning rate from the first param group, if available."""
        try:
            opt = getattr(self, '_optimizer_ref', None)
            if opt is None or not opt.param_groups:
                return
            lr = float(opt.param_groups[0].get('lr', 0.0))
            # Avoid spamming the progress bar; logger collects at high freq
            self.log('lr', lr, on_step=True, prog_bar=False, logger=True)
        except Exception:
            # Best-effort logging; never break training if LR logging fails
            pass

    # --------------------------- Scheduler validation ---------------------------
    def _validate_and_normalize_scheduler_cfg(self):
        """Validate scheduler config and return a normalized dict with defaults.

        Returns a dictionary with keys:
            use_cosine_annealing (bool), T_0 (int), T_mult (int), eta_min (float),
            last_epoch (int), monitor (str)
        """
        # Fetch scheduler config safely
        sch = OmegaConf.select(self.config, 'scheduler', default={})

        # Helper to read attributes from DictConfig or plain dict
        def _get(obj, key, default=None):
            try:
                if isinstance(obj, dict):
                    return obj.get(key, default)
                return getattr(obj, key, default)
            except Exception:
                return default

        use_cosine = bool(_get(sch, 'use_cosine_annealing', False))

        monitor = _get(sch, 'monitor', 'val_loss')
        if monitor is None or str(monitor).strip() == '':
            monitor = 'val_loss'

        if not use_cosine:
            # Only monitor is relevant for plateau fallback
            return {
                'use_cosine_annealing': False,
                'monitor': monitor,
            }

        # Validate CosineAnnealingWarmRestarts params
        # T_0: int >= 1
        T_0 = _get(sch, 'T_0', None)
        if T_0 is None:
            raise ValueError("scheduler.T_0 must be set for CosineAnnealingWarmRestarts")
        try:
            T_0 = int(T_0)
        except Exception:
            raise ValueError(f"scheduler.T_0 must be an int >= 1, got {T_0!r}")
        if T_0 < 1:
            raise ValueError(f"scheduler.T_0 must be >= 1, got {T_0}")

        # T_mult: int >= 1
        T_mult = _get(sch, 'T_mult', 1)
        try:
            T_mult = int(T_mult)
        except Exception:
            raise ValueError(f"scheduler.T_mult must be an int >= 1, got {T_mult!r}")
        if T_mult < 1:
            raise ValueError(f"scheduler.T_mult must be >= 1, got {T_mult}")

        # eta_min: float >= 0
        eta_min = _get(sch, 'eta_min', 0.0)
        try:
            eta_min = float(eta_min)
        except Exception:
            raise ValueError(f"scheduler.eta_min must be a float >= 0, got {eta_min!r}")
        if eta_min < 0.0:
            raise ValueError(f"scheduler.eta_min must be >= 0, got {eta_min}")

        # Optional last_epoch from config (used as default if no resume info)
        last_epoch = _get(sch, 'last_epoch', -1)
        try:
            last_epoch = int(last_epoch)
        except Exception:
            last_epoch = -1

        return {
            'use_cosine_annealing': True,
            'T_0': T_0,
            'T_mult': T_mult,
            'eta_min': eta_min,
            'last_epoch': last_epoch,
            'monitor': monitor,
        }

    def _resolve_last_epoch_for_resume(self, default: int = -1) -> int:
        """Resolve last_epoch for scheduler when resuming training.

        Priority:
        1) config.training.resume_from_epoch if set
        2) provided default (e.g., from scheduler.last_epoch in config)
        """
        # Try to fetch from training config if provided
        resume_epoch = OmegaConf.select(self.config, 'training.resume_from_epoch', default=None)
        if resume_epoch is not None:
            try:
                return int(resume_epoch)
            except Exception:
                pass
        return int(default) if default is not None else -1