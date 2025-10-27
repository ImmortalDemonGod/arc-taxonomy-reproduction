"""
Checkpoint loading and configuration sanitization utilities.

Handles conversion from OmegaConf (PyTorch Lightning) to clean dataclasses.
"""
import torch
from pathlib import Path
from typing import Any
from omegaconf import OmegaConf, DictConfig
from .config import ModelConfig, ContextEncoderConfig, BridgeConfig


def load_and_sanitize_config_from_checkpoint(ckpt_path: str | Path) -> ModelConfig:
    """
    Load checkpoint and extract hyper_parameters as sanitized ModelConfig.
    
    Handles OmegaConf → dataclass conversion with proper defaults and validation.
    
    Args:
        ckpt_path: Path to PyTorch Lightning checkpoint file
        
    Returns:
        ModelConfig: Sanitized configuration dataclass
        
    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        KeyError: If required config fields are missing
        ValueError: If config values are invalid
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    # Load checkpoint (need weights_only=False for OmegaConf)
    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    if 'hyper_parameters' not in checkpoint:
        raise KeyError("Checkpoint missing 'hyper_parameters' key")
    
    hparams = checkpoint['hyper_parameters']
    
    # Convert OmegaConf to plain dict if needed
    if isinstance(hparams, DictConfig):
        hparams = OmegaConf.to_container(hparams, resolve=True)
    
    # jarc_reactor nests model params under 'model' key
    if 'model' in hparams:
        model_params = hparams['model']
    else:
        model_params = hparams
    
    # Extract and validate core architecture parameters
    config_dict = {}
    
    # Encoder/Decoder dimensions (ModelConfig uses jarc_reactor field names)
    config_dict['d_model'] = _get_required(model_params, 'd_model', int)
    config_dict['encoder_layers'] = _get_required(model_params, 'encoder_layers', int)
    config_dict['decoder_layers'] = _get_required(model_params, 'decoder_layers', int)
    config_dict['n_head'] = _get_required(model_params, 'n_head', int)
    config_dict['d_ff'] = _get_required(model_params, 'd_ff', int)
    config_dict['dropout_rate'] = _get_optional(model_params, 'dropout_rate', float, 0.1)
    
    # Grid parameters (ModelConfig uses max_h/max_w directly)
    config_dict['max_h'] = _get_required(model_params, 'max_h', int)
    config_dict['max_w'] = _get_required(model_params, 'max_w', int)
    config_dict['vocab_size'] = _get_optional(model_params, 'vocab_size', int, 11)  # 0-9 colors + PAD
    
    # Context encoder configuration
    context_config = _extract_context_encoder_config(model_params)
    config_dict['context_encoder'] = context_config if context_config else None
    
    # Bridge configuration (wrapped in ConditioningConfig)
    bridge_config = _extract_bridge_config(model_params)
    if bridge_config:
        from .config import ConditioningConfig
        config_dict['conditioning'] = ConditioningConfig(bridge=bridge_config)
    else:
        config_dict['conditioning'] = None
    
    # Instantiate ModelConfig
    try:
        model_config = ModelConfig(**config_dict)
    except Exception as e:
        raise ValueError(f"Failed to create ModelConfig from checkpoint: {e}")
    
    print(f"✅ Loaded config: d_model={model_config.d_model}, "
          f"enc_layers={model_config.encoder_layers}, "
          f"dec_layers={model_config.decoder_layers}")
    
    return model_config


def _get_required(hparams: dict, key: str, expected_type: type) -> Any:
    """Get required parameter with type validation."""
    if key not in hparams:
        raise KeyError(f"Required parameter '{key}' missing from checkpoint config")
    
    value = hparams[key]
    if not isinstance(value, expected_type):
        try:
            value = expected_type(value)
        except Exception as e:
            raise ValueError(f"Parameter '{key}' must be {expected_type.__name__}, got {type(value).__name__}: {e}")
    
    return value


def _get_optional(hparams: dict, key: str, expected_type: type, default: Any) -> Any:
    """Get optional parameter with type validation and default."""
    if key not in hparams:
        return default
    
    value = hparams[key]
    if not isinstance(value, expected_type):
        try:
            value = expected_type(value)
        except Exception:
            return default
    
    return value


def _extract_context_encoder_config(model_params: dict) -> ContextEncoderConfig | None:
    """Extract context encoder configuration if present."""
    # jarc_reactor always has context_encoder in model params
    if 'context_encoder' not in model_params:
        return None
    
    context_params = model_params['context_encoder']
    if isinstance(context_params, DictConfig):
        context_params = OmegaConf.to_container(context_params, resolve=True)
    
    # ContextEncoderConfig fields match jarc_reactor structure exactly
    # Just pass through the relevant fields
    config_dict = {}
    for field_name in ['grid_height', 'grid_width', 'vocab_size', 'pad_token_id', 'd_model',
                       'n_head', 'pixel_layers', 'grid_layers', 'pe_type', 'pool_type',
                       'dynamic_pairs', 'attn_dropout', 'ffn_dropout', 'pe_dropout',
                       'dropout_rate', 'use_positional_encoding', 'order_sensitive',
                       'comp_mode', 'order_comp_use_layernorm']:
        if field_name in context_params:
            config_dict[field_name] = context_params[field_name]
    
    config = ContextEncoderConfig(**config_dict)
    
    return config


def _extract_bridge_config(model_params: dict) -> BridgeConfig | None:
    """Extract bridge configuration if present."""
    # jarc_reactor nests bridge under 'conditioning'
    if 'conditioning' not in model_params:
        return None
    
    conditioning = model_params['conditioning']
    if isinstance(conditioning, DictConfig):
        conditioning = OmegaConf.to_container(conditioning, resolve=True)
    
    if 'bridge' not in conditioning:
        return None
    
    bridge_params = conditioning['bridge']
    if isinstance(bridge_params, DictConfig):
        bridge_params = OmegaConf.to_container(bridge_params, resolve=True)
    
    # BridgeConfig fields match jarc_reactor structure exactly
    # Pass through relevant fields
    config_dict = {}
    for field_name in ['type', 'apply_to_encoder', 'apply_to_decoder', 'tokens', 'heads', 'hidden_factor']:
        if field_name in bridge_params:
            config_dict[field_name] = bridge_params[field_name]
    
    config = BridgeConfig(**config_dict)
    
    return config


def load_champion_checkpoint(ckpt_path: str | Path) -> tuple[ModelConfig, dict]:
    """
    Load champion_bootstrap.ckpt and return (config, state_dict).
    
    This is the main entry point for loading the trained checkpoint.
    
    Args:
        ckpt_path: Path to champion_bootstrap.ckpt
        
    Returns:
        (ModelConfig, state_dict): Configuration and model weights
    """
    config = load_and_sanitize_config_from_checkpoint(ckpt_path)
    
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['state_dict']
    
    # Remove Lightning wrapper prefixes if present
    clean_state_dict = {}
    for key, value in state_dict.items():
        # Remove 'model.' or 'core_model.' prefix
        if key.startswith('model.'):
            clean_key = key[6:]  # Remove 'model.'
        elif key.startswith('core_model.'):
            clean_key = key[11:]  # Remove 'core_model.'
        else:
            clean_key = key
        
        clean_state_dict[clean_key] = value
    
    print(f"✅ Loaded state_dict with {len(clean_state_dict)} keys")
    
    return config, clean_state_dict
