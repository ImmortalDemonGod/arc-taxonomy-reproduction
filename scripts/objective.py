"""
Optuna objective function for visual classifier HPO.

This module contains the reusable training logic that can be called by both:
1. The HPO conductor (optimize.py) for hyperparameter search
2. The standalone training script (3_train_task_encoder.py) for single runs
"""
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import optuna

# Make reproduction/src importable
REPRO_DIR = Path(__file__).resolve().parent.parent
if str(REPRO_DIR) not in sys.path:
    sys.path.insert(0, str(REPRO_DIR))

from src.models.task_encoder_cnn import TaskEncoderCNN
from src.models.task_encoder_advanced import TaskEncoderAdvanced


def seed_everything(seed: int):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute accuracy from logits and targets."""
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / total if total > 0 else 0.0


def run_training_trial(
    trial: Optional[optuna.trial.Trial],
    hparams: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    centroids: torch.Tensor,
    device: str,
    output_dir: Path,
) -> float:
    """
    Run a single training trial with given hyperparameters.
    
    Args:
        trial: Optuna trial object (None for standalone runs)
        hparams: Dictionary of hyperparameters
        train_loader: Training data loader
        val_loader: Validation data loader
        centroids: Category centroids tensor
        device: Device to train on
        output_dir: Directory to save checkpoints
    
    Returns:
        Dict with best_val_acc and final epoch metrics (train_loss, train_acc, val_loss, val_acc, val_cat_acc)
    """
    essential_categories = ["S1", "S2", "S3", "C1", "C2", "K1", "L1", "A1", "A2"]
    # Extract hyperparameters
    encoder_type = hparams.get("encoder_type", "cnn")
    embed_dim = hparams.get("embed_dim", 256)
    lr = hparams.get("lr", 1e-3)
    weight_decay = hparams.get("weight_decay", 0.0)
    label_smoothing = hparams.get("label_smoothing", 0.0)
    epochs = hparams.get("epochs", 20)
    early_stop_patience = hparams.get("early_stop_patience", 4)
    use_scheduler = hparams.get("use_scheduler", True)
    
    # Build model based on encoder type
    if encoder_type == "context":
        model = TaskEncoderAdvanced(
            embed_dim=embed_dim,
            num_demos=3,
            context_d_model=hparams.get("ctx_d_model", 256),
            n_head=hparams.get("ctx_n_head", 8),
            pixel_layers=hparams.get("ctx_pixel_layers", 3),
            grid_layers=hparams.get("ctx_grid_layers", 2),
            dropout_rate=hparams.get("ctx_dropout", 0.1),
        ).to(device)
    else:  # cnn
        model = TaskEncoderCNN(
            embed_dim=embed_dim,
            num_demos=3,
            width_mult=hparams.get("width_mult", 1.0),
            depth=hparams.get("depth", 3),
            mlp_hidden=hparams.get("mlp_hidden", 512),
            demo_agg=hparams.get("demo_agg", "mean"),
            use_coords=hparams.get("use_coords", False),
        ).to(device)
    
    # Centroid projection (learned)
    num_categories, centroid_dim = centroids.shape
    centroid_proj = nn.Linear(centroid_dim, embed_dim, bias=False).to(device)
    
    # Similarity computation settings
    use_cosine = hparams.get("use_cosine", False)
    temperature = hparams.get("temperature", 10.0)
    
    # Optimizer
    params = list(model.parameters()) + list(centroid_proj.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    
    # Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    # Scheduler
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-6
        )
    
    # Training loop
    best_val_acc = 0.0
    no_improve = 0
    best_epoch_metrics = None
    
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        centroid_proj.train()
        train_loss = 0.0
        train_acc = 0.0
        steps = 0
        
        for demo_in, demo_out, cat_idx in train_loader:
            demo_in = demo_in.to(device)
            demo_out = demo_out.to(device)
            cat_idx = cat_idx.to(device)
            
            optimizer.zero_grad()
            emb = model(demo_in, demo_out)
            proj_centroids = centroid_proj(centroids)
            
            # Compute logits with optional cosine similarity
            if use_cosine:
                emb_n = F.normalize(emb, dim=-1)
                cent_n = F.normalize(proj_centroids, dim=-1)
                logits = (emb_n @ cent_n.t().contiguous()) * temperature
            else:
                logits = emb @ proj_centroids.t().contiguous()
            
            loss = criterion(logits, cat_idx)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += accuracy_from_logits(logits, cat_idx)
            steps += 1
        
        train_loss /= max(1, steps)
        train_acc /= max(1, steps)
        
        # Validate
        model.eval()
        centroid_proj.eval()
        val_loss = 0.0
        val_acc = 0.0
        vsteps = 0
        
        # Per-category tracking
        correct_by_cat = torch.zeros(len(essential_categories), dtype=torch.long)
        total_by_cat = torch.zeros(len(essential_categories), dtype=torch.long)
        
        with torch.no_grad():
            for demo_in, demo_out, cat_idx in val_loader:
                demo_in = demo_in.to(device)
                demo_out = demo_out.to(device)
                cat_idx = cat_idx.to(device)
                
                emb = model(demo_in, demo_out)
                proj_centroids = centroid_proj(centroids)
                
                # Compute logits with optional cosine similarity
                if use_cosine:
                    emb_n = F.normalize(emb, dim=-1)
                    cent_n = F.normalize(proj_centroids, dim=-1)
                    logits = (emb_n @ cent_n.t().contiguous()) * temperature
                else:
                    logits = emb @ proj_centroids.t().contiguous()
                
                loss = criterion(logits, cat_idx)
                
                val_loss += loss.item()
                val_acc += accuracy_from_logits(logits, cat_idx)
                vsteps += 1
                
                # Track per-category accuracy
                preds = torch.argmax(logits, dim=-1)
                for k in range(len(essential_categories)):
                    mask = (cat_idx == k)
                    total_by_cat[k] += mask.sum().item()
                    correct_by_cat[k] += (preds[mask] == k).sum().item()
        
        val_loss /= max(1, vsteps)
        val_acc /= max(1, vsteps)
        
        # Compute per-category accuracy
        val_cat_acc = {}
        for i, name in enumerate(essential_categories):
            tot = int(total_by_cat[i].item())
            cor = int(correct_by_cat[i].item())
            val_cat_acc[name] = (cor / tot) if tot > 0 else 0.0
        
        # Scheduler step
        if scheduler is not None:
            scheduler.step(val_acc)
        
        # Report intermediate value for pruning
        if trial is not None:
            trial.report(val_acc, epoch)
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        # Track best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            # Store best epoch metrics
            best_epoch_metrics = {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_cat_acc": val_cat_acc,
                "epoch": epoch
            }
            # Save checkpoint
            if output_dir is not None:
                ckpt_path = output_dir / "best_model.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'centroid_proj_state_dict': centroid_proj.state_dict(),
                    'hparams': hparams,
                    'val_acc': val_acc,
                    'epoch': epoch,
                }, ckpt_path)
        else:
            no_improve += 1
        
        # Early stopping
        if no_improve >= early_stop_patience:
            break
    
    # Return metrics dict
    if best_epoch_metrics is None:
        # Fallback if no improvement (shouldn't happen)
        best_epoch_metrics = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_cat_acc": val_cat_acc,
            "epoch": epoch
        }
    
    return best_epoch_metrics


class Objective:
    """
    Optuna objective callable for HPO.
    
    This class:
    1. Samples hyperparameters from the search space (respecting conditions)
    2. Creates trial-specific data loaders
    3. Calls run_training_trial
    4. Returns the validation accuracy
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        train_dataset,
        val_dataset,
        centroids: torch.Tensor,
        device: str,
        base_output_dir: Path,
    ):
        """
        Args:
            config: Full config dict from YAML
            train_dataset: Training dataset
            val_dataset: Validation dataset
            centroids: Category centroids tensor
            device: Device to train on
            base_output_dir: Base directory for trial outputs
        """
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.centroids = centroids
        self.device = device
        self.base_output_dir = base_output_dir
    
    def __call__(self, trial: optuna.trial.Trial) -> float:
        """
        Optuna objective function.
        
        Args:
            trial: Optuna trial object
        
        Returns:
            Validation accuracy to maximize
        """
        from src.data.arc_task_dataset import collate_arc_tasks
        
        # Start with fixed parameters
        hparams = dict(self.config["fixed"])
        sampled_params = {}
        
        # Sample hyperparameters from search space
        param_ranges = self.config["param_ranges"]
        
        for param_name, param_config in param_ranges.items():
            # Check condition
            if "condition" in param_config:
                from src.hpo.config_schema import check_condition
                if not check_condition(param_config["condition"], sampled_params):
                    continue
            
            # Sample based on type
            param_type = param_config["type"]
            if param_type == "int":
                value = trial.suggest_int(
                    param_name,
                    int(param_config["low"]),
                    int(param_config["high"]),
                    log=param_config.get("log", False)
                )
            elif param_type == "float":
                value = trial.suggest_float(
                    param_name,
                    float(param_config["low"]),
                    float(param_config["high"]),
                    log=param_config.get("log", False)
                )
            elif param_type == "categorical":
                value = trial.suggest_categorical(
                    param_name,
                    param_config["choices"]
                )
            else:
                raise ValueError(f"Unknown parameter type: {param_type}")
            
            sampled_params[param_name] = value
        
        # Merge sampled params into hparams
        hparams.update(sampled_params)
        
        # Create trial-specific data loaders
        batch_size = hparams.get("batch_size", 8)
        num_workers = hparams.get("num_workers", 2)
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_arc_tasks
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_arc_tasks
        )
        
        # Create trial output directory
        trial_dir = self.base_output_dir / f"trial_{trial.number}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        
        # Run training
        metrics = run_training_trial(
            trial=trial,
            hparams=hparams,
            train_loader=train_loader,
            val_loader=val_loader,
            centroids=self.centroids,
            device=self.device,
            output_dir=trial_dir,
        )
        
        # Save all metrics to trial user attributes for later analysis
        trial.set_user_attr("train_loss", metrics["train_loss"])
        trial.set_user_attr("train_acc", metrics["train_acc"])
        trial.set_user_attr("val_loss", metrics["val_loss"])
        trial.set_user_attr("val_acc", metrics["val_acc"])
        trial.set_user_attr("best_epoch", metrics["epoch"])
        
        # Save per-category validation accuracy
        for cat_name, cat_acc in metrics["val_cat_acc"].items():
            trial.set_user_attr(f"val_acc_{cat_name}", cat_acc)
        
        # Return validation accuracy for Optuna to maximize
        return metrics["val_acc"]
