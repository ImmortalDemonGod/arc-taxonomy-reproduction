"""
PyTorch Lightning module for Exp 2: E-D + Grid2D PE + PermInvariant Embedding

This wraps the EDWithGrid2DPEAndPermInv architecture with training logic.
"""
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from typing import Tuple, Dict, Any

from .ed_with_grid2d_pe_and_perminv import EDWithGrid2DPEAndPermInv


class Exp2PermInvLightningModule(pl.LightningModule):
    """
    Lightning module for Exp 2: E-D + Grid2D PE + PermInvariant Embedding.
    
    Tests the contribution of color-permutation equivariance.
    """
    
    def __init__(
        self,
        vocab_size: int = 11,
        d_model: int = 168,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 3,
        num_heads: int = 4,
        d_ff: int = 672,
        max_grid_size: int = 30,
        dropout: float = 0.167,
        learning_rate: float = 0.0018498849832733245,
        weight_decay: float = 0.0,
        beta1: float = 0.95,
        beta2: float = 0.999,
        max_epochs: int = 100,
        pad_token: int = 10,
    ):
        """Initialize Exp 2 model."""
        super().__init__()
        self.save_hyperparameters()
        
        # Create model
        self.model = EDWithGrid2DPEAndPermInv(
            vocab_size=vocab_size,
            d_model=d_model,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            max_grid_size=max_grid_size,
            dropout=dropout,
            pad_idx=pad_token,
        )
        self.pad_token = pad_token
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_token)
        
        # For per-category metric collection
        self.validation_step_outputs = []
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Grid2D PE auto-computes from sequence length, shapes not needed
        batch_size, src_len = src.shape
        tgt_len = tgt.size(1) if tgt.dim() > 1 else 1
        # Dummy shapes (not used by model)
        src_shape = (1, src_len)  
        tgt_shape = (1, tgt_len)
        return self.model(src, tgt, src_shape, tgt_shape)
    
    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        """Training step."""
        src, tgt, task_ids = batch
        
        # Create shifted target (input to decoder)
        tgt_input = tgt[:, :-1]  # Remove last token
        tgt_shifted = tgt[:, 1:]  # Remove first token (target labels)
        
        # Forward pass
        logits = self(src, tgt_input)
        
        # Compute loss
        batch_size, seq_len, vocab_size = logits.shape
        loss = self.criterion(
            logits.reshape(-1, vocab_size),
            tgt_shifted.reshape(-1)
        )
        
        # Log
        self.log('train_loss_step', loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train_loss_epoch', loss, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx: int):
        """Validation step."""
        from src.evaluation.metrics import compute_grid_accuracy
        
        src, tgt, task_ids = batch
        
        # Create shifted target
        tgt_input = tgt[:, :-1]
        tgt_shifted = tgt[:, 1:]
        
        # Forward pass
        logits = self(src, tgt_input)
        
        # Compute loss
        batch_size, seq_len, vocab_size = logits.shape
        loss = self.criterion(
            logits.reshape(-1, vocab_size),
            tgt_shifted.reshape(-1)
        )
        
        # Get predictions
        preds = logits.argmax(dim=-1)
        
        # Compute grid-level accuracy metrics
        grid_metrics = compute_grid_accuracy(preds, tgt_shifted, self.hparams.pad_token)
        self.log('val_grid_accuracy', grid_metrics['grid_accuracy'], batch_size=batch_size, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_cell_accuracy', grid_metrics['cell_accuracy'], batch_size=batch_size, prog_bar=True, on_step=False, on_epoch=True)
        
        # Compute per-example cell counts for category aggregation
        valid_mask = (tgt_shifted != self.hparams.pad_token)
        correct_cells = (preds == tgt_shifted) & valid_mask
        cell_correct_counts = correct_cells.sum(dim=1)  # Per-example
        cell_total_counts = valid_mask.sum(dim=1)  # Per-example
        
        # Store per-task metrics for category aggregation
        step_output = {
            'task_ids': task_ids,
            'grid_correct': grid_metrics['grid_correct'],
            'cell_correct_counts': cell_correct_counts,
            'cell_total_counts': cell_total_counts,
        }
        
        # Add transformation metrics (CENTRALIZED in validation_helpers)
        from .validation_helpers import add_transformation_metrics
        src_shifted = src[:, 1:] if src.size(1) == tgt.size(1) else src[:, :tgt_shifted.size(1)]
        add_transformation_metrics(step_output, src_shifted, tgt_shifted, preds, cell_correct_counts, cell_total_counts, self, batch_size)
        
        self.validation_step_outputs.append(step_output)
        
        # Log validation loss (CENTRALIZED in validation_helpers for reliable checkpointing)
        from .validation_helpers import log_validation_loss
        log_validation_loss(self, loss, batch_size)
        
        return loss
    
    def on_validation_epoch_end(self) -> None:
        """Aggregate and print per-category metrics at end of validation epoch."""
        from .validation_helpers import load_task_categories, aggregate_validation_metrics, print_category_table
        
        if not self.validation_step_outputs:
            return
        
        # Load task categories and aggregate metrics
        task_categories = load_task_categories()
        category_stats = aggregate_validation_metrics(self.validation_step_outputs, task_categories)
        print_category_table(category_stats, self.current_epoch)
        
        # Clear for next epoch
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        from .validation_helpers import create_trial69_optimizer_and_scheduler
        return create_trial69_optimizer_and_scheduler(self)
