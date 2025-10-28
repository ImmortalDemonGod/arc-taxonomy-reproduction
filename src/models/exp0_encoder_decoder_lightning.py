"""
PyTorch Lightning module for Encoder-Decoder baseline (Exp 0).

Simplified for ablation experiments - works directly with data loader outputs.
"""
import torch
import pytorch_lightning as pl
# Using torch.optim directly to import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .encoder_decoder_baseline import GenericEncoderDecoder, create_encoder_decoder_model
from ..evaluation.metrics import compute_grid_accuracy, compute_copy_metrics_on_batch


class Exp0EncoderDecoderLightningModule(pl.LightningModule):
    """
    Lightning module for encoder-decoder baseline.
    
    Simplified for smoke tests and ablation experiments.
    Works directly with (src, tgt) tuple from data loader.
    """
    
    def __init__(
        self,
        vocab_size: int = 11,
        d_model: int = 128,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        num_heads: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        learning_rate: float = 0.0018498849832733245,  # Trial 69 optimized
        weight_decay: float = 0.0,  # Trial 69 had none
        beta1: float = 0.95,  # Trial 69 optimized
        beta2: float = 0.999,
        max_epochs: int = 100,
        pad_token: int = 10,
    ):
        """Initialize encoder-decoder Lightning module."""
        super().__init__()
        self.save_hyperparameters()
        
        # Create model
        self.model = create_encoder_decoder_model(
            vocab_size=vocab_size,
            d_model=d_model,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.pad_token = pad_token
        
        # For per-category metric collection
        self.validation_step_outputs = []
        
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Forward pass through model."""
        return self.model(src, tgt)
    
    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """
        Training step with proper target shifting.
        
        Args:
            batch: Tuple of (src, tgt) from data loader
            batch_idx: Batch index
            
        Returns:
            Loss tensor
        """
        src, tgt = batch
        
        # Target shifting for proper next-token prediction
        batch_size = tgt.size(0)
        
        if tgt.size(1) <= 1:
            return None
        
        tgt_input = tgt[:, :-1]   # Decoder input: remove last token
        tgt_output = tgt[:, 1:]   # Loss target: predict next token
        
        # Forward pass with shifted target
        logits = self(src, tgt_input)
        
        # Cross-entropy loss on shifted target
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_output.reshape(-1),
            ignore_index=self.pad_token,
        )
        
        self.log('train_loss', loss, batch_size=batch_size, prog_bar=True, on_step=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Validation step with proper target shifting."""
        src, tgt, task_ids = batch
        
        # Target shifting
        batch_size = tgt.size(0)
        
        if tgt.size(1) <= 1:
            return None
        
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        logits = self(src, tgt_input)
        
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_output.reshape(-1),
            ignore_index=self.pad_token,
        )
        
        # Get predictions
        preds = logits.argmax(dim=-1)
        
        # Compute grid-level accuracy metrics
        grid_metrics = compute_grid_accuracy(preds, tgt_output, self.pad_token)
        self.log('val_grid_accuracy', grid_metrics['grid_accuracy'], batch_size=batch_size, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_cell_accuracy', grid_metrics['cell_accuracy'], batch_size=batch_size, prog_bar=True, on_step=False, on_epoch=True)
        
        # Compute per-example cell counts for category aggregation
        valid_mask = (tgt_output != self.pad_token)
        correct_cells = (preds == tgt_output) & valid_mask
        cell_correct_counts = correct_cells.sum(dim=1)  # Per-example
        cell_total_counts = valid_mask.sum(dim=1)  # Per-example
        
        # Compute transformation quality metrics
        if src.size(1) > 1:
            src_shifted = src[:, 1:] if src.size(1) == tgt.size(1) else src[:, :tgt_output.size(1)]
            try:
                copy_metrics = compute_copy_metrics_on_batch(src_shifted, tgt_output, preds)
                self.log('val_change_recall', copy_metrics['change_recall'], batch_size=batch_size, prog_bar=False, on_step=False, on_epoch=True)
                self.log('val_transformation_f1', copy_metrics['transformation_f1'], batch_size=batch_size, prog_bar=False, on_step=False, on_epoch=True)
                
                # Transformation quality score: F1 * cell_accuracy
                transformation_quality_score = copy_metrics['transformation_f1'] * grid_metrics['cell_accuracy']
                self.log('val_transformation_quality_score', transformation_quality_score, batch_size=batch_size, prog_bar=True, on_step=False, on_epoch=True)
            except Exception:
                pass
        
        # Store per-task metrics for category aggregation
        self.validation_step_outputs.append({
            'task_ids': task_ids,
            'grid_correct': grid_metrics['grid_correct'],
            'cell_correct_counts': cell_correct_counts,
            'cell_total_counts': cell_total_counts,
        })
        
        self.log('val_loss', loss, batch_size=batch_size, prog_bar=False, on_step=False, on_epoch=True)
        
        return loss
    
    def on_validation_epoch_end(self):
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
        """Configure optimizer and learning rate scheduler to match Trial 69."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=(self.hparams.beta1, self.hparams.beta2),
            weight_decay=self.hparams.weight_decay,
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=6,
            T_mult=1,
            eta_min=1.6816632143867157e-06,
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            },
        }
