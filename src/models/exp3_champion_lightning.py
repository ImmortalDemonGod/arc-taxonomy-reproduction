"""
PyTorch Lightning module for Champion architecture (Exp 3).

Handles context pairs for full champion model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .champion_architecture import ChampionArchitecture, create_champion_architecture
from ..evaluation.metrics import compute_grid_accuracy


class Exp3ChampionLightningModule(pl.LightningModule):
    """
    Lightning module for Champion model with context pairs.
    
    Works directly with (src, tgt, ctx_in, ctx_out) tuple from data loader.
    """
    
    def __init__(
        self,
        vocab_size: int = 11,
        d_model: int = 160,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 3,
        num_heads: int = 4,
        d_ff: int = 640,
        max_grid_size: int = 30,
        dropout: float = 0.1,
        learning_rate: float = 0.0018498849832733245,  # Trial 69 optimized value
        weight_decay: float = 0.0,  # Trial 69 had NO weight decay
        beta1: float = 0.95,  # Trial 69 optimized value
        beta2: float = 0.999,
        max_epochs: int = 100,
        pad_token: int = 10,
        use_context: bool = True,
        use_bridge: bool = True,
    ):
        """Initialize champion Lightning module."""
        super().__init__()
        self.save_hyperparameters()
        
        # For tracking per-category metrics during validation
        self.validation_step_outputs = []
        
        # Create model
        self.model = create_champion_architecture(
            vocab_size=vocab_size,
            d_model=d_model,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            max_grid_size=max_grid_size,
            dropout=dropout,
            use_context=use_context,
            use_bridge=use_bridge,
        )
        
        self.pad_token = pad_token
        self.use_context = use_context
        
    def forward(
        self, 
        src: torch.Tensor, 
        tgt: torch.Tensor,
        src_grid_shape: tuple,
        tgt_grid_shape: tuple,
        ctx_input: torch.Tensor = None,
        ctx_output: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass through model."""
        return self.model(src, tgt, src_grid_shape, tgt_grid_shape, ctx_input, ctx_output)
    
    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """
        Training step with proper target shifting.
        
        Args:
            batch: Tuple of (src, tgt, ctx_in, ctx_out, src_shapes, tgt_shapes, task_ids) from data loader
            batch_idx: Batch index
            
        Returns:
            Loss tensor
        """
        src, tgt, ctx_in, ctx_out, src_shapes, tgt_shapes, task_ids = batch
        
        # Target shifting for proper next-token prediction
        # Decoder input: tgt[:-1] (remove last token)
        # Loss target: tgt[1:] (remove first token, predict next)
        batch_size = tgt.size(0)
        
        if tgt.size(1) <= 1:
            # Skip examples with only 1 token (can't shift)
            return None
        
        tgt_input = tgt[:, :-1]   # Decoder sees all but last token
        tgt_output = tgt[:, 1:]   # Predict next token
        
        # Use actual grid shapes from data loader
        src_shape = src_shapes[0]
        tgt_shape = tgt_shapes[0]
        
        # Forward pass with shifted target
        logits = self(src, tgt_input, src_shape, tgt_shape, ctx_in, ctx_out)
        
        # Cross-entropy loss on shifted target
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_output.reshape(-1),
            ignore_index=self.pad_token,
        )
        
        self.log('train_loss', loss, batch_size=batch_size, prog_bar=True, on_step=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Validation step with grid-level and transformation metrics."""
        src, tgt, ctx_in, ctx_out, src_shapes, tgt_shapes, task_ids = batch
        
        # Target shifting for proper next-token prediction
        batch_size = tgt.size(0)
        
        if tgt.size(1) <= 1:
            return None
        
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        # Use actual grid shapes from data loader
        src_shape = src_shapes[0]
        tgt_shape = tgt_shapes[0]
        
        logits = self(src, tgt_input, src_shape, tgt_shape, ctx_in, ctx_out)
        
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_output.reshape(-1),
            ignore_index=self.pad_token,
        )
        
        # Get predictions
        preds = logits.argmax(dim=-1)
        
        # Compute grid-level accuracy metrics (HEADLINE METRICS)
        grid_metrics = compute_grid_accuracy(preds, tgt_output, self.pad_token)
        self.log('val_grid_accuracy', grid_metrics['grid_accuracy'], batch_size=batch_size, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_cell_accuracy', grid_metrics['cell_accuracy'], batch_size=batch_size, prog_bar=True, on_step=False, on_epoch=True)
        
        # Compute per-example cell accuracy
        # CRITICAL: Use same tensors as grid_metrics to ensure consistency
        valid_mask = (tgt_output != self.pad_token)
        correct_cells = (preds == tgt_output) & valid_mask
        
        # Per-example cell accuracy counts
        cell_correct_counts = correct_cells.view(correct_cells.size(0), -1).sum(dim=1)
        cell_total_counts = valid_mask.view(valid_mask.size(0), -1).sum(dim=1)
        
        
        # Store for per-category metrics at epoch end
        step_output = {
            'task_ids': task_ids,
            'grid_correct': grid_metrics['grid_correct'],  # Boolean per example
            'cell_correct_counts': cell_correct_counts,     # Correct cells per example
            'cell_total_counts': cell_total_counts,         # Total valid cells per example
        }
        
        # Add transformation metrics (CENTRALIZED in validation_helpers)
        from .validation_helpers import add_transformation_metrics
        src_shifted = src[:, 1:] if src.size(1) == tgt.size(1) else src[:, :tgt_output.size(1)]
        add_transformation_metrics(step_output, src_shifted, tgt_output, preds, cell_correct_counts, cell_total_counts, self, batch_size)
        
        self.validation_step_outputs.append(step_output)
        
        # Log validation loss (CENTRALIZED in validation_helpers for reliable checkpointing)
        from .validation_helpers import log_validation_loss
        log_validation_loss(self, loss, batch_size)
        
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
        from .validation_helpers import create_trial69_optimizer_and_scheduler
        return create_trial69_optimizer_and_scheduler(self)
