"""
PyTorch Lightning module for Champion architecture (Exp 3).

Handles context pairs for full champion model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .champion_architecture import ChampionArchitecture, create_champion_architecture
from ..evaluation.metrics import compute_grid_accuracy, compute_copy_metrics_on_batch


class ChampionLightningModule(pl.LightningModule):
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
        valid_mask = (tgt_output != self.pad_token)
        correct_cells = (preds == tgt_output) & valid_mask
        
        # Per-example cell accuracy
        cell_correct_counts = correct_cells.view(correct_cells.size(0), -1).sum(dim=1)
        cell_total_counts = valid_mask.view(valid_mask.size(0), -1).sum(dim=1)
        
        # Store for per-category metrics at epoch end
        self.validation_step_outputs.append({
            'task_ids': task_ids,
            'grid_correct': grid_metrics['grid_correct'],  # Boolean per example
            'cell_correct_counts': cell_correct_counts,     # Correct cells per example
            'cell_total_counts': cell_total_counts,         # Total valid cells per example
        })
        
        # Compute transformation quality metrics
        # Note: src needs to be truncated to match tgt_output (due to target shifting)
        if src.size(1) > 1:
            src_shifted = src[:, 1:] if src.size(1) == tgt.size(1) else src[:, :tgt_output.size(1)]
            try:
                copy_metrics = compute_copy_metrics_on_batch(src_shifted, tgt_output, preds)
                self.log('val_change_recall', copy_metrics['change_recall'], batch_size=batch_size, prog_bar=False, on_step=False, on_epoch=True)
                self.log('val_change_precision', copy_metrics['change_precision'], batch_size=batch_size, prog_bar=False, on_step=False, on_epoch=True)
                self.log('val_transformation_f1', copy_metrics['transformation_f1'], batch_size=batch_size, prog_bar=False, on_step=False, on_epoch=True)
                self.log('val_copy_rate', copy_metrics['copy_rate'], batch_size=batch_size, prog_bar=False, on_step=False, on_epoch=True)
                
                # Transformation quality score: F1 * cell_accuracy
                # Combines transformation detection with prediction accuracy
                transformation_quality_score = copy_metrics['transformation_f1'] * grid_metrics['cell_accuracy']
                self.log('val_transformation_quality_score', transformation_quality_score, batch_size=batch_size, prog_bar=True, on_step=False, on_epoch=True)
            except Exception as e:
                # Transformation metrics may fail if shapes don't align - log but don't crash
                pass
        
        self.log('val_loss', loss, batch_size=batch_size, prog_bar=False, on_step=False, on_epoch=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        """Compute and print per-category accuracy at end of validation epoch."""
        if not self.validation_step_outputs:
            return
        
        # Load task categories (try to get from dataloader)
        task_categories = {}
        if hasattr(self.trainer, 'datamodule'):
            task_categories = getattr(self.trainer.datamodule, 'task_categories', {})
        
        # If not available, try to load from data directory
        if not task_categories and len(self.validation_step_outputs) > 0:
            # Try to infer data directory from validation
            try:
                from pathlib import Path
                import json
                # Assume we're in reproduction/ directory
                data_dir = Path("data/distributional_alignment")
                categories_file = data_dir / "task_categories.json"
                if categories_file.exists():
                    with open(categories_file) as f:
                        task_categories = json.load(f)
            except:
                pass
        
        # Aggregate per-category metrics
        from collections import defaultdict
        category_stats = defaultdict(lambda: {
            'grid_correct': 0, 
            'grid_total': 0,
            'cell_correct': 0,
            'cell_total': 0
        })
        
        for output in self.validation_step_outputs:
            task_ids = output['task_ids']
            grid_correct = output['grid_correct']
            cell_correct_counts = output['cell_correct_counts']
            cell_total_counts = output['cell_total_counts']
            
            for task_id, is_grid_correct, cell_correct, cell_total in zip(
                task_ids, grid_correct, cell_correct_counts, cell_total_counts
            ):
                category = task_categories.get(task_id, 'unknown')
                category_stats[category]['grid_correct'] += int(is_grid_correct)
                category_stats[category]['grid_total'] += 1
                category_stats[category]['cell_correct'] += int(cell_correct)
                category_stats[category]['cell_total'] += int(cell_total)
        
        # Print per-category accuracy
        # IMPORTANT: Use explicit print + flush to avoid Paperspace log truncation
        if category_stats:
            import sys
            
            # Force newlines to separate from progress bar
            print("\n\n")
            sys.stdout.flush()
            
            print("="*90)
            print("PER-CATEGORY VALIDATION ACCURACY (Epoch {})".format(self.current_epoch))
            print("="*90)
            print(f"{'Category':<12} {'Grids':<8} {'Grid Acc':<12} {'Cell Acc':<12}")
            print("-"*90)
            
            for category in sorted(category_stats.keys()):
                stats = category_stats[category]
                grid_acc = (stats['grid_correct'] / stats['grid_total'] * 100) if stats['grid_total'] > 0 else 0
                cell_acc = (stats['cell_correct'] / stats['cell_total'] * 100) if stats['cell_total'] > 0 else 0
                print(f"{category:<12} {stats['grid_total']:<8} {grid_acc:>10.2f}%  {cell_acc:>10.2f}%")
            
            # Overall
            total_grid_correct = sum(s['grid_correct'] for s in category_stats.values())
            total_grids = sum(s['grid_total'] for s in category_stats.values())
            total_cell_correct = sum(s['cell_correct'] for s in category_stats.values())
            total_cells = sum(s['cell_total'] for s in category_stats.values())
            
            overall_grid_acc = (total_grid_correct / total_grids * 100) if total_grids > 0 else 0
            overall_cell_acc = (total_cell_correct / total_cells * 100) if total_cells > 0 else 0
            
            print("-"*90)
            print(f"{'OVERALL':<12} {total_grids:<8} {overall_grid_acc:>10.2f}%  {overall_cell_acc:>10.2f}%")
            print("="*90)
            print("\n")
            
            # Explicit flush to ensure it appears in Paperspace logs
            sys.stdout.flush()
        
        # Clear for next epoch
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler to match Trial 69."""
        # Use Adam (not AdamW) with Trial 69's optimized hyperparameters
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=(self.hparams.beta1, self.hparams.beta2),
            weight_decay=self.hparams.weight_decay,  # 0.0 for Trial 69
        )
        
        # Use CosineAnnealingWarmRestarts (not CosineAnnealingLR) to match Trial 69
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=6,  # Trial 69 value
            T_mult=1,  # Trial 69 value
            eta_min=1.6816632143867157e-06,  # Trial 69 value
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            },
        }
