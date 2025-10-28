"""
PyTorch Lightning module for Exp 1: Encoder-Decoder + Grid2D PE

This wraps the EncoderDecoderWithGrid2DPE architecture with training logic.
"""
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from typing import Tuple, Dict, Any

from .ed_with_grid2d_pe import EncoderDecoderWithGrid2DPE


class Exp1Grid2DPELightningModule(pl.LightningModule):
    """
    Lightning module for Exp 1: E-D + Grid2D Positional Encoding.
    
    Tests the contribution of spatial 2D bias over generic 1D encoding.
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
        sep_token: int = 0,
    ):
        """Initialize Exp 1 model."""
        super().__init__()
        self.save_hyperparameters()
        
        self.sep_token = sep_token
        self.pad_token = pad_token
        
        # For per-category metric collection
        self.validation_step_outputs = []
        
        # Create model
        self.model = EncoderDecoderWithGrid2DPE(
            vocab_size=vocab_size,
            d_model=d_model,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            max_grid_size=max_grid_size,
            dropout=dropout,
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_token)
        
        # Metrics storage
        self.validation_outputs = []
    
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
        src, tgt = batch
        
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
        
        # Grid accuracy (exact match)
        predictions = logits.argmax(dim=-1)
        grid_correct = (predictions == tgt_shifted).all(dim=1).float().sum()
        
        # Cell accuracy (per-token)
        cell_correct = (predictions == tgt_shifted).float().sum()
        total_cells = (tgt_shifted != self.hparams.pad_token).sum()
        
        # Store outputs
        self.validation_outputs.append({
            'val_loss': loss,
            'grid_correct': grid_correct,
            'total_grids': batch_size,
            'cell_correct': cell_correct,
            'total_cells': total_cells,
        })
        
        return {'val_loss': loss}
    
    def on_validation_epoch_end(self) -> None:
        """Aggregate validation metrics."""
        # Aggregate
        avg_loss = torch.stack([x['val_loss'] for x in self.validation_outputs]).mean()
        total_grid_correct = sum(x['grid_correct'] for x in self.validation_outputs)
        total_grids = sum(x['total_grids'] for x in self.validation_outputs)
        total_cell_correct = sum(x['cell_correct'] for x in self.validation_outputs)
        total_cells = sum(x['total_cells'] for x in self.validation_outputs)
        
        # Compute accuracies
        grid_accuracy = total_grid_correct / total_grids if total_grids > 0 else 0.0
        cell_accuracy = total_cell_correct / total_cells if total_cells > 0 else 0.0
        
        # Log
        self.log('val_loss', avg_loss, prog_bar=True)
        self.log('val_grid_accuracy', grid_accuracy, prog_bar=True)
        self.log('val_cell_accuracy', cell_accuracy, prog_bar=True)
        
        # Clear outputs
        self.validation_outputs.clear()
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=(self.hparams.beta1, self.hparams.beta2),
            weight_decay=self.hparams.weight_decay,
        )
        
        scheduler = CosineAnnealingWarmRestarts(
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
            }
        }
