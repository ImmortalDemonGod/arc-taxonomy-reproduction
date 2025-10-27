"""
PyTorch Lightning module for Decoder-Only baseline (Exp -1).

Provides training/validation loop with logging for ablation experiments.
"""
import torch
import pytorch_lightning as pl
# Using torch.optim directly to match Trial 69 configuration
from typing import Optional, Dict, Any

from .decoder_only_baseline import (
    create_decoder_only_model,
    compute_decoder_only_loss,
    flatten_grid_to_sequence,
)


class DecoderOnlyLightningModule(pl.LightningModule):
    """
    Lightning module for training decoder-only baseline.
    
    Follows cs336 pedagogical style: clean separation of model logic
    and training infrastructure.
    """
    
    def __init__(
        self,
        vocab_size: int = 11,
        context_length: int = 512,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        d_ff: int = 512,
        rope_theta: float = 10000.0,
        dropout: float = 0.1,
        learning_rate: float = 0.0018498849832733245,  # Trial 69 optimized
        weight_decay: float = 0.0,  # Trial 69 had none
        beta1: float = 0.95,  # Trial 69 optimized
        beta2: float = 0.999,
        max_epochs: int = 100,
        sep_token: int = 10,
        pad_token: int = 10,
    ):
        """
        Initialize decoder-only Lightning module.
        
        Args:
            vocab_size: Number of tokens (11 for ARC)
            context_length: Maximum sequence length
            d_model: Model dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            d_ff: Feedforward dimension
            rope_theta: RoPE base frequency
            dropout: Dropout rate
            learning_rate: Adam learning rate
            max_epochs: Total epochs for cosine schedule
            sep_token: Separator token ID
            pad_token: Padding token ID
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Create model
        self.model = create_decoder_only_model(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            rope_theta=rope_theta,
            dropout=dropout,
        )
        
        self.sep_token = sep_token
        self.pad_token = pad_token
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through model."""
        return self.model(input_ids)
    
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Training step.
        
        Args:
            batch: Token sequences (B, L) - includes input SEP output
            batch_idx: Batch index
            
        Returns:
            Loss tensor
        """
        # Batch is just the sequences tensor
        sequences = batch
        
        # Create input (all but last token) and targets (all but first token)
        input_ids = sequences[:, :-1]
        targets = sequences[:, 1:]
        
        # Forward pass
        logits = self(input_ids)
        
        # Compute loss (ignore pad tokens)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=self.pad_token,
        )
        
        # Log metrics
        batch_size = sequences.size(0)
        self.log('train_loss', loss, batch_size=batch_size, prog_bar=True, on_step=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Validation step."""
        sequences = batch
        
        # Create input and targets
        input_ids = sequences[:, :-1]
        targets = sequences[:, 1:]
        
        # Forward pass
        logits = self(input_ids)
        
        # Compute loss
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=self.pad_token,
        )
        
        # Compute accuracy (ignoring pad tokens)
        batch_size = sequences.size(0)
        preds = logits.argmax(dim=-1)
        non_pad_mask = targets != self.pad_token
        if non_pad_mask.sum() > 0:
            correct = (preds == targets) & non_pad_mask
            accuracy = correct.sum().float() / non_pad_mask.sum()
            self.log('val_accuracy', accuracy, batch_size=batch_size, prog_bar=True, on_step=False, on_epoch=True)
        
        self.log('val_loss', loss, batch_size=batch_size, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
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
    
    def predict_grid(
        self,
        input_grid: torch.Tensor,
        max_output_size: int = 30,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate output grid from input grid (inference).
        
        Args:
            input_grid: Input grid tensor (H, W)
            max_output_size: Maximum tokens to generate for output
            temperature: Sampling temperature
            
        Returns:
            Predicted output grid tokens (flat sequence)
        """
        self.eval()
        with torch.no_grad():
            # Flatten input and add SEP
            input_flat = input_grid.flatten()
            sep = torch.tensor([self.sep_token], dtype=input_flat.dtype, device=input_flat.device)
            
            # Start with input + SEP
            sequence = torch.cat([input_flat, sep])
            sequence = sequence.unsqueeze(0)  # Add batch dim
            
            # Autoregressively generate output
            for _ in range(max_output_size):
                # Get logits
                logits = self(sequence)
                
                # Get next token prediction
                next_logits = logits[0, -1, :] / temperature
                next_token = next_logits.argmax()
                
                # Append to sequence
                sequence = torch.cat([
                    sequence,
                    next_token.unsqueeze(0).unsqueeze(0)
                ], dim=1)
                
                # Stop if we predict a pad/sep token (heuristic)
                if next_token.item() == self.pad_token:
                    break
            
            # Extract output portion (after input + SEP)
            output_flat = sequence[0, len(input_flat)+1:]
            
            return output_flat
