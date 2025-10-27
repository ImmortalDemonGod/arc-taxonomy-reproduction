"""
PyTorch Lightning module for Encoder-Decoder baseline (Exp 0).

Simplified for ablation experiments - works directly with data loader outputs.
"""
import torch
import pytorch_lightning as pl
# Using torch.optim directly to match Trial 69 configuration
import torch.nn.functional as F

from .encoder_decoder_baseline import create_encoder_decoder_model


class EncoderDecoderLightningModule(pl.LightningModule):
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
        
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Forward pass through model."""
        return self.model(src, tgt)
    
    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """
        Training step.
        
        Args:
            batch: Tuple of (src, tgt) from data loader
            batch_idx: Batch index
            
        Returns:
            Loss tensor
        """
        src, tgt = batch
        
        # Forward pass - model predicts next token
        # So input to decoder is tgt, output should predict tgt shifted by 1
        logits = self(src, tgt)
        
        # Simple cross-entropy loss (ignore pad tokens)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1),
            ignore_index=self.pad_token,
        )
        
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Validation step."""
        src, tgt = batch
        
        logits = self(src, tgt)
        
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1),
            ignore_index=self.pad_token,
        )
        
        # Compute accuracy
        preds = logits.argmax(dim=-1)
        non_pad_mask = tgt != self.pad_token
        if non_pad_mask.sum() > 0:
            correct = (preds == tgt) & non_pad_mask
            accuracy = correct.sum().float() / non_pad_mask.sum()
            self.log('val_accuracy', accuracy, prog_bar=True, on_step=False, on_epoch=True)
        
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        
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
