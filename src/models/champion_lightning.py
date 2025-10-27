"""
PyTorch Lightning module for Champion architecture (Exp 3).

Handles context pairs for full champion model.
"""
import torch
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

from .champion_architecture import create_champion_architecture


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
            batch: Tuple of (src, tgt, ctx_in, ctx_out, src_shapes, tgt_shapes) from data loader
            batch_idx: Batch index
            
        Returns:
            Loss tensor
        """
        src, tgt, ctx_in, ctx_out, src_shapes, tgt_shapes = batch
        
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
        """Validation step with proper target shifting."""
        src, tgt, ctx_in, ctx_out, src_shapes, tgt_shapes = batch
        
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
        
        # Compute accuracy on shifted target
        preds = logits.argmax(dim=-1)
        non_pad_mask = tgt_output != self.pad_token
        if non_pad_mask.sum() > 0:
            correct = (preds == tgt_output) & non_pad_mask
            accuracy = correct.sum().float() / non_pad_mask.sum()
            self.log('val_accuracy', accuracy, batch_size=batch_size, prog_bar=True, on_step=False, on_epoch=True)
        
        self.log('val_loss', loss, batch_size=batch_size, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
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
