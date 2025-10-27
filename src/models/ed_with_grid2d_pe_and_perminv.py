"""
E-D + Grid2D PE + PermInvariant Embedding (Experiment 2)

Extends Exp 1 by replacing standard embedding with PermInvariantEmbedding.

This tests the value of color-permutation equivariance for ARC tasks.
Expected improvement: +10-15% over Exp 1.
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from ..positional_encoding import Grid2DPositionalEncoding
from ..embedding import PermInvariantEmbedding


class EDWithGrid2DPEAndPermInv(nn.Module):
    """
    Encoder-decoder with Grid2D PE and PermInvariant embedding.
    
    Key additions over Exp 1:
    - PermInvariantEmbedding instead of nn.Embedding
    - Preserves color permutation equivariance
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        num_heads: int,
        d_ff: int,
        max_grid_size: int = 30,
        dropout: float = 0.1,
        pad_idx: int = 10,
    ):
        """
        Initialize model with PermInvariant embedding.
        
        Args:
            vocab_size: Number of tokens (11 for ARC)
            d_model: Model embedding dimension
            num_encoder_layers: Encoder depth
            num_decoder_layers: Decoder depth
            num_heads: Number of attention heads
            d_ff: Feedforward dimension
            max_grid_size: Maximum grid dimension
            dropout: Dropout rate
            pad_idx: Padding token index
        """
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_grid_size = max_grid_size
        self.pad_idx = pad_idx
        
        # PermInvariant Embedding (replaces nn.Embedding)
        self.embedding = PermInvariantEmbedding(
            d_model=d_model,
            vocab_size=vocab_size,
            pad_idx=pad_idx,
        )
        
        # Grid2D Positional Encoding
        self.pos_encoder = Grid2DPositionalEncoding(
            d_model=d_model,
            max_height=max_grid_size,
            max_width=max_grid_size,
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # Initialize weights (PermInvariant already has its own init)
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights (excluding PermInvariant)."""
        for name, p in self.named_parameters():
            if 'embedding' not in name and p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_grid_shape: tuple[int, int],
        tgt_grid_shape: tuple[int, int],
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass with PermInvariant embedding.
        
        Args:
            src: Source sequence (batch_size, src_len)
            tgt: Target sequence (batch_size, tgt_len)
            src_grid_shape: (height, width) of source grid
            tgt_grid_shape: (height, width) of target grid
            src_mask: Source attention mask
            tgt_mask: Target attention mask (causal)
            src_key_padding_mask: Source padding mask
            tgt_key_padding_mask: Target padding mask
            
        Returns:
            Logits of shape (batch_size, tgt_len, vocab_size)
        """
        # Embed with PermInvariant
        src_emb = self.embedding(src) * (self.d_model ** 0.5)
        tgt_emb = self.embedding(tgt) * (self.d_model ** 0.5)
        
        # Add Grid2D positional encoding + dropout
        src_emb = self.pos_encoder(src_emb)
        src_emb = self.dropout(src_emb)
        
        tgt_emb = self.pos_encoder(tgt_emb)
        tgt_emb = self.dropout(tgt_emb)
        
        # Create causal mask for decoder
        if tgt_mask is None:
            tgt_len = tgt.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                tgt_len, device=tgt.device
            )
        
        # Encode
        memory = self.encoder(
            src_emb,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
        )
        
        # Decode
        output = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=None,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        
        # Project to vocabulary
        logits = self.output_proj(output)
        
        return logits


def create_ed_with_grid2d_pe_and_perminv(
    vocab_size: int = 11,
    d_model: int = 128,
    num_encoder_layers: int = 2,
    num_decoder_layers: int = 2,
    num_heads: int = 4,
    d_ff: int = 512,
    max_grid_size: int = 30,
    dropout: float = 0.1,
    pad_idx: int = 10,
) -> EDWithGrid2DPEAndPermInv:
    """
    Factory function for creating E-D model with Grid2D PE and PermInvariant.
    
    Following cs336 style with clean interface.
    """
    return EDWithGrid2DPEAndPermInv(
        vocab_size=vocab_size,
        d_model=d_model,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        max_grid_size=max_grid_size,
        dropout=dropout,
        pad_idx=pad_idx,
    )
