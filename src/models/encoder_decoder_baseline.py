"""
Generic Encoder-Decoder Baseline Model (Experiment 0)

Standard transformer encoder-decoder using PyTorch's built-in components.
Following cs336 pedagogical style: clean, well-documented, standard implementation.

This baseline uses:
- PyTorch TransformerEncoder and TransformerDecoder
- 1D sinusoidal positional encoding
- Standard embeddings
- No specialized components (Grid2D PE, PermInvariant, Context)

Purpose: Isolate the value of encoder-decoder architecture before adding
specialized components. Expected to significantly outperform decoder-only (~15-20%).
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from ..positional_encoding_1d import PositionalEncoding1D


class GenericEncoderDecoder(nn.Module):
    """
    Generic encoder-decoder transformer for ARC tasks.
    
    Uses standard PyTorch components with 1D sinusoidal PE.
    Processes flattened 2D grids as 1D sequences.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        max_len: int = 2000,
    ):
        """
        Initialize encoder-decoder model.
        
        Args:
            vocab_size: Number of tokens (11 for ARC)
            d_model: Model embedding dimension
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            num_heads: Number of attention heads
            d_ff: Feedforward dimension
            dropout: Dropout probability
            max_len: Maximum sequence length
        """
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Embeddings (shared for encoder and decoder inputs)
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding1D(d_model, max_len, dropout)
        
        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm architecture
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
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights following best practices."""
        # Xavier/Glorot initialization for linear layers
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass.
        
        Args:
            src: Source sequence (batch_size, src_len)
            tgt: Target sequence (batch_size, tgt_len)
            src_mask: Source attention mask
            tgt_mask: Target attention mask (causal)
            src_key_padding_mask: Source padding mask
            tgt_key_padding_mask: Target padding mask
            
        Returns:
            Logits of shape (batch_size, tgt_len, vocab_size)
        """
        # Embed and add positional encoding
        src_emb = self.embedding(src) * (self.d_model ** 0.5)  # Scale embeddings
        src_emb = self.pos_encoder(src_emb)
        
        tgt_emb = self.embedding(tgt) * (self.d_model ** 0.5)
        tgt_emb = self.pos_encoder(tgt_emb)
        
        # Create causal mask for decoder if not provided
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


def create_encoder_decoder_model(
    vocab_size: int = 11,
    d_model: int = 128,
    num_encoder_layers: int = 2,
    num_decoder_layers: int = 2,
    num_heads: int = 4,
    d_ff: int = 512,
    dropout: float = 0.1,
    max_len: int = 2000,
) -> GenericEncoderDecoder:
    """
    Factory function for creating encoder-decoder model.
    
    Provides clean interface following cs336 style.
    
    Args:
        vocab_size: Number of tokens
        d_model: Model dimension
        num_encoder_layers: Encoder depth
        num_decoder_layers: Decoder depth
        num_heads: Attention heads
        d_ff: Feedforward dimension
        dropout: Dropout rate
        max_len: Max sequence length
        
    Returns:
        GenericEncoderDecoder model
    """
    return GenericEncoderDecoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=dropout,
        max_len=max_len,
    )


def prepare_grid_batch(
    input_grids: list[Tensor],
    output_grids: list[Tensor],
    pad_token: int = 10,
) -> tuple[Tensor, Tensor]:
    """
    Prepare batch of grids for encoder-decoder model.
    
    Flattens 2D grids into 1D sequences and pads to uniform length.
    
    Args:
        input_grids: List of 2D input grids
        output_grids: List of 2D output grids
        pad_token: Padding token ID
        
    Returns:
        (src, tgt): Source and target sequences
            - src: (batch_size, max_input_len)
            - tgt: (batch_size, max_output_len)
    """
    batch_size = len(input_grids)
    
    # Flatten grids
    src_seqs = [grid.flatten() for grid in input_grids]
    tgt_seqs = [grid.flatten() for grid in output_grids]
    
    # Get max lengths
    max_src_len = max(len(seq) for seq in src_seqs)
    max_tgt_len = max(len(seq) for seq in tgt_seqs)
    
    # Pad sequences
    src = torch.full((batch_size, max_src_len), pad_token, dtype=torch.long)
    tgt = torch.full((batch_size, max_tgt_len), pad_token, dtype=torch.long)
    
    for i in range(batch_size):
        src_len = len(src_seqs[i])
        tgt_len = len(tgt_seqs[i])
        src[i, :src_len] = src_seqs[i]
        tgt[i, :tgt_len] = tgt_seqs[i]
    
    return src, tgt


def create_padding_mask(sequences: Tensor, pad_token: int) -> Tensor:
    """
    Create padding mask for sequences.
    
    Args:
        sequences: Token sequences (batch_size, seq_len)
        pad_token: Padding token ID
        
    Returns:
        Boolean mask where True indicates padding positions
    """
    return sequences == pad_token
