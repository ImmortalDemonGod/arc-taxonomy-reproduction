"""
Champion Architecture - Full Model (Experiment 3)

Matches champion_bootstrap.ckpt architecture:
- PermInvariantEmbedding
- Grid2D PE
- Encoder-Decoder with standard PyTorch components
- ContextEncoder (processes context pairs)
- Bridge (ConcatMLP, integrates context into decoder)

This is a simplified version for architecture validation.
The full champion uses a custom TransformerModel with bridge integration 
at each decoder layer, but this version applies bridge after full decoder
for testing purposes.
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple

from ..positional_encoding import Grid2DPositionalEncoding
from ..embedding import PermInvariantEmbedding
from ..context import ContextEncoderModule
from ..bridge import ConcatMLPBridge
from ..config import ContextEncoderConfig, BridgeConfig


class ChampionArchitecture(nn.Module):
    """
    Champion architecture with all components.
    
    This is a pedagogical implementation for validation purposes.
    The actual champion uses more sophisticated integration, but this
    captures the essential architecture for testing.
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
        encoder_dropout: float = None,  # Separate encoder dropout (Trial 69: 0.1)
        decoder_dropout: float = None,  # Separate decoder dropout (Trial 69: 0.015)
        pad_idx: int = 10,
        context_config: Optional[ContextEncoderConfig] = None,
        bridge_config: Optional[BridgeConfig] = None,
    ):
        """
        Initialize champion architecture.
        
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
            context_config: Configuration for context encoder
            bridge_config: Configuration for bridge module
        """
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_grid_size = max_grid_size
        self.pad_idx = pad_idx
        
        # Use separate dropout rates if provided, otherwise use general dropout
        self.encoder_dropout = encoder_dropout if encoder_dropout is not None else dropout
        self.decoder_dropout = decoder_dropout if decoder_dropout is not None else dropout
        
        # PermInvariant Embedding
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
        
        # Encoder (with separate dropout rate)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=self.encoder_dropout,  # Trial 69: 0.1
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # Decoder (with separate dropout rate)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=self.decoder_dropout,  # Trial 69: 0.015
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # Context Encoder (if provided)
        self.has_context = context_config is not None
        if self.has_context:
            self.context_encoder = ContextEncoderModule(context_config)
            d_ctx = context_config.d_model
        else:
            self.context_encoder = None
            d_ctx = 0
        
        # Bridge (if provided)
        self.has_bridge = bridge_config is not None and self.has_context
        if self.has_bridge:
            # ConcatMLP bridge - simplified for architecture validation
            # Full champion uses more complex integration, but this captures essentials
            self.bridge = ConcatMLPBridge(
                d_model=d_model,
                d_ctx=d_ctx,
            )
        else:
            self.bridge = None
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights (excluding PermInvariant and Context)."""
        for name, p in self.named_parameters():
            if 'embedding' not in name and 'context_encoder' not in name and p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_grid_shape: Tuple[int, int],
        tgt_grid_shape: Tuple[int, int],
        ctx_input: Optional[Tensor] = None,
        ctx_output: Optional[Tensor] = None,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass with context and bridge.
        
        Args:
            src: Source sequence (batch_size, src_len)
            tgt: Target sequence (batch_size, tgt_len)
            src_grid_shape: (height, width) of source grid
            tgt_grid_shape: (height, width) of target grid
            ctx_input: Context input grids (batch_size, num_pairs, H, W)
            ctx_output: Context output grids (batch_size, num_pairs, H, W)
            src_mask: Source attention mask
            tgt_mask: Target attention mask (causal)
            src_key_padding_mask: Source padding mask
            tgt_key_padding_mask: Target padding mask
            
        Returns:
            Logits of shape (batch_size, tgt_len, vocab_size)
        """
        # Process context if provided
        context_emb = None
        if self.has_context and ctx_input is not None and ctx_output is not None:
            context_emb = self.context_encoder(ctx_input, ctx_output)
        
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
        decoder_output = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=None,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        
        # Apply bridge if available
        if self.has_bridge and context_emb is not None:
            decoder_output = self.bridge(
                decoder_output,
                context_emb,
                pad_valid_mask=None,  # Simplified for testing
            )
        
        # Project to vocabulary
        logits = self.output_proj(decoder_output)
        
        return logits


def create_champion_architecture(
    vocab_size: int = 11,
    d_model: int = 160,
    num_encoder_layers: int = 1,
    num_decoder_layers: int = 3,
    num_heads: int = 4,
    d_ff: int = 640,
    max_grid_size: int = 30,
    dropout: float = 0.1,
    encoder_dropout: float = 0.1,  # Trial 69 value
    decoder_dropout: float = 0.014891948478374184,  # Trial 69 value
    pad_idx: int = 10,
    use_context: bool = True,
    use_bridge: bool = True,
) -> ChampionArchitecture:
    """
    Factory function for creating champion architecture.
    
    Default parameters match champion_bootstrap.ckpt (Trial 69).
    """
    # Create context config (matching champion)
    context_config = None
    if use_context:
        context_config = ContextEncoderConfig(
            grid_height=30,
            grid_width=30,
            vocab_size=11,
            pad_token_id=10,
            d_model=32,
            n_head=8,  # Trial 69 value
            pixel_layers=3,  # Trial 69 value
            grid_layers=2,
            dynamic_pairs=False,  # Champion uses fixed 2 pairs
            dropout_rate=0.0,  # Trial 69 value
        )
    
    # Create bridge config (matching champion)
    bridge_config = None
    if use_bridge and use_context:
        bridge_config = BridgeConfig(
            type='concat_mlp',
            tokens=2,
            heads=8,
            hidden_factor=1.705,
            apply_to_encoder=False,
            apply_to_decoder=True,
        )
    
    return ChampionArchitecture(
        vocab_size=vocab_size,
        d_model=d_model,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        max_grid_size=max_grid_size,
        dropout=dropout,
        encoder_dropout=encoder_dropout,  # Trial 69: 0.1
        decoder_dropout=decoder_dropout,  # Trial 69: 0.015
        pad_idx=pad_idx,
        context_config=context_config,
        bridge_config=bridge_config,
    )
