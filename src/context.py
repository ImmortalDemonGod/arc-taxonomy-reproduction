"""
Context Encoder Module.

Processes pairs of (input, output) grids from demonstration examples
to create a context embedding that conditions the main model.

Ported from jarc_reactor for standalone reproduction package.
"""
import torch
import torch.nn as nn
import math
from typing import Optional

from .positional_encoding import Grid2DPositionalEncoding
from .config import ContextEncoderConfig
from .embedding import PermInvariantEmbedding

__all__ = ["ContextEncoderModule"]


class ContextEncoderModule(nn.Module):
    """
    Context encoder that processes input/output grid pairs into a single embedding.
    
    - Pixel-level embedding with Grid2D positional encoding
    - Intra-grid self-attention over pixels
    - Masked mean pooling (excludes PAD tokens)
    - Grid-level cross-attention (output attends to input)
    - Attention-weighted pooling over context pairs
    - Supports fixed pairs (champion: 2) or dynamic pairs
    """
    
    def __init__(self, config: ContextEncoderConfig):
        super().__init__()
        self.config = config

        # Embedding layer parameterised by config-driven vocab size and padding id
        vocab_size = getattr(config, "vocab_size", 11)
        pad_id = getattr(config, "pad_token_id", 10)
        self.embedding = PermInvariantEmbedding(
            d_model=config.d_model, 
            vocab_size=vocab_size, 
            pad_idx=pad_id
        )

        # 2D Positional Encoding
        self.pos_encoder = Grid2DPositionalEncoding(
            config.d_model, 
            config.grid_height, 
            config.grid_width
        )

        # Self-attention layer using TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_head,
            dropout=config.dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # --- NEW modules for combined context fusion ---
        # 1. Grid-type embedding (0 = ctx_in, 1 = ctx_out)
        self.grid_type_embedding = nn.Embedding(2, config.d_model)
        # Optional grid position embedding (small table; order sensitivity for multi-pair contexts)
        self.grid_pos_embedding = nn.Embedding(16, config.d_model)

        # Dynamic pair handling toggle
        self.dynamic_pairs = getattr(config, 'dynamic_pairs', False)
        # 2. Cross-attention at grid level (output queries → input keys/values)
        self.cross_attn = nn.MultiheadAttention(config.d_model, config.n_head, batch_first=True)
        # 3. Attention pooling over context pairs
        self.attn_proj = nn.Linear(config.d_model, 1, bias=False)
        # Expose last cross-attention weights for diagnostics (shape [B, C_out, C_in] when averaged)
        self.last_cross_attn = None

    def forward(self, ctx_input, ctx_output):
        """Process context pairs → single context embedding [batch, d_model]."""
        # Concatenate context pairs. Shape: [B, N, H, W] -> [B, 2*N, H, W]
        x = torch.cat([ctx_input, ctx_output], dim=1)
        
        batch_size, num_grids, height, width = x.shape

        # Validate grid size against configured maximums to prevent positional
        # encoding table overflow and incorrect encodings.
        max_h = int(self.config.grid_height)
        max_w = int(self.config.grid_width)
        if height > max_h or width > max_w:
            raise ValueError(
                f"Grid size exceeds configured maximum; got {height}x{width}, "
                f"but config supports up to {max_h}x{max_w}. "
                "Increase ContextEncoderConfig.grid_height/width in config."
            )

        # Reshape for pixel embedding. Shape: [B, num_grids, H, W] -> [B * num_grids, H * W]
        x_flat = x.view(batch_size * num_grids, height * width)

        # Embed pixels and add positional encoding
        # Shape: [B * num_grids, H * W] -> [B * num_grids, H * W, d_model]
        embedded = self.embedding(x_flat.long()) * math.sqrt(self.config.d_model)
        if self.config.use_positional_encoding:
            pos_encoded = self.pos_encoder(embedded)
        else:
            pos_encoded = embedded

        # Intra-grid self-attention over pixels (pads remain but contribute via masking later)
        attended = self.transformer_encoder(pos_encoded)

        # Masked mean pooling over valid (non-pad) pixels.
        # Shape: [B * num_grids, H * W, d_model] -> [B * num_grids, d_model]
        pixel_mask = (x_flat != self.embedding.padding_idx).unsqueeze(-1)           # [B*num_grids, H*W, 1]
        pixel_mask_float = pixel_mask.float()
        sum_embeddings = (attended * pixel_mask_float).sum(dim=1)
        valid_counts = pixel_mask_float.sum(dim=1).clamp(min=1.0)                   # avoid div/0
        grid_embeddings = sum_embeddings / valid_counts

        # Reshape to a sequence of grids. Shape: [B * num_grids, d_model] -> [B, num_grids, d_model]
        grid_sequence = grid_embeddings.view(batch_size, num_grids, self.config.d_model)
        # Add a small positional hint so that the order of grids can be learned when needed
        pos_ids = torch.arange(num_grids, device=x.device).unsqueeze(0).expand(batch_size, -1)
        grid_sequence = grid_sequence + self.grid_pos_embedding(pos_ids)

        # -------------------------------------------------------------
        # Combined fusion strategy with optional dynamic pair counts
        # -------------------------------------------------------------
        if self.dynamic_pairs:
            # ctx_input & ctx_output may have different counts; handle generically
            num_in = ctx_input.size(1)
            num_out = ctx_output.size(1)
            total = num_in + num_out

            # Add type embeddings
            type_ids = torch.cat([
                torch.zeros(num_in, dtype=torch.long, device=x.device),
                torch.ones(num_out, dtype=torch.long, device=x.device)
            ])
            type_ids = type_ids.unsqueeze(0).expand(batch_size, -1)  # [B,total]
            grid_sequence = grid_sequence + self.grid_type_embedding(type_ids)

            # Split according to actual sizes
            input_emb = grid_sequence[:, :num_in, :]
            output_emb = grid_sequence[:, num_in:, :]
            # Store raw role embeddings (mean over grids) before cross-attn/MLP for diagnostics
            try:
                raw_in = input_emb.mean(dim=1).detach()
                raw_out = output_emb.mean(dim=1).detach()
                self.last_raw_roles = (raw_in, raw_out)
            except Exception:
                self.last_raw_roles = None

            crossed, attn_w = self.cross_attn(
                query=output_emb, key=input_emb, value=input_emb,
                need_weights=True, average_attn_weights=True
            )
            # Store averaged attention weights for diagnostics
            try:
                self.last_cross_attn = attn_w.detach()
            except Exception:
                self.last_cross_attn = attn_w
            # Champion uses order_sensitive=false: simple fusion
            fused = crossed + output_emb

            scores = torch.softmax(self.attn_proj(torch.tanh(fused)), dim=1)
            context_embedding = (scores * fused).sum(dim=1)
            return context_embedding
        else:
            if num_grids % 2 != 0:
                raise ValueError("Expected even number of grids (input+output pairs), got %d" % num_grids)
            num_pairs = num_grids // 2

            type_ids = torch.cat([
                torch.zeros(num_pairs, dtype=torch.long, device=x.device),
                torch.ones(num_pairs, dtype=torch.long, device=x.device)
            ]).unsqueeze(0).expand(batch_size, -1)
            grid_sequence = grid_sequence + self.grid_type_embedding(type_ids)

            input_emb, output_emb = torch.split(grid_sequence, num_pairs, dim=1)
            # Store raw role embeddings (mean over grids) before cross-attn/MLP for diagnostics
            try:
                raw_in = input_emb.mean(dim=1).detach()
                raw_out = output_emb.mean(dim=1).detach()
                self.last_raw_roles = (raw_in, raw_out)
            except Exception:
                self.last_raw_roles = None
            crossed, attn_w = self.cross_attn(
                query=output_emb, key=input_emb, value=input_emb,
                need_weights=True, average_attn_weights=True
            )
            try:
                self.last_cross_attn = attn_w.detach()
            except Exception:
                self.last_cross_attn = attn_w
            # Champion uses order_sensitive=false: simple fusion
            pair_emb = crossed + output_emb
            scores = torch.softmax(self.attn_proj(torch.tanh(pair_emb)), dim=1)
            context_embedding = (scores * pair_emb).sum(dim=1)
            return context_embedding
