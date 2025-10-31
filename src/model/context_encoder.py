# model/context_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math

from ..utils.positional_encoding import Grid2DPositionalEncoding
from ..config_schema import ContextEncoderConfig
from jarc_reactor.models.perm_embedding import PermInvariantEmbedding

logger = logging.getLogger(__name__)

class ContextEncoderModule(nn.Module):
    """Context encoder that processes grids by embedding pixels with positional encoding."""
    def __init__(self, config: ContextEncoderConfig):
        super().__init__()
        self.config = config

        # Embedding layer parameterised by config-driven vocab size and padding id
        vocab_size = getattr(config, "vocab_size", 11)
        pad_id = getattr(config, "pad_token_id", 10)
        self.embedding = PermInvariantEmbedding(d_model=config.d_model, vocab_size=vocab_size, pad_idx=pad_id)

        # 2D Positional Encoding
        self.pos_encoder = Grid2DPositionalEncoding(config.d_model, config.grid_height, config.grid_width)

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

        # Order-sensitive composition path (Hydra-gated)
        self.order_sensitive = bool(getattr(config, 'order_sensitive', False))
        if self.order_sensitive:
            d = int(config.d_model)
            # Compose [out, crossed, crossed-out, crossed*out] -> d_model
            use_ln = bool(getattr(config, 'order_comp_use_layernorm', True))
            comp_layers = [
                nn.Linear(4 * d, d),
                nn.ReLU(),
            ]
            if use_ln:
                comp_layers.append(nn.LayerNorm(d))
            self.order_comp = nn.Sequential(*comp_layers)
            # Additional projection for explicit raw concatenation path
            self.order_concat_proj = nn.Linear(2 * d, d)

        logger.info(f"Initialized ContextEncoderModule with config: {config}")

    def forward(self, ctx_input, ctx_output):
        """
        Process context pairs to create a single context embedding.

        Args:
            ctx_input: [batch_size, num_context_pairs, height, width]
            ctx_output: [batch_size, num_context_pairs, height, width]

        Returns:
            context_embedding: [batch_size, d_model]
        """
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
            if self.order_sensitive:
                comp_mode = getattr(self.config, 'comp_mode', 'cross_out_diff_prod')
                if comp_mode == 'raw_diff_skip':
                    # Deterministic, normalization-free order-sensitive fusion from raw roles
                    fused = output_emb + (output_emb - input_emb) + 0.5 * (output_emb * input_emb)
                elif comp_mode == 'raw_concat_mlp':
                    # Concatenate raw roles and their interactions; feed through MLP
                    comp_in = torch.cat([
                        output_emb,
                        input_emb,
                        output_emb - input_emb,
                        output_emb * input_emb,
                    ], dim=-1)
                    fused = self.order_comp(comp_in)
                    # Short-circuit final aggregation for this mode: simple mean over pairs
                    context_embedding = fused.mean(dim=1)
                    return context_embedding
                elif comp_mode == 'raw_diff_mean':
                    # Minimal deterministic aggregator: mean difference of roles
                    context_embedding = (output_emb - input_emb).mean(dim=1)
                    return context_embedding
                elif comp_mode == 'raw_concat_proj':
                    # Explicit order-preserving concatenation over role means, projected back to d
                    mean_out = output_emb.mean(dim=1)
                    mean_in = input_emb.mean(dim=1)
                    pair = torch.cat([mean_out, mean_in], dim=-1)
                    context_embedding = self.order_concat_proj(pair)
                    return context_embedding
                elif comp_mode == 'raw_out_only':
                    # Return the output-role mean directly as context embedding
                    context_embedding = output_emb.mean(dim=1)
                    return context_embedding
                elif comp_mode == 'raw_out_plus_diff':
                    # Minimal relational signal: output mean + mean difference
                    mean_out = output_emb.mean(dim=1)
                    mean_diff = (output_emb - input_emb).mean(dim=1)
                    context_embedding = mean_out + mean_diff
                    return context_embedding
                elif comp_mode == 'raw_unit_diff':
                    # Unit-normalized relational signal: normalized mean difference
                    mean_diff = (output_emb - input_emb).mean(dim=1)
                    context_embedding = F.normalize(mean_diff, p=2, dim=-1)
                    return context_embedding
                else:
                    comp_in = torch.cat([
                        output_emb,
                        crossed,
                        crossed - output_emb,
                        crossed * output_emb,
                    ], dim=-1)
                    fused = self.order_comp(comp_in)
            else:
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
            if self.order_sensitive:
                comp_mode = getattr(self.config, 'comp_mode', 'cross_out_diff_prod')
                if comp_mode == 'raw_diff_skip':
                    pair_emb = output_emb + (output_emb - input_emb) + 0.5 * (output_emb * input_emb)
                elif comp_mode == 'raw_concat_mlp':
                    comp_in = torch.cat([
                        output_emb,
                        input_emb,
                        output_emb - input_emb,
                        output_emb * input_emb,
                    ], dim=-1)
                    pair_emb = self.order_comp(comp_in)
                    # Short-circuit final aggregation for this mode: simple mean over pairs
                    context_embedding = pair_emb.mean(dim=1)
                    return context_embedding
                elif comp_mode == 'raw_diff_mean':
                    context_embedding = (output_emb - input_emb).mean(dim=1)
                    return context_embedding
                elif comp_mode == 'raw_concat_proj':
                    mean_out = output_emb.mean(dim=1)
                    mean_in = input_emb.mean(dim=1)
                    pair = torch.cat([mean_out, mean_in], dim=-1)
                    context_embedding = self.order_concat_proj(pair)
                    return context_embedding
                elif comp_mode == 'raw_out_only':
                    context_embedding = output_emb.mean(dim=1)
                    return context_embedding
                elif comp_mode == 'raw_out_plus_diff':
                    mean_out = output_emb.mean(dim=1)
                    mean_diff = (output_emb - input_emb).mean(dim=1)
                    context_embedding = mean_out + mean_diff
                    return context_embedding
                elif comp_mode == 'raw_unit_diff':
                    mean_diff = (output_emb - input_emb).mean(dim=1)
                    context_embedding = F.normalize(mean_diff, p=2, dim=-1)
                    return context_embedding
                else:
                    comp_in = torch.cat([
                        output_emb,
                        crossed,
                        crossed - output_emb,
                        crossed * output_emb,
                    ], dim=-1)
                    pair_emb = self.order_comp(comp_in)
            else:
                pair_emb = crossed + output_emb
            scores = torch.softmax(self.attn_proj(torch.tanh(pair_emb)), dim=1)
            context_embedding = (scores * pair_emb).sum(dim=1)
            return context_embedding
