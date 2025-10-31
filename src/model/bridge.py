# models/bridge.py
import torch
import torch.nn as nn
from typing import Optional


class ContextBridgeBase(nn.Module):
    """Abstract base for context-token integration bridges.

    Forward signature mirrors the integration points in TransformerModel:
      - x: [B, L, d_model]
      - context: [B, d_ctx]
      - pad_valid_mask: optional [B, L] boolean (True for valid tokens)
      - step: optional global step for schedules
      - is_eval: whether in evaluation mode
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor],
        pad_valid_mask: Optional[torch.Tensor] = None,
        *,
        step: Optional[int] = None,
        is_eval: Optional[bool] = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class IdentityBridge(ContextBridgeBase):
    """No-op bridge: returns x unchanged, ignores context."""

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor],
        pad_valid_mask: Optional[torch.Tensor] = None,
        *,
        step: Optional[int] = None,
        is_eval: Optional[bool] = None,
    ) -> torch.Tensor:
        return x


class ConcatMLPBridge(ContextBridgeBase):
    """Concatenation + MLP + LayerNorm bridge.

    This replicates the existing integration path in TransformerModel:
        nn.Sequential(
            nn.Linear(d_model + d_ctx, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model)
        )

    Phase 1 requirement: preserve exact behavior; do not apply PAD gating or schedules here.
    """

    def __init__(self, d_model: int, d_ctx: int, external_modules: Optional[dict] = None):
        super().__init__()
        self.d_model = int(d_model)
        self.d_ctx = int(d_ctx)
        # If external modules are provided (Linear, Activation, LayerNorm), use them without
        # registering as children to avoid double-parenting. Otherwise, create internal modules.
        self._external = None
        if external_modules is not None:
            # Expect keys: 'proj' (Linear), 'act' (Module), 'ln' (LayerNorm)
            self._external = {
                'proj': external_modules.get('proj'),
                'act': external_modules.get('act'),
                'ln': external_modules.get('ln'),
            }
            # Sanity: ensure provided modules match expected types
            assert isinstance(self._external['proj'], nn.Linear)
            assert isinstance(self._external['act'], nn.Module)
            assert isinstance(self._external['ln'], nn.LayerNorm)
        else:
            self.proj = nn.Linear(self.d_model + self.d_ctx, self.d_model)
            self.act = nn.ReLU()
            self.ln = nn.LayerNorm(self.d_model)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor],
        pad_valid_mask: Optional[torch.Tensor] = None,
        *,
        step: Optional[int] = None,
        is_eval: Optional[bool] = None,
    ) -> torch.Tensor:
        if context is None:
            return x
        # Expand context to per-token and concatenate
        B, L, _ = x.shape
        ctx_exp = context.unsqueeze(1).expand(B, L, -1)
        z = torch.cat([x, ctx_exp], dim=-1)
        if self._external is not None:
            out = self._external['proj'](z)
            out = self._external['act'](out)
            out = self._external['ln'](out)
        else:
            out = self.proj(z)
            out = self.act(out)
            out = self.ln(out)
        return out


class ContextTokenHead(nn.Module):
    """Project a batch context vector [B, d_ctx] into P token embeddings [B, P, d_model]."""

    def __init__(self, d_ctx: int, d_model: int, tokens: int = 2, dropout: float = 0.0):
        super().__init__()
        self.d_ctx = int(d_ctx)
        self.d_model = int(d_model)
        self.tokens = int(tokens)
        hidden = max(self.d_model, self.d_ctx)
        self.proj1 = nn.Linear(self.d_ctx, hidden)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(hidden, self.tokens * self.d_model)
        self.drop = nn.Dropout(p=float(dropout))

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        # context: [B, d_ctx]
        h = self.proj1(context)
        h = self.act(h)
        h = self.drop(h)
        h = self.proj2(h)  # [B, P*D]
        B = h.size(0)
        return h.view(B, self.tokens, self.d_model)


class CrossAttnBridge(ContextBridgeBase):
    """Token-wise micro cross-attention over learned context tokens, with gated residual mixing.

    x' = x + gate(context, mask) * MLP([x, Attn(x, ctx_tok), x - Attn, x * Attn])
    """

    def __init__(
        self,
        d_model: int,
        d_ctx: int,
        tokens: int = 2,
        heads: int = 2,
        hidden_factor: float = 2.0,
        dropout: float = 0.1,
        gate_pad_positions: bool = True,
        alpha_max: float = 1.0,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.d_ctx = int(d_ctx)
        self.gate_pad_positions = bool(gate_pad_positions)
        self.alpha_max = float(alpha_max)
        self.warmup_steps = 0  # default no warmup; can be set by caller via attribute

        self.ctx_tokens = ContextTokenHead(d_ctx=self.d_ctx, d_model=self.d_model, tokens=int(tokens), dropout=float(dropout))
        self.q_ln = nn.LayerNorm(self.d_model)
        self.mha = nn.MultiheadAttention(self.d_model, num_heads=int(heads), dropout=float(dropout), batch_first=True)

        hidden = int(hidden_factor * self.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(4 * self.d_model, hidden),
            nn.ReLU(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(hidden, self.d_model),
        )
        self.out_ln = nn.LayerNorm(self.d_model)
        # Gate from context: [B, d_ctx] -> [B, d_model]
        self.gate_proj = nn.Linear(self.d_ctx, self.d_model)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor],
        pad_valid_mask: Optional[torch.Tensor] = None,
        *,
        step: Optional[int] = None,
        is_eval: Optional[bool] = None,
    ) -> torch.Tensor:
        if context is None:
            return x
        B, L, D = x.shape
        # Build context tokens
        ctx_tok = self.ctx_tokens(context)  # [B, P, D]
        # Cross-attention: queries are normalized token states
        x_q = self.q_ln(x)
        attn_out, _ = self.mha(query=x_q, key=ctx_tok, value=ctx_tok, need_weights=False)
        # Feature mixing
        mix_in = torch.cat([x, attn_out, x - attn_out, x * attn_out], dim=-1)
        upd = self.mlp(mix_in)
        # Gate
        gate = self.sigmoid(self.gate_proj(context))  # [B, D]
        # Warmup schedule
        if is_eval:
            alpha = self.alpha_max
        else:
            if step is None or self.warmup_steps <= 0:
                alpha = self.alpha_max
            else:
                # linear warmup 0..alpha_max
                alpha = min(1.0, float(step) / float(self.warmup_steps)) * self.alpha_max
        gate = gate * alpha
        gate = gate.unsqueeze(1).expand(B, L, D)
        if self.gate_pad_positions and pad_valid_mask is not None:
            # pad_valid_mask True for valid tokens; zero-out residual for PAD positions
            gate = gate * pad_valid_mask.unsqueeze(-1).to(gate.dtype)
        out = x + gate * upd
        out = self.out_ln(out)
        return out


class HybridBridge(ContextBridgeBase):
    """Hybrid concat + cross-attn bridge with optional HyperFiLM and warmup schedule.

    Pipeline:
      1) Optional HyperFiLM: y = (1 + gain*gamma) * LN(x) + gain*beta.
      2) Concat path: Linear+ReLU over [x | ctx] -> upd_c.
      3) Cross-attn path: LN(x) -> MHA(q=x, k/v=ctx_tok) -> MLP mix -> upd_a.
      4) Combine: x' = y + gate * (upd_c + upd_a), with PAD gating and warmup.
    """

    def __init__(
        self,
        d_model: int,
        d_ctx: int,
        tokens: int = 2,
        heads: int = 2,
        hidden_factor: float = 2.0,
        dropout: float = 0.1,
        gate_pad_positions: bool = True,
        alpha_max: float = 1.0,
        film_enabled: bool = False,
        film_gain: float = 1.0,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.d_ctx = int(d_ctx)
        self.gate_pad_positions = bool(gate_pad_positions)
        self.alpha_max = float(alpha_max)
        self.warmup_steps = 0
        self.film_enabled = bool(film_enabled)
        self.film_gain = float(film_gain)

        # Concat path
        self.concat_proj = nn.Linear(self.d_model + self.d_ctx, self.d_model)
        self.concat_act = nn.ReLU()
        self.concat_ln = nn.LayerNorm(self.d_model)

        # Cross-attn path
        self.ctx_tokens = ContextTokenHead(d_ctx=self.d_ctx, d_model=self.d_model, tokens=int(tokens), dropout=float(dropout))
        self.q_ln = nn.LayerNorm(self.d_model)
        self.mha = nn.MultiheadAttention(self.d_model, num_heads=int(heads), dropout=float(dropout), batch_first=True)
        hidden = int(hidden_factor * self.d_model)
        self.mix_mlp = nn.Sequential(
            nn.Linear(4 * self.d_model, hidden),
            nn.ReLU(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(hidden, self.d_model),
        )
        self.out_ln = nn.LayerNorm(self.d_model)

        # HyperFiLM
        if self.film_enabled:
            self.film_ln = nn.LayerNorm(self.d_model)
            self.film_proj = nn.Linear(self.d_ctx, 2 * self.d_model)
        else:
            self.film_ln = None
            self.film_proj = None

        # Gate from context
        self.gate_proj = nn.Linear(self.d_ctx, self.d_model)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor],
        pad_valid_mask: Optional[torch.Tensor] = None,
        *,
        step: Optional[int] = None,
        is_eval: Optional[bool] = None,
    ) -> torch.Tensor:
        if context is None:
            return x
        B, L, D = x.shape

        # 1) HyperFiLM (optional)
        base = x
        if self.film_enabled:
            ln_x = self.film_ln(x)
            gb = self.film_proj(context)  # [B, 2D]
            gamma, beta = gb.chunk(2, dim=-1)
            base = (1.0 + self.film_gain * gamma).unsqueeze(1) * ln_x + (self.film_gain * beta).unsqueeze(1)

        # 2) Concat path
        ctx_exp = context.unsqueeze(1).expand(B, L, -1)
        z = torch.cat([x, ctx_exp], dim=-1)
        upd_c = self.concat_ln(self.concat_act(self.concat_proj(z)))

        # 3) Cross-attn path
        ctx_tok = self.ctx_tokens(context)
        x_q = self.q_ln(x)
        attn_out, _ = self.mha(query=x_q, key=ctx_tok, value=ctx_tok, need_weights=False)
        mix_in = torch.cat([x, attn_out, x - attn_out, x * attn_out], dim=-1)
        upd_a = self.mix_mlp(mix_in)

        # 4) Combine with gate and warmup schedule
        gate = self.sigmoid(self.gate_proj(context))  # [B, D]
        if is_eval:
            alpha = self.alpha_max
        else:
            if step is None or self.warmup_steps <= 0:
                alpha = self.alpha_max
            else:
                alpha = min(1.0, float(step) / float(self.warmup_steps)) * self.alpha_max
        gate = gate * alpha
        gate = gate.unsqueeze(1).expand(B, L, D)
        if self.gate_pad_positions and pad_valid_mask is not None:
            gate = gate * pad_valid_mask.unsqueeze(-1).to(gate.dtype)

        out = base + gate * (upd_c + upd_a)
        out = self.out_ln(out)
        return out
