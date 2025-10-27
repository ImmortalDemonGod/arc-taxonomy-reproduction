"""
Context Bridge Modules.

Ported from jarc_reactor for standalone reproduction package.
"""
import torch
import torch.nn as nn
from typing import Optional

__all__ = ["ContextBridgeBase", "IdentityBridge", "ConcatMLPBridge"]


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
    
    def __init__(self, d_model: int, d_ctx: int):
        super().__init__()
        self.d_model = d_model
        self.d_ctx = d_ctx

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
