"""
Permutation-Invariant Color Embedding for ARC Tasks.

Adapted from jarc_reactor for standalone reproduction package.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn

__all__ = ["PermInvariantEmbedding"]


class PermInvariantEmbedding(nn.Module):
    """
    Color-permutation equivariant embedding for ARC tasks.
    
    - Single shared weight matrix: [vocab_size, d_model]
    - Forward performs table lookup: weight[color_indices]
    - Initialization: Kaiming uniform (good for ReLU activations)
    - Preserves color permutation: emb(P(x)) = P(emb(x)) for any permutation P
    - Default vocab_size=11 (10 colors + PAD), pad_idx=10
    """
    
    def __init__(self, d_model: int, vocab_size: int = 11, pad_idx: int = 10):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.pad_idx = int(pad_idx)
        self.padding_idx = self.pad_idx  # Legacy attribute name for compatibility
        
        # Single shared weight matrix: [vocab_size, d_model]
        # All color indices share this same projection matrix
        self.G = nn.Parameter(torch.empty(vocab_size, d_model))
        
        # Initialize using Kaiming uniform (good for ReLU activations)
        nn.init.kaiming_uniform_(self.G, a=math.sqrt(5))
    
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Table lookup: weight[idx]."""
        return self.G[idx]
