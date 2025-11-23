"""Utility functions for the ARC taxonomy reproduction package.

This package provides utility functions including numerically-stable softmax.
"""
import torch
from jaxtyping import Float


def softmax(in_features: Float[torch.Tensor, " ..."], dim: int) -> Float[torch.Tensor, " ..."]:
    """
    Numerically-stable softmax over the specified dimension using the subtract-max trick.

    Policy:
    - Upcast intermediates to float32 for stability.
    - Subtract the max along `dim` before exponentiation to avoid overflow.
    - Cast the final probabilities back to the original dtype of the input tensor.
    """
    x = in_features
    orig_dtype = x.dtype
    x32 = x.float()
    max_vals = x32.max(dim=dim, keepdim=True).values
    shifted = x32 - max_vals
    exps = torch.exp(shifted)
    sums = exps.sum(dim=dim, keepdim=True)
    out = exps / sums
    return out.to(orig_dtype)


__all__ = ['softmax']
