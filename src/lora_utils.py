"""
LoRA utilities for visual classifier.

Simple, standalone implementations needed for Phase 0.
Following cs336 style: clear, minimal, well-documented.
"""
import torch
from pathlib import Path
from typing import Dict


def is_peft_available() -> bool:
    """Check if PEFT library is installed."""
    try:
        import peft  # noqa: F401
        return True
    except ImportError:
        return False


def flatten_adapter(adapter_path: str) -> torch.Tensor:
    """
    Flatten LoRA adapter weights to 1D vector.
    
    Args:
        adapter_path: Path to adapter directory containing adapter_model.safetensors
        
    Returns:
        1D tensor with all parameters concatenated in sorted key order
    """
    adapter_dir = Path(adapter_path)
    
    # Load safetensors
    st_path = adapter_dir / "adapter_model.safetensors"
    if not st_path.exists():
        raise FileNotFoundError(f"No adapter_model.safetensors in {adapter_dir}")
    
    try:
        from safetensors.torch import load_file
    except ImportError:
        raise ImportError("safetensors required: pip install safetensors")
    
    state_dict = load_file(str(st_path), device="cpu")
    
    # Concatenate all tensors in sorted key order
    flat_parts = []
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key]
        flat_parts.append(tensor.reshape(-1).to(torch.float32).cpu())
    
    if not flat_parts:
        return torch.empty(0, dtype=torch.float32)
    
    return torch.cat(flat_parts, dim=0)
