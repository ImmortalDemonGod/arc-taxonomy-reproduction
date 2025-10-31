"""
LoRA utilities for visual classifier.

Simple, standalone implementations needed for Phase 0.
Following cs336 style: clear, minimal, well-documented.
"""
import torch
from pathlib import Path
from typing import Dict, Iterable


def is_peft_available() -> bool:
    """Check if PEFT library is installed."""
    try:
        import peft  # noqa: F401
        return True
    except ImportError:
        return False


def _load_adapter_state_dict(adapter_dir: Path) -> Dict[str, torch.Tensor]:
    """Load a PEFT LoRA adapter state dict from a directory.

    Prefers `adapter_model.safetensors`, falls back to `adapter_model.bin`.
    Returns tensors on CPU.
    """
    adapter_dir = Path(adapter_dir)
    st_path = adapter_dir / "adapter_model.safetensors"
    bin_path = adapter_dir / "adapter_model.bin"

    if st_path.exists():
        try:
            from safetensors.torch import load_file as safe_load_file
        except Exception as e:
            raise RuntimeError(
                "safetensors is required to read adapter_model.safetensors; install the 'safetensors' package"
            ) from e
        state = safe_load_file(str(st_path), device="cpu")
        return {k: v.to("cpu") for k, v in state.items()}
    if bin_path.exists():
        state = torch.load(bin_path, map_location="cpu")
        if not isinstance(state, dict):
            raise ValueError(f"Unexpected state type in {bin_path}: {type(state)}")
        return {k: (v.to("cpu") if isinstance(v, torch.Tensor) else v) for k, v in state.items()}

    raise FileNotFoundError(
        f"No adapter weights found under {adapter_dir} (expected adapter_model.safetensors or adapter_model.bin)"
    )


def flatten_adapter(adapter_path: str) -> torch.Tensor:
    """
    Flatten LoRA adapter weights to 1D vector in deterministic key order.
    """
    sd = _load_adapter_state_dict(Path(adapter_path))
    flat_parts = []
    for key in sorted(sd.keys()):
        tensor = sd[key]
        if not isinstance(tensor, torch.Tensor):
            continue
        flat_parts.append(tensor.reshape(-1).to(torch.float32).cpu())
    if not flat_parts:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(flat_parts, dim=0)


def average_adapters(adapter_paths: Iterable[str]) -> Dict[str, torch.Tensor]:
    """Compute element-wise mean of tensor entries across multiple adapters.

    Returns a state dict with same keys/shapes/dtypes on CPU.
    """
    paths = [str(p) for p in adapter_paths]
    if len(paths) == 0:
        raise ValueError("average_adapters() requires at least one adapter path")

    ref_sd = _load_adapter_state_dict(Path(paths[0]))
    ref_keys = [k for k, v in ref_sd.items() if isinstance(v, torch.Tensor)]

    avg_sd: Dict[str, torch.Tensor] = {}
    for k in ref_keys:
        t = ref_sd[k]
        avg_sd[k] = torch.zeros_like(t, device="cpu")

    count = 0
    for p in paths:
        cur_sd = _load_adapter_state_dict(Path(p))
        cur_keys = [k for k, v in cur_sd.items() if isinstance(v, torch.Tensor)]
        if set(cur_keys) != set(ref_keys):
            raise ValueError(
                f"State dict keys mismatch for {p}; expected {len(ref_keys)} keys, got {len(cur_keys)}"
            )
        for k in ref_keys:
            if cur_sd[k].shape != ref_sd[k].shape:
                raise ValueError(
                    f"Tensor shape mismatch for key '{k}': {cur_sd[k].shape} vs {ref_sd[k].shape} in {p}"
                )
        for k in ref_keys:
            avg_sd[k] += cur_sd[k].to("cpu")
        count += 1

    if count == 0:
        return avg_sd

    for k in ref_keys:
        avg_sd[k] = avg_sd[k] / float(count)

    return avg_sd
