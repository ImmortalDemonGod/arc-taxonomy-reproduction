"""
Unit tests for LoRA utilities.

Following TDD: test first, implementation already exists.
"""
import tempfile
from pathlib import Path

import torch
import pytest

from src.lora_utils import is_peft_available, flatten_adapter


def test_is_peft_available():
    """Test PEFT availability check."""
    result = is_peft_available()
    assert isinstance(result, bool)


@pytest.fixture
def mock_adapter():
    """Create a mock adapter directory with safetensors file."""
    import numpy as np
    
    # Create temp directory
    tmpdir = Path(tempfile.mkdtemp())
    
    # Create mock adapter weights
    state_dict = {
        'layer1.weight': torch.randn(10, 5),
        'layer2.weight': torch.randn(5, 3),
        'layer3.bias': torch.randn(3)
    }
    
    # Save as safetensors
    try:
        from safetensors.torch import save_file
        save_file(state_dict, tmpdir / "adapter_model.safetensors")
    except ImportError:
        pytest.skip("safetensors not installed")
    
    return tmpdir, state_dict


def test_flatten_adapter(mock_adapter):
    """Test adapter flattening."""
    adapter_dir, original_state = mock_adapter
    
    flat = flatten_adapter(str(adapter_dir))
    
    # Check it's a 1D tensor
    assert flat.ndim == 1
    assert flat.dtype == torch.float32
    
    # Check size matches total params
    expected_size = sum(t.numel() for t in original_state.values())
    assert flat.shape[0] == expected_size


def test_flatten_adapter_missing_file():
    """Test error handling for missing adapter file."""
    tmpdir = Path(tempfile.mkdtemp())
    
    with pytest.raises(FileNotFoundError):
        flatten_adapter(str(tmpdir))


def test_flatten_adapter_deterministic(mock_adapter):
    """Test flattening is deterministic (same order each time)."""
    adapter_dir, _ = mock_adapter
    
    flat1 = flatten_adapter(str(adapter_dir))
    flat2 = flatten_adapter(str(adapter_dir))
    
    assert torch.allclose(flat1, flat2), "Flattening should be deterministic"
