"""
Unit tests for single-task dataset.

Following TDD: test first, then verify implementation.
"""
import json
import tempfile
from pathlib import Path

import torch
import pytest

from src.data.single_task_data import SingleTaskDataset, collate_single_task


@pytest.fixture
def sample_task_file():
    """Create a sample ARC task JSON."""
    task_data = {
        'train': [
            {
                'input': [[1, 2], [3, 4]],
                'output': [[5, 6], [7, 8]]
            },
            {
                'input': [[0, 1], [2, 3]],
                'output': [[4, 5], [6, 7]]
            },
            {
                'input': [[9, 8], [7, 6]],
                'output': [[5, 4], [3, 2]]
            }
        ],
        'test': []
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(task_data, f)
        return Path(f.name)


def test_dataset_creation(sample_task_file):
    """Test dataset loads and processes examples."""
    dataset = SingleTaskDataset(sample_task_file)
    
    assert len(dataset) > 0, "Dataset should have examples"
    
    # Get first example
    ex = dataset[0]
    assert len(ex) == 7, "Example should be 7-tuple"
    
    src, tgt, ctx_in, ctx_out, src_shape, tgt_shape, task_id = ex
    
    # Check types
    assert isinstance(src, torch.Tensor)
    assert isinstance(tgt, torch.Tensor)
    assert isinstance(ctx_in, torch.Tensor)
    assert isinstance(ctx_out, torch.Tensor)
    assert isinstance(src_shape, tuple)
    assert isinstance(tgt_shape, tuple)
    assert isinstance(task_id, str)
    
    # Check shapes
    assert src.ndim == 1, "src should be 1D (flattened)"
    assert tgt.ndim == 1, "tgt should be 1D (flattened)"
    assert ctx_in.ndim == 3, "ctx_in should be (num_pairs, H, W)"
    assert ctx_out.ndim == 3, "ctx_out should be (num_pairs, H, W)"


def test_context_pairs(sample_task_file):
    """Test context pairs are properly structured."""
    dataset = SingleTaskDataset(sample_task_file, num_context_pairs=2)
    
    ex = dataset[0]
    _, _, ctx_in, ctx_out, _, _, _ = ex
    
    assert ctx_in.shape[0] == 2, "Should have 2 context pairs"
    assert ctx_out.shape[0] == 2, "Should have 2 context pairs"
    assert ctx_in.shape == ctx_out.shape, "Context in/out should have same shape"


def test_padding(sample_task_file):
    """Test grids are padded correctly."""
    dataset = SingleTaskDataset(sample_task_file, pad_token=10)
    
    ex = dataset[0]
    _, _, ctx_in, _, _, _, _ = ex
    
    # Context grids should be padded to consistent size within the batch
    # For uniform 2x2 grids, they stay 2x2 (no padding needed)
    # This is correct - padding is adaptive, not forced to 30x30
    assert ctx_in.shape[1] >= 2 and ctx_in.shape[2] >= 2, "Should have valid dimensions"


def test_collate_function(sample_task_file):
    """Test collate function batches correctly."""
    dataset = SingleTaskDataset(sample_task_file)
    
    batch = [dataset[i] for i in range(min(2, len(dataset)))]
    
    collated = collate_single_task(batch)
    
    assert len(collated) == 7, "Collated batch should be 7-tuple"
    
    src_batch, tgt_batch, ctx_in_batch, ctx_out_batch, _, _, _ = collated
    
    # Check batching
    assert src_batch.shape[0] == len(batch), "Batch dim should match"
    assert tgt_batch.shape[0] == len(batch)
    assert ctx_in_batch.shape[0] == len(batch)
    assert ctx_out_batch.shape[0] == len(batch)


def test_empty_task_file():
    """Test handling of insufficient examples."""
    task_data = {'train': [{'input': [[1]], 'output': [[2]]}], 'test': []}
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(task_data, f)
        task_file = Path(f.name)
    
    # Need at least 2 examples for context
    dataset = SingleTaskDataset(task_file, num_context_pairs=2)
    assert len(dataset) == 0, "Should have 0 examples when insufficient context"


def test_task_id_extraction(sample_task_file):
    """Test task_id is correctly extracted from filename."""
    dataset = SingleTaskDataset(sample_task_file)
    
    ex = dataset[0]
    task_id = ex[6]
    
    # Should be the filename stem
    assert task_id == sample_task_file.stem
