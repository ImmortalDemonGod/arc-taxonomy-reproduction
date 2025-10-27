"""
Test Decoder-Only Data Loader with Real ARC Data

Following TDD: Test with actual data to verify implementation.
"""
import pytest
import torch
from pathlib import Path
from src.data.decoder_only_data import (
    DecoderOnlyARCDataset,
    collate_decoder_only,
    create_decoder_only_dataloader,
)


# Use synthetic data for testing
DATA_DIR = Path("/Users/tomriddle1/Holistic-Performance-Enhancement/cultivation/systems/arc_reactor/data/synthetic_data/distributional_alignment")


def test_dataset_creation():
    """Test dataset can be created with real ARC files."""
    task_files = list(DATA_DIR.glob("*.json"))[:5]  # Use first 5 files
    
    assert len(task_files) > 0, "Should find ARC task files"
    
    dataset = DecoderOnlyARCDataset(task_files, max_seq_len=512)
    
    assert len(dataset) > 0, "Dataset should contain examples"
    print(f"Loaded {len(dataset)} examples from {len(task_files)} files")


def test_sequence_format():
    """Test sequences match expected format: [input] [SEP] [output]."""
    task_files = list(DATA_DIR.glob("*.json"))[:2]
    dataset = DecoderOnlyARCDataset(task_files, sep_token=10, max_seq_len=512)
    
    # Get first sequence
    seq = dataset[0]
    
    # Verify it's 1D tensor of longs
    assert seq.dim() == 1, "Sequence should be 1D"
    assert seq.dtype == torch.long, "Sequence should be long dtype"
    
    # Check for SEP token (10)
    sep_positions = (seq == 10).nonzero(as_tuple=True)[0]
    assert len(sep_positions) > 0, "Should contain at least one SEP token"
    
    print(f"Sequence length: {seq.size(0)}")
    print(f"SEP token at position: {sep_positions[0].item()}")


def test_collate_function():
    """Test collate function pads sequences correctly."""
    # Create mock sequences of different lengths
    seq1 = torch.tensor([1, 2, 3], dtype=torch.long)
    seq2 = torch.tensor([4, 5, 6, 7, 8], dtype=torch.long)
    seq3 = torch.tensor([9], dtype=torch.long)
    
    batch = [seq1, seq2, seq3]
    padded = collate_decoder_only(batch, pad_token=10)
    
    # Check shape
    assert padded.shape == (3, 5), f"Expected (3, 5), got {padded.shape}"
    
    # Check padding
    assert (padded[0, 3:] == 10).all(), "First sequence should be padded"
    assert (padded[2, 1:] == 10).all(), "Third sequence should be padded"
    
    # Check original values preserved
    assert (padded[0, :3] == seq1).all()
    assert (padded[1] == seq2).all()
    assert padded[2, 0] == 9


def test_dataloader_creation():
    """Test full dataloader pipeline."""
    task_files = list(DATA_DIR.glob("*.json"))[:3]
    
    dataloader = create_decoder_only_dataloader(
        task_files,
        batch_size=4,
        shuffle=False,
    )
    
    # Get first batch
    batch = next(iter(dataloader))
    
    # Verify batch properties
    assert isinstance(batch, torch.Tensor)
    assert batch.dim() == 2, "Batch should be 2D"
    assert batch.dtype == torch.long
    assert batch.size(0) <= 4, "Batch size should be ≤ 4"
    
    print(f"Batch shape: {batch.shape}")
    print(f"Batch dtype: {batch.dtype}")


def test_contract_satisfaction():
    """
    CRITICAL: Verify data loader satisfies contract from test_data_loaders.py
    
    Contract requirements:
    1. Sequence format: [INPUT] [SEP] [OUTPUT]
    2. Batch shape: (batch_size, seq_len)
    3. dtype: torch.long
    """
    task_files = list(DATA_DIR.glob("*.json"))[:5]
    dataloader = create_decoder_only_dataloader(
        task_files,
        batch_size=8,
        shuffle=False,
    )
    
    batch = next(iter(dataloader))
    
    # Contract check 1: Batch shape
    assert batch.dim() == 2, "FAILED: Batch should be (batch_size, seq_len)"
    
    # Contract check 2: dtype
    assert batch.dtype == torch.long, "FAILED: Batch should be torch.long"
    
    # Contract check 3: Contains SEP tokens
    has_sep = (batch == 10).any(dim=1).all()
    assert has_sep, "FAILED: All sequences should contain SEP token"
    
    print("✅ All contract requirements satisfied")
    print(f"   Batch shape: {batch.shape}")
    print(f"   Dtype: {batch.dtype}")


if __name__ == "__main__":
    print("Testing Decoder-Only Data Loader...")
    print("=" * 70)
    
    pytest.main([__file__, "-v", "-s"])
