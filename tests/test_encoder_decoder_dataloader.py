"""
Test Encoder-Decoder Data Loader

Quick validation with real ARC data.
"""
import pytest
import torch
from pathlib import Path
from src.data.encoder_decoder_data import (
    EncoderDecoderARCDataset,
    collate_encoder_decoder,
    create_encoder_decoder_dataloader,
)


DATA_DIR = Path("/Users/tomriddle1/Holistic-Performance-Enhancement/cultivation/systems/arc_reactor/data/synthetic_data/distributional_alignment")


def test_encoder_decoder_dataset():
    """Test dataset creates src/tgt pairs."""
    task_files = list(DATA_DIR.glob("*.json"))[:5]
    dataset = EncoderDecoderARCDataset(task_files)
    
    assert len(dataset) > 0
    
    # Get first example
    src, tgt = dataset[0]
    
    assert src.dim() == 1, "src should be 1D"
    assert tgt.dim() == 1, "tgt should be 1D"
    assert src.dtype == torch.long
    assert tgt.dtype == torch.long
    
    print(f"Loaded {len(dataset)} examples")
    print(f"Example: src len={src.size(0)}, tgt len={tgt.size(0)}")


def test_encoder_decoder_dataloader():
    """Test full dataloader pipeline."""
    task_files = list(DATA_DIR.glob("*.json"))[:3]
    dataloader = create_encoder_decoder_dataloader(
        task_files,
        batch_size=4,
        shuffle=False,
    )
    
    src_batch, tgt_batch = next(iter(dataloader))
    
    # Verify batch properties
    assert src_batch.dim() == 2, "src_batch should be (B, L_src)"
    assert tgt_batch.dim() == 2, "tgt_batch should be (B, L_tgt)"
    assert src_batch.dtype == torch.long
    assert tgt_batch.dtype == torch.long
    
    print(f"Batch shapes: src={src_batch.shape}, tgt={tgt_batch.shape}")


def test_contract_satisfaction():
    """
    CRITICAL: Verify contract from test_data_loaders.py
    
    Contract requirements:
    1. Separate src/tgt format
    2. Batch shapes: (B, L_src), (B, L_tgt)
    3. dtype: torch.long
    """
    task_files = list(DATA_DIR.glob("*.json"))[:5]
    dataloader = create_encoder_decoder_dataloader(task_files, batch_size=8)
    
    src_batch, tgt_batch = next(iter(dataloader))
    
    # Contract checks
    assert src_batch.dim() == 2, "FAILED: src should be (B, L_src)"
    assert tgt_batch.dim() == 2, "FAILED: tgt should be (B, L_tgt)"
    assert src_batch.dtype == torch.long, "FAILED: dtype should be long"
    assert tgt_batch.dtype == torch.long, "FAILED: dtype should be long"
    
    print("✅ All contract requirements satisfied")
    print(f"   src shape: {src_batch.shape}")
    print(f"   tgt shape: {tgt_batch.shape}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
