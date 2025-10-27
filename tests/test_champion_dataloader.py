"""
Test Champion Data Loader (Context Pairs)

Critical validation with real ARC data.
"""
import pytest
import torch
from pathlib import Path
from src.data.champion_data import (
    ChampionARCDataset,
    collate_champion,
    create_champion_dataloader,
)


DATA_DIR = Path("/Users/tomriddle1/Holistic-Performance-Enhancement/cultivation/systems/arc_reactor/data/synthetic_data/distributional_alignment")


def test_champion_dataset():
    """Test dataset creates src/tgt + context pairs."""
    task_files = list(DATA_DIR.glob("*.json"))[:10]  # More files for context
    dataset = ChampionARCDataset(task_files, num_context_pairs=2)
    
    assert len(dataset) > 0, "Should load examples"
    
    # Get first example
    src, tgt, ctx_input, ctx_output, src_shape, tgt_shape, task_id = dataset[0]
    
    assert src.dim() == 1, "src should be 1D"
    assert tgt.dim() == 1, "tgt should be 1D"
    assert ctx_input.dim() == 3, "ctx_input should be (num_pairs, H, W)"
    assert ctx_output.dim() == 3, "ctx_output should be (num_pairs, H, W)"
    
    # CRITICAL: Context pairs must have matching H, W
    assert ctx_input.shape == ctx_output.shape, "Context shapes must match!"
    
    print(f"Loaded {len(dataset)} examples")
    print(f"Context shape: {ctx_input.shape}")
    print(f"Grid shapes: src={src_shape}, tgt={tgt_shape}")


def test_champion_dataloader():
    """Test full dataloader pipeline."""
    task_files = list(DATA_DIR.glob("*.json"))[:10]
    dataloader = create_champion_dataloader(
        task_files,
        batch_size=4,
        shuffle=False,
    )
    
    src, tgt, ctx_in, ctx_out, src_shapes, tgt_shapes, task_ids = next(iter(dataloader))
    
    # Verify batch properties
    assert src.dim() == 2, "src should be (B, L)"
    assert tgt.dim() == 2, "tgt should be (B, L)"
    assert ctx_in.dim() == 4, "ctx_in should be (B, num_pairs, H, W)"
    assert ctx_out.dim() == 4, "ctx_out should be (B, num_pairs, H, W)"
    
    # CRITICAL: Context pairs have matching shape
    assert ctx_in.shape == ctx_out.shape, "Context batch shapes must match!"
    
    print(f"Batch shapes:")
    print(f"  src: {src.shape}, tgt: {tgt.shape}")
    print(f"  ctx_in: {ctx_in.shape}, ctx_out: {ctx_out.shape}")
    print(f"  Grid shapes: {src_shapes}, {tgt_shapes}")


def test_contract_satisfaction():
    """
    CRITICAL: Verify contract from test_data_loaders.py
    
    Contract requirements:
    1. Context pairs: (B, num_pairs, H, W)
    2. Context input/output must have matching H, W
    3. dtype: torch.long
    """
    task_files = list(DATA_DIR.glob("*.json"))[:10]
    dataloader = create_champion_dataloader(
        task_files,
        batch_size=8,
        num_context_pairs=2,
    )
    
    src_batch, tgt_batch, ctx_input_batch, ctx_output_batch, src_shapes, tgt_shapes, task_ids = next(iter(dataloader))
    
    # Contract checks
    assert ctx_input_batch.dim() == 4, "FAILED: ctx_in should be (B, N, H, W)"
    assert ctx_output_batch.dim() == 4, "FAILED: ctx_out should be (B, N, H, W)"
    assert ctx_input_batch.shape == ctx_output_batch.shape, "FAILED: Context shapes must match"
    assert ctx_input_batch.dtype == torch.long, "FAILED: dtype should be long"
    
    # Verify num_pairs
    assert ctx_input_batch.size(1) == 2, "FAILED: Should have 2 context pairs"
    
    # Verify matching H, W
    assert ctx_input_batch.shape[2:] == ctx_output_batch.shape[2:], "FAILED: H, W must match"
    
    # Verify shapes are provided
    assert len(src_shapes) == src_batch.size(0), "FAILED: Should have shape for each example"
    assert len(tgt_shapes) == tgt_batch.size(0), "FAILED: Should have shape for each example"
    
    print("✅ All contract requirements satisfied")
    print(f"   src: {src_batch.shape}, tgt: {tgt_batch.shape}")
    print(f"   ctx_in: {ctx_input_batch.shape}")
    print(f"   ctx_out: {ctx_output_batch.shape}")
    print(f"   Grid shapes: src={src_shapes[0]}, tgt={tgt_shapes[0]}")
    print(f"   H, W match: {ctx_input_batch.shape[2:] == ctx_output_batch.shape[2:]}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
