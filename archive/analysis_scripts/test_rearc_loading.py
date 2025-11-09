#!/usr/bin/env python3
"""
Lightweight test to verify re-arc data loading still works after ARC-AGI-2 fix.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data.champion_data import ChampionARCDataset

def test_rearc_loading():
    """Test that re-arc data can still be loaded properly."""
    print("\n" + "="*80)
    print("Testing Re-ARC Data Loading (Regression Test)")
    print("="*80 + "\n")
    
    data_dir = Path(__file__).parent / "data" / "distributional_alignment"
    
    # Load split manifest
    print("1. Loading split manifest...")
    with open(data_dir / "split_manifest.json") as f:
        split_info = json.load(f)
    print(f"   ✓ Train files: {len(split_info['train_files'])}")
    print(f"   ✓ Val files: {len(split_info['val_files'])}")
    
    # Get first 5 files for testing
    train_files = [data_dir / fname for fname in split_info["train_files"][:5]]
    
    # Check format
    print("\n2. Checking re-arc task format...")
    with open(train_files[0]) as f:
        sample_task = json.load(f)
    print(f"   Keys: {sample_task.keys()}")
    print(f"   Train examples: {len(sample_task['train'])}")
    print(f"   Test examples: {len(sample_task['test'])}")
    print(f"   Train[0] has output: {'output' in sample_task['train'][0]}")
    if len(sample_task['test']) > 0:
        print(f"   Test[0] has output: {'output' in sample_task['test'][0]}")
    else:
        print(f"   Test[0] has output: N/A (no test examples)")
    
    # Test dataset loading
    print("\n3. Testing ChampionARCDataset loading...")
    dataset = ChampionARCDataset(
        task_files=train_files,
        num_context_pairs=2,
        max_grid_size=30,
        pad_token=10,
    )
    
    print(f"   ✓ Dataset created successfully")
    print(f"   Total examples loaded: {len(dataset)}")
    
    # Test getting a sample
    print("\n4. Testing sample retrieval...")
    if len(dataset) > 0:
        sample = dataset[0]
        src, tgt, ctx_input, ctx_output, src_shape, tgt_shape, task_id = sample
        print(f"   ✓ Sample retrieved successfully")
        print(f"   Task ID: {task_id}")
        print(f"   Source shape: {src_shape}")
        print(f"   Target shape: {tgt_shape}")
        print(f"   Context input shape: {ctx_input.shape}")
        print(f"   Context output shape: {ctx_output.shape}")
    else:
        print("   ✗ No examples loaded!")
        return False
    
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED - Re-ARC data loading still works correctly!")
    print("="*80 + "\n")
    return True

if __name__ == "__main__":
    try:
        success = test_rearc_loading()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
