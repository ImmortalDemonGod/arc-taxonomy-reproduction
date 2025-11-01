#!/usr/bin/env python3
"""
Lightweight test to verify ARC-AGI-2 data loading without running the model.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data.champion_data import ChampionARCDataset

def test_arc_agi_loading():
    """Test that ARC-AGI-2 data can be loaded properly."""
    print("\n" + "="*80)
    print("Testing ARC-AGI-2 Data Loading")
    print("="*80 + "\n")
    
    # Path to ARC-AGI-2 data
    arc_data_dir = Path("/Users/tomriddle1/Holistic-Performance-Enhancement/cultivation/data/raw/arc_prize_2025")
    
    # Load training challenges
    print("1. Loading training challenges JSON...")
    with open(arc_data_dir / "arc-agi_training_challenges.json") as f:
        training_data = json.load(f)
    print(f"   ✓ Found {len(training_data)} training tasks")
    
    # Load evaluation challenges
    print("\n2. Loading evaluation challenges JSON...")
    with open(arc_data_dir / "arc-agi_evaluation_challenges.json") as f:
        eval_data = json.load(f)
    print(f"   ✓ Found {len(eval_data)} evaluation tasks")
    
    # Check format of a sample task
    print("\n3. Checking task format...")
    sample_task_id = list(training_data.keys())[0]
    sample_task = training_data[sample_task_id]
    print(f"   Sample task ID: {sample_task_id}")
    print(f"   Keys: {sample_task.keys()}")
    print(f"   Train examples: {len(sample_task['train'])}")
    print(f"   Test examples: {len(sample_task['test'])}")
    print(f"   Train[0] has output: {'output' in sample_task['train'][0]}")
    print(f"   Test[0] has output: {'output' in sample_task['test'][0]}")
    
    # Create temp files (needed for dataset loader)
    print("\n4. Creating temporary task files...")
    temp_dir = Path(__file__).parent / "data" / "arc_agi_temp_test"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Just test with 5 tasks
    test_tasks = list(training_data.items())[:5]
    train_files = []
    
    for task_id, task_data in test_tasks:
        task_file = temp_dir / f"{task_id}.json"
        with open(task_file, 'w') as f:
            json.dump(task_data, f)
        train_files.append(task_file)
    
    print(f"   ✓ Created {len(train_files)} temp task files")
    
    # Test dataset loading
    print("\n5. Testing ChampionARCDataset loading...")
    dataset = ChampionARCDataset(
        task_files=train_files,
        num_context_pairs=2,
        max_grid_size=30,
        pad_token=10,
    )
    
    print(f"   ✓ Dataset created successfully")
    print(f"   Total examples loaded: {len(dataset)}")
    
    # Test getting a sample
    print("\n6. Testing sample retrieval...")
    if len(dataset) > 0:
        sample = dataset[0]
        src, tgt, ctx_input, ctx_output, src_shape, tgt_shape, task_id = sample
        print(f"   ✓ Sample retrieved successfully")
        print(f"   Task ID: {task_id}")
        print(f"   Source shape: {src_shape}")
        print(f"   Target shape: {tgt_shape}")
        print(f"   Context input shape: {ctx_input.shape}")
        print(f"   Context output shape: {ctx_output.shape}")
        print(f"   Source tensor length: {len(src)}")
        print(f"   Target tensor length: {len(tgt)}")
    else:
        print("   ✗ No examples loaded!")
        return False
    
    # Clean up temp files
    print("\n7. Cleaning up...")
    import shutil
    shutil.rmtree(temp_dir)
    print(f"   ✓ Removed temp directory")
    
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED - ARC-AGI-2 data loading works correctly!")
    print("="*80 + "\n")
    return True

if __name__ == "__main__":
    try:
        success = test_arc_agi_loading()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
