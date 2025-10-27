#!/usr/bin/env python3
"""
Create a size-aware stratified train/val split for distributional_alignment dataset.

Two-tier stratification:
1. By taxonomy category (A1, A2, C1, etc.)
2. By output grid size within each category (tiny, small, medium, large)

This ensures validation set is representative across both categories AND difficulty levels.
"""
import json
from pathlib import Path
from collections import defaultdict
import random
import statistics


def get_size_bin(mean_size: float) -> str:
    """
    Categorize output size into bins.
    
    Args:
        mean_size: Mean output grid size in cells
        
    Returns:
        Size bin label
    """
    if mean_size <= 10:
        return 'tiny'
    elif mean_size <= 50:
        return 'small'
    elif mean_size <= 200:
        return 'medium'
    else:
        return 'large'


def compute_task_output_sizes(data_dir: Path) -> dict:
    """
    Compute mean output size for each task.
    
    Args:
        data_dir: Directory containing task JSON files
        
    Returns:
        Dict mapping task_id -> mean output size
    """
    task_sizes = {}
    
    for task_file in data_dir.glob("*.json"):
        # Skip metadata files
        if task_file.name in ['task_categories.json', 'split_manifest.json', 'generation_statistics.json']:
            continue
        
        task_id = task_file.stem
        
        try:
            with open(task_file) as f:
                task_data = json.load(f)
            
            # Compute sizes across all examples
            sizes = []
            for ex in task_data.get('train', []):
                output = ex.get('output', [])
                h = len(output)
                w = len(output[0]) if output else 0
                sizes.append(h * w)
            
            if sizes:
                task_sizes[task_id] = statistics.mean(sizes)
            else:
                print(f"Warning: No training examples in {task_id}")
                task_sizes[task_id] = 0
                
        except Exception as e:
            print(f"Error processing {task_file}: {e}")
            continue
    
    return task_sizes


def create_size_aware_stratified_split(
    task_categories_file: Path,
    data_dir: Path,
    output_file: Path,
    train_ratio: float = 0.8,
    random_seed: int = 42
):
    """
    Create stratified train/val split with two-tier stratification.
    
    Args:
        task_categories_file: Path to task_categories.json
        data_dir: Directory containing task JSON files
        output_file: Path to save split_manifest.json
        train_ratio: Fraction of tasks for training (default 0.8)
        random_seed: Random seed for reproducibility
    """
    # Set seed for reproducibility
    random.seed(random_seed)
    
    # Load task categories
    with open(task_categories_file) as f:
        categories = json.load(f)
    
    # Compute output sizes
    print("Computing output sizes for all tasks...")
    task_sizes = compute_task_output_sizes(data_dir)
    
    # Group tasks by category AND size bin
    category_size_bins = defaultdict(lambda: defaultdict(list))
    
    for task_id, category in categories.items():
        if task_id not in task_sizes:
            print(f"Warning: Task {task_id} not found in data directory")
            continue
            
        size = task_sizes[task_id]
        size_bin = get_size_bin(size)
        category_size_bins[category][size_bin].append(task_id)
    
    # Stratified split per category-size bin
    train_files = []
    val_files = []
    
    print("\nCreating size-aware stratified split:")
    print(f"{'Category':<12} {'Size Bin':<8} {'Total':<6} {'Train':<6} {'Val':<6} {'Train %':<8}")
    print("-" * 70)
    
    for category in sorted(category_size_bins.keys()):
        for size_bin in sorted(category_size_bins[category].keys()):
            tasks = sorted(category_size_bins[category][size_bin])
            random.shuffle(tasks)  # Shuffle with fixed seed
            
            n_total = len(tasks)
            n_train = max(1, int(n_total * train_ratio))  # At least 1 train task
            n_val = n_total - n_train
            
            train_tasks = [f"{tid}.json" for tid in tasks[:n_train]]
            val_tasks = [f"{tid}.json" for tid in tasks[n_train:]]
            
            train_files.extend(train_tasks)
            val_files.extend(val_tasks)
            
            train_pct = (n_train / n_total * 100) if n_total > 0 else 0
            print(f"{category:<12} {size_bin:<8} {n_total:<6} {n_train:<6} {n_val:<6} {train_pct:<8.1f}")
    
    print("-" * 70)
    total = len(categories)
    print(f"{'TOTAL':<21} {total:<6} {len(train_files):<6} {len(val_files):<6} {len(train_files)/total*100:<8.1f}")
    
    # Sort for deterministic output
    train_files.sort()
    val_files.sort()
    
    # Calculate total examples (150 per task)
    train_examples = len(train_files) * 150
    val_examples = len(val_files) * 150
    
    # Create manifest
    manifest = {
        "random_seed": random_seed,
        "train_ratio": train_ratio,
        "total_tasks": len(categories),
        "train_tasks": len(train_files),
        "val_tasks": len(val_files),
        "train_examples": train_examples,
        "val_examples": val_examples,
        "samples_per_task": 150,
        "stratification": "two_tier",  # category + size
        "train_files": train_files,
        "val_files": val_files,
    }
    
    # Save manifest
    with open(output_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n✅ Saved split manifest to: {output_file}")
    print(f"   Train: {len(train_files)} tasks ({train_examples:,} examples)")
    print(f"   Val:   {len(val_files)} tasks ({val_examples:,} examples)")
    
    return manifest


if __name__ == "__main__":
    # Paths
    data_dir = Path(__file__).parent.parent / "data" / "distributional_alignment"
    categories_file = data_dir / "task_categories.json"
    output_file = data_dir / "split_manifest.json"
    
    # Create split
    manifest = create_size_aware_stratified_split(
        categories_file,
        data_dir,
        output_file,
        train_ratio=0.8,
        random_seed=42
    )
