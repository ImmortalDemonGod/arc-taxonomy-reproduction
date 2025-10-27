#!/usr/bin/env python3
"""
Create a stratified train/val split for distributional_alignment dataset.

Ensures each taxonomy category has approximately 80/20 train/val split.
"""
import json
from pathlib import Path
from collections import defaultdict
import random

def create_stratified_split(
    task_categories_file: Path,
    output_file: Path,
    train_ratio: float = 0.8,
    random_seed: int = 42
):
    """
    Create stratified train/val split maintaining taxonomy balance.
    
    Args:
        task_categories_file: Path to task_categories.json
        output_file: Path to save split_manifest.json
        train_ratio: Fraction of tasks for training (default 0.8)
        random_seed: Random seed for reproducibility
    """
    # Set seed for reproducibility
    random.seed(random_seed)
    
    # Load task categories
    with open(task_categories_file) as f:
        categories = json.load(f)
    
    # Group tasks by category
    category_tasks = defaultdict(list)
    for task_id, category in categories.items():
        category_tasks[category].append(f"{task_id}.json")
    
    # Stratified split per category
    train_files = []
    val_files = []
    
    print("Creating stratified split:")
    print(f"{'Category':<12} {'Total':<8} {'Train':<8} {'Val':<8} {'Train %':<10}")
    print("-" * 60)
    
    for category in sorted(category_tasks.keys()):
        tasks = sorted(category_tasks[category])  # Sort for reproducibility
        random.shuffle(tasks)  # Shuffle with fixed seed
        
        n_total = len(tasks)
        n_train = max(1, int(n_total * train_ratio))  # At least 1 train task
        n_val = n_total - n_train
        
        train_tasks = tasks[:n_train]
        val_tasks = tasks[n_train:]
        
        train_files.extend(train_tasks)
        val_files.extend(val_tasks)
        
        train_pct = (n_train / n_total * 100) if n_total > 0 else 0
        print(f"{category:<12} {n_total:<8} {n_train:<8} {n_val:<8} {train_pct:<10.1f}")
    
    print("-" * 60)
    total = len(categories)
    print(f"{'TOTAL':<12} {total:<8} {len(train_files):<8} {len(val_files):<8} {len(train_files)/total*100:<10.1f}")
    
    # Sort for deterministic output
    train_files.sort()
    val_files.sort()
    
    # Calculate total examples (15 per task)
    train_examples = len(train_files) * 15
    val_examples = len(val_files) * 15
    
    # Create manifest
    manifest = {
        "random_seed": random_seed,
        "train_ratio": train_ratio,
        "total_tasks": len(categories),
        "train_tasks": len(train_files),
        "val_tasks": len(val_files),
        "train_examples": train_examples,
        "val_examples": val_examples,
        "train_files": train_files,
        "val_files": val_files,
    }
    
    # Save manifest
    with open(output_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n✅ Saved split manifest to: {output_file}")
    print(f"   Train: {len(train_files)} tasks ({train_examples} examples)")
    print(f"   Val:   {len(val_files)} tasks ({val_examples} examples)")
    
    return manifest


if __name__ == "__main__":
    # Paths
    data_dir = Path(__file__).parent.parent / "data" / "distributional_alignment"
    categories_file = data_dir / "task_categories.json"
    output_file = data_dir / "split_manifest.json"
    
    # Create split
    manifest = create_stratified_split(
        categories_file,
        output_file,
        train_ratio=0.8,
        random_seed=42
    )
