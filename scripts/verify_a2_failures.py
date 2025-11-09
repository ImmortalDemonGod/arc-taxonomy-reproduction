#!/usr/bin/env python3
"""
Verify A2 Failure Quantification for Section 7.2

This script verifies the claim that a significant portion of A2 (Spatial Packing)
tasks remained at 0.0% grid accuracy despite 400 examples of targeted LoRA training.

Source Data:
    - reproduction/outputs/atomic_lora_training_summary.json (LoRA results)
    - data/taxonomy_classification/tasks_by_category.json (category mappings)

Output: Verification of A2 failure statistics and examples

Usage:
    python reproduction/scripts/verify_a2_failures.py
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_category_mapping(filepath: Path) -> Dict[str, List[str]]:
    """Load the task-to-category mapping."""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_lora_summary(filepath: Path) -> dict:
    """Load the atomic LoRA training summary JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def analyze_a2_failures(
    lora_data: dict,
    category_mapping: Dict[str, List[str]]
) -> Tuple[int, int, List[Tuple[str, float, float]]]:
    """
    Analyze A2 task performance in LoRA fine-tuning.
    
    Args:
        lora_data: Parsed JSON from atomic_lora_training_summary.json
        category_mapping: Task-to-category mapping
    
    Returns:
        Tuple of:
        - Number of A2 tasks attempted in LoRA training
        - Number of A2 tasks with 0.0% grid accuracy
        - List of (task_id, cell_acc, grid_acc) for all A2 tasks
    """
    a2_task_ids = set(category_mapping.get('A2', []))
    
    a2_attempted = 0
    a2_zero_grid = 0
    a2_results = []
    
    for task_id, task_data in lora_data.get('tasks', {}).items():
        # Only consider A2 tasks that were successfully trained
        if task_id not in a2_task_ids:
            continue
        
        if task_data.get('status') != 'success':
            continue
        
        a2_attempted += 1
        metadata = task_data.get('metadata', {})
        
        cell_acc = metadata.get('final_cell_accuracy', 0.0)
        grid_acc = metadata.get('final_grid_accuracy', 0.0)
        
        a2_results.append((task_id, cell_acc, grid_acc))
        
        if grid_acc == 0.0:
            a2_zero_grid += 1
    
    return a2_attempted, a2_zero_grid, a2_results


def print_statistics(
    attempted: int,
    zero_grid: int,
    results: List[Tuple[str, float, float]],
    all_a2_tasks: List[str]
):
    """Print A2 failure statistics in publication format."""
    percentage = (zero_grid / attempted * 100) if attempted > 0 else 0
    
    print("=" * 80)
    print("A2 FAILURE QUANTIFICATION - Section 7.2")
    print("=" * 80)
    print()
    print(f"Total A2 tasks in taxonomy: {len(all_a2_tasks)}")
    print(f"A2 tasks attempted in LoRA training: {attempted}")
    print(f"A2 tasks with 0.0% grid accuracy: {zero_grid}")
    print(f"Failure rate: {percentage:.1f}%")
    print()
    
    # Sort by grid accuracy ascending, then cell accuracy descending
    results_sorted = sorted(results, key=lambda x: (x[2], -x[1]))
    
    print("-" * 80)
    print("ALL A2 TASKS (sorted by grid accuracy)")
    print("-" * 80)
    
    for i, (task_id, cell_acc, grid_acc) in enumerate(results_sorted, 1):
        status = "❌ TOTAL FAILURE" if grid_acc == 0.0 else ""
        print(f"{i:2d}. {task_id}: {cell_acc:6.2f}% cell / {grid_acc:5.2f}% grid  {status}")
    
    print()
    print("-" * 80)
    print("PAPER CITATION")
    print("-" * 80)
    print(f'"{zero_grid} of the {attempted} attempted A2 tasks ({percentage:.1f}%)')
    print('remained at absolute 0.0% grid accuracy despite 400 examples of')
    print('targeted training, demonstrating an impenetrable architectural barrier')
    print('for a significant portion of this Very Low affinity category."')
    print()
    
    # Identify A2 tasks not in LoRA training
    trained_ids = {r[0] for r in results}
    untrained = [tid for tid in all_a2_tasks if tid not in trained_ids]
    
    if untrained:
        print("-" * 80)
        print(f"NOTE: {len(untrained)} A2 tasks were NOT included in LoRA training:")
        print("-" * 80)
        for task_id in untrained:
            print(f"  - {task_id}")
        print()


def main():
    """Main execution function."""
    # Determine file paths relative to script location
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent
    
    lora_file = repo_root / "reproduction" / "outputs" / "atomic_lora_training_summary.json"
    mapping_file = repo_root / "data" / "taxonomy_classification" / "tasks_by_category.json"
    
    # Verify files exist
    if not lora_file.exists():
        print(f"ERROR: LoRA data file not found at {lora_file}")
        return 1
    
    if not mapping_file.exists():
        print(f"ERROR: Category mapping file not found at {mapping_file}")
        return 1
    
    # Load data
    print(f"Loading LoRA data from: {lora_file}")
    print(f"Loading category mapping from: {mapping_file}")
    print()
    
    lora_data = load_lora_summary(lora_file)
    category_mapping = load_category_mapping(mapping_file)
    
    # Analyze A2 performance
    attempted, zero_grid, results = analyze_a2_failures(lora_data, category_mapping)
    
    # Print results
    print_statistics(attempted, zero_grid, results, category_mapping.get('A2', []))
    
    # Success
    return 0


if __name__ == "__main__":
    exit(main())
