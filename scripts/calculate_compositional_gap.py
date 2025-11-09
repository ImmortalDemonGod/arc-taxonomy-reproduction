#!/usr/bin/env python3
"""
Calculate Compositional Gap Statistics for Section 7.1

This script provides reproducible calculation of the compositional gap statistic
cited in the paper: the number of tasks achieving high local pattern recognition
(>80% cell accuracy) but failing at global composition (<10% grid accuracy).

Source: reproduction/outputs/atomic_lora_training_summary.json
Output: Printed statistics and top extreme examples

Usage:
    python reproduction/scripts/calculate_compositional_gap.py
"""

import json
from pathlib import Path
from typing import List, Tuple


def load_lora_summary(filepath: Path) -> dict:
    """Load the atomic LoRA training summary JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def calculate_compositional_gap(
    data: dict,
    cell_threshold: float = 80.0,
    grid_threshold: float = 10.0
) -> Tuple[int, int, List[Tuple[str, float, float]]]:
    """
    Calculate compositional gap statistics.
    
    Args:
        data: Parsed JSON from atomic_lora_training_summary.json
        cell_threshold: Minimum cell accuracy threshold (default: 80%)
        grid_threshold: Maximum grid accuracy threshold (default: 10%)
    
    Returns:
        Tuple of:
        - Number of tasks with compositional gap
        - Total number of completed tasks
        - List of (task_id, cell_acc, grid_acc) for gap tasks
    """
    completed_tasks = 0
    gap_tasks = []
    
    for task_id, task_data in data.get('tasks', {}).items():
        # Only consider successfully completed tasks
        if task_data.get('status') != 'success':
            continue
        
        completed_tasks += 1
        metadata = task_data.get('metadata', {})
        
        cell_acc = metadata.get('final_cell_accuracy', 0.0)
        grid_acc = metadata.get('final_grid_accuracy', 0.0)
        
        # Check if task exhibits compositional gap
        if cell_acc > cell_threshold and grid_acc < grid_threshold:
            gap_tasks.append((task_id, cell_acc, grid_acc))
    
    return len(gap_tasks), completed_tasks, gap_tasks


def print_statistics(gap_count: int, total_count: int, gap_tasks: List[Tuple[str, float, float]]):
    """Print compositional gap statistics in publication format."""
    percentage = (gap_count / total_count * 100) if total_count > 0 else 0
    
    print("=" * 80)
    print("COMPOSITIONAL GAP STATISTICS - Section 7.1")
    print("=" * 80)
    print()
    print(f"Total completed LoRA training runs: {total_count}")
    print(f"Tasks with compositional gap: {gap_count}")
    print(f"Percentage: {percentage:.1f}%")
    print()
    print("Definition: Tasks with >80% cell accuracy AND <10% grid accuracy")
    print()
    
    # Sort by cell accuracy descending, then by grid accuracy ascending
    gap_tasks_sorted = sorted(gap_tasks, key=lambda x: (-x[1], x[2]))
    
    print("-" * 80)
    print("TOP 10 EXTREME EXAMPLES (>90% cell, 0% grid)")
    print("-" * 80)
    
    extreme_examples = [t for t in gap_tasks_sorted if t[1] > 90.0 and t[2] == 0.0]
    
    if extreme_examples:
        for i, (task_id, cell_acc, grid_acc) in enumerate(extreme_examples[:10], 1):
            print(f"{i:2d}. {task_id}: {cell_acc:6.2f}% cell / {grid_acc:5.2f}% grid")
    else:
        print("No tasks found with >90% cell and 0% grid")
    
    print()
    print("-" * 80)
    print("PAPER CITATION")
    print("-" * 80)
    print(f'"We found that {gap_count} of {total_count} fine-tuned tasks ({percentage:.1f}%)')
    print('achieve >80% cell-level accuracy but <10% grid-level accuracy."')
    print()


def main():
    """Main execution function."""
    # Determine file path relative to script location
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent
    data_file = repo_root / "reproduction" / "outputs" / "atomic_lora_training_summary.json"
    
    # Verify file exists
    if not data_file.exists():
        print(f"ERROR: Data file not found at {data_file}")
        print(f"Expected location: reproduction/outputs/atomic_lora_training_summary.json")
        return 1
    
    # Load and analyze data
    print(f"Loading data from: {data_file}")
    print()
    
    data = load_lora_summary(data_file)
    gap_count, total_count, gap_tasks = calculate_compositional_gap(data)
    
    # Print results
    print_statistics(gap_count, total_count, gap_tasks)
    
    # Success
    return 0


if __name__ == "__main__":
    exit(main())
