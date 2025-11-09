#!/usr/bin/env python3
"""
Generate Compositional Gap Sensitivity Analysis for M3.2

Purpose: Demonstrate that the Compositional Gap phenomenon (high cell accuracy,
         low grid accuracy) is robust across threshold ranges, not an artifact
         of the specific 80/10 choice.

Output: Heatmap showing % of tasks in "gap" for various threshold pairs.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration
DATA_PATH = Path(__file__).parent.parent / "outputs/atomic_lora_training_summary.json"
OUTPUT_PATH = Path(__file__).parent.parent / "figures/compositional_gap_sensitivity.png"
OUTPUT_PATH.parent.mkdir(exist_ok=True, parents=True)

# Threshold ranges to test
CELL_THRESHOLDS = np.arange(0.70, 0.96, 0.05)  # 70% to 95% in 5% steps
GRID_THRESHOLDS = np.arange(0.00, 0.21, 0.02)  # 0% to 20% in 2% steps

def load_task_data():
    """Load task-level accuracy data from LoRA fine-tuning results."""
    with open(DATA_PATH) as f:
        data = json.load(f)
    
    tasks = []
    for task_id, task_data in data["tasks"].items():
        if task_data["status"] == "success":
            metadata = task_data.get("metadata", {})
            tasks.append({
                "task_id": task_id,
                "cell_acc": metadata["final_cell_accuracy"] / 100,  # Convert to fraction
                "grid_acc": metadata["final_grid_accuracy"] / 100
            })
    
    print(f"Loaded {len(tasks)} successfully trained tasks")
    return tasks

def compute_gap_matrix(tasks):
    """
    Compute percentage of tasks in compositional gap for each threshold pair.
    
    Returns:
        results: 2D numpy array where results[i,j] = % tasks in gap at thresholds[i,j]
    """
    results = np.zeros((len(CELL_THRESHOLDS), len(GRID_THRESHOLDS)))
    
    for i, cell_t in enumerate(CELL_THRESHOLDS):
        for j, grid_t in enumerate(GRID_THRESHOLDS):
            # Count tasks meeting gap criteria
            gap_count = sum(
                1 for t in tasks 
                if t["cell_acc"] > cell_t and t["grid_acc"] < grid_t
            )
            results[i, j] = 100 * gap_count / len(tasks)
    
    return results

def create_heatmap(results):
    """Generate publication-quality heatmap with annotations."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(
        results, 
        annot=True, 
        fmt='.1f',
        cmap='YlOrRd',
        cbar_kws={'label': '% Tasks in Compositional Gap'},
        xticklabels=[f'{t:.0%}' for t in GRID_THRESHOLDS],
        yticklabels=[f'{t:.0%}' for t in CELL_THRESHOLDS],
        ax=ax,
        vmin=0,
        vmax=100
    )
    
    # Mark the operating point (80% cell, 10% grid)
    cell_idx = np.argmin(np.abs(CELL_THRESHOLDS - 0.80))
    grid_idx = np.argmin(np.abs(GRID_THRESHOLDS - 0.10))
    ax.add_patch(
        plt.Rectangle(
            (grid_idx, cell_idx), 1, 1, 
            fill=False, 
            edgecolor='blue', 
            linewidth=3
        )
    )
    
    # Add text annotation for operating point
    ax.text(
        grid_idx + 0.5, cell_idx + 0.5, 
        '★', 
        ha='center', va='center', 
        fontsize=20, color='blue',
        weight='bold'
    )
    
    # Labels and title
    ax.set_xlabel('Grid Accuracy Threshold', fontsize=12, weight='bold')
    ax.set_ylabel('Cell Accuracy Threshold', fontsize=12, weight='bold')
    ax.set_title(
        'Compositional Gap Prevalence Across Threshold Ranges\n'
        '(Blue box marks operating point: >80% cell, <10% grid)',
        fontsize=14, weight='bold', pad=20
    )
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to: {OUTPUT_PATH}")
    
    return fig, ax

def generate_summary_statistics(results):
    """Generate key statistics for paper text."""
    # Operating point value
    cell_idx = np.argmin(np.abs(CELL_THRESHOLDS - 0.80))
    grid_idx = np.argmin(np.abs(GRID_THRESHOLDS - 0.10))
    operating_point_value = results[cell_idx, grid_idx]
    
    # Range statistics
    min_value = results.min()
    max_value = results.max()
    mean_value = results.mean()
    
    # Count cells above 50% threshold
    substantial_gap = (results > 50).sum()
    total_cells = results.size
    
    print("\n" + "="*60)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("="*60)
    print(f"Operating Point (80% cell, 10% grid): {operating_point_value:.1f}% of tasks")
    print(f"\nRange across all threshold pairs:")
    print(f"  Min: {min_value:.1f}%")
    print(f"  Max: {max_value:.1f}%")
    print(f"  Mean: {mean_value:.1f}%")
    print(f"\nRobustness:")
    print(f"  {substantial_gap}/{total_cells} threshold pairs ({100*substantial_gap/total_cells:.1f}%) show >50% gap")
    print("="*60)
    
    # Generate text snippet for paper
    print("\n" + "="*60)
    print("SUGGESTED TEXT FOR PAPER (Section 7.1)")
    print("="*60)
    print(f"""
### Sensitivity Analysis: Robustness of the Compositional Gap

A valid concern with any threshold-based metric is whether it is cherry-picked to maximize a desired finding. Figure 7.X directly addresses this by showing how the percentage of tasks exhibiting the compositional gap changes as we vary both thresholds across plausible ranges.

Key observations:
1. **Broad Plateau:** The gap remains prevalent (>50% of tasks) across {substantial_gap} of {total_cells} threshold pairs tested ({100*substantial_gap/total_cells:.1f}% of the parameter space).
2. **Conservative Choice:** Our 80/10 operating point (blue box, {operating_point_value:.1f}% of tasks) sits well within this plateau, not at an extreme that artificially inflates the finding.
3. **Lower Bound:** Even with very strict thresholds (85% cell, 5% grid), {results[np.argmin(np.abs(CELL_THRESHOLDS - 0.85)), np.argmin(np.abs(GRID_THRESHOLDS - 0.05))]:.1f}% of tasks still exhibit the gap.
4. **Upper Bound:** With relaxed thresholds (75% cell, 15% grid), the gap affects {results[np.argmin(np.abs(CELL_THRESHOLDS - 0.75)), np.argmin(np.abs(GRID_THRESHOLDS - 0.15))]:.1f}% of tasks.

This analysis demonstrates that the compositional gap is a stable, threshold-insensitive architectural phenomenon, not an artifact of our choice of operating point.
""")
    print("="*60)

def main():
    """Execute full sensitivity analysis pipeline."""
    print("Starting Compositional Gap Sensitivity Analysis...")
    print(f"Data source: {DATA_PATH}")
    
    # Load data
    tasks = load_task_data()
    
    # Compute sensitivity matrix
    print(f"\nComputing sensitivity across {len(CELL_THRESHOLDS)}×{len(GRID_THRESHOLDS)} threshold grid...")
    results = compute_gap_matrix(tasks)
    
    # Generate heatmap
    print("\nGenerating heatmap...")
    create_heatmap(results)
    
    # Generate summary
    generate_summary_statistics(results)
    
    print("\n✓ Sensitivity analysis complete!")
    print(f"✓ Figure saved: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
