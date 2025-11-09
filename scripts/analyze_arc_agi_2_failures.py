#!/usr/bin/env python3
"""
Analyze ARC-AGI-2 Failures by Category for Section 7.3

This script analyzes champion model failures on ARC-AGI-2 by category
to compute the percentage of failures in low-affinity categories (S3, A1, A2).

Data Source:
    - reproduction/outputs/arc_agi_2_experiments/summary/arc_agi_2_per_task_aggregated.csv
      (includes embedded category labels from visual classifier predictions)

Output: Statistics for Section 7.3 "Framework Validation via Prediction"

Usage:
    python reproduction/scripts/analyze_arc_agi_2_failures.py
"""

from pathlib import Path
import pandas as pd
import numpy as np

# Paths
BASE = Path(__file__).resolve().parents[2]
PER_TASK_FILE = BASE / "reproduction" / "outputs" / "arc_agi_2_experiments" / "summary" / "arc_agi_2_per_task_aggregated.csv"

# Low affinity categories for standard Transformers
LOW_AFFINITY_CATEGORIES = {"S3", "A1", "A2"}  # From Neural Affinity Framework


def main():
    """Main analysis function."""
    
    # Load data
    print("Loading champion performance data...")
    perf = pd.read_csv(PER_TASK_FILE)
    
    # Filter for exp2 (baseline champion on re-arc)
    exp2_perf = perf[perf["experiment_name"].str.startswith("exp2_champion")].copy()
    
    print(f"Found {len(exp2_perf)} task-experiment pairs")
    
    # Aggregate across seeds
    exp2_agg = exp2_perf.groupby("task_id").agg({
        "grid_accuracy": "mean",
        "cell_accuracy": "mean",
        "category": "first"  # Category should be same across seeds
    }).reset_index()
    
    print(f"Aggregated to {len(exp2_agg)} unique tasks")
    print(f"Categories found: {sorted(exp2_agg['category'].unique())}")
    
    # Define failures (grid accuracy = 0)
    exp2_agg["is_failure"] = exp2_agg["grid_accuracy"] == 0.0
    
    # Classify by affinity
    exp2_agg["is_low_affinity"] = exp2_agg["category"].isin(LOW_AFFINITY_CATEGORIES)
    
    merged = exp2_agg  # Rename for consistency with rest of code
    
    # Compute statistics
    total_tasks = len(merged)
    total_failures = merged["is_failure"].sum()
    failures_in_low_affinity = (merged["is_failure"] & merged["is_low_affinity"]).sum()
    
    if total_failures > 0:
        pct_failures_low_affinity = (failures_in_low_affinity / total_failures * 100)
    else:
        pct_failures_low_affinity = 0.0
    
    # Breakdown by category
    category_stats = merged.groupby("category").agg({
        "is_failure": ["sum", "count"]
    }).reset_index()
    category_stats.columns = ["category", "failures", "total"]
    category_stats["failure_rate"] = (category_stats["failures"] / category_stats["total"] * 100)
    category_stats["is_low_affinity"] = category_stats["category"].isin(LOW_AFFINITY_CATEGORIES)
    category_stats = category_stats.sort_values("failure_rate", ascending=False)
    
    # Print results
    print("=" * 80)
    print("ARC-AGI-2 FAILURE ANALYSIS BY CATEGORY - Section 7.3")
    print("=" * 80)
    print()
    print(f"Total ARC-AGI-2 tasks analyzed: {total_tasks}")
    print(f"Total failures (0% grid accuracy): {total_failures}")
    print(f"Failures in low-affinity categories (S3, A1, A2): {failures_in_low_affinity}")
    print(f"Percentage of failures in low-affinity: {pct_failures_low_affinity:.1f}%")
    print()
    
    print("-" * 80)
    print("FAILURE RATE BY CATEGORY")
    print("-" * 80)
    for _, row in category_stats.iterrows():
        affinity_marker = "🔴 LOW" if row["is_low_affinity"] else "🟢 MED/HIGH"
        print(f"{affinity_marker} {row['category']:3s}: {row['failures']:3.0f}/{row['total']:3.0f} failures ({row['failure_rate']:5.1f}%)")
    print()
    
    print("-" * 80)
    print("PAPER CITATION - Section 7.3")
    print("-" * 80)
    print(f'"Our framework predicts that failures should concentrate in low-affinity')
    print(f'categories. We validate this by showing that {pct_failures_low_affinity:.1f}% of')
    print(f'champion model failures on ARC-AGI-2 occur in S3, A1, and A2 categories,')
    print(f'as predicted by our visual classifier. This prospective prediction demonstrates')
    print(f'the framework has predictive power beyond post-hoc explanation."')
    print()
    
    # Also compute: what percentage of low-affinity tasks are failures?
    low_affinity_tasks = merged[merged["is_low_affinity"]]
    low_affinity_failure_rate = (low_affinity_tasks["is_failure"].sum() / len(low_affinity_tasks) * 100)
    
    other_tasks = merged[~merged["is_low_affinity"]]
    other_failure_rate = (other_tasks["is_failure"].sum() / len(other_tasks) * 100) if len(other_tasks) > 0 else 0.0
    
    print("-" * 80)
    print("AFFINITY-LEVEL COMPARISON")
    print("-" * 80)
    print(f"Low-affinity (S3, A1, A2):")
    print(f"  Tasks: {len(low_affinity_tasks)}")
    print(f"  Failures: {low_affinity_tasks['is_failure'].sum()}")
    print(f"  Failure rate: {low_affinity_failure_rate:.1f}%")
    print()
    print(f"Medium/High-affinity (C1, C2, S1, S2, K1, L1):")
    print(f"  Tasks: {len(other_tasks)}")
    print(f"  Failures: {other_tasks['is_failure'].sum()}")
    print(f"  Failure rate: {other_failure_rate:.1f}%")
    print()
    
    return 0


if __name__ == "__main__":
    exit(main())
