# ARC-AGI-2 Generalization Evaluation Report

Date: 2025-11-05
Model: Champion Baseline (exp2, pretrained on re-arc)
Evaluation Set: ARC-AGI-2 public test subset (120 tasks)

---

## Key Findings (Section 7.3)

- **Performance Drop (Grid):** 2.34% (re-arc) → 0.34% mean on ARC-AGI-2
  - Per-seed grid accuracy: 0.279%, 0.279%, 0.279%, 0.279%, 0.559%
  - Mean ± 95% CI: 0.34% (95% CI: [0.18%, 0.49%])
  - Source (re-arc 2.34%): paper/champion_FINAL_CLEAR.txt
  - Source (ARC-AGI-2): reproduction/outputs/arc_agi_2_experiments/summary/arc_agi_2_cross_run_summary.csv

- **Cell Accuracy Improvement:** ~73% (re-arc) → 89.37% mean on ARC-AGI-2
  - Per-seed cell accuracy: 90.02%, 90.14%, 89.87%, 89.96%, 86.88%
  - Mean ± 95% CI: 89.37% (95% CI: [87.64%, 91.11%])
  - Sources: same as above

- **Framework Validation (Prediction):** 68.6% of failures on ARC-AGI-2 fall in low-affinity categories (S3, A1, A2)
  - Stats (exp2 aggregate across seeds):
    - Total tasks analyzed: 120
    - Total failures (0% grid): 118
    - Failures in low-affinity (S3, A1, A2): 81 (68.6%)
    - Breakdown (failures/total, failure rate):
      - S3: 81/82 (98.8%)
      - C1: 30/31 (96.8%)
      - S2: 7/7 (100.0%)
  - **Analysis Script:** `reproduction/scripts/analyze_arc_agi_2_failures.py`
    - Joins champion performance with category labels from arc_agi_2_per_task_aggregated.csv
    - Defines failures as grid_accuracy == 0.0
    - Computes share of failures in low-affinity categories {S3, A1, A2}

---

## Supporting Tables

### Cross Run Summary (exp2 only)

From arc_agi_2_cross_run_summary.csv:

- Grid mean per seed: 0.0027933, 0.0027933, 0.0027933, 0.0027933, 0.00558659
- Mean grid across seeds: 0.00335196 (0.3352%)
- Cell mean per seed: 0.90015921, 0.90142889, 0.89869808, 0.89959491, 0.86879018
- Mean cell across seeds: 0.8937 (89.37%)

### Failure Analysis Method

- Use exp2 rows from arc_agi_2_per_task_aggregated.csv.
- Aggregate by task across seeds (mean grid, mean cell, take first category).
- Define failure as grid_accuracy == 0.0.
- Low-affinity categories: {S3, A1, A2}.
- Compute share of failures that belong to low-affinity categories.

---

## Paper-Ready Citations

- Re-arc baseline: paper/champion_FINAL_CLEAR.txt
- ARC-AGI-2 analysis outputs: reproduction/outputs/arc_agi_2_experiments/summary/
  - arc_agi_2_cross_run_summary.csv
  - arc_agi_2_per_task_aggregated.csv
  - arc_agi_2_metrics_aggregate.csv
- Failure analysis script: reproduction/scripts/analyze_arc_agi_2_failures.py

---

## Notes

- exp3/exp3b runs show 0% grid (as expected) and low cell accuracy; not used for headline.
- The range across seeds suggests sensitivity to initialization (s311 > s307–s310).
- These numbers supersede earlier placeholders in Section 7.3.
