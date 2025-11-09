# Ablation Study (30-Epoch Slice) – Systematic Analysis

- **Scope**
  - New logs at: `/Users/tomriddle1/Holistic-Performance-Enhancement/cultivation/systems/arc_reactor/publications/arc_taxonomy_2025/reproduction/logs/per_task_metrics`
  - Analysis script + console output saved:
    - Script: `reproduction/analyze_30epoch_ablation.py`
    - Output: `reproduction/analyze_30epoch_ablation_output.txt`
  - No reruns performed. Results computed from existing CSVs.

- **Method (Fair slice)**
  - Unequal training lengths exist (e.g., Exp0 has 51 epochs). To compare fairly without reruns, we compute best-epoch metrics using only epochs `<= 29` for all experiments.
  - Metrics aggregated from per-category CSVs: overall grid/cell accuracies.

---

## Executive Summary

- **Encoder–Decoder (Exp0)** is the strongest within the 0–29 epoch slice.
- **Grid2D PE (Exp1)** shows a small negative effect versus Exp0.
- **Permutation Invariance (Exp2)** shows a marginal positive effect versus Exp1.
- **Context System (Exp3)** is unstable: peaks at epoch 0 and degrades sharply with training.

---

## Data Inventory and Training Lengths

- **Exp0 (E–D)**: 51 per-epoch CSVs (epochs 0–50). Config shows `max_epochs: 100` but run produced 51 epochs.
- **Exp1 (Grid2D)**: 30 per-epoch CSVs (epochs 0–29).
- **Exp2 (PermInv)**: 30 per-epoch CSVs (epochs 0–29).
- **Exp3 (Context)**:
  - s307: 30 per-epoch CSVs (0–29)
  - s308: 30 per-epoch CSVs (0–29)
  - s309: 19 per-epoch CSVs (0–18)

Notes:
- Exp3 Lightning hparams show `d_model: 160`, while Exp0/1/2 runs use `d_model: 168`. This is typical of an ablation; totals are in the same range.

---

## Results (Best Epoch within 0–29)

- **Exp0 (E–D)**
  - Best epoch: 23
  - Grid: 3.3255%
  - Cell: 90.1212%

- **Exp1 (+Grid2D)**
  - Best epoch: 17
  - Grid: 3.1685%
  - Cell: 89.8650%

- **Exp2 (+PermInv)**
  - Best epoch: 17
  - Grid: 3.2458%
  - Cell: 89.6442%

- **Exp3 (Context)**
  - s307: Best epoch 0 → Grid 3.1662%, Cell 71.0067%
  - s308: Best epoch 0 → Grid 3.1662%, Cell 69.2548%
  - s309: Best epoch 7 → Grid 3.1689%, Cell 69.7270%

---

## Component Contributions (Grid Accuracy, best ≤ 29)

- **Grid2D vs E–D**: −0.1570 pp (3.1685 − 3.3255)
- **PermInv vs Grid2D**: +0.0773 pp (3.2458 − 3.1685)
- **Context vs PermInv (mean of 3 seeds)**: −0.0787 pp

Interpretation:
- Grid2D shows a small negative change vs E–D under this budget.
- PermInv shows a small positive change vs Grid2D; effect size is marginal.
- Context, averaged across s307/s308/s309, is slightly negative vs Exp2 and exhibits training instability (see below).

---

## Training Dynamics Highlights

- **E–D (Exp0)**: Stable improvement through the slice; high cell accuracy (~90%).
- **Grid2D (Exp1) & PermInv (Exp2)**: Modest, stable curves; small deltas relative to E–D.
- **Context (Exp3)**:
  - All seeds peak at or near epoch 0 on grid accuracy (~3.166%) but rapidly degrade during training.
  - Final epoch grid/cell (typical): ~0.6124% / ~11.8795%.
  - Indicates severe instability with this configuration and budget.

---

## Conclusions (Bounded by Observed Data)

- **Supported**
  - E–D provides the strongest performance and stable training within the first 30 epochs.

- **Weak/Negative Evidence**
  - Grid2D shows a slight negative impact vs E–D in this dataset/training budget.
  - PermInv offers only a small positive delta vs Grid2D.

- **Context System**
  - Unstable with current settings: performance collapses from the initialization peak.
  - Even restricting to best ≤29, Context does not provide a consistent advantage.

---

## Limitations and Notes

- **Unequal training lengths** handled by analyzing the fair slice (epochs ≤ 29). Exp0 also achieves 3.3310% at epoch 41 when using its full run; conclusions above are based on the 0–29 slice only.
- **Model width** differs (Context d_model=160 vs 168 for others). Parameter counts are in the same ballpark; observed instability likely reflects optimization/architecture interaction rather than capacity alone.

---

### Naming clarification (ablation logs)

- Due to a legacy naming bug in the context-only training script, some context-only runs were stored under directories named `exp3_champion_rearc_sXXX`.
- In this analysis, any directories matching `logs/per_task_metrics/exp3_champion_rearc_s*` are context-only ablations (Grid2D=False, PermInv=False), not full Champion runs.
- The training script has been updated to name future context-only runs as `exp3_context_only_{dataset}_s{seed}` and to print an unambiguous configuration line in the console.

---

## Reproducibility

- Script used: `reproduction/analyze_30epoch_ablation.py`
- Console output: `reproduction/analyze_30epoch_ablation_output.txt`
- Data directory: `reproduction/logs/per_task_metrics/`

Run (from `reproduction/`):
```
python analyze_30epoch_ablation.py | tee analyze_30epoch_ablation_output.txt
```

This report summarizes the key findings for the paper without requiring additional experiments.
