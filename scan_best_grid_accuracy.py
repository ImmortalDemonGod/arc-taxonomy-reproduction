#!/usr/bin/env python3
"""
Scan all known per_task_metrics directories and report the highest grid accuracy observed,
with provenance (path, experiment folder, epoch).

Outputs a human-readable summary file.
"""
from pathlib import Path
import pandas as pd
import sys
import re

ROOT = Path(__file__).parent
# Candidate locations (discovered via find_by_name)
CANDIDATE_DIRS = [
    ROOT / "../data/ablation_study/data_for_paper/logs_run3/logs/per_task_metrics",
    ROOT / "../data/ablation_study/data_for_paper/logs_run4/logs/per_task_metrics",
    ROOT / "../data/data_for_paper/logs/per_task_metrics",
    ROOT / "../data/data_for_paper/logs_champion_1000/per_task_metrics",
    ROOT / "../data/data_for_paper/logs_champion_150/per_task_metrics",
    ROOT / "../data/data_for_paper/logs_run2/logs/per_task_metrics",
    ROOT / "logs/per_task_metrics",
    ROOT / "outputs/ablation/logs/per_task_metrics",
    ROOT / "outputs/arc_agi_2_experiments/archive/logs_arc_agi_2_run_1/per_task_metrics",
    ROOT / "outputs/arc_agi_2_experiments/archive/logs_arc_agi_2_run_2/per_task_metrics",
    ROOT / "outputs/arc_agi_2_experiments/archive/logs_arc_agi_2_run_3/per_task_metrics",
    ROOT / "outputs/arc_agi_2_experiments/per_task_metrics",
]

OUT_TXT = ROOT / "BEST_GRID_ACCURACY_SUMMARY.txt"


def classify_dataset(path: Path) -> str:
    p = str(path)
    if "arc_agi_2" in p or "arc-agi-2" in p:
        return "ARC-AGI-2 (fine-tune)"
    if "rearc" in p or "distributional_alignment" in p:
        return "Re-ARC (pretrain-like)"
    # Heuristics for paper dirs
    if "logs_champion_" in p:
        return "Champion (paper logs)"
    if "ablation_study" in p:
        return "Ablation (paper logs)"
    if "reproduction/logs" in p:
        return "Reproduction (current runs)"
    return "Unclassified"


def best_epoch_for_run(run_dir: Path):
    """Return a dict with best epoch stats or None if no data."""
    csvs = sorted(run_dir.glob("*per_category.csv"))
    if not csvs:
        return None
    # Aggregate per epoch across categories
    df = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
    if not {'epoch','grid_correct','grid_total','cell_correct','cell_total'} <= set(df.columns):
        return None
    agg = df.groupby('epoch', as_index=False).agg({
        'grid_correct':'sum','grid_total':'sum','cell_correct':'sum','cell_total':'sum'
    })
    agg['grid_acc'] = 100.0 * agg['grid_correct'] / agg['grid_total']
    agg['cell_acc'] = 100.0 * agg['cell_correct'] / agg['cell_total']
    best = agg.loc[agg['grid_acc'].idxmax()].to_dict()
    best['epochs_count'] = int(agg.shape[0])
    return best


def main():
    rows = []
    for base in CANDIDATE_DIRS:
        base = base.resolve()
        if not base.exists():
            continue
        for run_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
            best = best_epoch_for_run(run_dir)
            if best is None:
                continue
            rows.append({
                'dataset': classify_dataset(base),
                'base_dir': str(base),
                'run_dir': str(run_dir),
                'experiment': run_dir.name,
                'best_epoch': int(best['epoch']),
                'grid_acc': float(best['grid_acc']),
                'cell_acc': float(best['cell_acc']),
                'epochs_count': best['epochs_count'],
            })
    if not rows:
        OUT_TXT.write_text("No per_category.csv files found in candidate directories.\n")
        print(f"Wrote: {OUT_TXT}")
        return

    import pandas as pd
    r = pd.DataFrame(rows)
    # Overall top 10 by grid_acc
    top_overall = r.sort_values('grid_acc', ascending=False).head(10)
    # Top by dataset class
    top_by_ds = r.sort_values(['dataset','grid_acc'], ascending=[True, False]).groupby('dataset').head(5)

    # Save report
    with OUT_TXT.open('w') as f:
        f.write("BEST GRID ACCURACY — GLOBAL SCAN\n")
        f.write("================================\n\n")
        f.write("Top 10 overall (by grid accuracy, any epoch):\n\n")
        for i, row in top_overall.iterrows():
            f.write(f"- {row['grid_acc']:.4f}% grid @ epoch {row['best_epoch']} | {row['experiment']} | {row['dataset']}\n")
            f.write(f"  run_dir: {row['run_dir']}\n")
        f.write("\n")

        f.write("Top by dataset class:\n\n")
        for ds, grp in top_by_ds.groupby('dataset'):
            f.write(f"[{ds}]\n")
            for i, row in grp.sort_values('grid_acc', ascending=False).iterrows():
                f.write(f"- {row['grid_acc']:.4f}% grid @ epoch {row['best_epoch']} | {row['experiment']}\n")
                f.write(f"  run_dir: {row['run_dir']}\n")
            f.write("\n")

        # Absolute best
        best_row = r.loc[r['grid_acc'].idxmax()]
        f.write("Absolute best observed:\n\n")
        f.write(f"- {best_row['grid_acc']:.4f}% grid @ epoch {int(best_row['best_epoch'])} | {best_row['experiment']} | {best_row['dataset']}\n")
        f.write(f"  run_dir: {best_row['run_dir']}\n")

    print(f"Wrote: {OUT_TXT}")

if __name__ == '__main__':
    main()
