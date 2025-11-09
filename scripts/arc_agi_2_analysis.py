from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu
import warnings
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # type: ignore
except Exception:
    EventAccumulator = None  # type: ignore


REPRO_ROOT = Path(__file__).resolve().parent.parent  # reproduction root
OUTPUTS_ROOT = REPRO_ROOT / "outputs" / "arc_agi_2_experiments"
SUMMARY_DIR = OUTPUTS_ROOT / "summary"
PER_TASK_ROOT = OUTPUTS_ROOT / "per_task_metrics"  # Updated path for new layout


def _list_runs(outputs_root: Path) -> List[Path]:
    """List all experiment variant directories directly in outputs_root.
    
    New layout: exp2_champion_arc-agi-2_s307_csv, exp3_champion_arc-agi-2_s308_csv, etc.
    Old layout (archived): logs_arc_agi_2_run_1, logs_arc_agi_2_run_2, etc.
    """
    if not outputs_root.exists():
        return []
    
    # New layout: exp*_csv directories directly in outputs_root
    variant_dirs = sorted([
        p for p in outputs_root.iterdir() 
        if p.is_dir() and (
            p.name.startswith("exp2_champion_arc-agi-2") or
            p.name.startswith("exp3_champion_arc-agi-2") or
            p.name.startswith("exp3b_merged_lora_arc-agi-2")
        ) and p.name.endswith("_csv")
    ])
    
    return variant_dirs


def _variant_dirs(run_dir: Path) -> Dict[str, Path]:
    """In new layout, run_dir IS the variant directory (e.g., exp2_champion_arc-agi-2_s307_csv).
    
    We return {variant_name: version_0_path} where variant_name is the run_dir name.
    """
    version_0 = run_dir / "version_0"
    if version_0.exists():
        return {run_dir.name: version_0}
    return {}


def _read_metrics_csv(csv_path: Path, run_id: str, variant: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return pd.DataFrame()
    df["run"] = run_id
    df["variant"] = variant
    # Merge extras if present (e.g., val_total_grids) from global_metrics.csv
    # Only merge columns that don't already exist in metrics.csv
    try:
        extras_path = csv_path.parent / "global_metrics.csv"
        if extras_path.exists():
            extras = pd.read_csv(extras_path)
            if "epoch" in extras.columns:
                extras["epoch"] = pd.to_numeric(extras["epoch"], errors="coerce")
                # Only keep columns from extras that aren't in df (except epoch for merge key)
                extra_cols = ["epoch"] + [c for c in extras.columns if c not in df.columns and c != "epoch"]
                if len(extra_cols) > 1:  # More than just epoch
                    df = df.merge(extras[extra_cols], on="epoch", how="left")
    except Exception:
        pass
    # Harmonize transformation metric naming across sources
    if "val_transformation_quality" in df.columns and "val_transformation_f1" not in df.columns:
        df.rename(columns={"val_transformation_quality": "val_transformation_f1"}, inplace=True)
    for c in [
        "epoch",
        "val_grid_accuracy",
        "val_cell_accuracy",
        "val_loss",
        "val_copy_rate",
        "val_change_recall",
        "val_transformation_f1",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _read_hparams_yaml(yaml_path: Path, run_id: str, variant: str) -> pd.DataFrame:
    try:
        with open(yaml_path, "r") as f:
            hp = yaml.safe_load(f) or {}
    except Exception:
        hp = {}
    flat = {"run": run_id, "variant": variant}
    for k in [
        "vocab_size",
        "d_model",
        "num_encoder_layers",
        "num_decoder_layers",
        "num_heads",
        "d_ff",
        "max_grid_size",
        "dropout",
        "learning_rate",
        "weight_decay",
        "beta1",
        "beta2",
        "max_epochs",
        "pad_token",
        "use_context",
        "use_bridge",
    ]:
        if k in hp:
            flat[k] = hp[k]
    return pd.DataFrame([flat])


def build_manifest_and_metrics(outputs_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    runs = _list_runs(outputs_root)
    manifest_rows: List[pd.DataFrame] = []
    metrics_rows: List[pd.DataFrame] = []
    for run_dir in runs:
        run_id = run_dir.name
        vdirs = _variant_dirs(run_dir)
        for variant, vdir in vdirs.items():
            mpath = vdir / "metrics.csv"
            hpath = vdir / "hparams.yaml"
            if mpath.exists():
                metrics_rows.append(_read_metrics_csv(mpath, run_id, variant))
            if hpath.exists():
                manifest_rows.append(_read_hparams_yaml(hpath, run_id, variant))
    manifest = pd.concat(manifest_rows, ignore_index=True) if manifest_rows else pd.DataFrame()
    metrics = pd.concat(metrics_rows, ignore_index=True) if metrics_rows else pd.DataFrame()
    return manifest, metrics


def _collect_per_task_frames(per_task_root: Path) -> pd.DataFrame:
    if not per_task_root.exists():
        return pd.DataFrame()
    frames: List[pd.DataFrame] = []
    for exp_dir in sorted(per_task_root.iterdir()):
        if not exp_dir.is_dir():
            continue
        name = exp_dir.name
        if not (
            name.startswith("exp2_champion_arc-agi-2")
            or name.startswith("exp3_champion_arc-agi-2")
            or name.startswith("exp3b_merged_lora_arc-agi-2")
        ):
            continue
        for csvf in sorted(exp_dir.glob("*_per_task.csv")):
            try:
                df = pd.read_csv(csvf)
                df["experiment_name"] = name
                frames.append(df)
            except Exception:
                continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def compute_hparams_audit(manifest: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if manifest is None or manifest.empty:
        return pd.DataFrame(), pd.DataFrame()
    keys = [
        "vocab_size",
        "d_model",
        "num_encoder_layers",
        "num_decoder_layers",
        "num_heads",
        "d_ff",
        "max_grid_size",
        "dropout",
        "learning_rate",
        "weight_decay",
        "beta1",
        "beta2",
        "max_epochs",
        "pad_token",
        "use_context",
        "use_bridge",
    ]
    def _set_str(vals: List) -> str:
        uniq = sorted({str(v) for v in vals if pd.notna(v)})
        return ";".join(uniq)
    by_variant = (
        manifest.groupby("variant")[keys]
        .agg(lambda col: _set_str(col.tolist()))
        .reset_index()
    )
    rows = []
    for k in keys:
        vals = manifest.groupby("variant")[k].agg(lambda x: sorted(set(x))).to_dict()
        distinct = sorted({str(v) for vlist in vals.values() for v in vlist if v is not None})
        if len(distinct) > 1:
            rows.append({
                "key": k,
                "variants": ";".join(sorted(vals.keys())),
                "values_by_variant": ";".join(f"{vv}:{','.join(map(str, sorted(set(vals[vv]))))}" for vv in sorted(vals.keys())),
                "distinct_values": ",".join(distinct),
            })
    confounds = pd.DataFrame(rows)
    return by_variant, confounds


def compute_training_dynamics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics is None or metrics.empty:
        return pd.DataFrame()
    cols = ["run", "variant", "epoch", "val_grid_accuracy", "val_cell_accuracy"]
    df = metrics[[c for c in cols if c in metrics.columns]].dropna(subset=["val_grid_accuracy"])  # type: ignore
    def _per_group(g: pd.DataFrame) -> pd.Series:
        g = g.sort_values("epoch")
        best_idx = g["val_grid_accuracy"].idxmax()
        best = g.loc[best_idx]
        final = g.iloc[-1]
        auc_grid = float(np.trapezoid(y=g["val_grid_accuracy"].fillna(0.0).values, x=g["epoch"].values))
        auc_cell = float(np.trapezoid(y=g.get("val_cell_accuracy", pd.Series([np.nan]*len(g))).fillna(0.0).values, x=g["epoch"].values))
        return pd.Series({
            "n_epochs": len(g),
            "best_epoch_grid": int(best["epoch"]),
            "best_grid": float(best["val_grid_accuracy"]),
            "final_epoch": int(final["epoch"]),
            "final_grid": float(final["val_grid_accuracy"]),
            "stability_grid": float(final["val_grid_accuracy"]) - float(best["val_grid_accuracy"]),
            "auc_grid": auc_grid,
            "auc_cell": auc_cell,
        })
    out = df.groupby(["run", "variant"]).apply(_per_group).reset_index()
    return out


def compute_dissociation(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics is None or metrics.empty:
        return pd.DataFrame()
    if "val_cell_accuracy" not in metrics.columns:
        return pd.DataFrame()
    df = metrics.dropna(subset=["val_grid_accuracy", "val_cell_accuracy"]).copy()
    def _per_group(g: pd.DataFrame) -> pd.Series:
        g = g.sort_values("epoch")
        best_grid_row = g.loc[g["val_grid_accuracy"].idxmax()]
        best_cell_row = g.loc[g["val_cell_accuracy"].idxmax()]
        try:
            corr = float(g[["val_grid_accuracy", "val_cell_accuracy"]].corr(method="pearson").iloc[0, 1])
        except Exception:
            corr = float("nan")
        return pd.Series({
            "best_epoch_grid": int(best_grid_row["epoch"]),
            "best_epoch_cell": int(best_cell_row["epoch"]),
            "epoch_lag_cell_minus_grid": int(best_cell_row["epoch"]) - int(best_grid_row["epoch"]),
            "pearson_grid_cell": corr,
        })
    out = df.groupby(["run", "variant"]).apply(_per_group).reset_index()
    return out


def compute_cross_run_summary(metrics: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if metrics is None or metrics.empty:
        return pd.DataFrame(), pd.DataFrame()
    df = metrics.dropna(subset=["val_grid_accuracy"]).copy()
    best_by_run = (
        df.sort_values(["run", "variant", "val_grid_accuracy", "epoch"], ascending=[True, True, False, True])
        .groupby(["run", "variant"], as_index=False)
        .first()
    )
    rows = []
    for variant, g in best_by_run.groupby("variant"):
        arr_grid = g["val_grid_accuracy"].astype(float).values
        arr_cell = g.get("val_cell_accuracy", pd.Series([np.nan]*len(g))).astype(float).values
        n = len(arr_grid)
        mean_g = float(np.nanmean(arr_grid)) if n else float("nan")
        std_g = float(np.nanstd(arr_grid, ddof=1)) if n > 1 else float("nan")
        se_g = std_g / np.sqrt(n) if n and not np.isnan(std_g) else float("nan")
        ci95_g = 1.96 * se_g if not np.isnan(se_g) else float("nan")
        mean_c = float(np.nanmean(arr_cell)) if n else float("nan")
        std_c = float(np.nanstd(arr_cell, ddof=1)) if n > 1 else float("nan")
        se_c = std_c / np.sqrt(n) if n and not np.isnan(std_c) else float("nan")
        ci95_c = 1.96 * se_c if not np.isnan(se_c) else float("nan")
        rows.append({
            "variant": variant,
            "n_runs": n,
            "grid_mean": mean_g,
            "grid_std": std_g,
            "grid_ci95": ci95_g,
            "cell_mean": mean_c,
            "cell_std": std_c,
            "cell_ci95": ci95_c,
        })
    summary = pd.DataFrame(rows)
    pairs = [("exp3_champion_arc-agi-2_csv", "exp2_champion_arc-agi-2_csv"),
             ("exp3b_merged_lora_arc-agi-2_csv", "exp3_champion_arc-agi-2_csv"),
             ("exp3b_merged_lora_arc-agi-2_csv", "exp2_champion_arc-agi-2_csv")]
    tests = []
    for a, b in pairs:
        ga = best_by_run.loc[best_by_run["variant"] == a, "val_grid_accuracy"].astype(float).values
        gb = best_by_run.loc[best_by_run["variant"] == b, "val_grid_accuracy"].astype(float).values
        if len(ga) and len(gb):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    t_stat, t_p = ttest_ind(ga, gb, equal_var=False, nan_policy="omit")
                except Exception:
                    t_stat, t_p = float("nan"), float("nan")
                try:
                    mw_stat, mw_p = mannwhitneyu(ga, gb, alternative="two-sided")
                except Exception:
                    mw_stat, mw_p = float("nan"), float("nan")
        else:
            t_stat, t_p, mw_stat, mw_p = float("nan"), float("nan"), float("nan"), float("nan")
        tests.append({"variant_A": a, "variant_B": b, "t_stat": t_stat, "t_p": t_p, "mw_stat": mw_stat, "mw_p": mw_p})
    tests_df = pd.DataFrame(tests)
    return summary, tests_df


def compute_lora_effect(best_by_run: pd.DataFrame) -> pd.DataFrame:
    if best_by_run is None or best_by_run.empty:
        return pd.DataFrame()
    import re
    def _run_num(s: str) -> int:
        m = re.search(r"run_(\d+)", str(s))
        return int(m.group(1)) if m else -1
    tmp = best_by_run.copy()
    tmp["run_num"] = tmp["run"].map(_run_num)
    a = tmp[tmp["variant"] == "exp3_champion_arc-agi-2_csv"]["val_grid_accuracy"].groupby(tmp["run_num"]).first()
    b = tmp[tmp["variant"] == "exp3b_merged_lora_arc-agi-2_csv"]["val_grid_accuracy"].groupby(tmp["run_num"]).first()
    df = pd.DataFrame({"run_num": sorted(set(a.index).union(set(b.index)))})
    df["exp3_grid"] = df["run_num"].map(a.to_dict())
    df["exp3b_grid"] = df["run_num"].map(b.to_dict())
    df["delta_grid"] = df["exp3b_grid"] - df["exp3_grid"]
    return df


def save_summaries(manifest: pd.DataFrame, metrics: pd.DataFrame, per_task: pd.DataFrame, summary_dir: Path) -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)
    if not manifest.empty:
        manifest.to_csv(summary_dir / "arc_agi_2_manifest.csv", index=False)
        by_variant, confounds = compute_hparams_audit(manifest)
        if not by_variant.empty:
            by_variant.to_csv(summary_dir / "arc_agi_2_hparams_by_variant.csv", index=False)
        if confounds is not None and not confounds.empty:
            confounds.to_csv(summary_dir / "arc_agi_2_hparams_confound_report.csv", index=False)
    if not metrics.empty:
        metrics.to_csv(summary_dir / "arc_agi_2_metrics_aggregate.csv", index=False)
        best = (
            metrics.dropna(subset=["val_grid_accuracy"])
            .sort_values(["run", "variant", "val_grid_accuracy", "epoch"], ascending=[True, True, False, True])
            .groupby(["run", "variant"], as_index=False)
            .first()
        )
        best.to_csv(summary_dir / "arc_agi_2_best_by_grid_accuracy.csv", index=False)
        dyn = compute_training_dynamics(metrics)
        if not dyn.empty:
            dyn.to_csv(summary_dir / "arc_agi_2_training_dynamics.csv", index=False)
        diss = compute_dissociation(metrics)
        if not diss.empty:
            diss.to_csv(summary_dir / "arc_agi_2_dissociation.csv", index=False)
        nz = compute_nonzero_grid(metrics)
        if not nz.empty:
            nz.to_csv(summary_dir / "arc_agi_2_nonzero_grid.csv", index=False)
        cor = compute_correlations(metrics)
        if not cor.empty:
            cor.to_csv(summary_dir / "arc_agi_2_correlations.csv", index=False)
        conv = compute_convergence_and_overfit(metrics)
        if not conv.empty:
            conv.to_csv(summary_dir / "arc_agi_2_convergence.csv", index=False)
        cross, tests = compute_cross_run_summary(metrics)
        if not cross.empty:
            cross.to_csv(summary_dir / "arc_agi_2_cross_run_summary.csv", index=False)
        if tests is not None and not tests.empty:
            tests.to_csv(summary_dir / "arc_agi_2_significance_tests.csv", index=False)
        if not best.empty:
            lora = compute_lora_effect(best)
            if not lora.empty:
                lora.to_csv(summary_dir / "arc_agi_2_lora_effect.csv", index=False)
        # Confound-controlled comparison: restrict to max_grid_size == 30 if manifest is available
        if manifest is not None and not manifest.empty and "max_grid_size" in manifest.columns:
            try:
                m_join = metrics.merge(
                    manifest[["run", "variant", "max_grid_size"]],
                    on=["run", "variant"], how="left"
                )
                m30 = m_join.loc[m_join["max_grid_size"] == 30]
                if not m30.empty:
                    cross30, tests30 = compute_cross_run_summary(m30)
                    if not cross30.empty:
                        cross30.to_csv(summary_dir / "arc_agi_2_cross_run_summary_confound30.csv", index=False)
                    if tests30 is not None and not tests30.empty:
                        tests30.to_csv(summary_dir / "arc_agi_2_significance_tests_confound30.csv", index=False)
            except Exception:
                pass
    if not per_task.empty:
        per_task.to_csv(summary_dir / "arc_agi_2_per_task_all_epochs.csv", index=False)
        agg = (
            per_task.groupby(["experiment_name", "task_id", "category"], as_index=False)
            .agg({
                "grid_accuracy": "mean",
                "cell_accuracy": "mean",
            })
        )
        agg.to_csv(summary_dir / "arc_agi_2_per_task_aggregated.csv", index=False)


def ingest_tb_scalars(outputs_root: Path) -> pd.DataFrame:
    """Parse TensorBoard event files across runs/variants for triangulation."""
    if EventAccumulator is None:
        return pd.DataFrame()
    events = list(outputs_root.glob("logs_arc_agi_2_run_*/*_tb/**/events.out.tfevents*"))
    rows: List[Dict[str, object]] = []
    tags_of_interest = {
        "val_grid_accuracy",
        "val_cell_accuracy",
        "val_loss",
        "lr-Adam",
        "val_shape_mixed",
    }
    for ev in events:
        try:
            parts = ev.relative_to(outputs_root).parts
            run_id = parts[0]
            var_tb = parts[1]
            variant = var_tb.replace("_tb", "_csv")
            ea = EventAccumulator(str(ev))
            ea.Reload()
            scalar_tags = set(ea.Tags().get("scalars", []))
            for tag in sorted(scalar_tags & tags_of_interest):
                for s in ea.Scalars(tag):
                    rows.append({
                        "run": run_id,
                        "variant": variant,
                        "tag": tag,
                        "step": int(s.step),
                        "value": float(s.value),
                    })
        except Exception:
            continue
    return pd.DataFrame(rows)


def compute_lr_dynamics(tb: pd.DataFrame) -> pd.DataFrame:
    if tb is None or tb.empty:
        return pd.DataFrame()
    need = {"val_grid_accuracy", "val_cell_accuracy"}
    if not set(tb["tag"].unique()).intersection(need):
        return pd.DataFrame()
    def _per_group(g: pd.DataFrame) -> pd.Series:
        try:
            g = g.sort_values("step")
            # Build series
            lr = g[g["tag"].str.lower().str.startswith("lr")][["step", "value"]].rename(columns={"value": "lr"})
            grid = g[g["tag"] == "val_grid_accuracy"][["step", "value"]].rename(columns={"value": "grid"})
            cell = g[g["tag"] == "val_cell_accuracy"][["step", "value"]].rename(columns={"value": "cell"})
            if lr.empty or (grid.empty and cell.empty):
                return pd.Series({"r_lr_grid": np.nan, "r_lr_cell": np.nan})
            # Align via nearest step within small tolerance
            r_lr_grid = np.nan
            r_lr_cell = np.nan
            if not grid.empty:
                m_g = pd.merge_asof(grid.sort_values("step"), lr.sort_values("step"), on="step", direction="nearest", tolerance=50)
                m_g = m_g.dropna()
                if not m_g.empty and m_g[["grid", "lr"]].nunique().min() > 1:
                    r_lr_grid = float(m_g[["grid", "lr"]].corr().iloc[0, 1])
            if not cell.empty:
                m_c = pd.merge_asof(cell.sort_values("step"), lr.sort_values("step"), on="step", direction="nearest", tolerance=50)
                m_c = m_c.dropna()
                if not m_c.empty and m_c[["cell", "lr"]].nunique().min() > 1:
                    r_lr_cell = float(m_c[["cell", "lr"]].corr().iloc[0, 1])
            return pd.Series({"r_lr_grid": r_lr_grid, "r_lr_cell": r_lr_cell})
        except Exception:
            return pd.Series({"r_lr_grid": np.nan, "r_lr_cell": np.nan})
    out = tb.groupby(["run", "variant"], as_index=False).apply(_per_group).reset_index()
    if {"level_0", "level_1"}.issubset(out.columns):
        out = out.drop(columns=["level_0", "level_1"], errors="ignore")
    return out

def compute_nonzero_grid(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics is None or metrics.empty or "val_grid_accuracy" not in metrics.columns:
        return pd.DataFrame()
    df = metrics.dropna(subset=["val_grid_accuracy"]).copy()
    def _streak(mask: pd.Series) -> int:
        m = mask.values.astype(bool)
        best = cur = 0
        for v in m:
            if v:
                cur += 1
                if cur > best:
                    best = cur
            else:
                cur = 0
        return best
    def _per_group(g: pd.DataFrame) -> pd.Series:
        g = g.sort_values("epoch")
        nz = g["val_grid_accuracy"] > 0
        earliest = g.loc[nz, "epoch"].min() if nz.any() else np.nan
        count = int(nz.sum())
        streak = int(_streak(nz))
        return pd.Series({"earliest_nonzero_epoch": earliest, "nonzero_epoch_count": count, "longest_nonzero_streak": streak})
    out = df.groupby(["run", "variant"]).apply(_per_group).reset_index()
    return out


def compute_correlations(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics is None or metrics.empty:
        return pd.DataFrame()
    cols_need = ["val_grid_accuracy", "val_cell_accuracy", "val_copy_rate", "val_change_recall", "val_transformation_f1", "epoch"]
    miss = [c for c in cols_need if c not in metrics.columns]
    df = metrics.copy()
    for c in cols_need:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    def _corr(a: pd.Series, b: pd.Series) -> float:
        try:
            c = pd.concat([a, b], axis=1).dropna()
            if c.empty or c.nunique().min() <= 1:
                return float("nan")
            return float(c.corr(method="pearson").iloc[0, 1])
        except Exception:
            return float("nan")
    def _slope_epoch(y: pd.Series, x: pd.Series) -> float:
        try:
            c = pd.concat([x, y], axis=1).dropna()
            if len(c) < 2:
                return float("nan")
            k = min(10, len(c))
            c = c.sort_values(x.name).iloc[:k]
            p = np.polyfit(c[x.name].values, c[y.name].values, 1)
            return float(p[0])
        except Exception:
            return float("nan")
    def _per_group(g: pd.DataFrame) -> pd.Series:
        g = g.sort_values("epoch")
        r_gc = _corr(g.get("val_grid_accuracy"), g.get("val_cell_accuracy"))
        r_g_copy = _corr(g.get("val_grid_accuracy"), g.get("val_copy_rate"))
        r_g_change = _corr(g.get("val_grid_accuracy"), g.get("val_change_recall"))
        r_g_f1 = _corr(g.get("val_grid_accuracy"), g.get("val_transformation_f1"))
        slope_cell = _slope_epoch(g.get("val_cell_accuracy"), g.get("epoch"))
        slope_grid = _slope_epoch(g.get("val_grid_accuracy"), g.get("epoch"))
        return pd.Series({
            "r_grid_cell": r_gc,
            "r_grid_copy_rate": r_g_copy,
            "r_grid_change_recall": r_g_change,
            "r_grid_transformation_f1": r_g_f1,
            "early_slope_cell": slope_cell,
            "early_slope_grid": slope_grid,
        })
    out = df.groupby(["run", "variant"]).apply(_per_group).reset_index()
    return out


def compute_convergence_and_overfit(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics is None or metrics.empty:
        return pd.DataFrame()
    need = ["epoch", "val_cell_accuracy", "val_loss"]
    for c in need:
        if c not in metrics.columns:
            return pd.DataFrame()
    df = metrics.dropna(subset=["val_cell_accuracy"]).copy()
    def _per_group(g: pd.DataFrame) -> pd.Series:
        g = g.sort_values("epoch")
        best_cell_idx = g["val_cell_accuracy"].idxmax()
        best_cell_row = g.loc[best_cell_idx]
        target = 0.8 * float(best_cell_row["val_cell_accuracy"]) if pd.notna(best_cell_row["val_cell_accuracy"]) else np.nan
        t80 = g.loc[g["val_cell_accuracy"] >= target, "epoch"].min() if pd.notna(target) else np.nan
        final = g.iloc[-1]
        loss_delta = float(final.get("val_loss", np.nan)) - float(best_cell_row.get("val_loss", np.nan))
        cell_delta = float(final.get("val_cell_accuracy", np.nan)) - float(best_cell_row.get("val_cell_accuracy", np.nan))
        return pd.Series({
            "best_cell_epoch": float(best_cell_row["epoch"]),
            "best_cell": float(best_cell_row["val_cell_accuracy"]),
            "time_to_80pct_cell": float(t80) if pd.notna(t80) else np.nan,
            "loss_delta_final_minus_at_cell_peak": loss_delta,
            "cell_delta_final_minus_peak": cell_delta,
        })
    out = df.groupby(["run", "variant"]).apply(_per_group).reset_index()
    return out


def write_report(summary_dir: Path, manifest: pd.DataFrame, metrics: pd.DataFrame) -> None:
    try:
        cross = pd.read_csv(summary_dir / "arc_agi_2_cross_run_summary.csv") if (summary_dir / "arc_agi_2_cross_run_summary.csv").exists() else pd.DataFrame()
        cross30 = pd.read_csv(summary_dir / "arc_agi_2_cross_run_summary_confound30.csv") if (summary_dir / "arc_agi_2_cross_run_summary_confound30.csv").exists() else pd.DataFrame()
        conf = pd.read_csv(summary_dir / "arc_agi_2_hparams_confound_report.csv") if (summary_dir / "arc_agi_2_hparams_confound_report.csv").exists() else pd.DataFrame()
        dyn = pd.read_csv(summary_dir / "arc_agi_2_training_dynamics.csv") if (summary_dir / "arc_agi_2_training_dynamics.csv").exists() else pd.DataFrame()
        diss = pd.read_csv(summary_dir / "arc_agi_2_dissociation.csv") if (summary_dir / "arc_agi_2_dissociation.csv").exists() else pd.DataFrame()
        lora = pd.read_csv(summary_dir / "arc_agi_2_lora_effect.csv") if (summary_dir / "arc_agi_2_lora_effect.csv").exists() else pd.DataFrame()
        nonzero = pd.read_csv(summary_dir / "arc_agi_2_nonzero_grid.csv") if (summary_dir / "arc_agi_2_nonzero_grid.csv").exists() else pd.DataFrame()
        lines = []
        lines.append("ARC-AGI-2 Analysis Report\n")
        if not conf.empty:
            lines.append("Confounds:\n")
            for _, r in conf.iterrows():
                lines.append(f"- {r['key']}: {r['values_by_variant']}\n")
        if not cross.empty:
            lines.append("\nCross-run (best-by-grid, all variants):\n")
            for _, r in cross.iterrows():
                lines.append(f"- {r['variant']}: grid_mean={r['grid_mean']:.6f}, cell_mean={r['cell_mean']:.6f}\n")
        if not cross30.empty:
            lines.append("\nCross-run (confound-controlled max_grid_size=30):\n")
            for _, r in cross30.iterrows():
                lines.append(f"- {r['variant']}: grid_mean={r['grid_mean']:.6f}, cell_mean={r['cell_mean']:.6f}\n")
        if not lora.empty:
            lines.append("\nLoRA effect (grid deltas):\n")
            for _, r in lora.iterrows():
                lines.append(f"- run {int(r['run_num'])}: Δgrid={r['delta_grid']:.6f}\n")
        if not dyn.empty:
            lines.append("\nTraining dynamics (AUC/grid):\n")
            for _, r in dyn.iterrows():
                lines.append(f"- {r['run']} {r['variant']}: auc_grid={r['auc_grid']:.6f}, auc_cell={r['auc_cell']:.2f}\n")
        if not diss.empty:
            lines.append("\nDissociation (epoch lags):\n")
            for _, r in diss.iterrows():
                lines.append(f"- {r['run']} {r['variant']}: lag={r['epoch_lag_cell_minus_grid']}\n")
        if not nonzero.empty:
            lines.append("\nNonzero grid statistics:\n")
            for _, r in nonzero.iterrows():
                lines.append(f"- {r['run']} {r['variant']}: earliest={r['earliest_nonzero_epoch']}, count={r['nonzero_epoch_count']}, streak={r['longest_nonzero_streak']}\n")
        with open(summary_dir / "arc_agi_2_report.txt", "w") as f:
            f.writelines(lines)
    except Exception:
        pass


def main(outputs_root: Path | None = None) -> None:
    root = Path(outputs_root) if outputs_root else OUTPUTS_ROOT
    manifest, metrics = build_manifest_and_metrics(root)
    per_task = _collect_per_task_frames(PER_TASK_ROOT)
    save_summaries(manifest, metrics, per_task, SUMMARY_DIR)
    # Write a compact human-readable report inspired by champion/ablation analyses
    write_report(SUMMARY_DIR, manifest, metrics)
    # TensorBoard triangulation
    tb = ingest_tb_scalars(root)
    if tb is not None and not tb.empty:
        tb.to_csv(SUMMARY_DIR / "arc_agi_2_tb_scalars.csv", index=False)
        lr_dyn = compute_lr_dynamics(tb)
        if lr_dyn is not None and not lr_dyn.empty:
            lr_dyn.to_csv(SUMMARY_DIR / "arc_agi_2_lr_dynamics.csv", index=False)
    print("Manifest rows:", 0 if manifest is None else len(manifest))
    print("Metrics rows:", 0 if metrics is None else len(metrics))
    print("Per-task rows:", 0 if per_task is None else len(per_task))
    print("Summary directory:", SUMMARY_DIR)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-root", type=str, default=str(OUTPUTS_ROOT))
    args = ap.parse_args()
    main(Path(args.outputs_root))
