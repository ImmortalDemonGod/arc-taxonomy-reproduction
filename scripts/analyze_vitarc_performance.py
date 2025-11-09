#!/usr/bin/env python3
"""
Analyze ViTARC performance data from Li et al. (2025) against our Neural Affinity taxonomy.

This script:
1. Extracts per-task solve rates from Li et al. (2025) Appendix E, Tables 6-9
2. Merges with our taxonomy classifications
3. Computes category-level statistics
4. Validates S3-A vs S3-B distinction
5. Generates publication-ready outputs

Pre-analysis findings:
- Smoking gun: A2 task 137eaa0f at 0.00% (validates ceiling)
- Outlier: A2 task 50846271 at 0.86% (specialist vs generalist difference)
- Expected coverage: ~399 tasks (may have 1 mismatch)
"""

import re
import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
VITARC_CSV_PATH = BASE_DIR / "data/external_validation/vitarc_appendix_tables.csv"
TAXONOMY_PATH = BASE_DIR / "data/taxonomy/all_tasks_classified.json"
S3_LOOKUP_PATH = BASE_DIR / "data/taxonomy/s3_lookup.json"
OUTPUT_DIR = BASE_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuration flag: prefer stable CSV over PDF parsing for reproducibility
USE_STABLE_CSV = True

# Affinity mapping based on V2 fine-tuning experiment category labels
# This tests: "Do our empirically-assigned category affinities predict external performance?"
# Source: v2_results_analysis.md - categories assigned based on predicted architectural alignment
AFFINITY_MAP = {
    # High Affinity (only C1 in V2 experiment)
    "C1": 4,  # Color Transformation - V2: 95.94% base accuracy
    
    # Medium Affinity  
    "C2": 3,  # Color Patterns - V2: 69.19% base
    "S1": 3,  # Spatial Elementary - V2: 96.89% base
    "S2": 3,  # Spatial Moderate - V2: 86.62% base
    "K1": 3,  # Scaling Operations - V2: 85.56% base
    "L1": 3,  # Logic/Set Operations - V2: 94.81% base (canonical: Medium)
    
    # Low Affinity
    "S3": 2,  # Topological/Graph - V2: 88.15% base
    "A1": 2,  # Iterative - V2: 78.54% base (canonical: Low)
    
    # Very Low Affinity
    "A2": 1,  # Search/Packing - V2: 53.61% base
    
    # Other categories
    "C3": 3,  # Assume medium
    "G1": 3,  # Assume medium
    "G2": 3,  # Assume medium
    "ambiguous": 0,  # Exclude from analysis
}

# ============================================================================
# Step 1: Extract text from PDF
# ============================================================================

def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from PDF using pdfplumber."""
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("pdfplumber required. Install: pip install pdfplumber")
    
    print(f"Extracting text from {pdf_path}...")
    text_parts = []
    
    with pdfplumber.open(pdf_path) as pdf:
        # Appendix E is likely in the later pages
        # We'll extract from all pages and search for Table 6-9
        for page in pdf.pages:
            text_parts.append(page.extract_text())
    
    full_text = "\n".join(text_parts)
    print(f"Extracted {len(full_text)} characters from PDF")
    return full_text

# ============================================================================
# Step 2: Parse ViTARC performance tables
# ============================================================================

def load_vitarc_data_from_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load ViTARC performance data from stable CSV file.
    
    This is the preferred method for reproducibility.
    """
    print(f"Loading ViTARC data from stable CSV: {csv_path}...")
    df = pd.read_csv(csv_path, index_col="task_id")
    print(f"Loaded {len(df)} task performance records")
    return df

def parse_vitarc_tables(pdf_text: str) -> pd.DataFrame:
    """
    Parse Tables 6-9 from Appendix E (fallback/verification method).
    
    Tables have TWO columns of tasks per row:
    ce22a75a  0.00  0.94  1.00    444801d8  0.00  0.98  1.00
    
    We use findall to capture ALL task occurrences in each line.
    """
    print("Parsing ViTARC performance tables from PDF...")
    
    # Regex pattern: task_id (8 hex chars) followed by 3 decimal scores
    # Use findall to get ALL matches per line (handles two-column format)
    pattern = re.compile(r"([a-f0-9]{8})\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")
    
    tasks = []
    for line in pdf_text.splitlines():
        # findall returns list of tuples, one per match
        matches = pattern.findall(line)
        for match in matches:
            task_id, vit_vanilla, vitarc_vt, vitarc = match
            tasks.append({
                "task_id": task_id,
                "vit_vanilla": float(vit_vanilla),
                "vitarc_vt": float(vitarc_vt),
                "vitarc_solve_rate": float(vitarc)
            })
    
    df = pd.DataFrame(tasks)
    
    # Remove duplicates (in case tasks appear in multiple tables)
    df = df.drop_duplicates(subset="task_id").set_index("task_id")
    
    print(f"Extracted {len(df)} task performance records")
    
    # Save raw extracted data
    raw_output = OUTPUT_DIR / "vitarc_performance_raw.csv"
    df.to_csv(raw_output)
    print(f"Saved raw data to {raw_output}")
    
    return df

# ============================================================================
# Step 3: Load taxonomy data and merge
# ============================================================================

def load_and_merge_taxonomy(vitarc_df: pd.DataFrame) -> pd.DataFrame:
    """Load taxonomy and S3 subtype data, merge with ViTARC performance."""
    print("Loading taxonomy classifications...")
    
    # Load main taxonomy
    with open(TAXONOMY_PATH, "r") as f:
        taxonomy = json.load(f)
    
    taxonomy_df = pd.DataFrame([
        {"task_id": task_id, "category": cat}
        for task_id, cat in taxonomy.items()
    ]).set_index("task_id")
    
    # Load S3 subtypes (extract just the classification string from the dict)
    with open(S3_LOOKUP_PATH, "r") as f:
        s3_lookup = json.load(f)
    
    s3_df = pd.DataFrame([
        {"task_id": task_id, "s3_subtype": data["classification"]}
        for task_id, data in s3_lookup.items()
    ]).set_index("task_id")
    
    # Merge all data
    merged = vitarc_df.join(taxonomy_df, how="inner")
    merged = merged.join(s3_df, how="left")
    
    # Add affinity scores
    merged["affinity_score"] = merged["category"].map(AFFINITY_MAP).fillna(0)
    
    # Create affinity level labels for readability
    affinity_labels = {4: "High", 3: "Medium", 2: "Low", 1: "Very Low", 0: "Excluded"}
    merged["affinity_level"] = merged["affinity_score"].map(affinity_labels)
    
    print(f"Merged dataset: {len(merged)} tasks with taxonomy labels")
    print(f"Coverage: {len(merged)}/{len(vitarc_df)} ViTARC tasks matched")
    
    # Report missing tasks
    missing_in_taxonomy = set(vitarc_df.index) - set(merged.index)
    if missing_in_taxonomy:
        print(f"WARNING: {len(missing_in_taxonomy)} tasks in ViTARC not in our taxonomy:")
        for task_id in sorted(missing_in_taxonomy)[:10]:  # Show first 10
            print(f"  - {task_id}")
    
    # Save merged data
    merged_output = OUTPUT_DIR / "vitarc_with_taxonomy.csv"
    merged.to_csv(merged_output)
    print(f"Saved merged data to {merged_output}")
    
    return merged

# ============================================================================
# Step 4: Compute statistics
# ============================================================================

def compute_statistics(df: pd.DataFrame) -> dict:
    """Compute all statistics needed for Section 7.5."""
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS")
    print("="*80)
    
    results = {}
    
    # Category-level aggregates
    print("\n1. Category-level performance:")
    category_stats = df.groupby("category")["vitarc_solve_rate"].agg([
        "count", "mean", "median", "std", "min", "max"
    ]).sort_values("mean", ascending=False)
    
    print(category_stats)
    results["category_stats"] = category_stats
    
    # Spearman correlation
    print("\n2. Spearman correlation (Affinity Score vs Solve Rate):")
    rho, p_value = spearmanr(df["affinity_score"], df["vitarc_solve_rate"])
    print(f"   ρ = {rho:.4f}, p = {p_value:.4e}")
    results["spearman_rho"] = rho
    results["spearman_p"] = p_value
    
    # S3 sub-classification validation
    print("\n3. S3-A vs S3-B validation:")
    s3_tasks = df[df["category"] == "S3"].copy()
    if len(s3_tasks) > 0 and "s3_subtype" in s3_tasks.columns:
        s3_stats = s3_tasks.groupby("s3_subtype")["vitarc_solve_rate"].agg([
            "count", "mean", "median", "std"
        ])
        print(s3_stats)
        results["s3_stats"] = s3_stats
        
        # Statistical test: Is S3-A significantly better than S3-B?
        s3a_rates = s3_tasks[s3_tasks["s3_subtype"] == "S3-A"]["vitarc_solve_rate"]
        s3b_rates = s3_tasks[s3_tasks["s3_subtype"] == "S3-B"]["vitarc_solve_rate"]
        if len(s3a_rates) > 0 and len(s3b_rates) > 0:
            u_stat, p_value = mannwhitneyu(s3a_rates, s3b_rates, alternative="two-sided")
            print(f"\n   Mann-Whitney U test (S3-A vs S3-B): U={u_stat:.1f}, p={p_value:.4f}")
            if p_value < 0.05:
                print(f"   ✓ Difference is statistically significant (p < 0.05)")
            else:
                print(f"   ✗ Difference is NOT statistically significant (p >= 0.05)")
            results["s3_mannwhitney_u"] = float(u_stat)
            results["s3_mannwhitney_p"] = float(p_value)
    else:
        print("   No S3 subtypes found")
    
    # Cross-Affinity Statistical Significance Tests
    print("\n4. Cross-Affinity Statistical Tests (V2 Category Labels):")
    
    # Filter out excluded categories
    df_test = df[df["affinity_score"] > 0].copy()
    
    # Group by affinity level
    affinity_groups = {
        "High (4)": df_test[df_test["affinity_score"] == 4]["vitarc_solve_rate"],
        "Medium (3)": df_test[df_test["affinity_score"] == 3]["vitarc_solve_rate"],
        "Low (2)": df_test[df_test["affinity_score"] == 2]["vitarc_solve_rate"],
        "Very Low (1)": df_test[df_test["affinity_score"] == 1]["vitarc_solve_rate"]
    }
    
    # High vs. Very Low (most critical comparison)
    if len(affinity_groups["High (4)"]) > 0 and len(affinity_groups["Very Low (1)"]) > 0:
        u_stat, p_val = mannwhitneyu(
            affinity_groups["High (4)"], 
            affinity_groups["Very Low (1)"], 
            alternative="greater"
        )
        
        # Effect size (Cohen's d)
        mean_h = affinity_groups["High (4)"].mean()
        mean_vl = affinity_groups["Very Low (1)"].mean()
        std_h = affinity_groups["High (4)"].std()
        std_vl = affinity_groups["Very Low (1)"].std()
        n_h = len(affinity_groups["High (4)"])
        n_vl = len(affinity_groups["Very Low (1)"])
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n_h - 1) * std_h**2 + (n_vl - 1) * std_vl**2) / (n_h + n_vl - 2))
        cohens_d = (mean_h - mean_vl) / pooled_std if pooled_std > 0 else 0
        
        print(f"   High vs. Very Low:")
        print(f"     Mann-Whitney U: p = {p_val:.4e} {'✓ SIGNIFICANT' if p_val < 0.05 else '✗ NOT significant'}")
        print(f"     Mean difference: {mean_h:.2%} (High) vs {mean_vl:.2%} (Very Low) = {mean_h - mean_vl:+.2%}")
        print(f"     Cohen's d: {cohens_d:.3f} ({'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'} effect)")
        print(f"     Sample sizes: n_High={n_h}, n_VeryLow={n_vl}")
        
        results["high_vs_verylow_p"] = float(p_val)
        results["high_vs_verylow_cohens_d"] = float(cohens_d)
        results["high_mean"] = float(mean_h)
        results["verylow_mean"] = float(mean_vl)
    
    # Additional pairwise tests
    print("\n   Pairwise comparisons:")
    pairs = [
        ("High (4)", "Medium (3)"),
        ("Medium (3)", "Low (2)"),
        ("Low (2)", "Very Low (1)")
    ]
    
    for level1, level2 in pairs:
        if len(affinity_groups[level1]) > 0 and len(affinity_groups[level2]) > 0:
            u, p = mannwhitneyu(affinity_groups[level1], affinity_groups[level2], alternative="two-sided")
            m1 = affinity_groups[level1].mean()
            m2 = affinity_groups[level2].mean()
            print(f"     {level1} vs {level2}: p={p:.4f}, Δ={m1-m2:+.2%}")
    
    # Smoking guns and outliers
    print("\n5. Key examples:")
    
    # A2 smoking gun (137eaa0f)
    if "137eaa0f" in df.index:
        smoking_gun = df.loc["137eaa0f"]
        print(f"   ✓ Smoking gun (137eaa0f): {smoking_gun['vitarc_solve_rate']:.2%} [Category: {smoking_gun['category']}]")
        results["smoking_gun_137eaa0f"] = float(smoking_gun["vitarc_solve_rate"])
    else:
        print(f"   ✗ Smoking gun (137eaa0f): NOT FOUND in ViTARC data")
    
    # A2 outlier (50846271)
    if "50846271" in df.index:
        outlier = df.loc["50846271"]
        print(f"   ✓ Outlier (50846271): {outlier['vitarc_solve_rate']:.2%} [Category: {outlier['category']}]")
        results["outlier_50846271"] = float(outlier["vitarc_solve_rate"])
    else:
        print(f"   ✗ Outlier (50846271): NOT FOUND in ViTARC data")
    
    # A2 total failures
    a2_tasks = df[df["category"] == "A2"]
    if len(a2_tasks) > 0:
        a2_zeros = int((a2_tasks["vitarc_solve_rate"] == 0.0).sum())
        print(f"   A2 tasks at 0.0%: {a2_zeros}/{len(a2_tasks)} ({a2_zeros/len(a2_tasks):.1%})")
        results["a2_total_failures"] = a2_zeros
        results["a2_total_tasks"] = len(a2_tasks)
    
    # 2x2 Quadrant Analysis: Affinity vs Performance
    print("\n6. Affinity vs. Performance Quadrant Analysis:")
    df_quad = df[df["affinity_score"] > 0].copy()
    
    # Define high/low performance using median split
    perf_median = df_quad["vitarc_solve_rate"].median()
    df_quad["performance_level"] = df_quad["vitarc_solve_rate"].apply(
        lambda x: "High Perf" if x > perf_median else "Low Perf"
    )
    
    # Cross-tabulation
    quadrant_counts = df_quad.groupby(["affinity_level", "performance_level"]).size().unstack(fill_value=0)
    print(quadrant_counts)
    
    # Compositional Gap proxy: High affinity but low performance
    if "High" in quadrant_counts.index and "Low Perf" in quadrant_counts.columns:
        high_aff_low_perf = int(quadrant_counts.loc["High", "Low Perf"])
        print(f"\n   Compositional Gap Proxy: {high_aff_low_perf} tasks with High Affinity but Low Performance")
        print(f"   (These tasks are architecturally well-matched but still fail - suggests compositional bottleneck)")
        results["compositional_gap_proxy_count"] = high_aff_low_perf
    
    results["quadrant_analysis"] = quadrant_counts.to_dict()
    
    # Save summary statistics
    stats_output = OUTPUT_DIR / "statistics_summary.json"
    with open(stats_output, "w") as f:
        # Convert non-serializable types
        serializable_results = {}
        for k, v in results.items():
            if isinstance(v, (pd.DataFrame, pd.Series)):
                serializable_results[k] = v.to_dict()
            elif isinstance(v, (int, float, str, bool)):
                serializable_results[k] = v
            else:
                # Convert numpy types
                serializable_results[k] = float(v) if hasattr(v, 'item') else v
        json.dump(serializable_results, f, indent=2)
    print(f"\nSaved statistics to {stats_output}")
    
    return results

# ============================================================================
# Step 5: Generate visualizations
# ============================================================================

def generate_visualizations(df: pd.DataFrame, stats: dict):
    """Create publication-ready figures."""
    print("\nGenerating visualizations...")
    
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
    
    # Figure 1: Category mean solve rates (bar chart)
    fig, ax = plt.subplots(figsize=(10, 6))
    category_means = stats["category_stats"]["mean"].sort_values(ascending=False)
    
    colors = ["#2ecc71" if v >= 0.6 else "#f39c12" if v >= 0.3 else "#e74c3c" 
              for v in category_means.values]
    
    category_means.plot(kind="bar", ax=ax, color=colors)
    ax.set_xlabel("ARC Taxonomy Category", fontsize=12)
    ax.set_ylabel("Mean ViTARC Solve Rate", fontsize=12)
    ax.set_title("External Validation: ViTARC Performance by Neural Affinity Category", 
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="50% threshold")
    ax.legend()
    plt.xticks(rotation=0)
    plt.tight_layout()
    
    fig_path = OUTPUT_DIR / "category_performance_barplot.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved bar chart to {fig_path}")
    plt.close()
    
    # Figure 2: Boxplot (Affinity Score vs Solve Rate) - Better for distribution visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Boxplot showing distribution per affinity level
    sns.boxplot(data=df, x="affinity_score", y="vitarc_solve_rate", ax=ax, 
                color="lightblue", width=0.5)
    # Overlay individual points
    sns.stripplot(data=df, x="affinity_score", y="vitarc_solve_rate", ax=ax,
                 color=".25", alpha=0.3, size=3)
    
    ax.set_xticklabels(["Very Low\n(A1, A2)", "Low\n(L1, S3)", "Medium\n(C2, S1, S2, K1)", "High\n(C1)"])
    ax.set_xlabel("Neural Affinity Level", fontsize=12)
    ax.set_ylabel("ViTARC Solve Rate", fontsize=12)
    ax.set_title(f"Neural Affinity Predicts Performance (ρ={stats['spearman_rho']:.3f}, p={stats['spearman_p']:.3f})", 
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    boxplot_path = OUTPUT_DIR / "affinity_correlation_boxplot.png"
    plt.savefig(boxplot_path, dpi=300, bbox_inches="tight")
    print(f"Saved boxplot to {boxplot_path}")
    plt.close()
    
    # Also keep scatter as supplementary figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for category in df["category"].unique():
        subset = df[df["category"] == category]
        ax.scatter(subset["affinity_score"], subset["vitarc_solve_rate"], 
                  label=category, alpha=0.6, s=50)
    
    ax.set_xlabel("Affinity Score (1=Very Low, 4=High)", fontsize=12)
    ax.set_ylabel("ViTARC Solve Rate", fontsize=12)
    ax.set_title(f"Per-Category Performance (ρ={stats['spearman_rho']:.3f})", 
                 fontsize=14, fontweight="bold")
    ax.legend(title="Category", bbox_to_anchor=(1.05, 1), loc="upper left", ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    scatter_path = OUTPUT_DIR / "affinity_correlation_scatter.png"
    plt.savefig(scatter_path, dpi=300, bbox_inches="tight")
    print(f"Saved scatter plot to {scatter_path}")
    plt.close()
    
    # Figure 3: S3-A vs S3-B comparison (if available)
    if "s3_stats" in stats and len(stats["s3_stats"]) > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        s3_means = stats["s3_stats"]["mean"]
        s3_means.plot(kind="bar", ax=ax, color=["#3498db", "#9b59b6"])
        ax.set_xlabel("S3 Subtype", fontsize=12)
        ax.set_ylabel("Mean ViTARC Solve Rate", fontsize=12)
        ax.set_title("S3 Sub-classification Validated: Pattern vs Graph Reasoning", 
                     fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1.0)
        plt.xticks(rotation=0)
        plt.tight_layout()
        
        s3_path = OUTPUT_DIR / "s3_subtype_comparison.png"
        plt.savefig(s3_path, dpi=300, bbox_inches="tight")
        print(f"Saved S3 comparison to {s3_path}")
        plt.close()
    
    # Figure 4: Affinity-level summary with error bars
    fig, ax = plt.subplots(figsize=(10, 6))
    df_summary = df[df["affinity_score"] > 0].copy()
    
    affinity_summary = df_summary.groupby("affinity_level")["vitarc_solve_rate"].agg([
        "mean", "std", "count"
    ]).reindex(["Very Low", "Low", "Medium", "High"])
    
    # Calculate standard error
    affinity_summary["se"] = affinity_summary["std"] / np.sqrt(affinity_summary["count"])
    
    x_pos = np.arange(len(affinity_summary))
    ax.bar(x_pos, affinity_summary["mean"], 
           yerr=affinity_summary["se"], 
           capsize=5, alpha=0.7,
           color=["#e74c3c", "#f39c12", "#3498db", "#2ecc71"])
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(affinity_summary.index)
    ax.set_ylabel("Mean ViTARC Solve Rate", fontsize=12)
    ax.set_xlabel("Category Affinity Level (V2-Based)", fontsize=12)
    ax.set_title("External Validation: Category Affinity Predicts Performance", 
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.0)
    
    # Add sample sizes
    for i, (idx, row) in enumerate(affinity_summary.iterrows()):
        ax.text(i, row["mean"] + row["se"] + 0.03, 
               f"n={int(row['count'])}", 
               ha='center', fontsize=9)
    
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    summary_path = OUTPUT_DIR / "affinity_summary_barplot.png"
    plt.savefig(summary_path, dpi=300, bbox_inches="tight")
    print(f"Saved affinity summary to {summary_path}")
    plt.close()

# ============================================================================
# Main execution
# ============================================================================

def main():
    """Run the complete analysis pipeline."""
    print("="*80)
    print("ViTARC External Validation Analysis")
    print("="*80)
    
    # Step 1 & 2: Load ViTARC data (prefer stable CSV for reproducibility)
    if USE_STABLE_CSV and VITARC_CSV_PATH.exists():
        vitarc_df = load_vitarc_data_from_csv(VITARC_CSV_PATH)
    else:
        print("\nFalling back to PDF extraction (CSV not found or disabled)...")
        pdf_text = extract_pdf_text(PDF_PATH)
        vitarc_df = parse_vitarc_tables(pdf_text)
    
    # Step 3: Merge with taxonomy
    merged_df = load_and_merge_taxonomy(vitarc_df)
    
    # Step 4: Compute statistics
    stats = compute_statistics(merged_df)
    
    # Step 5: Generate visualizations
    generate_visualizations(merged_df, stats)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nGenerated files:")
    for file in sorted(OUTPUT_DIR.glob("*")):
        print(f"  - {file.name}")
    
    print("\nNext steps:")
    print("1. Review statistics_summary.json for values to insert into Section 7.5")
    print("2. Check visualizations for publication-ready figures")
    print("3. Update main.md placeholders with actual results")

if __name__ == "__main__":
    main()
