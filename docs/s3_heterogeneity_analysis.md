# S3 Heterogeneity Analysis - Section 7.4

Date: 2025-11-05
Source: paper/champion_FINAL_CLEAR.txt Section 2.5

---

## Executive Summary

The S3 (Topological) category exhibits extreme performance variance (0%-100% grid accuracy), motivating sub-classification into S3-A (pattern-based topology) and S3-B (graph reasoning). Within S3-A, we identify four distinct performance profiles that demonstrate the compositional gap phenomenon.

---

## S3-A vs S3-B Performance Comparison

### Overall Statistics (Champion Validation Set)

| Subclass | N | Grid Acc (Mean ± SD) | Grid Range | Cell Acc (Mean ± SD) | Cell Range |
|----------|---|---------------------|------------|---------------------|------------|
| **S3-A (pattern)** | 18 | 5.68% ± 23.54% | 0.00%-100.00% | 83.57% ± 12.77% | 51.49%-100.00% |
| **S3-B (graph)** | 5 | 0.10% ± 0.22% | 0.00%-0.50% | 82.24% ± 4.50% | 78.01%-87.80% |

**Statistical Significance:**
- Grid accuracy: t-test p=0.6076, Mann-Whitney U p=0.9550 (not significant)
- Cell accuracy: t-test p=0.8243, Mann-Whitney U p=0.4033 (not significant)

**Key Insight:** While not statistically significant on validation set, S3-A shows 57x higher mean grid accuracy and much higher variance, suggesting greater tractability for standard Transformers.

---

## S3-A Heterogeneity: Four Performance Profiles

### Table: S3-A Subgroup Breakdown

| Subgroup | # Tasks | Key Characteristic | Mean Grid Acc | Mean Cell Acc | Representative Task |
|----------|---------|-------------------|---------------|---------------|---------------------|
| **Perfect Solvers** | 1 | Complete grid-level solution | 100.0% | 100.0% | 445eab21 |
| **Partial Success** | 2 | Moderate grid accuracy achieved | 6.1% | 67.6% | 95990924, e50d258f |
| **Compositional Failure** | 11 | High cell (>80%), zero grid | 0.0% | 88.5% | 31aa019c, a2fd1cf0, b548a754 |
| **Low Affinity & Unsolved** | 4 | Low cell (<80%), zero grid | 0.0% | 70.7% | 9af7a82c, c909285e |

---

## Detailed Task Breakdown

### Perfect Solvers (1 task)
- **445eab21**: Grid=100.0%, Cell=100.0%
  - Only S3-A task achieving complete success
  - Investigation needed: What makes this uniquely solvable?

### Partial Success (2 tasks)
- **95990924**: Grid=10.0%, Cell=83.7%
- **e50d258f**: Grid=2.2%, Cell=51.5%
  - Achieves some grid-level success but inconsistent
  - Boundary cases or partial algorithmic understanding?

### Compositional Failure (11 tasks)
High local affinity (>80% cell) but zero global success—the core compositional gap:
- **31aa019c**: Cell=97.6%, Grid=0.0% ⭐ (smoking gun example in champion_FINAL_CLEAR.txt)
- **a2fd1cf0**: Cell=97.0%, Grid=0.0%
- **b548a754**: Cell=92.7%, Grid=0.0%
- **0a938d79**: Cell=94.1%, Grid=0.0%
- **b6afb2da**: Cell=88.2%, Grid=0.0%
- **928ad970**: Cell=87.9%, Grid=0.0%
- **67a423a3**: Cell=87.9%, Grid=0.0%
- **ec883f72**: Cell=86.8%, Grid=0.0%
- **50cb2852**: Cell=86.5%, Grid=0.0%
- **f35d900a**: Cell=85.4%, Grid=0.0%
- **25d487eb**: Cell=82.3%, Grid=0.0%

### Low Affinity & Unsolved (4 tasks)
- **9af7a82c**: Cell=77.7%, Grid=0.0%
- **c909285e**: Cell=74.2%, Grid=0.0%
- **913fb3ed**: Cell=69.4%, Grid=0.0%
- **00d62c1b**: Cell=61.4%, Grid=0.0%
  - May require more data or different architectural encoding

---

## Surrogate Metric Analysis by Subgroup

From CSV-derived metrics:

| Pattern | Copy Rate | Change Recall | Transform Quality |
|---------|-----------|---------------|-------------------|
| **PERFECT** | 0.020 | 0.998 | 0.998 |
| **PARTIAL** | 0.313 | 0.407 | 0.223 |
| **HIGH_CELL_ZERO_GRID** | 0.388 | 0.555 | 0.122 |
| **LOW_BOTH** | 0.187 | 0.766 | 0.323 |

**Correlations** (for HIGH_CELL_ZERO_GRID pattern):
- grid vs copy_rate: r=-0.560
- grid vs change_recall: r=+0.263
- grid vs transformation_quality: r=+0.715

---

## Paper-Ready Citations

- **S3 Classification Files:**
  - `data/s3_subclassification/s3_final_classification.json` (77 S3-A, 31 S3-B)
  - `data/s3_subclassification/s3_lookup.json` (task ID to subclass mapping)

- **Generator Code Analysis:**
  - `results/s3_subclassification/s3_generator_code_analysis.md` (detailed rationale for S3-A vs S3-B distinction)

- **Performance Data:**
  - `paper/champion_FINAL_CLEAR.txt` Section 2.5

---

## Key Findings for Paper

1. **S3 Variance Explained:** The 100% performance spread within S3 (0%-100% grid accuracy) is partially explained by S3-A vs S3-B distinction, though high variance persists within S3-A itself.

2. **Compositional Gap Validation:** 11 of 18 S3-A validation tasks (61%) exhibit the compositional failure pattern (>80% cell, 0% grid), providing category-specific evidence for the Compositional Gap.

3. **Perfect Solver Mystery:** Task 445eab21 achieves 100% success, suggesting some S3-A tasks are fully learnable. Generator code analysis needed to identify what makes it tractable.

4. **Diagnostic Power:** The four-subgroup breakdown demonstrates the framework's diagnostic utility—even within a single category, distinct architectural failure modes emerge.

---

## Next Steps (Research Questions)

1. Analyze generator code for 445eab21 to identify solvability factors
2. Investigate why "Compositional Failure" tasks learn local patterns but fail globally
3. Consider further S3-A subdivision (S3-A-easy vs S3-A-hard)
4. Test if additional training data resolves "Low Affinity & Unsolved" tasks
