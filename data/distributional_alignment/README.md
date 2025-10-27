# Distributional Alignment Dataset

This directory contains the **Phase 1B** training data for the ARC Taxonomy reproduction package.

## Dataset Statistics

- **400 tasks** from re-arc synthetic dataset
- **150 samples per task** = **60,000 total examples**
- **456 MB** total size
- Train/Val split: **308/92 tasks** (77/23 split)
- Train examples: **46,200**
- Val examples: **13,800**

## Files in Repository

Due to GitHub's file size limits, **only metadata files are committed**:

- ✅ `task_categories.json` (8KB) - Maps task IDs to taxonomy categories
- ✅ `generation_statistics.json` (183B) - Dataset generation metadata
- ✅ `split_manifest.json` (9KB) - Train/val split with size-aware stratification

**Task JSON files (400 files, 456MB) are NOT committed** - regenerate them locally.

## Regenerating the Dataset

### Step 1: Generate Task Files (15-20 minutes)

From the reproduction package root:

```bash
cd /path/to/reproduction

# Generate 60,000 examples (400 tasks × 150 samples)
python3 scripts/generate_synthetic_arc_dataset.py \
  --mode distributional_alignment \
  --samples-per-task 150 \
  --output-dir data/distributional_alignment
```

**Note:** The `generate_synthetic_arc_dataset.py` script is included in this reproduction package and requires the `re-arc` submodule. The script will automatically initialize the submodule if needed.

This will create:
- 400 task JSON files (e.g., `007bbfb7.json`, `00d62c1b.json`, ...)
- Updates `generation_statistics.json`

### Step 2: Verify Data Generation

```bash
# Check file count (should be 403: 400 tasks + 3 metadata)
ls data/distributional_alignment/*.json | wc -l

# Check size (should be ~456MB)
du -sh data/distributional_alignment/

# Verify a task file has 150 examples
python3 -c "
import json
with open('data/distributional_alignment/007bbfb7.json') as f:
    task = json.load(f)
print(f'Train examples: {len(task[\"train\"])}')
print(f'Test examples: {len(task[\"test\"])}')
"
# Expected output: Train examples: 150, Test examples: 0
```

### Step 3: (Optional) Regenerate Split

The `split_manifest.json` is already committed with a size-aware stratified split. Only regenerate if you want a different random seed:

```bash
# Use default seed (42)
python3 scripts/create_size_aware_split.py

# Or specify a different seed by editing the script
```

### Step 4: Verify Split Quality

```bash
python3 scripts/verify_split.py
```

Expected output:
- ✅ Category balance: Most categories 75-85% train ratio
- ✅ Size distribution: Train/val mean difference <20%
- ✅ No degenerate bias: A1 and ambiguous have varied output sizes

## Size-Aware Stratification

Unlike the original Phase 1B training which used simple random sampling, this reproduction uses **two-tier stratification**:

1. **Tier 1**: By taxonomy category (A1, A2, C1, C2, K1, L1, S1, S2, S3, ambiguous)
2. **Tier 2**: By output grid size within category:
   - Tiny: ≤10 cells
   - Small: 11-50 cells  
   - Medium: 51-200 cells
   - Large: 201+ cells

This ensures validation metrics are representative across both categories AND difficulty levels, preventing the bug where validation contained only degenerate 1-cell tasks.

## Troubleshooting

### "No module named 're_arc'"

The re-arc submodule isn't initialized:

```bash
git submodule update --init --recursive
```

### "FileNotFoundError: task_categories.json"

You're running from the wrong directory. Always run from the reproduction package root.

### Data generation is slow

Expected time: **15-20 minutes** on modern hardware. Each task takes 2-5 seconds to generate 150 verified examples.

### Split verification fails

If category balance fails for small categories (A1, K1, ambiguous with <15 tasks), this is expected due to mathematical constraints. The important checks are:
- ✅ Size distribution similarity
- ✅ No degenerate task bias

## Dataset Provenance

Generated from: `cultivation/systems/arc_reactor/external/re-arc/` (commit TBD)
Generation date: October 27, 2025
Configuration: distributional_alignment mode, 150 samples/task, seed 42
Stratification: Two-tier (category + size), seed 42
