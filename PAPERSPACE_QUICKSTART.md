# Paperspace Quick Start Guide

**CRITICAL:** This workflow includes data regeneration to fix the sample count issue (was 15/task, now 150/task).

## After `git pull origin main`

```bash
# 1. Pull latest code
git pull origin main

# 2. REGENERATE DATA (removes old 15-sample data, generates new 150-sample data)
#    This is REQUIRED after pulling the latest changes
bash scripts/regenerate_data.sh

# Expected output:
#   - Removes old task files
#   - Generates 400 tasks × 150 samples = 60,000 examples
#   - Takes 15-20 minutes
#   - Exits with "Data regeneration complete!"

# 3. Run all three test suites (MANDATORY before training)
python -m pytest tests/              # Should see: 128 passed, 1 skipped
python scripts/test_all_training.py  # Should see: All 3 models PASSED
python verify_setup.py               # Should see: 7/8 or 8/8 checks pass

# 4. Start training
./run_training.sh champion
```

---

## Why Data Regeneration is Required

**The Problem:**
- Old data: 400 tasks × **15 samples** = 6,000 examples (training was suspiciously fast)
- New data: 400 tasks × **150 samples** = 60,000 examples (correct size)
- Git ignores the 456MB of task files (only commits small metadata files)
- After `git pull`, old 15-sample files remain on disk

**The Solution:**
- `scripts/regenerate_data.sh` safely removes old data and regenerates with 150 samples/task
- Also includes size-aware stratification to fix the validation bias bug

---

## Data Regeneration Details

### What Gets Removed
- 400 task JSON files (e.g., `007bbfb7.json`, `00d62c1b.json`, ...)
- Old `generation_statistics.json`

### What Gets Kept
- `task_categories.json` (taxonomy category mappings)
- `split_manifest.json` (size-aware train/val split)

### What Gets Generated
- 400 new task JSON files with 150 samples each
- New `generation_statistics.json`

### Safety Features
- Detects if data is already up-to-date (checks sample count)
- Prompts before regenerating if data looks correct
- Verifies 400 files were generated
- Exits with error code if generation fails

---

## Troubleshooting

### "ERROR: task_categories.json not found!"
You're not in the reproduction package root directory.
```bash
cd /notebooks/arc-taxonomy-reproduction  # Or wherever you cloned it
bash scripts/regenerate_data.sh
```

### Data generation is slow
**Expected:** 15-20 minutes on A6000 GPU node
- Each task takes 2-5 seconds to generate 150 verified examples
- Total: 400 tasks × 3 seconds avg = ~20 minutes
- This is normal!

### "No module named 're_arc'"
The re-arc submodule isn't initialized (should auto-initialize, but if not):
```bash
git submodule update --init --recursive
```

### Tests fail after regeneration
Check that data generation completed successfully:
```bash
ls data/distributional_alignment/*.json | wc -l
# Should output: 403 (400 tasks + 3 metadata files)

python3 -c "
import json
with open('data/distributional_alignment/007bbfb7.json') as f:
    task = json.load(f)
print(f'Samples in first task: {len(task[\"train\"])}')
"
# Should output: Samples in first task: 150
```

---

## Expected Results After Fix

### Training Speed
- **Before:** 20 seconds per epoch (suspiciously fast with 6K examples)
- **After:** ~20 minutes per epoch (correct with 60K examples)

### Validation Metrics
- **Before:** A1: 100% grid acc, 0% cell acc (impossible!)
- **Before:** ambiguous: 53% grid acc (suspiciously high)
- **After:** Reasonable metrics across all categories

### Per-Category Report (After Each Epoch)
```
==========================================================================================
PER-CATEGORY VALIDATION ACCURACY (Epoch 0)
==========================================================================================
Category     Grids    Grid Acc     Cell Acc    
------------------------------------------------------------------------------------------
A1           30             0-5%        10-30%    ← Fixed (was 100%/0%)
A2           90             5-10%       60-70%
C1           300            1-2%        70-80%
ambiguous    45             5-15%       70-80%    ← Fixed (was 53%)
...
```

---

## Complete Workflow (One-Liner for Copy-Paste)

```bash
git pull origin main && \
bash scripts/regenerate_data.sh && \
python -m pytest tests/ && \
python scripts/test_all_training.py && \
python verify_setup.py && \
./run_training.sh champion
```

---

## Storage Requirements

- **Local generation:** 456MB in `data/distributional_alignment/`
- **After training:** Additional ~100MB for checkpoints
- **Total:** ~600MB

---

**Last Updated:** October 27, 2025  
**Fix Version:** v1.0 - Size-Aware Stratification + 150 Samples/Task
