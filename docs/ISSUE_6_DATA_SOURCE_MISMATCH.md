# CRITICAL ISSUE #6: Data Source Mismatch

**Discovery Date:** October 28, 2025, 1:13 PM  
**Severity:** 🚨 **BLOCKER** - Would completely invalidate ablation study  
**Status:** Identified, not yet fixed

---

## The Problem

**Baseline and Exp 0 are configured to train on COMPLETELY DIFFERENT DATA than Exp 1, 2, 3 (Champion).**

### Current Configuration:

| Experiment | Data Path | Notes |
|------------|-----------|-------|
| **Baseline** | `data/tasks/*.json` | ❌ Directory doesn't exist! Script will fail. |
| **Exp 0** | `data/tasks/*.json` | ❌ Directory doesn't exist! Script will fail. |
| **Exp 1** | `data/distributional_alignment/` + split_manifest.json | ✅ Correct |
| **Exp 2** | `data/distributional_alignment/` + split_manifest.json | ✅ Correct |
| **Exp 3 (Champion)** | `data/distributional_alignment/` + split_manifest.json | ✅ Correct |

### Code Evidence:

**Baseline (`train_baseline_decoder_only.py`), lines 27-28:**
```python
data_dir = Path(__file__).parent.parent / "data" / "tasks"
task_files = sorted(list(data_dir.glob("*.json")))
```

**Exp 0 (`train_exp0_encoder_decoder.py`), lines 27-28:**
```python
data_dir = Path(__file__).parent.parent / "data" / "tasks"
task_files = sorted(list(data_dir.glob("*.json")))
```

**Exp 1, 2, 3 (Correct):**
```python
data_dir = Path(__file__).parent.parent / "data" / "distributional_alignment"

with open(data_dir / "split_manifest.json") as f:
    split_info = json.load(f)

train_files = [data_dir / fname for fname in split_info["train_files"]]
val_files = [data_dir / fname for fname in split_info["val_files"]]
```

---

## Impact

1. **Scripts won't run:** Baseline and Exp 0 will crash immediately because `data/tasks/` doesn't exist
2. **Invalid comparison:** Even if the directory existed, using different data would make all comparisons meaningless
3. **No train/val split:** Baseline and Exp 0 don't use split_manifest.json, so they would train on all data without proper validation

---

## The Fix

Update Baseline and Exp 0 to use the same data loading pattern as Exp 1, 2, 3:

**File:** `train_baseline_decoder_only.py`  
**File:** `train_exp0_encoder_decoder.py`

**Replace:**
```python
# Get data files
data_dir = Path(__file__).parent.parent / "data" / "tasks"
task_files = sorted(list(data_dir.glob("*.json")))

# Split into train/val (80/20)
split_idx = int(0.8 * len(task_files))
train_files = task_files[:split_idx]
val_files = task_files[split_idx:]
```

**With:**
```python
# Get data files - MUST match Champion's data source
import json
data_dir = Path(__file__).parent.parent / "data" / "distributional_alignment"

# Load split manifest to get train/val split  
with open(data_dir / "split_manifest.json") as f:
    split_info = json.load(f)

train_files = [data_dir / fname for fname in split_info["train_files"]]
val_files = [data_dir / fname for fname in split_info["val_files"]]
```

---

## Priority

**P0 - MUST FIX BEFORE ANY TRAINING**

This issue would make the entire ablation study invalid.
