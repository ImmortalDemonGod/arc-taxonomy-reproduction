# Ablation Redesign: Complete Setup Guide
## Independent Component Testing

**Date:** 2025-11-05  
**Design:** Test each component independently against E-D baseline  
**Goal:** Justify "minimal necessary additions" narrative

---

## Experimental Design

```
Exp0: E-D Baseline (reference point)
  ├─ Exp1: E-D + Grid2D only    (isolated Grid2D test)
  ├─ Exp2: E-D + PermInv only   (isolated PermInv test)  ← NEED TO CREATE
  ├─ Exp3: E-D + Context only   (isolated Context test)  ← NEED TO CREATE
  └─ Exp4: Champion (All)       (synergy test)
```

---

## Architecture Status Audit

### What EXISTS (verified):

| Exp | Description | Architecture File | Lightning Module | Training Script | Status |
|-----|-------------|-------------------|------------------|-----------------|--------|
| Exp0 | E-D Baseline | `encoder_decoder_baseline.py` | `exp0_encoder_decoder_lightning.py` | `train_exp0_encoder_decoder.py` | ✅ READY |
| Exp1 | E-D + Grid2D | `ed_with_grid2d_pe.py` | `exp1_grid2d_pe_lightning.py` | `train_exp1_grid2d_pe.py` | ✅ READY |
| Exp4 | Champion | `champion_architecture.py` | `exp3_champion_lightning.py` | `train_exp3_champion.py` | ✅ READY |

### What DOES NOT EXIST (needs creation):

| Exp | Description | Need to Create |
|-----|-------------|----------------|
| Exp2 | E-D + PermInv only | Architecture + Lightning + Script |
| Exp3 | E-D + Context only | Architecture + Lightning + Script |

### What EXISTS but is WRONG (old cumulative design):

| File | Problem | Action |
|------|---------|--------|
| `exp2_perminv_lightning.py` | Trains E-D + Grid2D + PermInv (cumulative) | DO NOT USE |
| `train_exp2_perminv.py` | Uses cumulative architecture | DO NOT USE |
| `train_exp2_grid2d_perminv.py` | Cumulative design | DO NOT USE |

---

## Step 1: Verify Existing Scripts Work

### Test Exp0 (E-D Baseline)

```bash
cd /Users/tomriddle1/Holistic-Performance-Enhancement/cultivation/systems/arc_reactor/publications/arc_taxonomy_2025/reproduction

# Quick smoke test (1 batch)
python scripts/train_exp0_encoder_decoder.py --fast_dev_run 1

# Expected output:
# - No errors
# - Loads 18 training files, 92 validation files
# - Model has ~1.71M params
# - Trains 1 batch successfully
```

### Test Exp1 (E-D + Grid2D)

```bash
# Quick smoke test
python scripts/train_exp1_grid2d_pe.py --fast_dev_run 1

# Expected:
# - No errors
# - Same data files
# - Model has ~1.71M params
# - Uses Grid2D PE
```

### Test Exp4 (Champion)

```bash
# Quick smoke test
python scripts/train_exp3_champion.py --fast_dev_run 1

# Expected:
# - No errors
# - Model has ~1.72M params
# - max_grid_size=30 (VERIFY THIS!)
```

**CRITICAL CHECK:**
```bash
# Verify max_grid_size=30 in Champion
grep "max_grid_size" scripts/train_exp3_champion.py
grep "max_grid_size" src/models/exp3_champion_lightning.py

# Should show: max_grid_size: int = 30
# NOT: max_grid_size: int = 35
```

---

## Step 2: Create Missing Architectures

### 2a. Create E-D + PermInv only (Exp2)

**File:** `src/models/ed_with_perminv.py`

**Architecture:**
- Encoder-Decoder: ✅
- Grid2D PE: ❌ NO (use standard 1D PE)
- PermInv Embedding: ✅
- Context System: ❌ NO

**Based on:** Copy `ed_with_grid2d_pe_and_perminv.py` and REMOVE Grid2D PE logic

```bash
# I'll create this file next
```

### 2b. Create E-D + Context only (Exp3)

**File:** `src/models/ed_with_context.py`

**Architecture:**
- Encoder-Decoder: ✅
- Grid2D PE: ❌ NO (use standard 1D PE)
- PermInv Embedding: ❌ NO (use standard embedding)
- Context System: ✅ (Encoder + Bridge)

**Based on:** Copy from `champion_architecture.py` and REMOVE Grid2D + PermInv

```bash
# I'll create this file next
```

---

## Step 3: Create Lightning Modules

### 3a. Lightning Module for Exp2 (E-D + PermInv)

**File:** `src/models/exp2_ed_perminv_lightning.py`

**Based on:** `exp0_encoder_decoder_lightning.py` but using `ed_with_perminv` architecture

### 3b. Lightning Module for Exp3 (E-D + Context)

**File:** `src/models/exp3_ed_context_lightning.py`

**Based on:** `exp3_champion_lightning.py` but using `ed_with_context` architecture

---

## Step 4: Create Training Scripts

### 4a. Training Script for Exp2

**File:** `scripts/train_exp2_ed_perminv.py`

### 4b. Training Script for Exp3

**File:** `scripts/train_exp3_ed_context.py`

---

## Step 5: Hyperparameter Verification

**CRITICAL:** All experiments must use IDENTICAL hyperparameters

### Required Settings (from Trial 69):

```python
learning_rate = 0.0018498849832733245
weight_decay = 0.0
beta1 = 0.95
beta2 = 0.999
dropout = 0.167 (or 0.1 for simpler architectures)
max_epochs = 200  # Doubled from 100
max_grid_size = 30  # CRITICAL!
seed = 307, 308, 309, 310, 311  # 5 seeds

# Architecture
d_model = 168
num_encoder_layers = 1
num_decoder_layers = 3
num_heads = 4
d_ff = 672
vocab_size = 11
pad_token = 10
```

### Verification Command:

```bash
# Check all scripts have matched hyperparameters
grep -A 10 "learning_rate" scripts/train_exp*.py

# All should show:
# learning_rate=0.0018498849832733245
# weight_decay=0.0
# beta1=0.95
```

---

## Step 6: Execution Plan

### Phase 1: Quick Validation (1-2 hours)

Run all 5 architectures with 1 seed, 10 epochs to verify no errors:

```bash
# Exp0
python scripts/train_exp0_encoder_decoder.py --seed 307 --max_epochs 10

# Exp1
python scripts/train_exp1_grid2d_pe.py --seed 307 --max_epochs 10

# Exp2
python scripts/train_exp2_ed_perminv.py --seed 307 --max_epochs 10

# Exp3
python scripts/train_exp3_ed_context.py --seed 307 --max_epochs 10

# Exp4
python scripts/train_exp3_champion.py --seed 307 --max_epochs 10
```

**Expected:** All complete without errors, metrics logged correctly

### Phase 2: Full Experiment (10-14 days)

Run all 5 architectures with 5 seeds, 200 epochs:

```bash
# Create master script
cat > run_all_ablations.sh << 'EOF'
#!/bin/bash
set -e

SEEDS="307 308 309 310 311"
MAX_EPOCHS=200

for seed in $SEEDS; do
    echo "=== Running all experiments with seed $seed ==="
    
    echo "Exp0: E-D Baseline..."
    python scripts/train_exp0_encoder_decoder.py --seed $seed --max_epochs $MAX_EPOCHS
    
    echo "Exp1: E-D + Grid2D..."
    python scripts/train_exp1_grid2d_pe.py --seed $seed --max_epochs $MAX_EPOCHS
    
    echo "Exp2: E-D + PermInv..."
    python scripts/train_exp2_ed_perminv.py --seed $seed --max_epochs $MAX_EPOCHS
    
    echo "Exp3: E-D + Context..."
    python scripts/train_exp3_ed_context.py --seed $seed --max_epochs $MAX_EPOCHS
    
    echo "Exp4: Champion..."
    python scripts/train_exp3_champion.py --seed $seed --max_epochs $MAX_EPOCHS
done

echo "All experiments complete!"
EOF

chmod +x run_all_ablations.sh
```

---

## Step 7: Monitoring and Verification

### During Training:

```bash
# Monitor GPU
watch -n 1 nvidia-smi

# Monitor logs
tail -f logs/exp0_encoder_decoder_csv/version_0/metrics.csv
tail -f logs/exp1_ed_grid2d_csv/version_0/metrics.csv
# etc.

# TensorBoard
tensorboard --logdir logs
```

### After Training:

```bash
# Check all experiments completed
ls -la checkpoints/

# Expected directories:
# checkpoints/exp0_encoder_decoder/
# checkpoints/exp1_ed_grid2d/
# checkpoints/exp2_ed_perminv/
# checkpoints/exp3_ed_context/
# checkpoints/exp3_champion/

# Verify 5 seeds per experiment
ls -1 logs/per_task_metrics/exp0/ | wc -l  # Should be 5
ls -1 logs/per_task_metrics/exp1/ | wc -l  # Should be 5
# etc.
```

---

## Step 8: Analysis

### Create Analysis Script:

**File:** `data/ablation_redesign_analysis.py`

**Based on:** Current `ablation_analysis.py` but modified for new design

**Changes needed:**
- Load data from new experiment structure
- Compare each Exp1/2/3 vs Exp0 (not cumulative)
- Calculate synergy: Exp4 vs (Exp0 + sum of individual gains)
- Generate report: `ABLATION_REDESIGN_REPORT.txt`

```bash
# Run analysis
python data/ablation_redesign_analysis.py

# Expected output:
# - ABLATION_REDESIGN_REPORT.txt with component contributions
# - Statistical tests for each component
# - Synergy analysis
```

---

## Computational Cost

```
5 architectures × 5 seeds × 200 epochs = 5,000 training runs

Estimated time per epoch: 3-4 minutes
Total sequential time: 5000 × 3.5 min = 291 hours = 12.1 days

Parallelization:
- 5 GPUs (1 arch per GPU): ~2.4 days
- 1 GPU (sequential): ~12 days
```

---

## Next Immediate Actions

1. ✅ Verify existing scripts work (Exp0, Exp1, Exp4)
2. ⏳ Create `src/models/ed_with_perminv.py`
3. ⏳ Create `src/models/exp2_ed_perminv_lightning.py`
4. ⏳ Create `scripts/train_exp2_ed_perminv.py`
5. ⏳ Create `src/models/ed_with_context.py`
6. ⏳ Create `src/models/exp3_ed_context_lightning.py`
7. ⏳ Create `scripts/train_exp3_ed_context.py`
8. ⏳ Run Phase 1 validation (quick 10-epoch runs)
9. ⏳ Run Phase 2 full experiment (200 epochs, 5 seeds)
10. ⏳ Analyze results and generate report

---

## Critical Checklist Before Starting

- [ ] Verified Exp0 trains successfully
- [ ] Verified Exp1 trains successfully
- [ ] Verified Exp4 trains successfully and uses max_grid_size=30
- [ ] Created Exp2 architecture (E-D + PermInv only)
- [ ] Created Exp2 Lightning module
- [ ] Created Exp2 training script
- [ ] Created Exp3 architecture (E-D + Context only)
- [ ] Created Exp3 Lightning module
- [ ] Created Exp3 training script
- [ ] Verified all hyperparameters match
- [ ] Ran Phase 1 validation (10 epochs)
- [ ] Confirmed no errors in Phase 1
- [ ] Ready to launch Phase 2 (full experiment)
