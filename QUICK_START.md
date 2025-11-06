# Ablation Redesign: Quick Start Commands

## Current Status (2025-11-05)

### What EXISTS and is READY:
- **Exp0:** E-D Baseline (`train_exp0_encoder_decoder.py`) 
- **Exp1:** E-D + Grid2D (`train_exp1_grid2d_pe.py`) 
- **Exp2:** E-D + PermInv ONLY (`train_exp2_perminv.py`) 
- **Exp3:** E-D + Context ONLY (`train_exp3_ed_context_only.py`) 
- **Exp4:** Champion (`train_exp3_champion.py`) uses max_grid_size=30

---

## Immediate Verification Commands

Run these from: `/Users/tomriddle1/Holistic-Performance-Enhancement/cultivation/systems/arc_reactor/publications/arc_taxonomy_2025/reproduction`

### 1. Test Existing Scripts (smoke tests)

```bash
# Test Exp0 (E-D Baseline)
python scripts/train_exp0_encoder_decoder.py --fast_dev_run 1

# Test Exp1 (E-D + Grid2D)  
python scripts/train_exp1_grid2d_pe.py --fast_dev_run 1

# Test Exp2 (E-D + PermInv ONLY)
python scripts/train_exp2_perminv.py --fast_dev_run 1

# Test Exp3 (E-D + Context ONLY)
python scripts/train_exp3_ed_context_only.py --fast_dev_run 1 --dataset rearc

# Test Exp4 (Champion, all components)
python scripts/train_exp3_champion.py --fast_dev_run 1
```

**Expected:** All should complete without errors

### 2. Verify max_grid_size=30 in Champion

```bash
grep "max_grid_size" scripts/train_exp3_champion.py
grep "max_grid_size" src/models/exp3_champion_lightning.py
```

**Expected:** All should show `max_grid_size=30` (not 35)

### 3. Check Hyperparameters Match

```bash
grep -A 5 "learning_rate=" scripts/train_exp0_encoder_decoder.py | grep learning
grep -A 5 "learning_rate=" scripts/train_exp1_grid2d_pe.py | grep learning
grep -A 5 "learning_rate=" scripts/train_exp2_perminv.py | grep learning
grep -A 5 "learning_rate=" scripts/train_exp3_ed_context_only.py | grep learning
grep -A 5 "learning_rate=" scripts/train_exp3_champion.py | grep learning
```

**Expected:** All should show: `learning_rate=0.0018498849832733245`

---

## Once All Files Created: Phase 1 Validation

Run quick 10-epoch test with 1 seed to verify no errors:

```bash
cd /Users/tomriddle1/Holistic-Performance-Enhancement/cultivation/systems/arc_reactor/publications/arc_taxonomy_2025/reproduction

# Exp0: E-D Baseline
python scripts/train_exp0_encoder_decoder.py --seed 307 --max_epochs 10

# Exp1: E-D + Grid2D
python scripts/train_exp1_grid2d_pe.py --seed 307 --max_epochs 10

# Exp2: E-D + PermInv ONLY
python scripts/train_exp2_perminv.py --seed 307 --max_epochs 10

# Exp3: E-D + Context ONLY
python scripts/train_exp3_ed_context_only.py --seed 307 --max_epochs 10 --dataset rearc

# Exp4: Champion
python scripts/train_exp3_champion.py --seed 307 --max_epochs 10
```

**Time:** ~30-60 minutes total  
**Purpose:** Verify all architectures train without errors

---

## Phase 2: Full Experiment (After Phase 1 Passes)

### Master Execution Script

```bash
#!/bin/bash
# Full ablation redesign experiment
# 5 architectures × 5 seeds × 200 epochs = 5,000 training runs
# Estimated time: 12 days on 1 GPU

set -e

SEEDS="307 308 309 310 311"
MAX_EPOCHS=200

cd /Users/tomriddle1/Holistic-Performance-Enhancement/cultivation/systems/arc_reactor/publications/arc_taxonomy_2025/reproduction

for seed in $SEEDS; do
    echo "========================================"
    echo "Running all experiments with seed $seed"
    echo "========================================"
    
    echo "[1/5] Exp0: E-D Baseline..."
    python scripts/train_exp0_encoder_decoder.py --seed $seed --max_epochs $MAX_EPOCHS
    
    echo "[2/5] Exp1: E-D + Grid2D..."
    python scripts/train_exp1_grid2d_pe.py --seed $seed --max_epochs $MAX_EPOCHS
    
    echo "[3/5] Exp2: E-D + PermInv ONLY..."
    python scripts/train_exp2_perminv.py --seed $seed --max_epochs $MAX_EPOCHS
    
    echo "[4/5] Exp3: E-D + Context ONLY..."
    python scripts/train_exp3_ed_context_only.py --seed $seed --max_epochs $MAX_EPOCHS --dataset rearc
    
    echo "[5/5] Exp4: Champion..."
    python scripts/train_exp3_champion.py --seed $seed --max_epochs $MAX_EPOCHS
    
    echo "Completed all 5 experiments for seed $seed"
    echo ""
done

echo "ALL EXPERIMENTS COMPLETE!"
echo "Total runs: 25 (5 architectures × 5 seeds)"
```

Save as: `run_all_ablations.sh`

```bash
chmod +x run_all_ablations.sh
./run_all_ablations.sh
```

---

## Monitoring During Execution

### GPU Usage
```bash
watch -n 1 nvidia-smi
```

### Log Monitoring
```bash
# Monitor metrics for each experiment
tail -f logs/exp0_encoder_decoder_csv/version_0/metrics.csv
tail -f logs/exp1_ed_grid2d_csv/version_0/metrics.csv
# etc.
```

### TensorBoard
```bash
tensorboard --logdir logs
# Open browser to http://localhost:6006
```

---

## After Completion: Analysis

```bash
# Run analysis script (to be created)
python data/ablation_redesign_analysis.py

# Expected output:
# - ABLATION_REDESIGN_REPORT.txt
# - Component contribution statistics
# - Synergy analysis (Champion vs sum of parts)
```

---

## Critical Checklist

Before starting Phase 2:

- [ ] All 5 scripts tested (Exp0, Exp1, Exp2, Exp3, Exp4)
- [ ] Champion uses max_grid_size=30 (verified)
- [ ] Phase 1 validation passed (10 epochs, no errors)
- [ ] Sufficient disk space (~50GB for logs/checkpoints)
- [ ] GPU available for ~12 days (or 5 GPUs for ~2.4 days)

---

## Expected Timeline

| Phase | Duration | Description |
|-------|----------|-------------|
| Setup | 2-4 hours | Create missing files, verify existing |
| Phase 1 | 1 hour | Quick validation (10 epochs, 1 seed) |
| Phase 2 | 12 days | Full experiment (200 epochs, 5 seeds, 1 GPU) |
| Analysis | 2 hours | Generate report and statistics |
| **Total** | **~12.5 days** | **From start to final report** |

With 5 GPUs parallelized: **~3 days total**

---

## What You'll Be Able to Claim After

With this redesign and 5 seeds:

✅ "Grid2D PE adds +X% to E-D baseline (95% CI: [a,b], p<0.05)"  
✅ "PermInv adds +Y% to E-D baseline (95% CI: [c,d], p<0.05)"  
✅ "Context System adds +Z% to E-D baseline (95% CI: [e,f], p<0.001)"  
✅ "Champion combines all components with W% synergy"  
✅ "Each component contributes independently to Champion performance"

**This directly supports your "minimal necessary additions" narrative.**
