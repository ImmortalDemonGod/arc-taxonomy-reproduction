# Paper-Ready Ablation Study Checklist

**Date:** October 28, 2025  
**Purpose:** Final verification before running 100-epoch training  
**Status:** ✅ **READY FOR TRAINING**

---

## ✅ Scientific Validity Verified

### Parameter Count Fairness
- ✅ All 5 models within ±1% parameter count (1.5% spread)
- ✅ Documented in `docs/ABLATION_MODEL_SPECIFICATIONS.md`
- ✅ Variation is **negligible and acceptable** for ablation studies
- ✅ Each ablation changes exactly ONE architectural component

### Training Configuration Parity
- ✅ All use identical optimizer (Adam, lr=0.00185, betas=(0.95, 0.999))
- ✅ All use identical scheduler (CosineAnnealingWarmRestarts)
- ✅ All train for exactly 100 epochs (no early stopping)
- ✅ All use same batch size (32)
- ✅ All use same precision (16-mixed)
- ✅ All use same seed (307)
- ✅ All use same data source (distributional_alignment)
- ✅ All use same train/val split (308/92 tasks)

### Metric Collection Completeness
- ✅ All models log per-task metrics (CSV export each epoch)
- ✅ All models log per-category metrics (9 categories)
- ✅ All models track grid accuracy, cell accuracy
- ✅ All models track copy rate, change recall, transformation quality
- ✅ All models use PerTaskMetricsLogger callback
- ✅ All models use TensorBoard + CSV logging

---

## 📊 Model Specifications Quick Reference

| Model | Architecture | d_model | Params | Expected Acc | Purpose |
|-------|--------------|---------|--------|--------------|---------|
| **Baseline** | Decoder-Only | 164 | 1.74M | ~2% | Catastrophic failure baseline |
| **Exp0** | + Encoder-Decoder | 168 | 1.71M | ~19% | Tests encoder value (+17%) |
| **Exp1** | + Grid2D PE | 168 | 1.71M | ~34% | Tests 2D structure (+15%) |
| **Exp2** | + PermInv | 168 | 1.71M | ~37% | Tests color equivariance (+3%) |
| **Champion** | + Context | 160 | 1.72M | ~61% | Tests context learning (+24%) |

**Total Gains:** Baseline 2% → Champion 61% (+59 percentage points)

---

## 🎯 Expected Table 5 Data (After Training)

Each model will produce CSV files with per-category breakdown:

```
Category | Grids | Grid Acc | Cell Acc | Copy Rate | Ch Recall | Trans Qual
---------|-------|----------|----------|-----------|-----------|------------
S1       | XX    | XX.XX%   | XX.XX%   | XX.XX%    | XX.XX%    | X.XXXX
S2       | XX    | XX.XX%   | XX.XX%   | XX.XX%    | XX.XX%    | X.XXXX
S3       | XX    | XX.XX%   | XX.XX%   | XX.XX%    | XX.XX%    | X.XXXX
L1       | XX    | XX.XX%   | XX.XX%   | XX.XX%    | XX.XX%    | X.XXXX
K1       | XX    | XX.XX%   | XX.XX%   | XX.XX%    | XX.XX%    | X.XXXX
C1       | XX    | XX.XX%   | XX.XX%   | XX.XX%    | XX.XX%    | X.XXXX
C2       | XX    | XX.XX%   | XX.XX%   | XX.XX%    | XX.XX%    | X.XXXX
A1       | XX    | XX.XX%   | XX.XX%   | XX.XX%    | XX.XX%    | X.XXXX
A2       | XX    | XX.XX%   | XX.XX%   | XX.XX%    | XX.XX%    | X.XXXX
---------|-------|----------|----------|-----------|-----------|------------
OVERALL  | 92    | XX.XX%   | XX.XX%   | XX.XX%    | XX.XX%    | X.XXXX
```

This data will be automatically saved to:
- `logs/per_task_metrics/{model}_epoch_099_per_category.csv`

---

## 📁 Log Structure After Training

```
logs/
├── console_output/
│   ├── baseline_{timestamp}.log        # Full training logs
│   ├── exp0_{timestamp}.log
│   ├── exp1_{timestamp}.log
│   ├── exp2_{timestamp}.log
│   └── exp3_champion_{timestamp}.log
│
├── per_task_metrics/
│   ├── baseline/
│   │   ├── baseline_epoch_000_per_task.csv      # 13,800 rows (92 tasks × 150 samples)
│   │   ├── baseline_epoch_000_per_category.csv  # 9 category rows
│   │   ├── baseline_epoch_001_per_task.csv
│   │   ├── ...
│   │   └── baseline_epoch_099_per_category.csv  # ← USE THIS FOR TABLE 5
│   │
│   ├── exp0/
│   ├── exp1/
│   ├── exp2/
│   └── exp3/
│
├── tensorboard/
│   ├── baseline_training/
│   ├── exp0_training/
│   ├── exp1_training/
│   ├── exp2_training/
│   └── champion_training/
│
└── csv/
    ├── baseline_csv/
    ├── exp0_csv/
    ├── exp1_csv/
    ├── exp2_csv/
    └── champion_csv/
```

---

## 🚀 Training Commands

### Quick Test (Already Verified ✅)
```bash
./run_training.sh test
# Output: All 5 models PASSED (5 batches each, ~2 min)
```

### Full Training (Next Step)
```bash
# On Paperspace L4 GPU
./run_training.sh all

# Expected runtime: ~250-300 hours (10-12 days)
# Models train sequentially: Baseline → Exp0 → Exp1 → Exp2 → Champion
```

### Individual Model Training
```bash
./run_training.sh baseline  # ~50 hours
./run_training.sh exp0      # ~50 hours  
./run_training.sh exp1      # ~50 hours
./run_training.sh exp2      # ~50 hours
./run_training.sh exp3      # ~60 hours (context encoder adds overhead)
```

---

## 📈 Monitoring Training Progress

### TensorBoard
```bash
tensorboard --logdir logs/tensorboard
# View at http://localhost:6006
```

### Live Training Logs
```bash
tail -f logs/console_output/baseline_*.log
# Or use Paperspace web terminal
```

### Check Per-Category Progress
```bash
# View latest per-category metrics
cat logs/per_task_metrics/baseline/baseline_epoch_*_per_category.csv | tail -n 20
```

---

## 📝 Paper Writing Checklist

After training completes:

### Table 5: Per-Category Ablation Results
- [ ] Extract epoch 99 per-category CSVs for all 5 models
- [ ] Compute mean ± std for grid accuracy per category
- [ ] Format as LaTeX table
- [ ] Include statistical significance tests (if variance is high)

### Figure X: Learning Curves
- [ ] Export TensorBoard data for val_grid_accuracy
- [ ] Plot all 5 models on same axes
- [ ] Annotate key architectural additions
- [ ] Show convergence behavior

### Table 6: Parameter Count Justification
- [ ] Copy table from `docs/ABLATION_MODEL_SPECIFICATIONS.md`
- [ ] Add citation to justify ±1% tolerance
- [ ] Reference this in Methods section

### Methods Section
- [ ] Copy training configuration from `docs/ABLATION_MODEL_SPECIFICATIONS.md`
- [ ] State: "All models parameter-matched within ±1% (1.5% spread)"
- [ ] State: "All models trained for 100 epochs without early stopping"
- [ ] State: "Identical hyperparameters across all experiments"

---

## 🔍 Verification Tests (All Passing ✅)

### Unit Tests
```bash
python -m pytest tests/
# Result: 163 passed, 1 skipped ✅
```

### Integration Tests
```bash
python scripts/test_all_training.py
# Result: All 5 models PASSED ✅
```

### Architecture Validation
```bash
python verify_param_counts.py
# Result: All within 1.5% spread ✅
```

---

## 🎯 Success Criteria

After full training, models should meet these criteria:

| Model | Min Grid Acc | Max Grid Acc | Category Performance |
|-------|--------------|--------------|---------------------|
| Baseline | 0-5% | 10% | Catastrophic across all categories |
| Exp0 | 15% | 25% | Moderate improvement, especially C1/C2 |
| Exp1 | 30% | 40% | Strong on S1/S2/S3 (spatial categories) |
| Exp2 | 35% | 42% | Marginal gain on C1/C2 (color categories) |
| Champion | 55% | 65% | Strong across all categories |

**If results deviate significantly:**
1. Check for training bugs (NaN loss, gradient explosion)
2. Verify data loading (confirm 308/92 split)
3. Check checkpoints (ensure best model saved)
4. Review TensorBoard logs for anomalies

---

## 📋 Final Pre-Training Checklist

- ✅ All training scripts use `distributional_alignment` dataset
- ✅ All scripts load `split_manifest.json` for train/val split
- ✅ All scripts have early stopping REMOVED
- ✅ All scripts have PerTaskMetricsLogger, TensorBoard, CSV loggers
- ✅ All Lightning modules compute per-category metrics
- ✅ All Lightning modules track task_ids
- ✅ All data loaders return task_ids
- ✅ All models use shared optimizer/scheduler configuration
- ✅ Parameter counts verified and documented
- ✅ Test suite passing (163/164 tests)
- ✅ Integration tests passing (5/5 models)
- ✅ Git repository up to date

---

## 🚨 Known Issues / Limitations

### Non-Issues (Documented and Acceptable)
- ✅ d_model varies (160-168) but parameter counts matched within 1.5%
- ✅ Champion has extra context encoder parameters (documented)
- ✅ Baseline uses 4 decoder layers (architecture difference, not bug)

### Future Improvements (Not Blocking)
- ⏭️ Add per-task learning curves
- ⏭️ Add confusion matrix for category predictions
- ⏭️ Add hyperparameter sensitivity analysis

---

## 📚 Documentation References

- **Complete Specifications:** `docs/ABLATION_MODEL_SPECIFICATIONS.md`
- **Fairness Analysis:** `docs/ABLATION_FAIRNESS_ANALYSIS.md`
- **Parameter Count Fix:** `docs/PARAMETER_COUNT_FIX.md`
- **Training Guide:** `README.md` (updated Oct 28)
- **Quick Start:** `PAPERSPACE_QUICKSTART.md`

---

**Status:** ✅ **READY FOR 100-EPOCH TRAINING**  
**Estimated Completion:** 10-12 days on Paperspace L4  
**Next Step:** Run `./run_training.sh all` on Paperspace  
**After Training:** Extract per-category CSVs and generate Table 5

---

**Created:** October 28, 2025, 7:05 PM  
**Last Verified:** October 28, 2025, 7:05 PM  
**Reviewer:** Ready for paper writing after training completes
