# HPO v3 Sweep Launch Checklist

**Study:** visual_classifier_v3_intelligent_refinement  
**Config:** `configs/hpo/visual_classifier_sweep_v3_intelligent.yaml`  
**Created:** October 31, 2025  
**Target Trials:** 100  
**Estimated Duration:** ~8-12 hours (with 4 workers)

---

## ✅ Pre-Launch Checklist

### **1. Configuration Validation**
- [x] Config file created
- [x] Syntax validated (`python scripts/validate_sweep_config.py ...`)
- [x] Design rationale documented
- [x] v2 vs v3 comparison documented
- [ ] Config reviewed by human
- [ ] Approved for launch

### **2. Infrastructure**
- [ ] Database connection verified
  ```bash
  psql "postgresql://doadmin:...@db-postgresql-nyc3-34697-do-user-15485406-0.l.db.ondigitalocean.com:25060/defaultdb?sslmode=require" -c "\dt"
  ```
- [ ] Compute resources allocated (GPU nodes)
- [ ] Disk space sufficient (>100GB recommended)
- [ ] Network bandwidth verified
- [ ] Dependencies installed (`pip list | grep optuna`)

### **3. Data Preparation**
- [ ] Training data exists: `data/distributional_alignment/`
- [ ] Labels file exists: `data/taxonomy_classification/all_tasks_classified.json`
- [ ] Centroids file exists: `outputs/visual_classifier/category_centroids_v3.npy`
- [ ] Data integrity verified (checksums, sample loading)

### **4. Code Readiness**
- [ ] Latest code pulled from repo
- [ ] Tests passing (`python -m pytest tests/`)
- [ ] No uncommitted changes in critical files
- [ ] Objective function verified (`scripts/objective.py`)

### **5. Monitoring Setup**
- [ ] Optuna dashboard accessible
- [ ] Log aggregation configured
- [ ] Alert system configured (optional)
- [ ] Progress tracking script ready

### **6. Baseline Validation**
- [ ] Run 1 trial manually to verify end-to-end
- [ ] Check trial completes successfully
- [ ] Verify metrics logged correctly
- [ ] Confirm no memory issues

---

## 🚀 Launch Commands

### **Option A: Single Worker (Serial)**
```bash
cd /path/to/reproduction

# Launch sweep
python scripts/run_hpo_sweep.py \
  --config configs/hpo/visual_classifier_sweep_v3_intelligent.yaml \
  --workers 1 \
  --timeout 3600

# Monitor progress
watch -n 60 'python ../../../jarc_reactor/optimization/inspect_optuna_db.py \
  --storage-url "postgresql://..." \
  --study "visual_classifier_v3_intelligent_refinement" \
  --analyze-ranges'
```

### **Option B: Multiple Workers (Parallel)**
```bash
# Terminal 1-4: Launch workers
for i in {1..4}; do
  python scripts/run_hpo_sweep.py \
    --config configs/hpo/visual_classifier_sweep_v3_intelligent.yaml \
    --workers 1 \
    --timeout 3600 \
    &> logs/hpo_v3_worker_${i}.log &
done

# Monitor all workers
tail -f logs/hpo_v3_worker_*.log
```

### **Option C: Distributed (Paperspace/Cloud)**
```bash
# On each node
export OPTUNA_STORAGE_URL="postgresql://..."
python scripts/run_hpo_sweep.py \
  --config configs/hpo/visual_classifier_sweep_v3_intelligent.yaml \
  --workers 1
```

---

## 📊 During-Sweep Monitoring

### **Every Hour: Quick Check**
```bash
python ../../../jarc_reactor/optimization/inspect_optuna_db.py \
  --storage-url "postgresql://..." \
  --list-studies
```

**Look for:**
- ✅ Trial count increasing
- ✅ Best value improving
- ⚠️ No failed trials (or <5%)
- ⚠️ Reasonable trial duration (<30 min/trial)

### **Every 4 Hours: Detailed Check**
```bash
# Check parameter exploration
python ../../../jarc_reactor/optimization/inspect_optuna_db.py \
  --storage-url "postgresql://..." \
  --study "visual_classifier_v3_intelligent_refinement" \
  --analyze-ranges
```

**Look for:**
- ✅ All parameters being explored
- ✅ No concentration at single value
- ⚠️ Encoder_type bias (75% CNN expected)
- ⚠️ Depth/width expanding beyond v2

### **Mid-Sweep Analysis (50 trials)**
```bash
bash scripts/analyze_current_hpo_sweep.sh

# Compare to v2
echo "v2 best: 0.286"
cat outputs/visual_classifier/hpo/analysis_*/full_analysis.log | grep "Best Trial"
```

**Decision Points:**
- **If best < 0.26:** Consider stopping (no improvement)
- **If best > 0.30:** On track (10% improvement)
- **If best > 0.32:** Excellent progress (12% improvement)

---

## 🛑 Emergency Stop Procedures

### **Graceful Stop (Finish Current Trials)**
```bash
# Create stop file
touch stop_sweep.flag

# Workers will check for this file and stop after current trial
```

### **Immediate Stop (Kill All Workers)**
```bash
pkill -f "run_hpo_sweep.py"

# Verify all processes stopped
ps aux | grep run_hpo_sweep
```

### **Database Cleanup (If Needed)**
```bash
# Mark incomplete trials as FAIL
python ../../../jarc_reactor/optimization/inspect_optuna_db.py \
  --storage-url "postgresql://..." \
  --study "visual_classifier_v3_intelligent_refinement" \
  --cleanup-incomplete
```

---

## 📈 Post-Sweep Analysis

### **Step 1: Immediate Summary (5 min)**
```bash
bash scripts/analyze_current_hpo_sweep.sh

# Key files:
# - outputs/visual_classifier/hpo/analysis_TIMESTAMP/refined_search_space_v4.yaml
# - outputs/visual_classifier/hpo/analysis_TIMESTAMP/directionality_*.csv
# - docs/HPO_SWEEP_ANALYSIS_FINDINGS_V3.md (to be created)
```

### **Step 2: Compare v2 vs v3 (15 min)**
Create comparison table:
```markdown
| Metric | v2 | v3 | Change |
|--------|----|----|--------|
| Best val_acc | 0.286 | X.XXX | +X% |
| Mean val_acc | 0.259 | X.XXX | +X% |
| Trials | 20 | 100 | +400% |
| CNN % in elite | 100% | XX% | ... |
| Context % in elite | 0% | XX% | ... |
```

### **Step 3: Extract Top 5 Configs (30 min)**
```bash
python scripts/extract_top_configs.py \
  --storage-url "postgresql://..." \
  --study "visual_classifier_v3_intelligent_refinement" \
  --top-k 5 \
  --output configs/champions/v3_top5.yaml
```

### **Step 4: Validate Champions (2 hours)**
```bash
# Train top 3 configs to convergence (100 epochs each)
for i in {1..3}; do
  python scripts/3_train_task_encoder.py \
    configs/champions/v3_config_${i}.yaml \
    --epochs 100 \
    --output outputs/visual_classifier/champion_v3_${i}/
done
```

---

## 🎯 Success Criteria

### **Minimum Success** ✅
- [ ] 100 trials completed
- [ ] Best val_acc > 0.30 (>5% improvement over v2)
- [ ] No catastrophic failures (>90% trials complete)
- [ ] Clear architecture winner (>70% elite share)

### **Full Success** ⭐
- [ ] Best val_acc > 0.32 (>12% improvement)
- [ ] CNN dominance validated OR context finds niche
- [ ] Depth/width boundaries resolved
- [ ] Clear parameter trends

### **Exceptional Success** 🏆
- [ ] Best val_acc > 0.35 (>22% improvement)
- [ ] Both architectures find optimal regions
- [ ] Ready for v4 production config
- [ ] Publishable results

---

## 📝 Documentation Requirements

### **During Sweep**
- [ ] Log all commands run
- [ ] Record any errors/warnings
- [ ] Note any manual interventions
- [ ] Save monitoring screenshots

### **After Sweep**
- [ ] Create `HPO_SWEEP_V3_RESULTS.md`
- [ ] Update `HPO_QUICK_REFERENCE.md` with v3 findings
- [ ] Save all analysis outputs
- [ ] Archive logs and configs
- [ ] Update main README with v3 summary

---

## 🚨 Common Issues & Solutions

### **Issue 1: Trials Failing Immediately**
**Symptoms:** Many trials with state=FAIL, duration <1 min  
**Diagnosis:**
```bash
tail -100 logs/hpo_v3_worker_1.log
```
**Common Causes:**
- Data path incorrect
- Missing dependencies
- GPU out of memory
- Database connection timeout

### **Issue 2: No Improvement Over Baseline**
**Symptoms:** Best val_acc stuck at ~23-25%  
**Diagnosis:**
```bash
python ../../../jarc_reactor/optimization/inspect_optuna_db.py \
  --storage-url "postgresql://..." \
  --study "visual_classifier_v3_intelligent_refinement" \
  --param-importances
```
**Possible Actions:**
- Check if parameters are actually varying
- Verify objective function is correct
- Inspect individual trial logs for issues

### **Issue 3: One Parameter Dominating**
**Symptoms:** 95%+ of trials using same value  
**Diagnosis:** Check parameter importance and ranges  
**Action:** May indicate problem with sampling or range definition

---

## 📞 Emergency Contacts

**Primary:** User (you)  
**Backup:** Check repo issues or restart from checkpoint  
**Database Admin:** DigitalOcean support (if DB issues)

---

## ✅ Final Pre-Launch Verification

**Before clicking "launch", verify:**

1. ✅ Config validated: `configs/hpo/visual_classifier_sweep_v3_intelligent.yaml`
2. ✅ Data accessible: `ls data/distributional_alignment/ | wc -l`
3. ✅ Database reachable: `psql ... -c "SELECT 1"`
4. ✅ GPU available: `nvidia-smi`
5. ✅ Disk space: `df -h .`
6. ✅ 1 test trial passes: `python scripts/run_single_trial.py`
7. ✅ Logs directory exists: `mkdir -p logs/`
8. ✅ This checklist reviewed: **YOU ARE HERE**

---

**Launch Authorization:**
- [ ] I have reviewed this checklist
- [ ] All critical items verified
- [ ] Ready to launch v3 sweep
- [ ] Estimated completion: ___ hours

**Launched by:** ___________  
**Date/Time:** ___________  
**Expected completion:** ___________

---

**Status:** 🟡 READY FOR LAUNCH (pending final approval)  
**Next Step:** Review checklist, verify items, launch sweep  
**Documentation:** Complete ✅
