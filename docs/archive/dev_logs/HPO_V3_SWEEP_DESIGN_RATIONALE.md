# HPO v3 Sweep Design Rationale

**Config:** `configs/hpo/visual_classifier_sweep_v3_intelligent.yaml`  
**Created:** October 31, 2025  
**Based on:** Systematic analysis of v2 sweep (20 trials)  
**Strategy:** Conservative refinement with intelligent expansion

---

## 🎯 Design Philosophy

### **Core Principle: Evidence-Based Refinement**

1. **Fix only what's certain** (100% elite convergence + strong effect)
2. **Expand where bounded** (elites hitting limits)
3. **Keep exploration where uncertain** (weak signals, small samples)
4. **Give fair trials** (context encoder with better hyperparams)

### **Anti-Patterns Avoided**

❌ **Premature elimination** - Context had only 5 trials, insufficient  
❌ **Over-narrowing** - Don't restrict based on weak signals (p>0.10)  
❌ **Under-exploration** - Must expand where elites hit boundaries  
❌ **Blind copying** - Auto-generated configs may be too aggressive

---

## 📊 Parameter-by-Parameter Decisions

### **FIXED PARAMETERS (2 total)**

#### **1. `use_scheduler: true`**
**Evidence:**
- ✅ 100% of elite trials used scheduler
- ✅ +5.0% performance boost (True: 0.275, False: 0.225)
- ✅ 15 trials with True vs 5 with False (clear preference)

**Decision:** FIX to `true`  
**Confidence:** ⭐⭐⭐ HIGH

#### **2. `label_smoothing: 0.0`**
**Evidence:**
- ✅ Negative correlation: ρ=-0.41, p=0.075
- ✅ Elite trials clustered at 0.0-0.05 range
- ✅ "Increase hurts" directionality

**Decision:** FIX to `0.0`  
**Confidence:** ⭐⭐ MODERATE (p=0.075, just outside α=0.05)

**Why not fix more?** Other parameters lack statistical power (n=20) or show 100% convergence that may be sampling artifacts.

---

### **BIASED SAMPLING (1 parameter)**

#### **3. `encoder_type: ["cnn", "cnn", "cnn", "context"]`**
**Evidence:**
- CNN: 15 trials (75%), mean = 0.272
- Context: 5 trials (25%), mean = 0.234
- Elite set: 100% CNN (5/5)
- Difference: +3.8% absolute, +16% relative

**Decision:** BIAS to 75% CNN, 25% context (3:1 ratio)  
**Confidence:** ⭐⭐⭐ HIGH for CNN superiority

**Why not eliminate context?**
1. **Small sample:** Only 5 context trials (underpowered)
2. **Suboptimal hyperparams:** Context trials may not have found good configs
3. **Scientific rigor:** Need fair comparison before elimination
4. **Theoretical value:** If context is important for paper, must validate properly

**Alternative considered:** Separate ablation study (rejected - adds complexity)

---

### **NARROWED RANGES (1 parameter)**

#### **4. `lr: [5e-5, 3e-4]`**
**Evidence:**
- Elite cluster: [6.7e-5, 2.5e-4]
- Original range: [5e-5, 0.01] (200× span)
- Effect: ρ=-0.15, p=0.53 (no clear trend)

**Decision:** Narrow to [5e-5, 3e-4] (6× span)  
**Confidence:** ⭐⭐ MODERATE

**Rationale:**
- Elite trials found narrow band
- No evidence for extreme values (0.001-0.01 unused)
- Still allow 20% headroom outside elite range
- Reduces wasted trials on known-bad regions

---

### **EXPANDED RANGES (6 parameters)**

#### **5. `depth: [3, 10]` (was [2, 6])**
**Evidence:**
- Elite trials: depth 5-6 (hitting upper bound)
- Correlation: ρ=+0.48, p=0.071 (moderate positive)
- Trend: "increase helps"

**Decision:** Expand upper bound 6→10  
**Confidence:** ⭐⭐⭐ HIGH (clear boundary hit)

**Rationale:**
- Elite trials maxed out at boundary
- Positive correlation suggests deeper helps
- Allow exploration of 7, 8, 9, 10
- Lower bound 3 (conservative, elites used 4+)

#### **6. `width_mult: [0.75, 5.0]` (was [0.5, 3.0])**
**Evidence:**
- Elite cluster: 2.3-3.0 (near upper bound)
- Correlation: ρ=+0.29 (weak positive)
- Elites never used <1.1

**Decision:** Expand 3.0→5.0, raise floor 0.5→0.75  
**Confidence:** ⭐⭐ MODERATE

**Rationale:**
- Elites crowded at high end
- Weak positive trend
- May continue improving beyond 3.0
- Raise floor to avoid wasted trials

#### **7. `temperature: [1.0, 30.0]` (was [5.0, 20.0])**
**Evidence:**
- Only 5 trials with use_cosine=True (conditional param)
- Weak effect: ρ=+0.21, p=0.74
- Insufficient data

**Decision:** Expand both bounds  
**Confidence:** ⚠️ LOW (very few samples)

**Rationale:**
- Original range may be too restrictive
- Literature suggests wider range may help
- Low cost (conditional parameter)

#### **8-10. Context Encoder Parameters**

**`ctx_d_model: [128, 256, 384, 512, 768]`** (added 768)  
**`ctx_pixel_layers: [1, 6]`** (was [1, 5])  
**`ctx_grid_layers: [1, 4]`** (was [1, 3])  
**`ctx_dropout: [0.0, 0.5]`** (was [0.0, 0.3])

**Evidence:**
- Context failed (0% in elite set)
- BUT only 5 trials total (underpowered)
- May have been capacity-limited

**Decision:** Expand all context parameters  
**Confidence:** ⭐ LOW (exploratory)

**Rationale:**
- Give context fair chance with more capacity
- If still fails with better hyperparams → justified elimination
- Cost: 25% of trials (25 trials out of 100)

---

### **KEPT BROAD (5 parameters)**

#### **11. `batch_size: [8, 16, 32]`**
**Evidence:**
- ⭐⭐⭐ STRONGEST SIGNAL: ρ=-0.67, p=0.001
- Elite convergence at 16
- Trend: smaller is better

**Decision:** Keep 16, add 8, keep 32 for validation  
**Confidence:** ⭐⭐⭐ HIGH

**Rationale:**
- Elite choice was 16 (100% convergence)
- But strong negative correlation suggests 8 may be better
- Keep 32 to validate trend continues
- Don't add 4 yet (too aggressive, memory constraints)

#### **12. `weight_decay: [1e-7, 0.1]`**
**Evidence:**
- Weak positive: ρ=+0.37, p=0.11
- No clear elite cluster

**Decision:** Keep original broad range  
**Confidence:** ⭐ LOW

**Rationale:**
- Weak signal (p>0.10)
- No evidence for restriction
- Regularization is task-dependent

#### **13. `embed_dim: [128, 256, 384, 512, 768]`**
**Evidence:**
- Negligible effect: ρ=+0.05, p=0.83
- Elites used 128 and 512

**Decision:** Keep all 5 choices  
**Confidence:** ⭐ LOW

**Rationale:**
- No evidence to restrict
- Pruned choices only used 2/5, but sample size too small
- Embedding dimension interacts with other params

#### **14-16. Binary Choices: `demo_agg`, `use_coords`, `use_cosine`**

**`demo_agg: ["flatten", "mean"]`**
- Evidence: flatten +7.1% better
- Decision: Keep both (validate with larger sample)
- Confidence: ⭐⭐ MODERATE

**`use_coords: [false, true]`**
- Evidence: False +3.5% better
- Decision: Keep both (moderate effect, small sample)
- Confidence: ⭐ LOW

**`use_cosine: [false, true]`**
- Evidence: False -1.6% penalty
- Decision: Keep both (small effect, validate trend)
- Confidence: ⭐ LOW

**Rationale for all:** Effects are moderate, sample sizes small, keep for validation

#### **17. `mlp_hidden: [256, 512, 1024, 2048]`**
**Evidence:**
- Weak effect: ρ=+0.23, p=0.41
- Elites used 256 and 2048 (extremes)

**Decision:** Keep all 4 choices  
**Confidence:** ⚠️ LOW

---

## 📈 Expected Outcomes

### **Primary Goals**
1. ✅ **Validate CNN dominance** with expanded capacity
2. ✅ **Fair context comparison** with better hyperparameters
3. ✅ **Explore batch_size=8** (strongest signal from v2)
4. ✅ **Test depth/width expansion** (boundary hits in v2)

### **Performance Targets**
- **Best trial:** 31-33% validation accuracy (10-15% improvement over v2's 28.6%)
- **Mean performance:** 28-29% (consistent improvement)
- **Elite set:** Should maintain or improve convergence patterns

### **Decision Criteria**

**After v3 completes:**

1. **If CNN still dominates (>80% of elites):**
   - ✅ Justified to fix encoder_type="cnn" in v4
   - ✅ Focus v4 on CNN-only refinement

2. **If context improves (>20% of elites):**
   - ⚠️ Need separate CNN vs Context ablation
   - ⚠️ May require architecture-specific sweeps

3. **If batch_size=8 dominates:**
   - ✅ Explore even smaller (batch_size=4)
   - ✅ May indicate need for gradient accumulation

4. **If depth/width still bounded:**
   - ⚠️ Further expansion needed
   - ⚠️ May indicate capacity-limited regime

---

## 🔍 Comparison: v2 vs v3

| Aspect | v2 (Original) | v3 (Intelligent) | Change |
|--------|---------------|------------------|--------|
| **Study size** | 150 trials planned, 20 completed | 100 trials | -33% (more focused) |
| **Fixed params** | 9 (data/training) | 11 (+2: scheduler, smoothing) | +2 |
| **Searchable params** | 17 | 13 | -4 (smarter) |
| **Search space size** | ~10^8 combinations | ~10^6 combinations | 100× smaller |
| **Encoder bias** | 50/50 CNN/context | 75/25 CNN/context | Evidence-based |
| **Depth range** | [2, 6] | [3, 10] | Expanded ✅ |
| **Width range** | [0.5, 3.0] | [0.75, 5.0] | Expanded ✅ |
| **LR range** | [5e-5, 0.01] | [5e-5, 3e-4] | Narrowed ✅ |
| **Batch sizes** | [8, 16, 32, 64] | [8, 16, 32] | Focused ✅ |
| **Context capacity** | Baseline | Expanded (+1 d_model, +1 layer) | Fair trial ✅ |

---

## ⚠️ Risks & Mitigations

### **Risk 1: Context Still Fails**
**Probability:** HIGH (70%)  
**Impact:** LOW (validates elimination)  
**Mitigation:** Document thoroughly for paper, shows due diligence

### **Risk 2: Deeper/Wider Models Hit Memory Limits**
**Probability:** MODERATE (30%)  
**Impact:** MODERATE (pruning will handle)  
**Mitigation:** Hyperband pruning will kill bad configs early

### **Risk 3: 100 Trials Insufficient**
**Probability:** LOW (20%)  
**Impact:** MODERATE (extend to 150)  
**Mitigation:** 13 parameters with biased sampling is tractable

### **Risk 4: No Improvement Over v2**
**Probability:** LOW (15%)  
**Impact:** HIGH (indicates task is hard)  
**Mitigation:** Validate task learnability separately

---

## 🎯 Success Criteria

### **Minimum Success:**
- ✅ Best trial improves by >5% relative (>30.0% accuracy)
- ✅ Clear winner emerges (CNN or context with >80% elite share)
- ✅ Boundary hits resolved (elites not clustered at limits)

### **Full Success:**
- ✅ Best trial >32% accuracy (12% relative improvement)
- ✅ CNN dominance confirmed with expanded capacity
- ✅ Clear parameter trends (depth, width, batch_size)
- ✅ Reproducible elite configurations

### **Exceptional Success:**
- ✅ Best trial >35% accuracy (22% relative improvement)
- ✅ Context finds competitive niche
- ✅ Clear path to v4 optimization

---

## 📚 References

**Analysis Documents:**
- `docs/HPO_SWEEP_ANALYSIS_FINDINGS.md` - Full v2 analysis
- `docs/HPO_QUICK_REFERENCE.md` - Quick reference
- `outputs/visual_classifier/hpo/analysis_20251031_135012/` - Raw data

**Key Findings:**
- Batch size: ρ=-0.67, p=0.001 (strongest signal)
- Depth: ρ=+0.48, p=0.071 (boundary hit at 6)
- Label smoothing: ρ=-0.41, p=0.075 (hurts)
- CNN vs context: 27.2% vs 23.4% (but n=5 for context)

---

## ✅ Pre-Launch Checklist

- [x] Config file created and documented
- [x] Design rationale documented
- [x] Comparison with v2 documented
- [ ] Config validated (syntax check)
- [ ] Database connection verified
- [ ] Compute resources allocated
- [ ] Monitoring dashboard set up
- [ ] Launch script prepared
- [ ] Estimated runtime calculated
- [ ] Checkpoint/resume mechanism tested

---

**Created:** October 31, 2025  
**Author:** Cascade AI (based on systematic v2 analysis)  
**Status:** Ready for validation and launch  
**Confidence:** HIGH - Evidence-based, conservative approach
