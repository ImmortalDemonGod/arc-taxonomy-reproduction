# Ablation Redesign: File Modification Plan

## Current State Analysis

### What EXISTS (checked 2025-11-06):

**Training Scripts:**
- ✅ `train_exp0_encoder_decoder.py` - E-D Baseline
- ✅ `train_exp1_grid2d_pe.py` - E-D + Grid2D  
- ⚠️ `train_exp2_perminv.py` - **E-D + Grid2D + PermInv** (CUMULATIVE, wrong for new design)
- ✅ `train_exp3_champion.py` - Champion (All components)

**Lightning Modules:**
- ✅ `exp0_encoder_decoder_lightning.py` - E-D Baseline
- ✅ `exp1_grid2d_pe_lightning.py` - E-D + Grid2D
- ⚠️ `exp2_perminv_lightning.py` - uses `EDWithGrid2DPEAndPermInv` (CUMULATIVE)
- ✅ `exp3_champion_lightning.py` - Champion

**Architecture Files:**
- ✅ `encoder_decoder_baseline.py` - E-D only
- ✅ `ed_with_grid2d_pe.py` - E-D + Grid2D
- ⚠️ `ed_with_grid2d_pe_and_perminv.py` - E-D + Grid2D + PermInv (CUMULATIVE)
- ✅ `champion_architecture.py` - Full Champion

---

## Required Modifications

### Option 1: Minimal Changes (RECOMMENDED)

**Keep existing cumulative design, just rename experiments for clarity:**

1. **Rename in analysis only**
   - Current Exp2 stays as "E-D + Grid2D + PermInv" (cumulative)
   - Analysis compares:
     - Exp0 vs Baseline → E-D contribution
     - Exp1 vs Exp0 → Grid2D contribution
     - Exp2 vs Exp1 → PermInv contribution (but when Grid2D present)
     - Exp3 vs Exp2 → Context contribution (but when Grid2D+PermInv present)

2. **Re-run Exp3 (Champion) only**
   - Fix: Ensure max_grid_size=30 (already done ✓)
   - Add: 5 seeds instead of 2
   - Time: ~3-5 days for just Champion

**Pros:** No code changes, use existing data
**Cons:** Components not tested independently (still order-dependent)

---

### Option 2: Create Independent Tests (What I was trying to do)

**Need 2 NEW architectures:**

1. **E-D + PermInv ONLY** (no Grid2D)
   - Copy `encoder_decoder_baseline.py`
   - Replace `nn.Embedding` with `PermInvariantEmbedding`
   - Keep standard 1D PE (not Grid2D)
   
2. **E-D + Context ONLY** (no Grid2D, no PermInv)
   - Copy from `champion_architecture.py`
   - Remove Grid2D PE logic
   - Remove PermInv logic
   - Keep Context System (Encoder + Bridge)

**Pros:** Clean independent component testing
**Cons:** Need to create new files, re-run everything (12 days)

---

## Decision

**Which option do you want?**

### A) Use existing cumulative data + theoretical justification
- **Time:** Ready now
- **Effort:** Just write the paper narrative
- **Quality:** Good enough for supporting evidence

### B) Create independent tests for clean claims
- **Time:** 2-3 weeks
- **Effort:** Create 2 architectures + 6 supporting files + re-run all
- **Quality:** Publication-grade ablation

---

## If Option B (Independent Tests):

### Files to CREATE:

1. `src/models/ed_with_perminv_only.py`
   - Based on: `encoder_decoder_baseline.py`
   - Add: PermInvariantEmbedding
   - Keep: Standard 1D PE

2. `src/models/exp2_ed_perminv_only_lightning.py`
   - Based on: `exp0_encoder_decoder_lightning.py`
   - Use: `ed_with_perminv_only` architecture

3. `scripts/train_exp2_ed_perminv_only.py`
   - Based on: `train_exp0_encoder_decoder.py`
   - Use: `Exp2EDPermInvOnlyLightningModule`

4. `src/models/ed_with_context_only.py`
   - Based on: `champion_architecture.py`
   - Remove: Grid2D PE, PermInv
   - Keep: Context System

5. `src/models/exp3_ed_context_only_lightning.py`
   - Based on: `exp3_champion_lightning.py`
   - Use: `ed_with_context_only` architecture

6. `scripts/train_exp3_ed_context_only.py`
   - Based on: `train_exp3_champion.py`
   - Use: `Exp3EDContextOnlyLightningModule`

### Files to RENAME (to avoid confusion):

- `train_exp2_perminv.py` → `train_exp2_cumulative_old.py` (preserve old)
- `train_exp3_champion.py` → `train_exp4_champion.py` (rename to Exp4)

---

## Next Action

**Tell me which option:** A or B?

- **Option A:** I'll update the analysis script to work with existing cumulative data
- **Option B:** I'll systematically create the 6 new files by copying and modifying existing patterns
