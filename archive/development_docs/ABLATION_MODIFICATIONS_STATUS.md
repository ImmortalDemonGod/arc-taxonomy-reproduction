# Ablation Redesign: Modification Status

## Completed: Exp2 (E-D + PermInv Only)

### Files Modified:

1. **`src/models/encoder_decoder_baseline.py`**
   - Added `use_perminv` parameter (default=False)
   - Added `pad_idx` parameter
   - Conditionally uses `PermInvariantEmbedding` when `use_perminv=True`
   - Updated factory function `create_encoder_decoder_model()`

2. **`src/models/exp0_encoder_decoder_lightning.py`**
   - Added `use_perminv` parameter (default=False)
   - Passes through to architecture

3. **`scripts/train_exp2_perminv.py`**
   - Changed from importing `Exp2PermInvLightningModule` (cumulative) 
   - Now imports `Exp0EncoderDecoderLightningModule`
   - Sets `use_perminv=True` to enable PermInv independently
   - Updated all print statements
   - Architecture: E-D + PermInv ONLY (NO Grid2D, NO Context)

### Test Command:
```bash
cd /Users/tomriddle1/Holistic-Performance-Enhancement/cultivation/systems/arc_reactor/publications/arc_taxonomy_2025/reproduction
python scripts/train_exp2_perminv.py --fast_dev_run 1
```

Expected output:
```
Exp 2: E-D + PermInvariant Embedding ONLY (Independent Test)
Components: E-D + PermInv ONLY (NO Grid2D, NO Context)
Architecture: E-D + PermInvariant Embedding ONLY
```

---

## Remaining: Exp3 (E-D + Context Only)

Need to create E-D + Context architecture WITHOUT Grid2D and PermInv.

### Option A: Modify Champion Architecture

Add flags to `champion_architecture.py`:
- `use_grid2d_pe` (default=True, set to False for Exp3)
- `use_perminv` (default=True, set to False for Exp3)
- Keep `use_context=True`

### Option B: Create Simplified Architecture File

Copy `champion_architecture.py` → `ed_with_context_only.py`
- Remove Grid2D PE logic
- Remove PermInv logic  
- Keep Context System (Encoder + Bridge)
- Use standard 1D PE and nn.Embedding

**Recommendation:** Option A (add flags to champion_architecture.py)
- Less duplication
- Single source of truth for Context System
- Easy to toggle components

---

## Current Ablation Lineup

After modifications:

| Exp | Name | Architecture | Status |
|-----|------|-------------|--------|
| Exp0 | E-D Baseline | E-D only | ✅ Ready |
| Exp1 | E-D + Grid2D | E-D + Grid2D PE | ✅ Ready |
| Exp2 | E-D + PermInv | E-D + PermInv ONLY | ✅ Modified |
| Exp3 | E-D + Context | E-D + Context ONLY | ⏳ Next |
| Exp4 | Champion | E-D + All | ✅ Ready (rename from exp3) |

---

## Next Steps

1. **Modify `champion_architecture.py`** to support flags
2. **Modify `exp3_champion_lightning.py`** to pass flags
3. **Create `scripts/train_exp3_ed_context_only.py`**
4. **Rename `train_exp3_champion.py`** → `train_exp4_champion.py`
5. **Test all scripts** with `--fast_dev_run 1`
6. **Run Phase 1 validation** (10 epochs, 1 seed)
7. **Launch Phase 2** (200 epochs, 5 seeds)

---

## Design Summary

**Independent Component Testing:**
```
E-D Baseline (Exp0)
  ├─ + Grid2D only    (Exp1) → Grid2D contribution
  ├─ + PermInv only   (Exp2) → PermInv contribution  ✅ DONE
  ├─ + Context only   (Exp3) → Context contribution  ⏳ NEXT
  └─ + All (Champion) (Exp4) → Synergy test
```

Each component tested independently to isolate its contribution!
