# ARC Taxonomy Reproduction - Metrics Requirements

**Date:** October 27, 2025  
**Status:** CRITICAL - Current metrics are completely wrong

---

## Problem Statement

The current reproduction package uses **simple token-level accuracy**, which is incorrect. The ARC Taxonomy paper uses **grid-level and transformation metrics** that are fundamentally different.

**Current (WRONG):**
- `train_loss`: CrossEntropyLoss
- `val_loss`: CrossEntropyLoss  
- `val_accuracy`: Simple token accuracy (% of correct tokens)

**Required (CORRECT):**
- `val_grid_accuracy`: 2.34% (% of completely correct grids)
- `val_cell_accuracy`: 90.39% (% of correct cells)
- `val_change_recall`: 70.32% (transformation detection)
- `val_dense_grid_objective`: 68.76% (composite primary metric)

---

## Required Metrics (from jarc_reactor/utils/metrics.py and train.py)

### 1. Grid-Level Metrics

#### `grid_accuracy` (PRIMARY - THIS IS THE HEADLINE METRIC)
**Definition:** Fraction of grids where ALL non-padding cells are predicted correctly

```python
# Grid is only counted as correct if 100% of non-padding cells match
grid_matches = (predictions == targets) | (targets == pad_token)
grid_accuracy = grid_matches.view(B, -1).all(dim=1).float().mean()
```

**Champion baseline:** 2.34% on 400 tasks  
**Interpretation:** Solving 1 complete grid out of ~43 validation tasks

#### `cell_accuracy` 
**Definition:** Fraction of non-padding cells predicted correctly (pixel-level)

```python
valid_mask = (targets != pad_token)
correct_cells = (predictions == targets) & valid_mask
cell_accuracy = correct_cells.sum() / valid_mask.sum()
```

**Champion baseline:** 90.39%  
**Interpretation:** Strong pixel-level predictions, but not perfect grids

#### `grid_tol_0p95`
**Definition:** Fraction of grids where per-grid cell accuracy >= 95%

```python
per_sample_correct = correct_cells.reshape(B, -1).sum(dim=1)
per_sample_valid = valid_mask.reshape(B, -1).sum(dim=1)
per_sample_acc = per_sample_correct / per_sample_valid
grid_tol_0p95 = (per_sample_acc >= 0.95).float().mean()
```

**Use:** Component of dense_grid_objective

---

### 2. Transformation Metrics (from compute_copy_metrics_on_batch)

#### `change_recall`
**Definition:** Of cells where target != source, fraction where prediction != source

```python
target_changes = (target != source)
pred_changes = (prediction != source)
change_recall = (pred_changes & target_changes).sum() / target_changes.sum()
```

**Champion baseline:** 70.32%  
**Interpretation:** Model detects 70% of transformations

#### `change_precision`
**Definition:** Of cells where prediction != source, fraction where target != source

```python
change_precision = (pred_changes & target_changes).sum() / pred_changes.sum()
```

#### `transformation_f1`
**Definition:** Harmonic mean of change_precision and change_recall

```python
transformation_f1 = 2 * (precision * recall) / (precision + recall)
```

#### `copy_rate`
**Definition:** Fraction of cells where prediction == source

---

### 3. Composite Objective (PRIMARY MONITORING METRIC)

#### `dense_grid_objective`
**Definition:** Weighted combination optimized for grid-solving correlation

```python
# Weights from CV optimization (Sept 21-22, 2025)
w_cell = 0.0   # Cell accuracy weight
w_tol  = 0.8   # Tolerance (95%) weight  
w_ex   = 0.2   # Exact grid accuracy weight

dense_grid_objective = (
    w_cell * cell_accuracy +
    w_tol * grid_tol_0p95 +
    w_ex * grid_accuracy
)
```

**Champion baseline:** 68.76%  
**Purpose:** Primary metric for early stopping and checkpoint selection  
**Why:** Best Spearman correlation with final grid accuracy

---

## Implementation Priority

### Phase 1: Essential Grid Metrics (BLOCKING)
**Must have before ANY training:**
1. ✅ `grid_accuracy` - The headline result (2.34%)
2. ✅ `cell_accuracy` - Pixel-level performance (90.39%)
3. ✅ `dense_grid_objective` - Primary monitoring metric (68.76%)

**Implementation:** Create `src/evaluation/metrics.py` with:
- `compute_grid_metrics(predictions, targets, pad_token)` → dict
- Returns: grid_accuracy, cell_accuracy, grid_tol_0p95

### Phase 2: Transformation Metrics (IMPORTANT)
**Needed for complete reproduction:**
4. ✅ `change_recall` - Transformation detection (70.32%)
5. ✅ `change_precision` - Change accuracy
6. ✅ `transformation_f1` - Combined transformation score
7. ✅ `copy_rate` - Bias detection

**Implementation:** Port `compute_copy_metrics_on_batch()` from jarc_reactor/utils/metrics.py

### Phase 3: Additional Diagnostics (NICE-TO-HAVE)
- `row_all_correct_rate` - Row-level accuracy
- `col_all_correct_rate` - Column-level accuracy
- Per-tolerance thresholds (0.90, 0.95, 0.99)

---

## Integration into Lightning Modules

### Current Structure (WRONG):
```python
def validation_step(self, batch, batch_idx):
    logits = self(...)
    loss = F.cross_entropy(...)
    accuracy = (predictions == targets).float().mean()  # ❌ WRONG
    self.log('val_accuracy', accuracy)
```

### Required Structure (CORRECT):
```python
def validation_step(self, batch, batch_idx):
    src, tgt, ... = batch
    
    # Forward pass
    logits = self(...)
    loss = F.cross_entropy(...)
    predictions = torch.argmax(logits, dim=-1)
    
    # Compute grid metrics
    metrics = compute_grid_metrics(predictions, tgt, self.pad_token)
    
    # Log grid-level metrics
    self.log('val_grid_accuracy', metrics['grid_accuracy'], ...)
    self.log('val_cell_accuracy', metrics['cell_accuracy'], ...)
    self.log('val_dense_grid_objective', metrics['dense_grid_objective'], ...)
    
    # Compute transformation metrics (if src available)
    if src is not None:
        copy_metrics = compute_copy_metrics_on_batch(src, tgt, predictions)
        self.log('val_change_recall', copy_metrics['change_recall'], ...)
        self.log('val_change_precision', copy_metrics['change_precision'], ...)
```

---

## Checkpoint Configuration

### Current (WRONG):
```python
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",  # ❌ Wrong metric
    mode="min"
)
```

### Required (CORRECT):
```python
checkpoint_callback = ModelCheckpoint(
    monitor="val_grid_accuracy",  # ✅ Primary metric
    mode="max"
)

early_stopping = EarlyStopping(
    monitor="val_grid_accuracy",  # ✅ Monitor grid solving
    mode="max",
    patience=7
)
```

---

## Expected Baseline Results (champion_bootstrap.ckpt on 400 tasks)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| `val_grid_accuracy` | 2.34% | Solving 1/43 grids completely |
| `val_cell_accuracy` | 90.39% | 90% pixels correct (strong) |
| `val_change_recall` | 70.32% | Detecting 70% of transformations |
| `val_dense_grid_objective` | 68.76% | Composite score (primary) |

**After fine-tuning on 18 tasks (V2 experiment):**
- Mean accuracy: 82.67% → 84.71%
- A2 (spatial packing): ~17% ceiling
- C1 (high affinity): 95-98%
- S3 (topology): 70-96% range

---

## Testing Requirements

### Metric Validation Tests:
```python
def test_grid_accuracy_perfect():
    """All grids perfect → 100%"""
    pred = torch.tensor([[1, 2], [3, 4]])
    tgt = torch.tensor([[1, 2], [3, 4]])
    metrics = compute_grid_metrics(pred, tgt, pad_token=10)
    assert metrics['grid_accuracy'] == 1.0

def test_grid_accuracy_one_wrong():
    """One cell wrong → 0% grid accuracy"""
    pred = torch.tensor([[1, 2], [3, 4]])
    tgt = torch.tensor([[1, 2], [3, 5]])  # One cell different
    metrics = compute_grid_metrics(pred, tgt, pad_token=10)
    assert metrics['grid_accuracy'] == 0.0
    assert metrics['cell_accuracy'] == 0.75  # 3/4 correct
```

---

## Action Items

### Immediate (BLOCKING TRAINING):
1. ☐ Create `src/evaluation/metrics.py`
2. ☐ Implement `compute_grid_metrics()` 
3. ☐ Implement `compute_dense_grid_objective()`
4. ☐ Update all 3 Lightning modules (champion, encoder_decoder, decoder_only)
5. ☐ Update checkpoint callbacks to monitor `val_grid_accuracy`
6. ☐ Add unit tests for metrics

### Follow-up (COMPLETENESS):
7. ☐ Port `compute_copy_metrics_on_batch()` from jarc_reactor
8. ☐ Add transformation metrics to validation steps
9. ☐ Update documentation with correct metric definitions

---

## References

**Source Code:**
- `jarc_reactor/utils/metrics.py` (lines 7-112): `compute_copy_metrics_on_batch()`
- `jarc_reactor/utils/train.py` (lines 715-774): `_compute_accuracy()`
- `jarc_reactor/utils/train.py` (lines 3340-3370): `dense_grid_objective` computation

**Documentation:**
- `docs/guides/ARC_TAXONOMY_TRAINING_PROVENANCE_FINAL.md` (lines 182-197): Champion baseline metrics
- Training logs: `outputs/logs/training/pretrain_bakeoff_t69.log`

---

**Status:** CRITICAL - Must implement before ANY training
**Estimated Effort:** 2-4 hours
**Priority:** P0 - BLOCKING
