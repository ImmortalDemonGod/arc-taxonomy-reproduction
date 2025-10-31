# HPO Sweep Configuration Coverage

This document details the complete hyperparameter search space for the visual classifier HPO sweep (v2_expanded).

## Study Configuration
- **Study name**: `visual_classifier_cnn_vs_context_v2_expanded`
- **Trials**: 150 (increased from 100 to cover expanded space)
- **Objective**: Maximize validation accuracy
- **Pruning**: Hyperband (min_resource=3, max_resource=20, reduction_factor=3)

## Architecture Coverage

### Shared Parameters (Both CNN and ContextEncoder)

| Parameter | Type | Range/Choices | Notes |
|-----------|------|---------------|-------|
| `encoder_type` | categorical | ["cnn", "context"] | Architectural switch |
| `lr` | float (log) | 5e-5 to 1e-2 | Extended range (was 1e-4 to 5e-3) |
| `batch_size` | categorical | [8, 16, 32, 64] | Added 64 |
| `weight_decay` | float (log) | 0.0 to 0.1 | Extended to 0.1, now log scale |
| `label_smoothing` | float | 0.0 to 0.2 | Extended to 0.2 |
| `embed_dim` | categorical | [128, 256, 384, 512, 768] | Added 384, 768 |
| `use_scheduler` | categorical | [true, false] | **NEW** - sweep scheduler usage |
| `use_cosine` | categorical | [true, false] | **NEW** - cosine similarity vs dot product |
| `temperature` | float | 5.0 to 20.0 | **NEW** - conditional on use_cosine=true |

### TaskEncoderCNN Specific Parameters

| Parameter | Type | Range/Choices | Notes |
|-----------|------|---------------|-------|
| `demo_agg` | categorical | ["flatten", "mean"] | **NEW** - was fixed at "mean" |
| `width_mult` | float | 0.5 to 3.0 | Broader (was 0.75-2.0) |
| `depth` | int | 2 to 6 | Extended to 6 (was 2-5) |
| `mlp_hidden` | categorical | [256, 512, 1024, 2048] | Added 2048 |
| `use_coords` | categorical | [true, false] | Coordinate position features |

### TaskEncoderAdvanced (ContextEncoder) Specific Parameters

| Parameter | Type | Range/Choices | Notes |
|-----------|------|---------------|-------|
| `ctx_d_model` | categorical | [128, 256, 384, 512] | Added 384, 512 (champion uses 512) |
| `ctx_n_head` | categorical | [4, 8, 16] | Added 16 |
| `ctx_pixel_layers` | int | 1 to 5 | Extended to 5 (was 2-4) |
| `ctx_grid_layers` | int | 1 to 3 | Extended to 3 (was 1-2) |
| `ctx_dropout` | float | 0.0 to 0.3 | Extended to 0.3 (was 0.0-0.2) |

## Fixed Parameters (Not Swept)

- `data_dir`: "data/distributional_alignment"
- `centroids`: "outputs/visual_classifier/category_centroids_v3.npy"
- `epochs`: 20
- `seed`: 42 (deterministic)
- `stratify`: true (stratified train/val split)
- `color_permute`: true (color augmentation)
- `random_demos`: true (randomize demo selection)
- `early_stop_patience`: 4
- `val_ratio`: 0.2
- `num_workers`: 2

## Key Improvements Over v1

### 1. **Complete Model Surface Coverage**
- **Added `demo_agg`** for CNN: Now sweeps "flatten" vs "mean" aggregation
- **Added similarity options**: `use_cosine` + `temperature` for metric learning
- **Added scheduler control**: `use_scheduler` to test with/without LR scheduling

### 2. **Broader Ranges**
- **Learning rate**: 5e-5 to 1e-2 (2x broader on both ends)
- **Embed dim**: Added 384 and 768 (enables testing larger models)
- **CNN width**: 0.5 to 3.0 (50% narrower to 3x wider)
- **CNN depth**: Up to 6 layers (was max 5)
- **Context d_model**: Up to 512 (matches champion architecture)
- **Context heads**: Up to 16 (for larger models)
- **Context layers**: More flexible (pixel 1-5, grid 1-3)
- **Dropout**: Up to 0.3 (stronger regularization)

### 3. **Conditional Logic**
- **Temperature** only sampled when `use_cosine=true`
- **CNN params** only when `encoder_type="cnn"`
- **Context params** only when `encoder_type="context"`

## Search Space Size Estimate

### CNN Configuration Space
- Continuous: lr (∞), weight_decay (∞), label_smoothing (∞), width_mult (∞), temperature (∞ if cosine)
- Discrete: 
  - encoder_type (1 choice: cnn)
  - batch_size (4)
  - embed_dim (5)
  - use_scheduler (2)
  - use_cosine (2)
  - demo_agg (2)
  - depth (5)
  - mlp_hidden (4)
  - use_coords (2)
- **Estimated discrete combinations**: ~6,400 (without continuous params)

### Context Configuration Space
- Continuous: lr (∞), weight_decay (∞), label_smoothing (∞), ctx_dropout (∞), temperature (∞ if cosine)
- Discrete:
  - encoder_type (1 choice: context)
  - batch_size (4)
  - embed_dim (5)
  - use_scheduler (2)
  - use_cosine (2)
  - ctx_d_model (4)
  - ctx_n_head (3)
  - ctx_pixel_layers (5)
  - ctx_grid_layers (3)
- **Estimated discrete combinations**: ~7,200 (without continuous params)

### Total Combined Space
- **Discrete combinations**: ~13,600
- **With continuous params**: Effectively infinite
- **150 trials**: Samples ~1.1% of discrete space (but uses TPE sampler for efficient exploration)

## Rationale for 150 Trials

With Optuna's TPE (Tree-structured Parzen Estimator) sampler and Hyperband pruning:
- **First 20-30 trials**: Random exploration
- **Trials 30-100**: Exploitation of promising regions
- **Trials 100-150**: Refinement and validation

This is sufficient to:
1. Identify best architecture (CNN vs Context)
2. Find optimal hyperparameters for each
3. Discover interaction effects (e.g., cosine + temperature, width + depth)
4. Validate findings with multiple runs in promising regions

## Expected Outcomes

1. **Best overall configuration** for each architecture
2. **Performance ceiling** for each (what's the max achievable?)
3. **Sensitivity analysis**: Which params matter most?
4. **Architecture comparison**: Does Context beat CNN? By how much?
5. **Per-category breakdown**: Which architecture handles which categories better?

## Next Steps After Sweep

1. **Extract top-5 trials** for each architecture
2. **Run extended training** (40+ epochs) on best configs
3. **Per-category analysis** on best models
4. **DIAGNOSE** failures: Which categories fail? Why?
5. **VALIDATE** hypotheses with targeted architectural modifications
