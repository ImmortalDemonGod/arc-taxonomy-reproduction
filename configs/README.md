# Configuration Files

This directory contains YAML configuration files for training, HPO sweeps, and experiments.

---

## Active Configurations

### LoRA Training
- **atomic_lora_training.yaml** - Configuration for atomic LoRA training
  - Used by: `scripts/train_atomic_loras.py`, `scripts/test_lora_minimal.py`
  - Purpose: Train task-specific LoRA adapters on 18 foundational tasks
  - Key params: lr=1.849e-4, rank=8, alpha=16, dropout=0.1
  - Training: 400 samples/task, 100 epochs, patience=10

### HPO Sweeps
- **hpo/visual_classifier_sweep.yaml** - Main HPO sweep configuration
  - Used by: `scripts/optimize.py` (default), HPO tests
  - Purpose: Optimize visual classifier hyperparameters
  - Storage: SQLite (local)
  
- **hpo/visual_classifier_sweep_v3_intelligent.yaml** - V3 intelligent sweep
  - Used by: `scripts/validate_sweep_config.py`
  - Purpose: Advanced HPO with conditional parameters
  - Features: Pruning, intelligent search space
  
- **hpo/test_sweep.yaml** - Testing/development sweep
  - Used by: HPO integration tests
  - Purpose: Fast HPO validation (smaller search space)

---

## Configuration Structure

### Typical Config Format

```yaml
model:
  d_model: 128
  num_layers: 4
  # ... architecture params

training:
  learning_rate: 1e-4
  max_epochs: 100
  batch_size: 32
  # ... training params

data:
  train_tasks: 308
  val_tasks: 92
  # ... data params
```

---

## Notes

- All ablation experiments (`train_baseline_decoder_only.py`, `train_exp0_encoder_decoder.py`, etc.) **hardcode hyperparameters** from Trial 69 directly in the training scripts
- This ensures exact reproducibility of published results
- HPO configs are for hyperparameter optimization, not reproduction
- LoRA config is the only one actively used for reproduction experiments

---

## Documentation

See `hpo/` subdirectory for detailed HPO documentation:
- `hpo/METRICS_TRACKING.md` - Metric definitions and tracking
- `hpo/STORAGE_UPDATE.md` - Database storage configuration
- `hpo/SWEEP_COVERAGE.md` - Search space coverage analysis
