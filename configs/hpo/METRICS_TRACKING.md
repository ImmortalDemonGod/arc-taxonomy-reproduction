# HPO Metrics Tracking

This document describes the comprehensive metrics saved for each trial in the HPO sweep.

## Trial User Attributes (Saved via Optuna)

Every trial saves the following metrics to `trial.user_attrs`, which are stored in the SQLite database and included in the JSON results file:

### Overall Metrics

| Attribute | Type | Description |
|-----------|------|-------------|
| `train_loss` | float | Training loss at best epoch |
| `train_acc` | float | Training accuracy at best epoch |
| `val_loss` | float | Validation loss at best epoch |
| `val_acc` | float | Validation accuracy at best epoch (optimization target) |
| `best_epoch` | int | Epoch number where best validation accuracy was achieved |

### Per-Category Validation Accuracy

Each category gets its own user attribute for fine-grained analysis:

| Attribute | Type | Description |
|-----------|------|-------------|
| `val_acc_S1` | float | Validation accuracy on S1 (Symmetry: Reflection) tasks |
| `val_acc_S2` | float | Validation accuracy on S2 (Symmetry: Rotation) tasks |
| `val_acc_S3` | float | Validation accuracy on S3 (Symmetry: Translation) tasks |
| `val_acc_C1` | float | Validation accuracy on C1 (Counting) tasks |
| `val_acc_C2` | float | Validation accuracy on C2 (Comparison) tasks |
| `val_acc_K1` | float | Validation accuracy on K1 (Knowledge: Shape) tasks |
| `val_acc_L1` | float | Validation accuracy on L1 (Logic) tasks |
| `val_acc_A1` | float | Validation accuracy on A1 (Abstraction: Patterns) tasks |
| `val_acc_A2` | float | Validation accuracy on A2 (Abstraction: Relations) tasks |

## Example Trial Record

Each trial in the results JSON will have this structure:

```json
{
  "number": 42,
  "value": 0.3125,
  "params": {
    "encoder_type": "cnn",
    "lr": 0.00123,
    "batch_size": 16,
    "embed_dim": 256,
    "width_mult": 1.5,
    "depth": 3,
    "mlp_hidden": 512,
    "use_coords": false,
    "demo_agg": "mean",
    "weight_decay": 0.001,
    "label_smoothing": 0.05,
    "use_scheduler": true,
    "use_cosine": false
  },
  "state": "COMPLETE",
  "user_attrs": {
    "train_loss": 1.842588,
    "train_acc": 0.310897,
    "val_loss": 2.064783,
    "val_acc": 0.3125,
    "best_epoch": 12,
    "val_acc_S1": 0.0,
    "val_acc_S2": 0.0,
    "val_acc_S3": 0.583333,
    "val_acc_C1": 0.304348,
    "val_acc_C2": 0.0,
    "val_acc_K1": 0.0,
    "val_acc_L1": 0.0,
    "val_acc_A1": 0.0,
    "val_acc_A2": 0.0
  }
}
```

## Access Pattern

### Via Optuna Study Object

```python
import optuna

study = optuna.load_study(
    study_name="visual_classifier_cnn_vs_context_v2_expanded",
    storage="sqlite:///path/to/hpo_visual_classifier_v2.db"
)

# Best trial
best = study.best_trial
print(f"Best val_acc: {best.value}")
print(f"Best epoch: {best.user_attrs['best_epoch']}")
print(f"S3 accuracy: {best.user_attrs['val_acc_S3']}")

# All trials
for trial in study.trials:
    if trial.state == optuna.trial.TrialState.COMPLETE:
        s3_acc = trial.user_attrs.get('val_acc_S3', 0.0)
        print(f"Trial {trial.number}: S3={s3_acc:.3f}")
```

### Via JSON Results File

```python
import json

with open('optimization_results.json') as f:
    results = json.load(f)

# Best trial
best = results['best_user_attrs']
print(f"Best S3 accuracy: {best['val_acc_S3']}")

# All trials
for trial in results['trials_summary']:
    attrs = trial['user_attrs']
    print(f"Trial {trial['number']}: S3={attrs.get('val_acc_S3', 0.0):.3f}")
```

## Analysis Use Cases

### 1. **Identify Architecture-Category Affinity**

Find which architecture (CNN vs Context) performs better on each category:

```python
cnn_trials = [t for t in study.trials if t.params['encoder_type'] == 'cnn']
context_trials = [t for t in study.trials if t.params['encoder_type'] == 'context']

for cat in ['S1', 'S2', 'S3', 'C1', 'C2', 'K1', 'L1', 'A1', 'A2']:
    cnn_best = max([t.user_attrs.get(f'val_acc_{cat}', 0.0) for t in cnn_trials])
    ctx_best = max([t.user_attrs.get(f'val_acc_{cat}', 0.0) for t in context_trials])
    print(f"{cat}: CNN={cnn_best:.3f}, Context={ctx_best:.3f}")
```

### 2. **Find Per-Category Specialists**

Identify which hyperparameter combinations excel at specific categories:

```python
# Best trials for S3 (symmetry-translation)
s3_trials = sorted(study.trials, 
                   key=lambda t: t.user_attrs.get('val_acc_S3', 0.0),
                   reverse=True)[:5]

print("Top 5 configurations for S3:")
for t in s3_trials:
    print(f"  Trial {t.number}: {t.user_attrs['val_acc_S3']:.3f}")
    print(f"    encoder: {t.params['encoder_type']}, lr: {t.params['lr']:.6f}")
```

### 3. **Diagnose Systematic Failures**

Find categories where all trials perform poorly:

```python
categories = ['S1', 'S2', 'S3', 'C1', 'C2', 'K1', 'L1', 'A1', 'A2']
for cat in categories:
    max_acc = max([t.user_attrs.get(f'val_acc_{cat}', 0.0) 
                   for t in study.trials 
                   if t.state == optuna.trial.TrialState.COMPLETE])
    if max_acc < 0.2:
        print(f"⚠️  {cat}: Best={max_acc:.3f} - SYSTEMATIC FAILURE")
```

### 4. **Hyperparameter-Category Correlation**

Analyze which hyperparameters correlate with success on specific categories:

```python
import pandas as pd

# Build dataframe
data = []
for t in study.trials:
    if t.state == optuna.trial.TrialState.COMPLETE:
        row = {**t.params, **t.user_attrs}
        data.append(row)

df = pd.DataFrame(data)

# Correlation with S3 accuracy
s3_corr = df[['lr', 'batch_size', 'embed_dim', 'val_acc_S3']].corr()['val_acc_S3']
print(s3_corr.sort_values(ascending=False))
```

## Integration with MEASURE → DIAGNOSE → VALIDATE Workflow

1. **MEASURE**: The sweep collects comprehensive per-category metrics across 150 trials
2. **DIAGNOSE**: Analysis scripts identify which categories fail for which architectures
3. **VALIDATE**: Targeted experiments test hypotheses about architectural modifications needed for failing categories

## Storage

- **SQLite database**: `outputs/visual_classifier/hpo_visual_classifier_v2.db`
- **JSON results**: `outputs/visual_classifier/hpo/<study_name>/optimization_results.json`
- **Model checkpoints**: `outputs/visual_classifier/hpo/<study_name>/trial_<N>/best_model.pt`
