# Storage URL Update for Visual Classifier HPO

## Change Summary

Updated both sweep configurations to use the same PostgreSQL database as the v3 architectural sweep for consistency and persistence.

### Files Modified

1. **`visual_classifier_sweep.yaml`** (main 150-trial sweep)
2. **`test_sweep.yaml`** (5-trial verification sweep)

### Previous Storage (SQLite)

```yaml
# Main sweep
storage_url: "sqlite:///outputs/visual_classifier/hpo_visual_classifier_v2.db"

# Test sweep
storage_url: "sqlite:///outputs/visual_classifier/hpo_test.db"
```

### New Storage (PostgreSQL)

```yaml
storage_url: "***REMOVED***"
```

## Benefits of PostgreSQL Storage

### 1. **Shared Database with Main Project**
- Same database used by `jarc-reactor-architectural-sweep_v3` (5000 trials)
- Enables cross-study analysis and comparison
- Consistent storage backend across all HPO experiments

### 2. **Persistence and Reliability**
- Remote database on DigitalOcean (managed PostgreSQL)
- Survives local machine crashes or restarts
- Concurrent access from multiple machines (if needed)
- Automatic backups

### 3. **Better for Large Sweeps**
- SQLite can have locking issues with concurrent writes
- PostgreSQL handles concurrent trials better
- More efficient for 150+ trial studies

### 4. **Query and Analysis**
- Can query the database directly with SQL
- Optuna Dashboard can connect to PostgreSQL for visualization
- Easier to export data for analysis

## Study Names (No Conflicts)

The visual classifier sweeps use different study names than the main architectural sweep:

| Study | Study Name |
|-------|------------|
| Main architectural sweep (v3) | `jarc-reactor-architectural-sweep_v3` |
| Visual classifier (main) | `visual_classifier_cnn_vs_context_v2_expanded` |
| Visual classifier (test) | `visual_classifier_test_v1` |

Each study is isolated by its unique study name, so they won't interfere with each other in the shared database.

## Dependencies

The environment already has the required PostgreSQL driver:
- ✅ `psycopg2` version 2.9.10 (installed)

## Usage

No changes needed to the command-line usage:

```bash
# Test sweep (5 trials)
python scripts/optimize.py --config configs/hpo/test_sweep.yaml

# Full sweep (150 trials)
python scripts/optimize.py --config configs/hpo/visual_classifier_sweep.yaml
```

The `optimize.py` script already handles both SQLite and PostgreSQL URLs correctly:
- SQLite: Creates local directory and adjusts path
- PostgreSQL: Uses URL as-is

## Loading Results

### From Optuna API

```python
import optuna

# Load study from PostgreSQL
study = optuna.load_study(
    study_name="visual_classifier_cnn_vs_context_v2_expanded",
    storage="***REMOVED***"
)

# Access trials
print(f"Number of trials: {len(study.trials)}")
print(f"Best value: {study.best_value}")
```

### From JSON Export

The `optimize.py` script still saves a JSON snapshot to the local filesystem:
- `outputs/visual_classifier/hpo/<study_name>/optimization_results.json`

This provides both:
1. **PostgreSQL database**: Persistent, queryable, shared
2. **JSON file**: Local backup, human-readable

## Migration Path (If Needed)

If you already have SQLite studies you want to migrate:

```python
import optuna

# Load from SQLite
old_study = optuna.load_study(
    study_name="old_study",
    storage="sqlite:///path/to/old.db"
)

# Create new study in PostgreSQL with same trials
new_study = optuna.create_study(
    study_name="migrated_study",
    storage="postgresql://...",
    direction=old_study.direction
)

# Copy trials
for trial in old_study.trials:
    new_study.add_trial(trial)
```

## Security Note

The PostgreSQL credentials are visible in the config file. This is the same pattern used in the v3 sweep configuration. For production use, consider:
- Environment variables for credentials
- Read-only credentials for analysis
- Separate databases for different project stages

For this research project, the shared database approach provides good traceability and consistency across experiments.
