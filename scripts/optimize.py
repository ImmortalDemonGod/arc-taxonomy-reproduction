"""
HPO Conductor: Systematic hyperparameter optimization for visual classifier.

This script orchestrates the MEASURE phase by running Optuna studies to find
optimal hyperparameters for both CNN and ContextEncoder architectures.

Usage:
    python scripts/optimize.py
    python scripts/optimize.py --config-name alternative_sweep
"""
import sys
import json
from pathlib import Path
import yaml
import argparse

import numpy as np
import torch
import optuna
from optuna.pruners import HyperbandPruner, MedianPruner

# Make reproduction/src importable
REPRO_DIR = Path(__file__).resolve().parent.parent
if str(REPRO_DIR) not in sys.path:
    sys.path.insert(0, str(REPRO_DIR))

from src.data.arc_task_dataset import ARCTaskDataset
from objective import Objective, seed_everything


def resolve_path(path_str: str, base_dir: Path) -> Path:
    """Resolve a path relative to base_dir if not absolute."""
    path = Path(path_str)
    if path.is_absolute() and path.exists():
        return path
    resolved = (base_dir / path).resolve()
    if resolved.exists():
        return resolved
    # Fallback for labels path - try multiple possible locations
    if 'all_tasks_classified.json' in str(path):
        # Try several common locations
        possible_paths = [
            # From reproduction/ go to data/ (same level)
            base_dir / 'data' / 'taxonomy_classification' / 'all_tasks_classified.json',
            # From reproduction/ go up to arc_reactor/ then into data/
            base_dir.parent.parent / 'data' / 'taxonomy_classification' / 'all_tasks_classified.json',
            # Absolute path from root (Paperspace-style)
            Path('/data/taxonomy_classification/all_tasks_classified.json'),
            # Try from parent directories
            base_dir.parent / 'data' / 'taxonomy_classification' / 'all_tasks_classified.json',
        ]
        for possible_path in possible_paths:
            try:
                if possible_path.exists():
                    return possible_path.resolve()
            except (OSError, RuntimeError):
                continue
    return resolved


def make_stratified_splits(files, labels_path, category_to_idx, val_ratio=0.2, seed=42):
    """Create stratified train/val splits."""
    import random
    with open(labels_path) as f:
        task_categories = json.load(f)
    buckets = {i: [] for i in category_to_idx.values()}
    for fp in files:
        tid = Path(fp).stem
        cat = task_categories.get(tid, None)
        if cat is None or cat == 'ambiguous':
            continue
        if cat not in category_to_idx:
            continue
        buckets[category_to_idx[cat]].append(fp)
    train, val = [], []
    for k, lst in buckets.items():
        rng = random.Random(seed + k)
        rng.shuffle(lst)
        n_val = int(len(lst) * val_ratio)
        val.extend(lst[:n_val])
        train.extend(lst[n_val:])
    return train, val


def main():
    parser = argparse.ArgumentParser(description="Run HPO for visual classifier")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/hpo/visual_classifier_sweep.yaml",
        help="Path to sweep config YAML"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Override number of trials"
    )
    args = parser.parse_args()
    
    # Load config
    config_path = REPRO_DIR / args.config
    if not config_path.exists():
        print(f"ERROR: Config not found at {config_path}")
        sys.exit(1)
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    print(f"Loaded config from: {config_path}")
    print(f"Study name: {config['study_name']}")
    
    # Override n_trials if specified
    if args.n_trials is not None:
        config['n_trials'] = args.n_trials
    
    # Seed for reproducibility
    seed = config['fixed'].get('seed', 42)
    seed_everything(seed)
    
    # Resolve paths
    data_dir = resolve_path(config['fixed']['data_dir'], REPRO_DIR)
    labels_path = resolve_path(config['fixed']['labels'], REPRO_DIR)
    centroids_path = resolve_path(config['fixed']['centroids'], REPRO_DIR)
    
    # Fallback for centroids
    if not centroids_path.exists():
        fallback = REPRO_DIR / 'outputs' / 'visual_classifier' / 'category_centroids_v3.npy'
        if fallback.exists():
            centroids_path = fallback
    
    print(f"Data dir: {data_dir}")
    print(f"Labels: {labels_path}")
    print(f"Centroids: {centroids_path}")
    
    # Load data
    task_files = sorted(list(data_dir.glob('*.json')))
    print(f"Found {len(task_files)} task files")
    
    # Category mapping
    essential_categories = ["S1", "S2", "S3", "C1", "C2", "K1", "L1", "A1", "A2"]
    cat_to_idx = {name: i for i, name in enumerate(essential_categories)}
    
    # Create splits
    if config['fixed'].get('stratify', True):
        train_files, val_files = make_stratified_splits(
            task_files,
            labels_path,
            cat_to_idx,
            val_ratio=config['fixed']['val_ratio']
        )
    else:
        import random
        files_copy = list(task_files)
        random.Random(seed).shuffle(files_copy)
        n_val = int(len(files_copy) * config['fixed']['val_ratio'])
        train_files = files_copy[n_val:]
        val_files = files_copy[:n_val]
    
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")
    
    # Create datasets
    train_ds = ARCTaskDataset(
        train_files,
        labels_path,
        max_grid_size=30,
        random_demos=config['fixed'].get('random_demos', True),
        color_permute=config['fixed'].get('color_permute', True)
    )
    val_ds = ARCTaskDataset(
        val_files,
        labels_path,
        max_grid_size=30,
        random_demos=False
    )
    
    # Load centroids
    centroids_np = np.load(str(centroids_path))
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    centroids = torch.tensor(centroids_np, dtype=torch.float32, device=device)
    print(f"Centroids shape: {centroids.shape}, Device: {device}")
    
    # Create output directory
    storage_url = config['storage_url']
    if storage_url.startswith('sqlite:///'):
        db_path = REPRO_DIR / storage_url.replace('sqlite:///', '')
        db_path.parent.mkdir(parents=True, exist_ok=True)
        storage_url = f"sqlite:///{db_path}"
    
    base_output_dir = REPRO_DIR / 'outputs' / 'visual_classifier' / 'hpo' / config['study_name']
    base_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {base_output_dir}")
    
    # Create pruner
    pruner_config = config['pruner']
    if pruner_config['type'] == 'hyperband':
        pruner = HyperbandPruner(
            min_resource=pruner_config['min_resource'],
            max_resource=pruner_config['max_resource'],
            reduction_factor=pruner_config['reduction_factor']
        )
    elif pruner_config['type'] == 'median':
        pruner = MedianPruner(
            n_startup_trials=pruner_config.get('n_startup_trials', 5),
            n_warmup_steps=pruner_config.get('n_warmup_steps', 3)
        )
    else:
        pruner = None
    
    # Create study
    study = optuna.create_study(
        study_name=config['study_name'],
        storage=storage_url,
        direction=config['direction'],
        pruner=pruner,
        load_if_exists=True
    )
    
    print(f"\nStarting optimization with {config['n_trials']} trials")
    print(f"Storage: {storage_url}")
    print(f"Pruner: {pruner_config['type']}")
    
    # Create objective
    objective = Objective(
        config=config,
        train_dataset=train_ds,
        val_dataset=val_ds,
        centroids=centroids,
        device=device,
        base_output_dir=base_output_dir
    )
    
    # Run optimization
    study.optimize(objective, n_trials=config['n_trials'])
    
    # Report results
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value:.4f}")
    print(f"Best params:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
    print(f"\nBest trial metrics:")
    print(f"  train_loss: {study.best_trial.user_attrs.get('train_loss', 'N/A'):.4f}")
    print(f"  train_acc: {study.best_trial.user_attrs.get('train_acc', 'N/A'):.4f}")
    print(f"  val_loss: {study.best_trial.user_attrs.get('val_loss', 'N/A'):.4f}")
    print(f"  val_acc: {study.best_trial.user_attrs.get('val_acc', 'N/A'):.4f}")
    print(f"  best_epoch: {study.best_trial.user_attrs.get('best_epoch', 'N/A')}")
    print(f"\nPer-category validation accuracy:")
    categories = ["S1", "S2", "S3", "C1", "C2", "K1", "L1", "A1", "A2"]
    for cat in categories:
        cat_acc = study.best_trial.user_attrs.get(f'val_acc_{cat}', 0.0)
        print(f"  {cat}: {cat_acc:.4f}")
    
    # Save results
    results = {
        'study_name': config['study_name'],
        'n_trials': len(study.trials),
        'best_trial': study.best_trial.number,
        'best_value': study.best_value,
        'best_params': study.best_trial.params,
        'best_user_attrs': dict(study.best_trial.user_attrs),
        'trials_summary': [
            {
                'number': t.number,
                'value': t.value,
                'params': t.params,
                'state': str(t.state),
                'user_attrs': dict(t.user_attrs)  # Include all metrics
            }
            for t in study.trials
        ]
    }
    
    results_path = base_output_dir / 'optimization_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    print(f"Best model checkpoint: {base_output_dir / f'trial_{study.best_trial.number}' / 'best_model.pt'}")


if __name__ == '__main__':
    main()
