#!/usr/bin/env python3
"""
Analyze the 30-epoch ablation study results.
Reads per_task_metrics CSVs and produces a clear summary.
"""
import pandas as pd
from pathlib import Path
import numpy as np

LOGS_DIR = Path("/Users/tomriddle1/Holistic-Performance-Enhancement/cultivation/systems/arc_reactor/publications/arc_taxonomy_2025/reproduction/logs/per_task_metrics")

EXPERIMENTS = {
    'Exp0_ED': 'exp0',
    'Exp1_Grid2D': 'exp1',
    'Exp2_PermInv': 'exp2',
    'Exp3_Context_s307': 'exp3_champion_rearc_s307',
    'Exp3_Context_s308': 'exp3_champion_rearc_s308',
    'Exp3_Context_s309': 'exp3_champion_rearc_s309',
    'Exp4_Champion_s307': 'exp3',  # This might be the full champion from the old runs
}

def load_experiment_metrics(exp_dir):
    """Load all epoch metrics for an experiment."""
    exp_path = LOGS_DIR / exp_dir
    if not exp_path.exists():
        return None
    
    all_data = []
    for csv_file in sorted(exp_path.glob('*per_category.csv')):
        try:
            df = pd.read_csv(csv_file)
            all_data.append(df)
        except Exception as e:
            print(f"  Warning: Failed to load {csv_file.name}: {e}")
    
    if not all_data:
        return None
    
    return pd.concat(all_data, ignore_index=True)

def analyze_experiment(name, exp_dir):
    """Analyze metrics for one experiment."""
    print(f"\n{'='*80}")
    print(f"{name} ({exp_dir})")
    print(f"{'='*80}")
    
    df = load_experiment_metrics(exp_dir)
    if df is None:
        print("  NO DATA FOUND")
        return None
    
    # Group by epoch and calculate overall metrics
    epoch_metrics = df.groupby('epoch').agg({
        'grid_correct': 'sum',
        'grid_total': 'sum',
        'cell_correct': 'sum',
        'cell_total': 'sum',
    }).reset_index()
    
    epoch_metrics['grid_accuracy'] = 100 * epoch_metrics['grid_correct'] / epoch_metrics['grid_total']
    epoch_metrics['cell_accuracy'] = 100 * epoch_metrics['cell_correct'] / epoch_metrics['cell_total']
    
    print(f"\nEpochs: {epoch_metrics['epoch'].min()} to {epoch_metrics['epoch'].max()}")
    print(f"Total epochs: {len(epoch_metrics)}")
    
    # Find best epoch by grid accuracy
    best_idx = epoch_metrics['grid_accuracy'].idxmax()
    best_epoch = epoch_metrics.loc[best_idx]
    
    print(f"\nBest Epoch (by grid accuracy): {int(best_epoch['epoch'])}")
    print(f"  Grid Accuracy: {best_epoch['grid_accuracy']:.4f}%")
    print(f"  Cell Accuracy: {best_epoch['cell_accuracy']:.4f}%")
    print(f"  Grid Correct: {int(best_epoch['grid_correct'])}/{int(best_epoch['grid_total'])}")
    
    # Final epoch
    final_idx = epoch_metrics['epoch'].idxmax()
    final_epoch = epoch_metrics.loc[final_idx]
    
    print(f"\nFinal Epoch: {int(final_epoch['epoch'])}")
    print(f"  Grid Accuracy: {final_epoch['grid_accuracy']:.4f}%")
    print(f"  Cell Accuracy: {final_epoch['cell_accuracy']:.4f}%")
    
    # Training dynamics
    first_epoch = epoch_metrics.loc[epoch_metrics['epoch'].idxmin()]
    print(f"\nTraining Dynamics:")
    print(f"  Epoch 0: {first_epoch['grid_accuracy']:.4f}%")
    print(f"  Best: {best_epoch['grid_accuracy']:.4f}% (epoch {int(best_epoch['epoch'])})")
    print(f"  Final: {final_epoch['grid_accuracy']:.4f}% (epoch {int(final_epoch['epoch'])})")
    print(f"  Peak-to-Final: {final_epoch['grid_accuracy'] - best_epoch['grid_accuracy']:.4f} pp")
    
    # Per-category breakdown at best epoch
    print(f"\nPer-Category Performance (Epoch {int(best_epoch['epoch'])}):")
    best_epoch_df = df[df['epoch'] == best_epoch['epoch']]
    for _, row in best_epoch_df.iterrows():
        cat_grid_acc = 100 * row['grid_correct'] / row['grid_total'] if row['grid_total'] > 0 else 0
        cat_cell_acc = 100 * row['cell_correct'] / row['cell_total'] if row['cell_total'] > 0 else 0
        print(f"  {row['category']}: {cat_grid_acc:.2f}% grid, {cat_cell_acc:.2f}% cell ({row['task_count']} tasks)")
    
    return {
        'name': name,
        'best_epoch': int(best_epoch['epoch']),
        'best_grid_acc': best_epoch['grid_accuracy'],
        'best_cell_acc': best_epoch['cell_accuracy'],
        'final_epoch': int(final_epoch['epoch']),
        'final_grid_acc': final_epoch['grid_accuracy'],
        'final_cell_acc': final_epoch['cell_accuracy'],
        'epoch_metrics': epoch_metrics
    }

def main():
    print("="*80)
    print("30-EPOCH ABLATION STUDY ANALYSIS")
    print("="*80)
    print(f"\nData directory: {LOGS_DIR}")
    
    results = {}
    for name, exp_dir in EXPERIMENTS.items():
        result = analyze_experiment(name, exp_dir)
        if result:
            results[name] = result
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON (Best Epoch Performance)")
    print(f"{'='*80}")
    print(f"\n{'Experiment':<25} {'Best Epoch':<12} {'Grid Acc %':<15} {'Cell Acc %':<15}")
    print("-"*80)
    
    for name in ['Exp0_ED', 'Exp1_Grid2D', 'Exp2_PermInv', 'Exp3_Context_s307', 'Exp3_Context_s308', 'Exp3_Context_s309']:
        if name in results:
            r = results[name]
            print(f"{name:<25} {r['best_epoch']:<12} {r['best_grid_acc']:<15.4f} {r['best_cell_acc']:<15.4f}")
    
    # Component deltas
    print(f"\n{'='*80}")
    print("COMPONENT CONTRIBUTIONS (Grid Accuracy)")
    print(f"{'='*80}")
    
    if 'Exp0_ED' in results:
        baseline = results['Exp0_ED']['best_grid_acc']
        print(f"\nBaseline (Exp0 E-D): {baseline:.4f}%")
        
        if 'Exp1_Grid2D' in results:
            delta = results['Exp1_Grid2D']['best_grid_acc'] - baseline
            print(f"  + Grid2D PE (Exp1): {delta:+.4f} pp → {results['Exp1_Grid2D']['best_grid_acc']:.4f}%")
            
            if 'Exp2_PermInv' in results:
                delta = results['Exp2_PermInv']['best_grid_acc'] - results['Exp1_Grid2D']['best_grid_acc']
                print(f"  + PermInv (Exp2):   {delta:+.4f} pp → {results['Exp2_PermInv']['best_grid_acc']:.4f}%")
                
                # Context is harder - we have multiple seeds
                exp3_accs = [results[k]['best_grid_acc'] for k in results if k.startswith('Exp3_Context')]
                if exp3_accs:
                    exp3_mean = np.mean(exp3_accs)
                    exp3_std = np.std(exp3_accs, ddof=1) if len(exp3_accs) > 1 else 0
                    delta = exp3_mean - results['Exp2_PermInv']['best_grid_acc']
                    print(f"  + Context (Exp3):   {delta:+.4f} pp → {exp3_mean:.4f}% ± {exp3_std:.4f}% (n={len(exp3_accs)})")

if __name__ == '__main__':
    main()
