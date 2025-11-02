"""
Per-task metrics logger callback.

Saves detailed per-task metrics to CSV files after each epoch.
Robust to GPU crashes - writes to disk immediately after each epoch.
"""
import csv
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List

import pytorch_lightning as pl
import torch


class PerTaskMetricsLogger(pl.Callback):
    """
    Logs per-task metrics to CSV files for later category-level analysis.
    
    Writes files after each epoch to survive GPU crashes.
    """
    
    def __init__(self, log_dir: str = "logs/per_task_metrics", experiment_name: str = "experiment"):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory to save CSV files
            experiment_name: Name of experiment to include in filenames
        """
        super().__init__()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        
        # Load task categories if available
        self.task_categories = self._load_task_categories()
        
        print(f"PerTaskMetricsLogger initialized")
        print(f"  Log directory: {self.log_dir.absolute()}")
        print(f"  Task categories loaded: {len(self.task_categories)} tasks")
    
    def _load_task_categories(self) -> Dict[str, str]:
        """Load task categories from JSON file."""
        mapping: Dict[str, str] = {}
        try:
            # repo root: .../reproduction
            base = Path(__file__).parent.parent
            rearc_categories = base / "data" / "distributional_alignment" / "task_categories.json"
            if rearc_categories.exists():
                with open(rearc_categories) as f:
                    mapping.update(json.load(f))
        except Exception as e:
            print(f"Warning: Could not load re-arc task categories: {e}")
        try:
            vcats_csv = base / "outputs" / "visual_classifier" / "results" / "arc2_classify_seed.csv"
            if vcats_csv.exists():
                with open(vcats_csv, newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        tid = row.get('task_id')
                        lab = row.get('pred_label')
                        if tid and lab:
                            mapping[tid] = lab
        except Exception as e:
            print(f"Warning: Could not load visual classifier categories: {e}")
        return mapping
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Save per-task metrics to CSV after each validation epoch.
        
        Creates two files per epoch:
        1. epoch_N_per_task.csv: Detailed per-task metrics
        2. epoch_N_per_category.csv: Aggregated category metrics
        """
        if not hasattr(pl_module, 'validation_step_outputs') or not pl_module.validation_step_outputs:
            return
        
        epoch = trainer.current_epoch
        
        # Aggregate metrics per task
        task_metrics = defaultdict(lambda: {
            'task_id': '',
            'category': 'unknown',
            'grid_correct': 0,
            'grid_total': 0,
            'cell_correct': 0,
            'cell_total': 0,
            'copy_rate_sum': 0.0,
            'change_recall_sum': 0.0,
            'trans_quality_sum': 0.0,
            'metric_count': 0,
        })
        
        # Collect all validation outputs
        for output in pl_module.validation_step_outputs:
            task_ids = output['task_ids']
            grid_correct = output['grid_correct']
            cell_correct_counts = output['cell_correct_counts']
            cell_total_counts = output['cell_total_counts']
            copy_rate = output.get('copy_rate', None)
            change_recall = output.get('change_recall', None)
            trans_quality = output.get('trans_quality', None)
            
            for idx, (task_id, is_grid_correct, cell_correct, cell_total) in enumerate(zip(
                task_ids, grid_correct, cell_correct_counts, cell_total_counts
            )):
                task_metrics[task_id]['task_id'] = task_id
                # Strip _eval/_train suffix for category lookup
                lookup_id = task_id.replace('_eval', '').replace('_train', '')
                task_metrics[task_id]['category'] = self.task_categories.get(lookup_id, 'unknown')
                task_metrics[task_id]['grid_correct'] += int(is_grid_correct)
                task_metrics[task_id]['grid_total'] += 1
                task_metrics[task_id]['cell_correct'] += int(cell_correct)
                task_metrics[task_id]['cell_total'] += int(cell_total)
                
                # Add transformation metrics if available
                if copy_rate is not None and idx < len(copy_rate):
                    task_metrics[task_id]['copy_rate_sum'] += float(copy_rate[idx])
                    task_metrics[task_id]['change_recall_sum'] += float(change_recall[idx])
                    task_metrics[task_id]['trans_quality_sum'] += float(trans_quality[idx])
                    task_metrics[task_id]['metric_count'] += 1
        
        # Write per-task CSV
        per_task_file = self.log_dir / f"{self.experiment_name}_epoch_{epoch:03d}_per_task.csv"
        self._write_per_task_csv(per_task_file, task_metrics, epoch)
        
        # Write per-category CSV
        per_category_file = self.log_dir / f"{self.experiment_name}_epoch_{epoch:03d}_per_category.csv"
        self._write_per_category_csv(per_category_file, task_metrics, epoch)
        
        print(f"\n✓ Saved per-task metrics to {per_task_file.name}")
        print(f"✓ Saved per-category metrics to {per_category_file.name}\n")

        try:
            self._write_global_metrics_csv(task_metrics, epoch)
        except Exception as e:
            print(f"Warning: Failed to write global metrics CSV: {e}")
    
    def _write_per_task_csv(self, filepath: Path, task_metrics: Dict[str, Dict[str, Any]], epoch: int) -> None:
        """Write per-task metrics to CSV."""
        fieldnames = [
            'epoch',
            'task_id',
            'category',
            'grid_accuracy',
            'cell_accuracy',
            'grid_correct',
            'grid_total',
            'cell_correct',
            'cell_total',
            'copy_rate',
            'change_recall',
            'transformation_quality',
        ]
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for task_id in sorted(task_metrics.keys()):
                metrics = task_metrics[task_id]
                
                grid_acc = (metrics['grid_correct'] / metrics['grid_total'] * 100) if metrics['grid_total'] > 0 else 0.0
                cell_acc = (metrics['cell_correct'] / metrics['cell_total'] * 100) if metrics['cell_total'] > 0 else 0.0
                copy_rate = (metrics['copy_rate_sum'] / metrics['metric_count']) if metrics['metric_count'] > 0 else 0.0
                change_recall = (metrics['change_recall_sum'] / metrics['metric_count']) if metrics['metric_count'] > 0 else 0.0
                trans_quality = (metrics['trans_quality_sum'] / metrics['metric_count']) if metrics['metric_count'] > 0 else 0.0
                
                writer.writerow({
                    'epoch': epoch,
                    'task_id': task_id,
                    'category': metrics['category'],
                    'grid_accuracy': f"{grid_acc:.4f}",
                    'cell_accuracy': f"{cell_acc:.4f}",
                    'grid_correct': metrics['grid_correct'],
                    'grid_total': metrics['grid_total'],
                    'cell_correct': metrics['cell_correct'],
                    'cell_total': metrics['cell_total'],
                    'copy_rate': f"{copy_rate:.4f}",
                    'change_recall': f"{change_recall:.4f}",
                    'transformation_quality': f"{trans_quality:.6f}",
                })
    
    def _write_per_category_csv(self, filepath: Path, task_metrics: Dict[str, Dict[str, Any]], epoch: int) -> None:
        """Write per-category aggregated metrics to CSV."""
        # Aggregate by category
        category_stats = defaultdict(lambda: {
            'grid_correct': 0,
            'grid_total': 0,
            'cell_correct': 0,
            'cell_total': 0,
            'copy_rate_sum': 0.0,
            'change_recall_sum': 0.0,
            'trans_quality_sum': 0.0,
            'metric_count': 0,
            'task_count': 0,
        })
        
        for task_id, metrics in task_metrics.items():
            category = metrics['category']
            category_stats[category]['grid_correct'] += metrics['grid_correct']
            category_stats[category]['grid_total'] += metrics['grid_total']
            category_stats[category]['cell_correct'] += metrics['cell_correct']
            category_stats[category]['cell_total'] += metrics['cell_total']
            category_stats[category]['copy_rate_sum'] += metrics['copy_rate_sum']
            category_stats[category]['change_recall_sum'] += metrics['change_recall_sum']
            category_stats[category]['trans_quality_sum'] += metrics['trans_quality_sum']
            category_stats[category]['metric_count'] += metrics['metric_count']
            category_stats[category]['task_count'] += 1
        
        fieldnames = [
            'epoch',
            'category',
            'task_count',
            'grid_accuracy',
            'cell_accuracy',
            'grid_correct',
            'grid_total',
            'cell_correct',
            'cell_total',
            'copy_rate',
            'change_recall',
            'transformation_quality',
        ]
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for category in sorted(category_stats.keys()):
                stats = category_stats[category]
                
                grid_acc = (stats['grid_correct'] / stats['grid_total'] * 100) if stats['grid_total'] > 0 else 0.0
                cell_acc = (stats['cell_correct'] / stats['cell_total'] * 100) if stats['cell_total'] > 0 else 0.0
                copy_rate = (stats['copy_rate_sum'] / stats['metric_count']) if stats['metric_count'] > 0 else 0.0
                change_recall = (stats['change_recall_sum'] / stats['metric_count']) if stats['metric_count'] > 0 else 0.0
                trans_quality = (stats['trans_quality_sum'] / stats['metric_count']) if stats['metric_count'] > 0 else 0.0
                
                writer.writerow({
                    'epoch': epoch,
                    'category': category,
                    'task_count': stats['task_count'],
                    'grid_accuracy': f"{grid_acc:.4f}",
                    'cell_accuracy': f"{cell_acc:.4f}",
                    'grid_correct': stats['grid_correct'],
                    'grid_total': stats['grid_total'],
                    'cell_correct': stats['cell_correct'],
                    'cell_total': stats['cell_total'],
                    'copy_rate': f"{copy_rate:.4f}",
                    'change_recall': f"{change_recall:.4f}",
                    'transformation_quality': f"{trans_quality:.6f}",
                })

    def _write_global_metrics_csv(self, task_metrics: Dict[str, Dict[str, Any]], epoch: int) -> None:
        # Aggregate across all tasks
        total_grid_correct = sum(m['grid_correct'] for m in task_metrics.values())
        total_grid = sum(m['grid_total'] for m in task_metrics.values())
        total_cell_correct = sum(m['cell_correct'] for m in task_metrics.values())
        total_cell = sum(m['cell_total'] for m in task_metrics.values())
        total_copy_sum = sum(m['copy_rate_sum'] for m in task_metrics.values())
        total_change_sum = sum(m['change_recall_sum'] for m in task_metrics.values())
        total_transq_sum = sum(m['trans_quality_sum'] for m in task_metrics.values())
        total_metric_count = sum(m['metric_count'] for m in task_metrics.values())

        # Compute decimals (0-1)
        val_grid_accuracy = (total_grid_correct / total_grid) if total_grid > 0 else 0.0
        val_cell_accuracy = (total_cell_correct / total_cell) if total_cell > 0 else 0.0
        val_copy_rate = (total_copy_sum / total_metric_count) if total_metric_count > 0 else 0.0
        val_change_recall = (total_change_sum / total_metric_count) if total_metric_count > 0 else 0.0
        val_transformation_quality = (total_transq_sum / total_metric_count) if total_metric_count > 0 else 0.0

        # Path: logs/{experiment_name}_csv/version_0/metrics.csv
        base = Path("logs") / f"{self.experiment_name}_csv" / "version_0"
        base.mkdir(parents=True, exist_ok=True)
        metrics_csv = base / "metrics.csv"
        # Also write denominators for analysis
        global_metrics_csv = base / "global_metrics.csv"

        # Prepare header
        fieldnames = [
            'epoch', 'val_grid_accuracy', 'val_cell_accuracy',
            'val_copy_rate', 'val_change_recall', 'val_transformation_quality'
        ]

        write_header = not metrics_csv.exists()
        with open(metrics_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({
                'epoch': epoch,
                'val_grid_accuracy': f"{val_grid_accuracy:.8f}",
                'val_cell_accuracy': f"{val_cell_accuracy:.8f}",
                'val_copy_rate': f"{val_copy_rate:.8f}",
                'val_change_recall': f"{val_change_recall:.8f}",
                'val_transformation_quality': f"{val_transformation_quality:.8f}",
            })
        # Write denominators to a separate file
        g_fieldnames = [
            'epoch', 'val_total_grids', 'val_total_cells',
            'val_grid_accuracy', 'val_cell_accuracy',
            'val_copy_rate', 'val_change_recall', 'val_transformation_quality'
        ]
        g_write_header = not global_metrics_csv.exists()
        with open(global_metrics_csv, 'a', newline='') as f:
            gw = csv.DictWriter(f, fieldnames=g_fieldnames)
            if g_write_header:
                gw.writeheader()
            gw.writerow({
                'epoch': epoch,
                'val_total_grids': total_grid,
                'val_total_cells': total_cell,
                'val_grid_accuracy': f"{val_grid_accuracy:.8f}",
                'val_cell_accuracy': f"{val_cell_accuracy:.8f}",
                'val_copy_rate': f"{val_copy_rate:.8f}",
                'val_change_recall': f"{val_change_recall:.8f}",
                'val_transformation_quality': f"{val_transformation_quality:.8f}",
            })
