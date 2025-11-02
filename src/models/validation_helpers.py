"""
Shared validation helpers for per-category metric aggregation.

All ablation models use this to ensure consistent metric collection.
"""
import sys
from pathlib import Path
from collections import defaultdict
import json
from typing import Dict, List, Any, Optional
import torch
import pytorch_lightning as pl


def create_trial69_optimizer_and_scheduler(pl_module: pl.LightningModule) -> dict:
    """
    Create optimizer and scheduler matching Trial 69 configuration.
    
    CENTRALIZED FUNCTION - All ablation models use identical optimizer config.
    
    Args:
        pl_module: Lightning module with hparams (learning_rate, beta1, beta2, weight_decay)
    
    Returns:
        Dict with optimizer and lr_scheduler config for PyTorch Lightning
    """
    optimizer = torch.optim.Adam(
        pl_module.parameters(),
        lr=pl_module.hparams.learning_rate,
        betas=(pl_module.hparams.beta1, pl_module.hparams.beta2),
        weight_decay=pl_module.hparams.weight_decay,
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=6,
        T_mult=1,
        eta_min=1.6816632143867157e-06,
    )
    
    return {
        'optimizer': optimizer,
        'lr_scheduler': {
            'scheduler': scheduler,
            'interval': 'epoch',
        },
    }


def log_validation_loss(pl_module: pl.LightningModule, loss: torch.Tensor, batch_size: int):
    """
    Log validation loss in a way that ModelCheckpoint can reliably monitor.
    
    CENTRALIZED FUNCTION - All ablation models use this to ensure checkpoints work.
    
    Issue: prog_bar=False can cause ModelCheckpoint to not see val_loss properly.
    Fix: Use prog_bar=True, sync_dist=True, logger=True for reliable checkpointing.
    
    Args:
        pl_module: Lightning module
        loss: Validation loss tensor
        batch_size: Batch size for proper reduction
    """
    pl_module.log(
        'val_loss',
        loss,
        batch_size=batch_size,
        prog_bar=True,  # Show in progress bar (helps debugging + ensures checkpoint sees it)
        on_step=False,
        on_epoch=True,
        sync_dist=True,  # Important for multi-GPU
        logger=True,  # Ensure TensorBoard/CSV loggers get it
    )


def add_transformation_metrics(
    step_output: Dict[str, Any],
    src: torch.Tensor,
    tgt: torch.Tensor,
    preds: torch.Tensor,
    cell_correct_counts: torch.Tensor,
    cell_total_counts: torch.Tensor,
    pl_module: pl.LightningModule,
    batch_size: int
) -> None:
    """
    Compute and add transformation metrics to step_output.
    
    CENTRALIZED FUNCTION - All models use this to ensure consistency.
    
    Args:
        step_output: Dict to add metrics to (modified in-place)
        src: Source input tensor
        tgt: Target output tensor
        preds: Model predictions
        cell_correct_counts: Per-example correct cell counts
        cell_total_counts: Per-example total cell counts
        pl_module: Lightning module (for logging)
        batch_size: Batch size (for logging)
    """
    from ..evaluation.metrics import compute_copy_metrics_on_batch
    
    if src.size(1) <= 1:
        return  # Skip if sequence too short
    
    try:
        copy_metrics = compute_copy_metrics_on_batch(src, tgt, preds)
        
        # Log aggregated metrics
        pl_module.log('val_change_recall', copy_metrics['change_recall'], batch_size=batch_size, prog_bar=False, on_step=False, on_epoch=True)
        pl_module.log('val_copy_rate', copy_metrics['copy_rate'], batch_size=batch_size, prog_bar=False, on_step=False, on_epoch=True)
        pl_module.log('val_transformation_f1', copy_metrics['transformation_f1'], batch_size=batch_size, prog_bar=False, on_step=False, on_epoch=True)
        
        # Store per-example metrics for category aggregation (tensors on CPU)
        step_output['copy_rate'] = torch.nan_to_num(copy_metrics['copy_rate_per_example'], nan=0.0).cpu()
        step_output['change_recall'] = torch.nan_to_num(copy_metrics['change_recall_per_example'], nan=0.0).cpu()
        trans_quality_per_example = copy_metrics['transformation_f1_per_example'] * (cell_correct_counts.float() / cell_total_counts.float())
        step_output['trans_quality'] = torch.nan_to_num(trans_quality_per_example, nan=0.0).cpu()
    except Exception:
        pass  # Silently skip if computation fails


def load_task_categories() -> Dict[str, str]:
    """
    Load task_categories.json mapping.
    
    Returns:
        Dict mapping task_id -> category
    """
    try:
        data_dir = Path("data/distributional_alignment")
        categories_file = data_dir / "task_categories.json"
        if categories_file.exists():
            with open(categories_file) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def aggregate_validation_metrics(
    validation_outputs: List[Dict[str, Any]],
    task_categories: Dict[str, str]
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate validation metrics by category.
    
    Args:
        validation_outputs: List of validation step outputs containing:
            - task_ids: List of task IDs
            - grid_correct: Tensor of correct grids (binary)
            - cell_correct_counts: Tensor of correct cells per example
            - cell_total_counts: Tensor of total cells per example
            - copy_rate (optional): Tensor of copy rates
            - change_recall (optional): Tensor of change recalls
            - trans_quality (optional): Tensor of transformation qualities
        task_categories: Dict mapping task_id -> category
    
    Returns:
        Dict of category -> stats dict
    """
    category_stats = defaultdict(lambda: {
        'grid_correct': 0,
        'grid_total': 0,
        'cell_correct': 0,
        'cell_total': 0,
        'copy_rate_sum': 0,
        'change_recall_sum': 0,
        'trans_quality_sum': 0,
        'metric_count': 0
    })
    
    for output in validation_outputs:
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
            # Try exact match first, then without suffix
            category = task_categories.get(task_id)
            if category is None:
                lookup_id = task_id.replace('_eval', '').replace('_train', '')
                category = task_categories.get(lookup_id, 'unknown')
            category_stats[category]['grid_correct'] += int(is_grid_correct)
            category_stats[category]['grid_total'] += 1
            category_stats[category]['cell_correct'] += int(cell_correct)
            category_stats[category]['cell_total'] += int(cell_total)
            
            # Add transformation metrics if available
            if copy_rate is not None and idx < len(copy_rate):
                category_stats[category]['copy_rate_sum'] += float(copy_rate[idx])
                category_stats[category]['change_recall_sum'] += float(change_recall[idx])
                category_stats[category]['trans_quality_sum'] += float(trans_quality[idx])
                category_stats[category]['metric_count'] += 1
    
    return category_stats


def print_category_table(category_stats: Dict[str, Dict], epoch: int):
    """
    Print per-category validation accuracy table.
    
    Args:
        category_stats: Dict of category -> stats from aggregate_validation_metrics
        epoch: Current epoch number
    """
    if not category_stats:
        return
    
    # Force newlines to separate from progress bar
    print("\n\n")
    sys.stdout.flush()
    
    print("="*140)
    print(f"PER-CATEGORY VALIDATION ACCURACY (Epoch {epoch})")
    print("="*140)
    print(f"{'Category':<12} {'Grids':<8} {'Grid Acc':<12} {'Cell Acc':<12} {'Copy Rate':<12} {'Ch Recall':<12} {'Trans Qual':<12}")
    print("-"*140)
    
    for category in sorted(category_stats.keys()):
        stats = category_stats[category]
        grid_acc = (stats['grid_correct'] / stats['grid_total'] * 100) if stats['grid_total'] > 0 else 0
        cell_acc = (stats['cell_correct'] / stats['cell_total'] * 100) if stats['cell_total'] > 0 else 0
        copy_rate = (stats['copy_rate_sum'] / stats['metric_count']) if stats['metric_count'] > 0 else 0
        change_recall = (stats['change_recall_sum'] / stats['metric_count']) if stats['metric_count'] > 0 else 0
        trans_quality = (stats['trans_quality_sum'] / stats['metric_count']) if stats['metric_count'] > 0 else 0
        print(f"{category:<12} {stats['grid_total']:<8} {grid_acc:>10.2f}%  {cell_acc:>10.2f}%  {copy_rate:>10.2f}%  {change_recall:>10.2f}%  {trans_quality:>10.4f}")
    
    # Overall
    total_grid_correct = sum(s['grid_correct'] for s in category_stats.values())
    total_grids = sum(s['grid_total'] for s in category_stats.values())
    total_cell_correct = sum(s['cell_correct'] for s in category_stats.values())
    total_cells = sum(s['cell_total'] for s in category_stats.values())
    total_copy_rate = sum(s['copy_rate_sum'] for s in category_stats.values())
    total_change_recall = sum(s['change_recall_sum'] for s in category_stats.values())
    total_trans_quality = sum(s['trans_quality_sum'] for s in category_stats.values())
    total_metric_count = sum(s['metric_count'] for s in category_stats.values())
    
    overall_grid_acc = (total_grid_correct / total_grids * 100) if total_grids > 0 else 0
    overall_cell_acc = (total_cell_correct / total_cells * 100) if total_cells > 0 else 0
    overall_copy_rate = (total_copy_rate / total_metric_count) if total_metric_count > 0 else 0
    overall_change_recall = (total_change_recall / total_metric_count) if total_metric_count > 0 else 0
    overall_trans_quality = (total_trans_quality / total_metric_count) if total_metric_count > 0 else 0
    
    print("-"*140)
    print(f"{'OVERALL':<12} {total_grids:<8} {overall_grid_acc:>10.2f}%  {overall_cell_acc:>10.2f}%  {overall_copy_rate:>10.2f}%  {overall_change_recall:>10.2f}%  {overall_trans_quality:>10.4f}")
    print("="*140)
    print("\n")
    
    # Explicit flush to ensure it appears in Paperspace logs
    sys.stdout.flush()
