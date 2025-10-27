"""ARC Taxonomy Reproduction Package - Grid and Transformation Metrics.

Critical metrics for evaluating ARC model performance:
- grid_accuracy: Fraction of completely correct grids (headline metric)
- cell_accuracy: Fraction of correct cells (pixel-level)
- transformation metrics: Change detection and transformation quality

Adapted from jarc_reactor for standalone reproduction.
"""
import torch
from typing import Dict


def compute_grid_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    pad_token: int = 10,
) -> Dict[str, torch.Tensor]:
    """Compute grid-level and cell-level accuracy metrics.
    
    Args:
        predictions: Model predictions [B, H, W] or [B, L]
        targets: Ground truth [B, H, W] or [B, L]
        pad_token: Token ID for padding (default: 10)
        
    Returns:
        dict with:
            - grid_accuracy: Fraction of grids where ALL non-padding cells are correct
            - cell_accuracy: Fraction of non-padding cells that are correct
    """
    # Ensure tensors are long
    predictions = predictions.to(torch.long)
    targets = targets.to(torch.long)
    
    # Create mask for non-padding elements
    valid_mask = (targets != pad_token)
    
    # Cell-wise accuracy
    correct_cells = (predictions == targets) & valid_mask
    cell_accuracy = correct_cells.float().sum() / (valid_mask.float().sum() + 1e-6)
    
    # Full grid accuracy (grid is correct only if ALL non-padded cells match)
    correct_or_pad = (correct_cells | ~valid_mask)
    
    # Support [B,H,W] or [B,T] by flattening non-batch dims
    if correct_or_pad.dim() >= 2:
        grid_matches = correct_or_pad.view(correct_or_pad.size(0), -1).all(dim=1)
    else:
        grid_matches = correct_or_pad.bool()
    
    grid_accuracy = grid_matches.float().mean()
    
    return {
        "grid_accuracy": grid_accuracy,
        "cell_accuracy": cell_accuracy,
    }


def compute_copy_metrics_on_batch(
    src: torch.Tensor,
    tgt: torch.Tensor,
    pred: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Compute copy-bias and transformation quality metrics averaged across a batch of grids.

    All tensors are expected to have shape ``[B, H, W]`` (or ``[B, L]`` which is
    treated identically). If the spatial dimensions differ between ``src``,
    ``tgt`` and ``pred`` the metrics are computed on the overlapping region
    defined by the minimum height/width.

    Returns:
        dict: Contains batch-averaged scalar tensors for:
            - ``copy_rate``: Fraction of cells where ``pred == src``.
            - ``change_recall``: Of cells where ``tgt != src``, the fraction where
              ``pred != src``. When no cells need to change, the recall is 1.0.
            - ``change_precision``: Of cells where ``pred != src``, the fraction where
              ``pred == tgt``. When no predictions differ from src, precision is 1.0.
            - ``transformation_f1``: Harmonic mean of precision and recall.
            - ``pct_changed_target``: Fraction of cells where ``tgt != src``.
            - ``pct_changed_pred``: Fraction of cells where ``pred != src``.
            - ``cell_accuracy``: Fraction of cells where ``pred == tgt``.
    """

    if src.dim() < 2:
        raise ValueError("src must be at least 2D (B, H, W)")

    # Ensure tensors are long for logical comparisons and located on same device.
    device = src.device
    src = src.to(device=device, dtype=torch.long)
    tgt = tgt.to(device=device, dtype=torch.long)
    pred = pred.to(device=device, dtype=torch.long)

    # Align shapes by cropping to the smallest overlapping region if necessary.
    if src.shape != tgt.shape or src.shape != pred.shape:
        if src.size(0) != tgt.size(0) or src.size(0) != pred.size(0):
            raise ValueError("src, tgt, pred must have the same batch size")
        h = min(src.size(-2), tgt.size(-2), pred.size(-2)) if src.dim() > 2 else None
        w = min(src.size(-1), tgt.size(-1), pred.size(-1))
        if src.dim() > 2 and h is not None:
            src = src[..., :h, :w]
            tgt = tgt[..., :h, :w]
            pred = pred[..., :h, :w]
        else:
            src = src[..., :w]
            tgt = tgt[..., :w]
            pred = pred[..., :w]

    # Flatten spatial dimensions for per-sample aggregations.
    batch_size = src.size(0)
    flat_src = src.reshape(batch_size, -1)
    flat_tgt = tgt.reshape(batch_size, -1)
    flat_pred = pred.reshape(batch_size, -1)

    total_cells = flat_src.size(1)
    if total_cells == 0:
        raise ValueError("src, tgt, pred must have non-zero spatial dimensions")

    copy_matches = (flat_pred == flat_src)
    pred_changes = (flat_pred != flat_src)
    target_changes = (flat_tgt != flat_src)
    correct_cells = (flat_pred == flat_tgt)

    copy_rate = copy_matches.float().mean(dim=1)
    pct_changed_pred = pred_changes.float().mean(dim=1)
    pct_changed_target = target_changes.float().mean(dim=1)
    cell_accuracy = correct_cells.float().mean(dim=1)

    target_change_counts = target_changes.sum(dim=1)
    pred_change_counts = pred_changes.sum(dim=1)
    change_attempt_counts = (pred_changes & target_changes).sum(dim=1)
    
    # Change Recall: TP / (TP + FN)
    # When there are no target changes the recall is defined as 1.0 (vacuously true).
    change_recall = torch.where(
        target_change_counts > 0,
        change_attempt_counts.float() / target_change_counts.float(),
        torch.ones_like(target_change_counts, dtype=torch.float32),
    )
    
    # Change Precision: TP / (TP + FP)
    # When model makes no changes, precision is defined as 1.0 (no false positives).
    change_precision = torch.where(
        pred_change_counts > 0,
        change_attempt_counts.float() / pred_change_counts.float(),
        torch.ones_like(pred_change_counts, dtype=torch.float32),
    )
    
    # Transformation F1: Harmonic mean of precision and recall
    # Avoids division by zero with epsilon
    eps = 1e-8
    transformation_f1 = 2.0 * (change_precision * change_recall) / (change_precision + change_recall + eps)

    # Return scalar tensors instead of Python floats to avoid graph breaks with torch.compile
    # PyTorch Lightning's log() method handles tensors properly
    def _mean_as_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Return batch mean as scalar tensor, or zero tensor if empty."""
        return tensor.mean() if tensor.numel() > 0 else torch.zeros((), device=tensor.device, dtype=tensor.dtype)

    return {
        "copy_rate": _mean_as_tensor(copy_rate),
        "change_recall": _mean_as_tensor(change_recall),
        "change_precision": _mean_as_tensor(change_precision),
        "transformation_f1": _mean_as_tensor(transformation_f1),
        "pct_changed_target": _mean_as_tensor(pct_changed_target),
        "pct_changed_pred": _mean_as_tensor(pct_changed_pred),
        "cell_accuracy": _mean_as_tensor(cell_accuracy),
    }
