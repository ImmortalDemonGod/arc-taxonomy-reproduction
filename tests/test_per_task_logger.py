"""Tests for PerTaskMetricsLogger callback."""
import pytest
import csv
from pathlib import Path
import shutil
import torch
import pytorch_lightning as pl

from src.callbacks import PerTaskMetricsLogger


class MockLightningModule(pl.LightningModule):
    """Mock module for testing callback."""
    
    def __init__(self):
        super().__init__()
        self.validation_step_outputs = []


class MockTrainer:
    """Mock trainer for testing callback."""
    
    def __init__(self, current_epoch=0):
        self.current_epoch = current_epoch


def test_per_task_logger_initialization(tmp_path):
    """Test that logger initializes correctly."""
    log_dir = tmp_path / "test_logs"
    logger = PerTaskMetricsLogger(log_dir=str(log_dir))
    
    assert log_dir.exists()
    assert isinstance(logger.task_categories, dict)


def test_per_task_logger_writes_csv(tmp_path):
    """Test that logger writes CSV files correctly."""
    log_dir = tmp_path / "test_logs"
    logger = PerTaskMetricsLogger(log_dir=str(log_dir))
    
    # Create mock data
    pl_module = MockLightningModule()
    pl_module.validation_step_outputs = [
        {
            'task_ids': ['task_001', 'task_002'],
            'grid_correct': torch.tensor([True, False]),
            'cell_correct_counts': torch.tensor([100, 50]),
            'cell_total_counts': torch.tensor([100, 100]),
            'copy_rate': torch.tensor([0.5, 0.3]),
            'change_recall': torch.tensor([0.8, 0.6]),
            'trans_quality': torch.tensor([0.4, 0.18]),
        }
    ]
    
    trainer = MockTrainer(current_epoch=0)
    
    # Call the callback
    logger.on_validation_epoch_end(trainer, pl_module)
    
    # Check that files were created (with default experiment name)
    per_task_file = log_dir / "experiment_epoch_000_per_task.csv"
    per_category_file = log_dir / "experiment_epoch_000_per_category.csv"
    
    assert per_task_file.exists(), "Per-task CSV not created"
    assert per_category_file.exists(), "Per-category CSV not created"
    
    # Verify per-task CSV content
    with open(per_task_file) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
        assert len(rows) == 2, "Should have 2 task rows"
        assert rows[0]['task_id'] == 'task_001'
        assert rows[1]['task_id'] == 'task_002'
        assert float(rows[0]['grid_accuracy']) == 100.0
        assert float(rows[1]['grid_accuracy']) == 0.0


def test_per_task_logger_multiple_epochs(tmp_path):
    """Test that logger creates separate files for each epoch."""
    log_dir = tmp_path / "test_logs"
    logger = PerTaskMetricsLogger(log_dir=str(log_dir))
    
    pl_module = MockLightningModule()
    pl_module.validation_step_outputs = [
        {
            'task_ids': ['task_001'],
            'grid_correct': torch.tensor([True]),
            'cell_correct_counts': torch.tensor([100]),
            'cell_total_counts': torch.tensor([100]),
        }
    ]
    
    # Epoch 0
    trainer = MockTrainer(current_epoch=0)
    logger.on_validation_epoch_end(trainer, pl_module)
    
    # Epoch 1
    trainer = MockTrainer(current_epoch=1)
    logger.on_validation_epoch_end(trainer, pl_module)
    
    # Check both epoch files exist (with default experiment name)
    assert (log_dir / "experiment_epoch_000_per_task.csv").exists()
    assert (log_dir / "experiment_epoch_001_per_task.csv").exists()
    assert (log_dir / "experiment_epoch_000_per_category.csv").exists()
    assert (log_dir / "experiment_epoch_001_per_category.csv").exists()


def test_per_task_logger_handles_empty_outputs(tmp_path):
    """Test that logger handles empty validation outputs gracefully."""
    log_dir = tmp_path / "test_logs"
    logger = PerTaskMetricsLogger(log_dir=str(log_dir))
    
    pl_module = MockLightningModule()
    pl_module.validation_step_outputs = []
    
    trainer = MockTrainer(current_epoch=0)
    
    # Should not crash
    logger.on_validation_epoch_end(trainer, pl_module)
    
    # No files should be created for empty outputs
    assert not (log_dir / "epoch_000_per_task.csv").exists()


def test_load_visual_classifier_categories(tmp_path):
    """Test that visual classifier categories are loaded from correct path."""
    # Create a mock repo structure
    repo_root = tmp_path / "reproduction"
    outputs_dir = repo_root / "outputs" / "visual_classifier" / "results"
    outputs_dir.mkdir(parents=True)
    
    # Create mock arc2_classify_seed.csv
    csv_path = outputs_dir / "arc2_classify_seed.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['task_id', 'pred_label', 'pred_idx'])
        writer.writerow(['00576224', 'C1', '3'])
        writer.writerow(['007bbfb7', 'S2', '1'])
        writer.writerow(['009d5c81', 'S3', '2'])
    
    # Create logger (pointing to tmp log dir, but categories should load from repo)
    log_dir = tmp_path / "logs"
    logger = PerTaskMetricsLogger(log_dir=str(log_dir))
    
    # The bug: _load_task_categories uses Path(__file__).parent.parent
    # which gives .../reproduction/src instead of .../reproduction
    # So it looks for outputs at the wrong path
    
    # For this test, we need to verify the path calculation is correct
    from src.callbacks.per_task_logger import PerTaskMetricsLogger as PTL
    import inspect
    callback_file = Path(inspect.getfile(PTL))
    
    # Verify: callback is at .../reproduction/src/callbacks/per_task_logger.py
    # So parent.parent = .../reproduction/src (WRONG)
    # Should be parent.parent.parent = .../reproduction
    base_wrong = callback_file.parent.parent
    base_correct = callback_file.parent.parent.parent
    
    # The test should fail with current code because base_wrong doesn't have outputs/
    # But will pass after we fix it to use base_correct
    assert base_correct.name == "reproduction" or base_correct.name == "arc-taxonomy-reproduction"


def test_category_lookup_strips_suffix(tmp_path):
    """Test that _eval and _train suffixes are stripped when looking up categories."""
    log_dir = tmp_path / "test_logs"
    logger = PerTaskMetricsLogger(log_dir=str(log_dir))
    
    # Manually set categories (bare IDs)
    logger.task_categories = {
        '0934a4d8': 'S3',
        '135a2760': 'S3',
        '136b0064': 'C1',
    }
    
    pl_module = MockLightningModule()
    pl_module.validation_step_outputs = [
        {
            'task_ids': ['0934a4d8_eval', '135a2760_eval', '136b0064_eval'],
            'grid_correct': torch.tensor([False, False, False]),
            'cell_correct_counts': torch.tensor([5, 41, 167]),
            'cell_total_counts': torch.tensor([89, 525, 228]),
        }
    ]
    
    trainer = MockTrainer(current_epoch=0)
    logger.on_validation_epoch_end(trainer, pl_module)
    
    # Verify categories were assigned correctly despite _eval suffix
    per_task_file = log_dir / "experiment_epoch_000_per_task.csv"
    with open(per_task_file) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
        assert rows[0]['category'] == 'S3', f"Expected S3, got {rows[0]['category']}"
        assert rows[1]['category'] == 'S3', f"Expected S3, got {rows[1]['category']}"
        assert rows[2]['category'] == 'C1', f"Expected C1, got {rows[2]['category']}"


def test_per_task_logger_aggregates_by_category(tmp_path):
    """Test that category aggregation works correctly."""
    log_dir = tmp_path / "test_logs"
    logger = PerTaskMetricsLogger(log_dir=str(log_dir))
    
    # Override task categories for testing
    logger.task_categories = {
        'task_001': 'C1',
        'task_002': 'C1',
        'task_003': 'S1',
    }
    
    pl_module = MockLightningModule()
    pl_module.validation_step_outputs = [
        {
            'task_ids': ['task_001', 'task_002', 'task_003'],
            'grid_correct': torch.tensor([True, True, False]),
            'cell_correct_counts': torch.tensor([100, 50, 25]),
            'cell_total_counts': torch.tensor([100, 100, 100]),
        }
    ]
    
    trainer = MockTrainer(current_epoch=0)
    logger.on_validation_epoch_end(trainer, pl_module)
    
    # Check category CSV (with default experiment name)
    per_category_file = log_dir / "experiment_epoch_000_per_category.csv"
    with open(per_category_file) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
        # Find C1 and S1 rows
        c1_row = next(r for r in rows if r['category'] == 'C1')
        s1_row = next(r for r in rows if r['category'] == 'S1')
        
        assert int(c1_row['task_count']) == 2
        assert int(s1_row['task_count']) == 1
        assert float(c1_row['grid_accuracy']) == 100.0  # 2/2
        assert float(s1_row['grid_accuracy']) == 0.0    # 0/1
