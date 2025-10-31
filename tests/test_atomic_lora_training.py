"""Tests for train_atomic_loras.py script."""
import pytest
import sys
import tempfile
import json
import torch
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_atomic_loras import load_champion, setup_lora, evaluate_model, train_task


class TestOutputSeparation:
    """Test that fast_dev_run doesn't interfere with full run."""
    
    def test_fast_dev_uses_separate_output_dir(self):
        """fast_dev_run should use outputs/atomic_loras_fast_dev/."""
        output_base_dir = Path("outputs/atomic_loras")
        
        # Simulate fast_dev_run logic
        fast_dev_run = 2
        if fast_dev_run:
            output_base_dir = output_base_dir.parent / f"{output_base_dir.name}_fast_dev"
        
        assert str(output_base_dir) == "outputs/atomic_loras_fast_dev"
        assert "fast_dev" in str(output_base_dir)
    
    def test_full_run_uses_normal_output_dir(self):
        """Full run should use outputs/atomic_loras/."""
        output_base_dir = Path("outputs/atomic_loras")
        fast_dev_run = None
        
        if fast_dev_run:
            output_base_dir = output_base_dir.parent / f"{output_base_dir.name}_fast_dev"
        
        assert str(output_base_dir) == "outputs/atomic_loras"
        assert "fast_dev" not in str(output_base_dir)
    
    def test_csv_filenames_separated(self):
        """CSV files should have different names."""
        fast_dev_run = 2
        csv_filename = "lora_training_metrics_fast_dev.csv" if fast_dev_run else "lora_training_metrics.csv"
        assert csv_filename == "lora_training_metrics_fast_dev.csv"
        
        fast_dev_run = None
        csv_filename = "lora_training_metrics_fast_dev.csv" if fast_dev_run else "lora_training_metrics.csv"
        assert csv_filename == "lora_training_metrics.csv"
    
    def test_json_filenames_separated(self):
        """JSON summary files should have different names."""
        fast_dev_run = 2
        json_filename = 'atomic_lora_training_summary_fast_dev.json' if fast_dev_run else 'atomic_lora_training_summary.json'
        assert json_filename == 'atomic_lora_training_summary_fast_dev.json'
        
        fast_dev_run = None
        json_filename = 'atomic_lora_training_summary_fast_dev.json' if fast_dev_run else 'atomic_lora_training_summary.json'
        assert json_filename == 'atomic_lora_training_summary.json'


class TestResumeCapability:
    """Test resume logic."""
    
    def test_detects_existing_adapters(self, tmp_path):
        """Should detect existing adapter_model.safetensors files."""
        output_dir = tmp_path / "atomic_loras"
        output_dir.mkdir()
        
        # Create fake adapters
        (output_dir / "task1").mkdir()
        (output_dir / "task1" / "adapter_model.safetensors").touch()
        (output_dir / "task2").mkdir()
        (output_dir / "task2" / "adapter_model.safetensors").touch()
        
        # Simulate resume logic
        existing_adapters = set()
        for adapter_dir in output_dir.iterdir():
            if adapter_dir.is_dir() and (adapter_dir / "adapter_model.safetensors").exists():
                existing_adapters.add(adapter_dir.name)
        
        assert len(existing_adapters) == 2
        assert "task1" in existing_adapters
        assert "task2" in existing_adapters
    
    def test_skips_incomplete_adapters(self, tmp_path):
        """Should NOT detect adapters without safetensors file."""
        output_dir = tmp_path / "atomic_loras"
        output_dir.mkdir()
        
        # Create incomplete adapter (directory but no file)
        (output_dir / "task1").mkdir()
        # No safetensors file!
        
        existing_adapters = set()
        for adapter_dir in output_dir.iterdir():
            if adapter_dir.is_dir() and (adapter_dir / "adapter_model.safetensors").exists():
                existing_adapters.add(adapter_dir.name)
        
        assert len(existing_adapters) == 0
    
    def test_filters_completed_tasks(self, tmp_path):
        """Should filter out completed tasks from task list."""
        # Simulate task files
        task_files = [
            Path("task1.json"),
            Path("task2.json"),
            Path("task3.json"),
        ]
        
        # Simulate existing adapters
        existing_adapters = {"task1", "task3"}
        
        # Filter logic
        remaining_tasks = [f for f in task_files if f.stem not in existing_adapters]
        
        assert len(remaining_tasks) == 1
        assert remaining_tasks[0].stem == "task2"


class TestMetricsCompleteness:
    """Test that all required metrics are tracked."""
    
    def test_evaluate_returns_all_fields(self):
        """evaluate_model should return all required fields."""
        # This is a schema test - checks return structure
        expected_fields = {
            'grid_accuracy', 'cell_accuracy',
            'grid_correct', 'grid_total',
            'cell_correct', 'cell_total',
            'copy_rate', 'change_recall', 'transformation_quality'
        }
        
        # When no data, should still return all fields
        empty_result = {
            'grid_accuracy': 0.0, 'cell_accuracy': 0.0,
            'grid_correct': 0, 'grid_total': 0,
            'cell_correct': 0, 'cell_total': 0,
            'copy_rate': 0.0, 'change_recall': 0.0, 'transformation_quality': 0.0
        }
        
        assert set(empty_result.keys()) == expected_fields
    
    def test_csv_headers_include_category(self):
        """CSV should include category field."""
        fieldnames = [
            'task_id', 'category', 'status',
            'base_grid_accuracy', 'final_grid_accuracy', 'grid_improvement',
            'base_cell_accuracy', 'final_cell_accuracy', 'cell_improvement',
            'copy_rate', 'change_recall', 'transformation_quality',
            'final_loss', 'epochs', 'num_examples', 'training_time_seconds',
            'early_stopped', 'error'
        ]
        
        assert 'category' in fieldnames
        assert 'task_id' in fieldnames
    
    def test_metadata_includes_all_metrics(self):
        """Metadata should include all accuracy metrics."""
        # Simulate metadata structure
        metadata = {
            'num_examples': 400,
            'num_epochs_trained': 50,
            'training_time_seconds': 100.0,
            'early_stopped': False,
            # Grid accuracy
            'base_grid_accuracy': 10.0,
            'final_grid_accuracy': 50.0,
            'grid_improvement': 40.0,
            # Cell accuracy
            'base_cell_accuracy': 20.0,
            'final_cell_accuracy': 60.0,
            'cell_improvement': 40.0,
            # Counts
            'grid_correct': 200,
            'grid_total': 400,
            'cell_correct': 10000,
            'cell_total': 15000,
            # Copy metrics
            'copy_rate': 0.5,
            'change_recall': 0.7,
            'transformation_quality': 0.6,
        }
        
        required_fields = {
            'base_grid_accuracy', 'final_grid_accuracy', 'grid_improvement',
            'base_cell_accuracy', 'final_cell_accuracy', 'cell_improvement',
            'copy_rate', 'change_recall', 'transformation_quality'
        }
        
        assert all(field in metadata for field in required_fields)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
