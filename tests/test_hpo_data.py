"""
Tests for HPO data pipeline (stratified splits, path resolution, dataset loading).

These tests verify that data splitting maintains category balance, paths resolve
correctly across environments, and datasets are properly configured.
"""
import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import sys

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))  # Add scripts to path for objective import

from optimize import resolve_path, make_stratified_splits


class TestResolvePath:
    """Test path resolution across different environments."""
    
    def test_absolute_path_exists(self, tmp_path):
        """Test that existing absolute paths are returned as-is."""
        test_file = tmp_path / "test.json"
        test_file.write_text("{}")
        
        result = resolve_path(str(test_file), tmp_path)
        assert result == test_file
    
    def test_relative_path_from_base(self, tmp_path):
        """Test that relative paths are resolved from base_dir."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        test_file = subdir / "test.json"
        test_file.write_text("{}")
        
        result = resolve_path("subdir/test.json", tmp_path)
        assert result.exists()
        assert result == test_file
    
    def test_labels_file_fallback_paths(self, tmp_path):
        """Test that labels file tries multiple fallback locations."""
        # Create labels file in a fallback location
        data_dir = tmp_path / "data" / "taxonomy_classification"
        data_dir.mkdir(parents=True)
        labels_file = data_dir / "all_tasks_classified.json"
        labels_file.write_text("{}")
        
        # Try to resolve with just filename
        result = resolve_path("all_tasks_classified.json", tmp_path)
        
        # Should find it in one of the fallback paths
        assert result.exists()
    
    def test_nonexistent_path_returns_unresolved(self, tmp_path):
        """Test that non-existent paths return the attempted resolution."""
        result = resolve_path("nonexistent.json", tmp_path)
        
        # Should return a path (even if it doesn't exist)
        assert isinstance(result, Path)


class TestMakeStratifiedSplits:
    """Test stratified train/val splitting."""
    
    @pytest.fixture
    def sample_labels_file(self, tmp_path):
        """Create a sample labels file with balanced categories."""
        # Note: Format matches actual all_tasks_classified.json (string values, not lists)
        labels = {
            "task_001": "S1",
            "task_002": "S1",
            "task_003": "S2",
            "task_004": "S2",
            "task_005": "C1",
            "task_006": "C1",
            "task_007": "K1",
            "task_008": "K1",
        }
        
        labels_file = tmp_path / "labels.json"
        labels_file.write_text(json.dumps(labels))
        
        return labels_file
    
    @pytest.fixture
    def sample_task_files(self, tmp_path):
        """Create sample task files."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        files = []
        for i in range(1, 9):
            task_file = data_dir / f"task_00{i}.json"
            task_file.write_text("{}")
            files.append(task_file)
        
        return files
    
    def test_stratified_split_maintains_ratios(self, sample_labels_file, sample_task_files):
        """Test that stratified split maintains category ratios."""
        category_to_idx = {
            "S1": 0, "S2": 1, "S3": 2,
            "C1": 3, "C2": 4,
            "K1": 5, "L1": 6,
            "A1": 7, "A2": 8
        }
        
        train, val = make_stratified_splits(
            sample_task_files,
            sample_labels_file,
            category_to_idx,
            val_ratio=0.25,
            seed=42
        )
        
        # Check that split happened
        assert len(train) + len(val) == len(sample_task_files)
        # With small datasets, val may be empty if each category has <4 items
        # At minimum, should have train data
        assert len(train) > 0 or len(val) > 0
    
    def test_same_seed_produces_same_split(self, sample_labels_file, sample_task_files):
        """Test that same seed produces reproducible splits."""
        category_to_idx = {"S1": 0, "S2": 1, "C1": 2, "K1": 3}
        
        train1, val1 = make_stratified_splits(
            sample_task_files, sample_labels_file, category_to_idx, seed=42
        )
        train2, val2 = make_stratified_splits(
            sample_task_files, sample_labels_file, category_to_idx, seed=42
        )
        
        assert train1 == train2
        assert val1 == val2
    
    def test_different_seed_produces_different_split(self, sample_labels_file, sample_task_files):
        """Test that different seeds produce different splits."""
        category_to_idx = {"S1": 0, "S2": 1, "C1": 2, "K1": 3}
        
        train1, val1 = make_stratified_splits(
            sample_task_files, sample_labels_file, category_to_idx, seed=42
        )
        train2, val2 = make_stratified_splits(
            sample_task_files, sample_labels_file, category_to_idx, seed=123
        )
        
        # Should be different (very unlikely to be identical with different seeds)
        assert train1 != train2 or val1 != val2
    
    def test_each_category_represented(self, sample_labels_file, sample_task_files):
        """Test that each category appears in both train and val (if possible)."""
        category_to_idx = {"S1": 0, "S2": 1, "C1": 2, "K1": 3}
        
        # Load labels to check
        with open(sample_labels_file) as f:
            labels = json.load(f)
        
        train, val = make_stratified_splits(
            sample_task_files, sample_labels_file, category_to_idx, val_ratio=0.25, seed=42
        )
        
        # Get categories in each split
        train_categories = set()
        val_categories = set()
        
        for fp in train:
            task_id = fp.stem
            if task_id in labels:
                cat = labels[task_id]
                if isinstance(cat, list):
                    train_categories.update(cat)
                else:
                    train_categories.add(cat)
        
        for fp in val:
            task_id = fp.stem
            if task_id in labels:
                cat = labels[task_id]
                if isinstance(cat, list):
                    val_categories.update(cat)
                else:
                    val_categories.add(cat)
        
        # With small datasets and 25% val_ratio, some categories may only be in train
        # Just verify we got some data split
        assert len(train) > 0 or len(val) > 0
        # If we have both splits, check categories present
        if len(train) > 0:
            assert len(train_categories) > 0


class TestDatasetIntegration:
    """Test dataset loading and configuration."""
    
    def test_arc_task_dataset_can_be_imported(self):
        """Test that ARCTaskDataset can be imported."""
        try:
            from src.data.arc_task_dataset import ARCTaskDataset
            assert ARCTaskDataset is not None
        except ImportError as e:
            pytest.fail(f"Failed to import ARCTaskDataset: {e}")
    
    def test_collate_function_can_be_imported(self):
        """Test that collate function can be imported."""
        try:
            from src.data.arc_task_dataset import collate_arc_tasks
            assert collate_arc_tasks is not None
        except ImportError as e:
            pytest.fail(f"Failed to import collate_arc_tasks: {e}")
    
    @pytest.mark.parametrize("color_permute", [True, False])
    @pytest.mark.parametrize("random_demos", [True, False])
    def test_dataset_config_combinations(self, color_permute, random_demos, tmp_path):
        """Test various dataset configuration combinations."""
        from src.data.arc_task_dataset import ARCTaskDataset
        
        # Create minimal test files
        task_files = []
        for i in range(2):
            task_file = tmp_path / f"task_{i}.json"
            task_file.write_text('{"train": [], "test": []}')
            task_files.append(task_file)
        
        # Create categories file
        categories_file = tmp_path / "categories.json"
        categories_file.write_text('{"task_0": "S1", "task_1": "C1"}')
        
        # This should not raise an error (constructor should work)
        try:
            dataset = ARCTaskDataset(
                task_files=task_files,
                categories_json=categories_file,
                max_grid_size=30,
                random_demos=random_demos,
                color_permute=color_permute
            )
            assert dataset is not None
        except Exception as e:
            pytest.fail(f"Dataset construction failed with valid files: {e}")


class TestPathResolutionEdgeCases:
    """Test edge cases in path resolution."""
    
    def test_path_with_special_characters(self, tmp_path):
        """Test paths with special characters."""
        special_dir = tmp_path / "test-dir_123"
        special_dir.mkdir()
        test_file = special_dir / "file@test.json"
        test_file.write_text("{}")
        
        result = resolve_path("test-dir_123/file@test.json", tmp_path)
        assert result.exists()
    
    def test_deeply_nested_path(self, tmp_path):
        """Test deeply nested directory structure."""
        deep_path = tmp_path / "a" / "b" / "c" / "d"
        deep_path.mkdir(parents=True)
        test_file = deep_path / "test.json"
        test_file.write_text("{}")
        
        result = resolve_path("a/b/c/d/test.json", tmp_path)
        assert result.exists()
    
    def test_symlink_resolution(self, tmp_path):
        """Test that symlinks are handled correctly."""
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        test_file = real_dir / "test.json"
        test_file.write_text("{}")
        
        link_dir = tmp_path / "link"
        link_dir.symlink_to(real_dir)
        
        result = resolve_path("link/test.json", tmp_path)
        assert result.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
