"""
Tests for HPO training logic (run_training_trial and related functions).

These tests cover the core training loop, model instantiation, optimizer/scheduler
creation, loss computation, and metrics tracking.
"""
import pytest
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import sys

# Add repo root and scripts to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from objective import (
    run_training_trial,
    accuracy_from_logits,
    seed_everything
)


class TestAccuracyFromLogits:
    """Test accuracy computation from logits."""
    
    def test_perfect_accuracy(self):
        """Test accuracy with perfect predictions."""
        logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
        targets = torch.tensor([0, 1])
        
        acc = accuracy_from_logits(logits, targets)
        assert acc == 1.0
    
    def test_zero_accuracy(self):
        """Test accuracy with all wrong predictions."""
        logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
        targets = torch.tensor([1, 0])  # All wrong
        
        acc = accuracy_from_logits(logits, targets)
        assert acc == 0.0
    
    def test_fifty_percent_accuracy(self):
        """Test accuracy with 50% correct."""
        logits = torch.tensor([[10.0, 0.0], [0.0, 10.0], [10.0, 0.0], [0.0, 10.0]])
        targets = torch.tensor([0, 1, 1, 0])  # 2 correct (indices 0,1), 2 wrong (indices 2,3)
        
        acc = accuracy_from_logits(logits, targets)
        assert acc == 0.5
    
    def test_empty_targets(self):
        """Test handling of empty tensors."""
        logits = torch.tensor([]).reshape(0, 3)
        targets = torch.tensor([]).long()
        
        acc = accuracy_from_logits(logits, targets)
        assert acc == 0.0


class TestSeedEverything:
    """Test deterministic seeding."""
    
    def test_seeding_makes_reproducible(self):
        """Test that seeding produces reproducible results."""
        seed_everything(42)
        rand1 = torch.rand(5)
        
        seed_everything(42)
        rand2 = torch.rand(5)
        
        assert torch.allclose(rand1, rand2)
    
    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        seed_everything(42)
        rand1 = torch.rand(5)
        
        seed_everything(123)
        rand2 = torch.rand(5)
        
        assert not torch.allclose(rand1, rand2)


class TestRunTrainingTrial:
    """Test the main training loop."""
    
    @pytest.fixture
    def mock_dataloader(self):
        """Create a mock dataloader with realistic batches."""
        # Create mock batch data
        batch_size = 4
        num_demos = 3
        grid_size = 30
        num_categories = 9
        
        # Mock batch: exactly 3 values (demo_inputs, demo_outputs, category_indices)
        # Shape: (batch_size, num_demos, grid_size, grid_size) - NO channel dimension
        demo_inputs = torch.randint(0, 10, (batch_size, num_demos, grid_size, grid_size))
        demo_outputs = torch.randint(0, 10, (batch_size, num_demos, grid_size, grid_size))
        category_indices = torch.randint(0, num_categories, (batch_size,))
        
        mock_batch = (demo_inputs, demo_outputs, category_indices)  # Exactly 3 values
        
        # Create mock dataloader
        mock_loader = Mock()
        mock_loader.__iter__ = Mock(return_value=iter([mock_batch]))
        mock_loader.__len__ = Mock(return_value=1)
        
        return mock_loader
    
    @pytest.fixture
    def hparams_cnn(self):
        """Hyperparameters for CNN encoder."""
        return {
            'encoder_type': 'cnn',
            'embed_dim': 128,
            'num_cnn_layers': 3,
            'cnn_channels': 64,
            'pool_type': 'avg',
            'lr': 0.001,
            'weight_decay': 1e-5,
            'batch_size': 4,
            'label_smoothing': 0.0,
            'use_scheduler': False,
            'use_cosine_similarity': False,
            'seed': 42,
        }
    
    @pytest.fixture
    def hparams_context(self):
        """Hyperparameters for Context encoder."""
        return {
            'encoder_type': 'context',
            'embed_dim': 256,
            'context_num_heads': 4,
            'context_num_layers': 2,
            'context_ff_dim': 512,
            'context_dropout': 0.1,
            'lr': 0.001,
            'weight_decay': 1e-5,
            'batch_size': 4,
            'label_smoothing': 0.0,
            'use_scheduler': False,
            'use_cosine_similarity': False,
            'seed': 42,
        }
    
    def test_training_trial_completes_cnn(self, mock_dataloader, hparams_cnn):
        """Test that training completes without errors for CNN encoder."""
        num_categories = 9
        centroids = torch.randn(num_categories, 128)  # Match embed_dim
        
        # Reduce to 1 epoch for fast test
        hparams_cnn['epochs'] = 1
        
        with patch('scripts.objective.TaskEncoderCNN') as MockCNN, \
             patch('scripts.objective.TaskEncoderAdvanced'):
            
            # Mock model
            mock_model = Mock(spec=nn.Module)
            mock_model.return_value = torch.randn(4, 128)  # batch_size x embed_dim
            mock_model.train = Mock()
            mock_model.eval = Mock()
            mock_model.to = Mock(return_value=mock_model)
            MockCNN.return_value = mock_model
            
            try:
                metrics = run_training_trial(
                    trial=None,
                    hparams=hparams_cnn,
                    train_loader=mock_dataloader,
                    val_loader=mock_dataloader,
                    centroids=centroids,
                    device='cpu',
                    output_dir=Path('/tmp/test_trial')
                )
                
                # Check that we got metrics back
                assert isinstance(metrics, dict)
                assert 'val_acc' in metrics
                assert 'train_loss' in metrics
                
            except Exception as e:
                pytest.fail(f"Training trial failed: {e}")
    
    def test_training_trial_completes_context(self, mock_dataloader, hparams_context):
        """Test that training completes without errors for Context encoder."""
        num_categories = 9
        centroids = torch.randn(num_categories, 256)  # Match embed_dim
        
        hparams_context['epochs'] = 1
        
        with patch('scripts.objective.TaskEncoderAdvanced') as MockContext, \
             patch('scripts.objective.TaskEncoderCNN'):
            
            # Mock model
            mock_model = Mock(spec=nn.Module)
            mock_model.return_value = torch.randn(4, 256)
            mock_model.train = Mock()
            mock_model.eval = Mock()
            mock_model.to = Mock(return_value=mock_model)
            MockContext.return_value = mock_model
            
            try:
                metrics = run_training_trial(
                    trial=None,
                    hparams=hparams_context,
                    train_loader=mock_dataloader,
                    val_loader=mock_dataloader,
                    centroids=centroids,
                    device='cpu',
                    output_dir=Path('/tmp/test_trial')
                )
                
                assert isinstance(metrics, dict)
                assert 'val_acc' in metrics
                
            except Exception as e:
                pytest.fail(f"Training trial failed: {e}")
    
    def test_cosine_similarity_mode(self, mock_dataloader, hparams_cnn):
        """Test training with cosine similarity instead of dot product."""
        hparams_cnn['use_cosine_similarity'] = True
        hparams_cnn['epochs'] = 1
        
        num_categories = 9
        centroids = torch.randn(num_categories, 128)
        
        with patch('scripts.objective.TaskEncoderCNN') as MockCNN:
            mock_model = Mock(spec=nn.Module)
            mock_model.return_value = torch.randn(4, 128)
            mock_model.train = Mock()
            mock_model.eval = Mock()
            mock_model.to = Mock(return_value=mock_model)
            MockCNN.return_value = mock_model
            
            try:
                metrics = run_training_trial(
                    trial=None,
                    hparams=hparams_cnn,
                    train_loader=mock_dataloader,
                    val_loader=mock_dataloader,
                    centroids=centroids,
                    device='cpu',
                    output_dir=Path('/tmp/test_trial')
                )
                assert 'val_acc' in metrics
            except Exception as e:
                pytest.fail(f"Cosine similarity mode failed: {e}")
    
    def test_scheduler_enabled(self, mock_dataloader, hparams_cnn):
        """Test training with learning rate scheduler."""
        hparams_cnn['use_scheduler'] = True
        hparams_cnn['epochs'] = 1
        
        num_categories = 9
        centroids = torch.randn(num_categories, 128)
        
        with patch('scripts.objective.TaskEncoderCNN') as MockCNN:
            mock_model = Mock(spec=nn.Module)
            mock_model.return_value = torch.randn(4, 128)
            mock_model.train = Mock()
            mock_model.eval = Mock()
            mock_model.to = Mock(return_value=mock_model)
            MockCNN.return_value = mock_model
            
            try:
                metrics = run_training_trial(
                    trial=None,
                    hparams=hparams_cnn,
                    train_loader=mock_dataloader,
                    val_loader=mock_dataloader,
                    centroids=centroids,
                    device='cpu',
                    output_dir=Path('/tmp/test_trial')
                )
                assert 'val_acc' in metrics
            except Exception as e:
                pytest.fail(f"Scheduler mode failed: {e}")
    
    def test_invalid_encoder_type_raises(self, mock_dataloader, hparams_cnn):
        """Test that invalid encoder type raises ValueError."""
        hparams_cnn['encoder_type'] = 'invalid_encoder'
        hparams_cnn['epochs'] = 1
        
        centroids = torch.randn(9, 128)
        
        with pytest.raises(ValueError, match="Unknown encoder_type.*invalid_encoder"):
            run_training_trial(
                trial=None,
                hparams=hparams_cnn,
                train_loader=mock_dataloader,
                val_loader=mock_dataloader,
                centroids=centroids,
                device='cpu',
                output_dir=Path('/tmp/test_trial')
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
