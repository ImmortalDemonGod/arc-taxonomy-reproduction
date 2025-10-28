"""Integration tests for Lightning module training steps.

Tests that training_step actually works without crashes.
These tests should have been written BEFORE the validation tests.

Critical: Tests MUST cover training_step, not just validation_step.
"""
import pytest
import torch
from pathlib import Path


@pytest.fixture
def mock_batch_decoder_only():
    """Create mock batch for decoder-only models."""
    batch_size = 4
    seq_len = 50
    sequences = torch.randint(0, 11, (batch_size, seq_len))
    task_ids = ['test_task_1', 'test_task_2', 'test_task_3', 'test_task_4']
    return sequences, task_ids


@pytest.fixture
def mock_batch_encoder_decoder():
    """Create mock batch for encoder-decoder models."""
    batch_size = 4
    src_len = 50
    tgt_len = 50
    src = torch.randint(0, 11, (batch_size, src_len))
    tgt = torch.randint(0, 11, (batch_size, tgt_len))
    task_ids = ['test_task_1', 'test_task_2', 'test_task_3', 'test_task_4']
    return src, tgt, task_ids


def test_baseline_training_step(mock_batch_decoder_only):
    """Test baseline decoder-only training_step doesn't crash."""
    from src.models.baseline_decoder_only_lightning import BaselineDecoderOnlyLightningModule
    
    # Create model
    model = BaselineDecoderOnlyLightningModule(
        vocab_size=11,
        d_model=64,
        num_layers=2,
        num_heads=2,
        d_ff=128,
        dropout=0.1,
        learning_rate=0.001,
        pad_token=10,
    )
    
    # Run training step
    loss = model.training_step(mock_batch_decoder_only, batch_idx=0)
    
    # Verify outputs
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_exp0_training_step(mock_batch_encoder_decoder):
    """Test exp0 encoder-decoder training_step doesn't crash."""
    from src.models.exp0_encoder_decoder_lightning import Exp0EncoderDecoderLightningModule
    
    # Create model
    model = Exp0EncoderDecoderLightningModule(
        vocab_size=11,
        d_model=64,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_heads=2,
        d_ff=128,
        dropout=0.1,
        learning_rate=0.001,
        pad_token=10,
    )
    
    # Run training step
    loss = model.training_step(mock_batch_encoder_decoder, batch_idx=0)
    
    # Verify outputs
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_exp1_training_step(mock_batch_encoder_decoder):
    """Test exp1 Grid2D PE training_step doesn't crash."""
    from src.models.exp1_grid2d_pe_lightning import Exp1Grid2DPELightningModule
    
    # Create model
    model = Exp1Grid2DPELightningModule(
        vocab_size=11,
        d_model=64,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_heads=2,
        d_ff=128,
        max_grid_size=30,
        dropout=0.1,
        learning_rate=0.001,
        pad_token=10,
        sep_token=0,
    )
    
    # Run training step
    loss = model.training_step(mock_batch_encoder_decoder, batch_idx=0)
    
    # Verify outputs
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_exp2_training_step(mock_batch_encoder_decoder):
    """Test exp2 PermInv training_step doesn't crash."""
    from src.models.exp2_perminv_lightning import Exp2PermInvLightningModule
    
    # Create model
    model = Exp2PermInvLightningModule(
        vocab_size=11,
        d_model=64,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_heads=2,
        d_ff=128,
        max_grid_size=30,
        dropout=0.1,
        learning_rate=0.001,
        pad_token=10,
    )
    
    # Run training step
    loss = model.training_step(mock_batch_encoder_decoder, batch_idx=0)
    
    # Verify outputs
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_baseline_forward_backward_pass(mock_batch_decoder_only):
    """Test baseline can do forward + backward pass."""
    from src.models.baseline_decoder_only_lightning import BaselineDecoderOnlyLightningModule
    
    model = BaselineDecoderOnlyLightningModule(
        vocab_size=11,
        d_model=64,
        num_layers=2,
        num_heads=2,
        d_ff=128,
        dropout=0.1,
        learning_rate=0.001,
        pad_token=10,
    )
    
    # Forward pass
    loss = model.training_step(mock_batch_decoder_only, batch_idx=0)
    
    # Backward pass
    loss.backward()
    
    # Check gradients exist
    has_gradients = False
    for param in model.parameters():
        if param.grad is not None:
            has_gradients = True
            break
    
    assert has_gradients, "No gradients computed"


def test_exp0_forward_backward_pass(mock_batch_encoder_decoder):
    """Test exp0 can do forward + backward pass."""
    from src.models.exp0_encoder_decoder_lightning import Exp0EncoderDecoderLightningModule
    
    model = Exp0EncoderDecoderLightningModule(
        vocab_size=11,
        d_model=64,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_heads=2,
        d_ff=128,
        dropout=0.1,
        learning_rate=0.001,
        pad_token=10,
    )
    
    # Forward pass
    loss = model.training_step(mock_batch_encoder_decoder, batch_idx=0)
    
    # Backward pass
    loss.backward()
    
    # Check gradients exist
    has_gradients = False
    for param in model.parameters():
        if param.grad is not None:
            has_gradients = True
            break
    
    assert has_gradients, "No gradients computed"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
