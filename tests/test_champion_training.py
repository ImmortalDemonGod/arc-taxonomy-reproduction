"""Tests for Champion (Exp 3) Lightning module.

The Champion model is the most important model in the ablation study.
This file tests training_step, validation_step, and optimizer configuration.

CRITICAL: This model had 0% coverage before these tests.
"""
import pytest
import torch
from pathlib import Path


@pytest.fixture
def mock_champion_batch():
    """Create mock batch for Champion model.
    
    Champion data loader returns:
    (src, tgt, ctx_input, ctx_output, src_shapes, tgt_shapes, task_ids)
    
    CRITICAL: Context pairs are 2D grids [B, num_pairs, H, W], not flattened!
    """
    batch_size = 4
    num_context_pairs = 2
    query_len = 50
    grid_h = 10
    grid_w = 10
    
    # Mock query src/tgt: [batch, seq_len]
    src = torch.randint(0, 11, (batch_size, query_len))
    tgt = torch.randint(0, 11, (batch_size, query_len))
    
    # Mock context pairs as 2D grids: [batch, num_pairs, H, W]
    ctx_input = torch.randint(0, 11, (batch_size, num_context_pairs, grid_h, grid_w))
    ctx_output = torch.randint(0, 11, (batch_size, num_context_pairs, grid_h, grid_w))
    
    # Mock shapes (as lists of tuples)
    src_shapes = [(10, 5) for _ in range(batch_size)]
    tgt_shapes = [(10, 5) for _ in range(batch_size)]
    
    # Task IDs
    task_ids = ['test_task_1', 'test_task_2', 'test_task_3', 'test_task_4']
    
    return src, tgt, ctx_input, ctx_output, src_shapes, tgt_shapes, task_ids


def test_champion_instantiation():
    """Test Champion model can be instantiated."""
    from src.models.exp3_champion_lightning import Exp3ChampionLightningModule
    
    model = Exp3ChampionLightningModule(
        vocab_size=11,
        d_model=64,
        num_encoder_layers=1,
        num_decoder_layers=2,
        num_heads=2,
        d_ff=128,
        max_grid_size=30,
        dropout=0.1,
        learning_rate=0.001,
    )
    
    assert model is not None
    assert model.hparams.vocab_size == 11
    assert model.hparams.d_model == 64


def test_champion_training_step(mock_champion_batch):
    """Test Champion training_step doesn't crash."""
    from src.models.exp3_champion_lightning import Exp3ChampionLightningModule
    
    model = Exp3ChampionLightningModule(
        vocab_size=11,
        d_model=64,
        num_encoder_layers=1,
        num_decoder_layers=2,
        num_heads=2,
        d_ff=128,
        max_grid_size=30,
        dropout=0.1,
        learning_rate=0.001,
    )
    
    # Run training step
    loss = model.training_step(mock_champion_batch, batch_idx=0)
    
    # Verify outputs
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_champion_validation_step(mock_champion_batch):
    """Test Champion validation_step doesn't crash."""
    from src.models.exp3_champion_lightning import Exp3ChampionLightningModule
    
    model = Exp3ChampionLightningModule(
        vocab_size=11,
        d_model=64,
        num_encoder_layers=1,
        num_decoder_layers=2,
        num_heads=2,
        d_ff=128,
        max_grid_size=30,
        dropout=0.1,
        learning_rate=0.001,
    )
    
    # Run validation step
    loss = model.validation_step(mock_champion_batch, batch_idx=0)
    
    # Verify outputs
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar
    assert len(model.validation_step_outputs) == 1
    
    # Verify stored metrics
    output = model.validation_step_outputs[0]
    assert 'task_ids' in output
    assert 'grid_correct' in output
    assert 'cell_correct_counts' in output
    assert 'cell_total_counts' in output


def test_champion_forward_backward(mock_champion_batch):
    """Test Champion can do forward + backward pass."""
    from src.models.exp3_champion_lightning import Exp3ChampionLightningModule
    
    model = Exp3ChampionLightningModule(
        vocab_size=11,
        d_model=64,
        num_encoder_layers=1,
        num_decoder_layers=2,
        num_heads=2,
        d_ff=128,
        max_grid_size=30,
        dropout=0.1,
        learning_rate=0.001,
    )
    
    # Forward pass
    loss = model.training_step(mock_champion_batch, batch_idx=0)
    
    # Backward pass
    loss.backward()
    
    # Check gradients exist
    has_gradients = False
    for param in model.parameters():
        if param.grad is not None:
            has_gradients = True
            break
    
    assert has_gradients, "No gradients computed"


def test_champion_configure_optimizers():
    """Test Champion configure_optimizers returns valid optimizer."""
    from src.models.exp3_champion_lightning import Exp3ChampionLightningModule
    
    model = Exp3ChampionLightningModule(
        vocab_size=11,
        d_model=64,
        num_encoder_layers=1,
        num_decoder_layers=2,
        num_heads=2,
        d_ff=128,
        max_grid_size=30,
        dropout=0.1,
        learning_rate=0.001,
    )
    
    # Get optimizer config
    config = model.configure_optimizers()
    
    # Verify structure
    assert 'optimizer' in config
    assert 'lr_scheduler' in config
    
    optimizer = config['optimizer']
    scheduler = config['lr_scheduler']
    
    # Verify optimizer
    assert optimizer is not None
    assert hasattr(optimizer, 'step')
    assert hasattr(optimizer, 'param_groups')
    assert len(optimizer.param_groups) > 0
    assert optimizer.param_groups[0]['lr'] == 0.001
    
    # Verify scheduler
    assert scheduler is not None
    assert 'scheduler' in scheduler
    assert 'interval' in scheduler
    assert scheduler['interval'] in ['step', 'epoch']  # Either is valid


def test_champion_has_epoch_end_method():
    """Test Champion has on_validation_epoch_end method."""
    from src.models.exp3_champion_lightning import Exp3ChampionLightningModule
    
    model = Exp3ChampionLightningModule(
        vocab_size=11,
        d_model=64,
        num_encoder_layers=1,
        num_decoder_layers=2,
        num_heads=2,
        d_ff=128,
        max_grid_size=30,
        dropout=0.1,
        learning_rate=0.001,
    )
    
    # Verify method exists
    assert hasattr(model, 'on_validation_epoch_end')
    assert callable(model.on_validation_epoch_end)
    
    # Note: Can't test actual execution without Trainer attached


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
