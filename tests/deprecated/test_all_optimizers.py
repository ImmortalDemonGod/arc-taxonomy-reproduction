"""Test configure_optimizers for all Lightning modules.

All models must have properly configured optimizers and schedulers.
This was identified as untested code (missing 20-26% coverage per model).
"""
import pytest
import torch


def test_baseline_configure_optimizers():
    """Test baseline optimizer configuration."""
    from src.models.baseline_decoder_only_lightning import BaselineDecoderOnlyLightningModule
    
    model = BaselineDecoderOnlyLightningModule(
        vocab_size=11,
        d_model=64,
        num_layers=2,
        num_heads=2,
        d_ff=128,
        dropout=0.1,
        learning_rate=0.001,
        weight_decay=0.01,
        beta1=0.9,
        beta2=0.999,
        max_epochs=100,
        pad_token=10,
    )
    
    config = model.configure_optimizers()
    
    # Verify structure
    assert 'optimizer' in config
    assert 'lr_scheduler' in config
    
    optimizer = config['optimizer']
    scheduler_config = config['lr_scheduler']
    
    # Verify optimizer
    assert optimizer is not None
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]['lr'] == 0.001
    assert optimizer.param_groups[0]['weight_decay'] == 0.01
    assert optimizer.param_groups[0]['betas'] == (0.9, 0.999)
    
    # Verify scheduler
    assert 'scheduler' in scheduler_config
    assert 'interval' in scheduler_config
    assert scheduler_config['interval'] in ['step', 'epoch']  # Either is valid


def test_exp0_configure_optimizers():
    """Test exp0 optimizer configuration."""
    from src.models.exp0_encoder_decoder_lightning import Exp0EncoderDecoderLightningModule
    
    model = Exp0EncoderDecoderLightningModule(
        vocab_size=11,
        d_model=64,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_heads=2,
        d_ff=128,
        dropout=0.1,
        learning_rate=0.002,
        weight_decay=0.02,
        beta1=0.95,
        beta2=0.998,
        max_epochs=50,
        pad_token=10,
    )
    
    config = model.configure_optimizers()
    
    # Verify structure
    assert 'optimizer' in config
    assert 'lr_scheduler' in config
    
    optimizer = config['optimizer']
    scheduler_config = config['lr_scheduler']
    
    # Verify optimizer
    assert optimizer is not None
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]['lr'] == 0.002
    assert optimizer.param_groups[0]['weight_decay'] == 0.02
    assert optimizer.param_groups[0]['betas'] == (0.95, 0.998)
    
    # Verify scheduler
    assert 'scheduler' in scheduler_config
    assert 'interval' in scheduler_config
    assert scheduler_config['interval'] in ['step', 'epoch']  # Either is valid


def test_exp1_configure_optimizers():
    """Test exp1 optimizer configuration."""
    from src.models.exp1_grid2d_pe_lightning import Exp1Grid2DPELightningModule
    
    model = Exp1Grid2DPELightningModule(
        vocab_size=11,
        d_model=64,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_heads=2,
        d_ff=128,
        max_grid_size=30,
        dropout=0.1,
        learning_rate=0.0015,
        weight_decay=0.0,
        beta1=0.9,
        beta2=0.999,
        max_epochs=100,
        pad_token=10,
        sep_token=0,
    )
    
    config = model.configure_optimizers()
    
    # Verify structure
    assert 'optimizer' in config
    assert 'lr_scheduler' in config
    
    optimizer = config['optimizer']
    scheduler_config = config['lr_scheduler']
    
    # Verify optimizer
    assert optimizer is not None
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]['lr'] == 0.0015
    assert optimizer.param_groups[0]['weight_decay'] == 0.0
    
    # Verify scheduler
    assert 'scheduler' in scheduler_config
    assert 'interval' in scheduler_config
    assert scheduler_config['interval'] in ['step', 'epoch']  # Either is valid


def test_exp2_configure_optimizers():
    """Test exp2 optimizer configuration."""
    from src.models.exp2_perminv_lightning import Exp2PermInvLightningModule
    
    model = Exp2PermInvLightningModule(
        vocab_size=11,
        d_model=64,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_heads=2,
        d_ff=128,
        max_grid_size=30,
        dropout=0.1,
        learning_rate=0.0018,
        weight_decay=0.0,
        beta1=0.95,
        beta2=0.999,
        max_epochs=100,
        pad_token=10,
    )
    
    config = model.configure_optimizers()
    
    # Verify structure
    assert 'optimizer' in config
    assert 'lr_scheduler' in config
    
    optimizer = config['optimizer']
    scheduler_config = config['lr_scheduler']
    
    # Verify optimizer
    assert optimizer is not None
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]['lr'] == 0.0018
    assert optimizer.param_groups[0]['weight_decay'] == 0.0
    assert optimizer.param_groups[0]['betas'] == (0.95, 0.999)
    
    # Verify scheduler
    assert 'scheduler' in scheduler_config
    assert 'interval' in scheduler_config
    assert scheduler_config['interval'] in ['step', 'epoch']  # Either is valid


def test_all_models_have_configure_optimizers():
    """Test that all 5 models have configure_optimizers method."""
    from src.models.baseline_decoder_only_lightning import BaselineDecoderOnlyLightningModule
    from src.models.exp0_encoder_decoder_lightning import Exp0EncoderDecoderLightningModule
    from src.models.exp1_grid2d_pe_lightning import Exp1Grid2DPELightningModule
    from src.models.exp2_perminv_lightning import Exp2PermInvLightningModule
    from src.models.exp3_champion_lightning import Exp3ChampionLightningModule
    
    models = [
        BaselineDecoderOnlyLightningModule(vocab_size=11, d_model=64, num_layers=2, num_heads=2, d_ff=128, dropout=0.1, learning_rate=0.001, pad_token=10),
        Exp0EncoderDecoderLightningModule(vocab_size=11, d_model=64, num_encoder_layers=2, num_decoder_layers=2, num_heads=2, d_ff=128, dropout=0.1, learning_rate=0.001, pad_token=10),
        Exp1Grid2DPELightningModule(vocab_size=11, d_model=64, num_encoder_layers=2, num_decoder_layers=2, num_heads=2, d_ff=128, max_grid_size=30, dropout=0.1, learning_rate=0.001, pad_token=10, sep_token=0),
        Exp2PermInvLightningModule(vocab_size=11, d_model=64, num_encoder_layers=2, num_decoder_layers=2, num_heads=2, d_ff=128, max_grid_size=30, dropout=0.1, learning_rate=0.001, pad_token=10),
        Exp3ChampionLightningModule(vocab_size=11, d_model=64, num_encoder_layers=1, num_decoder_layers=2, num_heads=2, d_ff=128, max_grid_size=30, dropout=0.1, learning_rate=0.001),
    ]
    
    for model in models:
        assert hasattr(model, 'configure_optimizers')
        assert callable(model.configure_optimizers)
        
        # Verify it returns valid config
        config = model.configure_optimizers()
        assert 'optimizer' in config
        assert 'lr_scheduler' in config


def test_optimizer_can_step():
    """Test that optimizers can actually step (update parameters)."""
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
    
    config = model.configure_optimizers()
    optimizer = config['optimizer']
    
    # Create dummy loss
    dummy_input = torch.randint(0, 11, (2, 10))
    output = model(dummy_input)
    loss = output.mean()
    
    # Backward
    loss.backward()
    
    # Check gradients exist
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad, "No gradients computed"
    
    # Take optimizer step
    optimizer.step()
    optimizer.zero_grad()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
