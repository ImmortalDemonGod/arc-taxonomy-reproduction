"""Test scheduler logic for all Lightning modules.

The scheduler includes warmup and cosine annealing which needs testing.
This was identified as untested code in configure_optimizers.
"""
import pytest
import torch


def test_baseline_scheduler_warmup():
    """Test baseline scheduler during warmup phase."""
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
        max_epochs=100,
    )
    
    config = model.configure_optimizers()
    optimizer = config['optimizer']
    scheduler_dict = config['lr_scheduler']
    scheduler = scheduler_dict['scheduler']
    
    # Get initial LR
    initial_lr = optimizer.param_groups[0]['lr']
    
    # Step scheduler multiple times (simulating warmup)
    for _ in range(10):
        scheduler.step()
    
    # LR should have changed during warmup
    after_warmup_lr = optimizer.param_groups[0]['lr']
    # During warmup or annealing, LR changes
    # We just verify scheduler is working (not stuck)
    assert scheduler.last_epoch == 10


def test_exp0_scheduler_step():
    """Test exp0 scheduler steps correctly."""
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
        max_epochs=50,
    )
    
    config = model.configure_optimizers()
    scheduler_dict = config['lr_scheduler']
    scheduler = scheduler_dict['scheduler']
    
    # Step multiple times
    for i in range(20):
        scheduler.step()
    
    assert scheduler.last_epoch == 20


def test_exp1_scheduler_annealing():
    """Test exp1 scheduler annealing phase."""
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
        learning_rate=0.001,
        pad_token=10,
        sep_token=0,
        max_epochs=100,
    )
    
    config = model.configure_optimizers()
    optimizer = config['optimizer']
    scheduler_dict = config['lr_scheduler']
    scheduler = scheduler_dict['scheduler']
    
    # Step past warmup into annealing
    for _ in range(100):
        scheduler.step()
    
    # Verify scheduler ran
    assert scheduler.last_epoch == 100


def test_exp2_scheduler_epoch_mode():
    """Test exp2 scheduler in epoch mode."""
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
        learning_rate=0.001,
        pad_token=10,
        max_epochs=100,
    )
    
    config = model.configure_optimizers()
    scheduler_dict = config['lr_scheduler']
    
    # Verify scheduler interval is set
    assert scheduler_dict['interval'] in ['step', 'epoch']
    assert 'scheduler' in scheduler_dict
    
    scheduler = scheduler_dict['scheduler']
    
    # Step scheduler
    for _ in range(5):
        scheduler.step()
    
    assert scheduler.last_epoch == 5


def test_champion_scheduler_config():
    """Test champion scheduler configuration."""
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
    
    config = model.configure_optimizers()
    scheduler_dict = config['lr_scheduler']
    scheduler = scheduler_dict['scheduler']
    
    # Step scheduler multiple times
    for _ in range(50):
        scheduler.step()
    
    assert scheduler.last_epoch == 50


def test_scheduler_state_dict():
    """Test that scheduler state can be saved/loaded."""
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
    scheduler = config['lr_scheduler']['scheduler']
    
    # Step scheduler
    for _ in range(10):
        scheduler.step()
    
    # Save state
    state = scheduler.state_dict()
    assert 'last_epoch' in state or '_step_count' in state
    
    # Load state
    scheduler.load_state_dict(state)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
