"""Integration tests for Lightning module validation steps.

These tests verify that validation_step and on_validation_epoch_end
actually work without KeyErrors or crashes.

Critical: Run these BEFORE deploying to cloud GPU.
"""
import pytest
import torch
from pathlib import Path

# Test data directory
DATA_DIR = Path(__file__).parent.parent / "data" / "distributional_alignment"


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


def test_baseline_validation_step(mock_batch_decoder_only):
    """Test baseline decoder-only validation_step doesn't crash."""
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
    
    # Run validation step
    loss = model.validation_step(mock_batch_decoder_only, batch_idx=0)
    
    # Verify outputs
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar
    assert len(model.validation_step_outputs) == 1
    
    # Verify stored metrics have correct keys
    output = model.validation_step_outputs[0]
    assert 'task_ids' in output
    assert 'grid_correct' in output
    assert 'cell_correct_counts' in output
    assert 'cell_total_counts' in output
    
    # Verify types
    assert isinstance(output['task_ids'], list)
    assert isinstance(output['grid_correct'], torch.Tensor)
    assert isinstance(output['cell_correct_counts'], torch.Tensor)
    assert isinstance(output['cell_total_counts'], torch.Tensor)


def test_exp0_validation_step(mock_batch_encoder_decoder):
    """Test exp0 encoder-decoder validation_step doesn't crash."""
    from src.models.exp0_encoder_decoder_lightning import Exp0EncoderDecoderLightningModule
    
    # Create model (exp0 doesn't have sep_token)
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
    
    # Run validation step
    loss = model.validation_step(mock_batch_encoder_decoder, batch_idx=0)
    
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


def test_exp1_validation_step(mock_batch_encoder_decoder):
    """Test exp1 Grid2D PE validation_step doesn't crash."""
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
    
    # Run validation step
    loss = model.validation_step(mock_batch_encoder_decoder, batch_idx=0)
    
    # Verify outputs
    assert isinstance(loss, torch.Tensor)
    assert len(model.validation_step_outputs) == 1
    
    # Verify stored metrics
    output = model.validation_step_outputs[0]
    assert 'task_ids' in output
    assert 'grid_correct' in output
    assert 'cell_correct_counts' in output
    assert 'cell_total_counts' in output


def test_exp2_validation_step(mock_batch_encoder_decoder):
    """Test exp2 PermInv validation_step doesn't crash."""
    from src.models.exp2_perminv_lightning import Exp2PermInvLightningModule
    
    # Create model (exp2 doesn't have sep_token)
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
    
    # Run validation step
    loss = model.validation_step(mock_batch_encoder_decoder, batch_idx=0)
    
    # Verify outputs
    assert isinstance(loss, torch.Tensor)
    assert len(model.validation_step_outputs) == 1
    
    # Verify stored metrics
    output = model.validation_step_outputs[0]
    assert 'task_ids' in output
    assert 'grid_correct' in output
    assert 'cell_correct_counts' in output
    assert 'cell_total_counts' in output


def test_baseline_epoch_end_doesnt_crash(mock_batch_decoder_only):
    """Test baseline on_validation_epoch_end doesn't crash."""
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
    
    # Simulate validation step
    model.validation_step(mock_batch_decoder_only, batch_idx=0)
    
    # Call epoch end (should not crash)
    model.on_validation_epoch_end()
    
    # Verify outputs cleared
    assert len(model.validation_step_outputs) == 0


def test_all_models_validation_pattern():
    """Test that all 4 models follow the same validation pattern."""
    from src.models.baseline_decoder_only_lightning import BaselineDecoderOnlyLightningModule
    from src.models.exp0_encoder_decoder_lightning import Exp0EncoderDecoderLightningModule
    from src.models.exp1_grid2d_pe_lightning import Exp1Grid2DPELightningModule
    from src.models.exp2_perminv_lightning import Exp2PermInvLightningModule
    
    models = [
        BaselineDecoderOnlyLightningModule(vocab_size=11, d_model=64, num_layers=2, num_heads=2, d_ff=128, dropout=0.1, learning_rate=0.001, pad_token=10),
        Exp0EncoderDecoderLightningModule(vocab_size=11, d_model=64, num_encoder_layers=2, num_decoder_layers=2, num_heads=2, d_ff=128, dropout=0.1, learning_rate=0.001, pad_token=10),
        Exp1Grid2DPELightningModule(vocab_size=11, d_model=64, num_encoder_layers=2, num_decoder_layers=2, num_heads=2, d_ff=128, max_grid_size=30, dropout=0.1, learning_rate=0.001, pad_token=10, sep_token=0),
        Exp2PermInvLightningModule(vocab_size=11, d_model=64, num_encoder_layers=2, num_decoder_layers=2, num_heads=2, d_ff=128, max_grid_size=30, dropout=0.1, learning_rate=0.001, pad_token=10),
    ]
    
    for model in models:
        # All should have validation_step_outputs
        assert hasattr(model, 'validation_step_outputs')
        assert isinstance(model.validation_step_outputs, list)
        
        # All should have on_validation_epoch_end
        assert hasattr(model, 'on_validation_epoch_end')
        assert callable(model.on_validation_epoch_end)


def test_trainer_csv_logger_header_preseed(tmp_path):
    """Regression test: CSVLogger must handle validation metrics keys.

    Scenario: The CSV logger creates a header on first use (often during sanity check).
    Later, validation logs add new keys (val_grid_accuracy, etc.) causing a header mismatch.
    Fix: pre-seed the CSV logger with those keys before training starts.
    This test verifies a short Trainer run finishes and metrics.csv contains expected columns.
    """
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import CSVLogger
    from torch.utils.data import Dataset, DataLoader
    from src.models.exp0_encoder_decoder_lightning import Exp0EncoderDecoderLightningModule
    import torch
    from pathlib import Path

    class DummyEDDataset(Dataset):
        def __len__(self):
            return 2
        def __getitem__(self, idx):
            src = torch.randint(0, 11, (50,), dtype=torch.long)
            tgt = torch.randint(0, 11, (50,), dtype=torch.long)
            task_id = f"task_{idx}"
            return src, tgt, task_id

    train_loader = DataLoader(DummyEDDataset(), batch_size=2, shuffle=False)
    val_loader = DataLoader(DummyEDDataset(), batch_size=2, shuffle=False)

    model = Exp0EncoderDecoderLightningModule(
        vocab_size=11,
        d_model=32,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        d_ff=64,
        dropout=0.0,
        learning_rate=1e-3,
        pad_token=10,
    )

    csv_logger = CSVLogger(save_dir=str(tmp_path), name="csv_header_test", version=None)
    # Pre-seed header with validation keys
    csv_logger.log_metrics({
        'val_grid_accuracy': 0.0,
        'val_cell_accuracy': 0.0,
        'val_change_recall': 0.0,
        'val_copy_rate': 0.0,
        'val_transformation_quality': 0.0,
    }, step=0)

    trainer = pl.Trainer(
        max_epochs=1,
        logger=csv_logger,
        enable_progress_bar=False,
        log_every_n_steps=1,
        fast_dev_run=1,
    )

    trainer.fit(model, train_loader, val_loader)

    # Locate metrics.csv
    root = Path(tmp_path) / "csv_header_test"
    assert root.exists()
    versions = [d for d in root.iterdir() if d.is_dir() and d.name.startswith("version_")]
    assert versions, f"No version directory under {root}"
    metrics_csv = versions[0] / "metrics.csv"
    assert metrics_csv.exists(), f"metrics.csv not found at {metrics_csv}"

    # Header must include validation keys
    header = metrics_csv.read_text().splitlines()[0]
    for key in [
        'val_grid_accuracy',
        'val_cell_accuracy',
        'val_change_recall',
        'val_copy_rate',
        'val_transformation_quality',
    ]:
        assert key in header, f"Missing column {key} in metrics.csv header"

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
