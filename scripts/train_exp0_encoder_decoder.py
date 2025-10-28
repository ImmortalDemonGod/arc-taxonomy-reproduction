"""
Training script for Encoder-Decoder baseline (Exp 0).

Trains from scratch on 18 foundational V2 tasks to test E-D architecture contribution.
"""
import sys
import argparse
from pathlib import Path
import json
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.exp0_encoder_decoder_lightning import Exp0EncoderDecoderLightningModule
from src.data.encoder_decoder_data import create_encoder_decoder_dataloader
from src.callbacks import PerTaskMetricsLogger


def main():
    """Train Encoder-Decoder baseline (Exp 0)."""
    # Parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast_dev_run', type=int, default=None,
                        help='Run fast_dev_run with N batches for testing')
    args, unknown = parser.parse_known_args()
    fast_dev_run = args.fast_dev_run

    
    # Set seed for reproducibility (Trial 69 used 307)
    pl.seed_everything(307, workers=True)
    
    # Set matmul precision for Tensor Cores (A6000 optimization)
    torch.set_float32_matmul_precision('high')
    
    # Get data files - MUST match Champion's data source
    data_dir = Path(__file__).parent.parent / "data" / "distributional_alignment"
    
    # Load split manifest to get train/val split
    with open(data_dir / "split_manifest.json") as f:
        split_info = json.load(f)
    
    train_files = [data_dir / fname for fname in split_info["train_files"]]
    val_files = [data_dir / fname for fname in split_info["val_files"]]
    
    print(f"\n{'='*70}")
    print(f"TRAINING: Exp 0 (Generic Encoder-Decoder)")
    print(f"{'='*70}")
    print(f"Dataset: distributional_alignment (same as Champion)")
    print(f"Train tasks: {len(train_files)}")
    print(f"Val tasks: {len(val_files)}")
    print(f"Total tasks: {len(train_files) + len(val_files)}")
    print(f"Architecture: Generic Encoder-Decoder")
    print(f"Training config: Trial 69 hyperparameters")
    print(f"{'='*70}\n")
    
    # Create data loaders with Trial 69 batch size
    train_loader = create_encoder_decoder_dataloader(
        train_files,
        batch_size=32,  # Trial 69 value
        shuffle=True,
        max_seq_len=900,  # 30x30 grid max
    )
    
    val_loader = create_encoder_decoder_dataloader(
        val_files,
        batch_size=32,
        shuffle=False,
        max_seq_len=900,
    )
    
    # Create model (PARAMETER-MATCHED to Champion: 1.71M params)
    model = Exp0EncoderDecoderLightningModule(
        vocab_size=11,
        d_model=168,  # Matched to Champion
        num_encoder_layers=1,  # Same structure as Champion
        num_decoder_layers=3,  # Same structure as Champion
        num_heads=4,
        d_ff=672,  # Matched to Champion
        dropout=0.167,
        learning_rate=0.0018498849832733245,  # Trial 69
        weight_decay=0.0,  # Trial 69
        beta1=0.95,  # Trial 69
        beta2=0.999,
        max_epochs=100,
        pad_token=10,
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/exp0_encoder_decoder",
        filename="encoder_decoder-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    
    per_task_logger = PerTaskMetricsLogger(log_dir="logs/per_task_metrics")
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    tb_logger = TensorBoardLogger(
        save_dir="logs",
        name="exp0_encoder_decoder_training",
        version=None,
    )
    
    csv_logger = CSVLogger(
        save_dir="logs",
        name="exp0_encoder_decoder_csv",
        version=None,
    )
    
    # Create trainer with Trial 69 configuration
    # Note: Using '16-mixed' for precision on Mac/MPS compatibility
    trainer = pl.Trainer(
        max_epochs=100,
        precision='16-mixed',  # Mixed precision (Trial 69 used 16, but '16-mixed' is modern syntax)
        gradient_clip_val=1.0,  # Trial 69
        deterministic=False,  # Set to False for MPS compatibility (Mac)
        callbacks=[checkpoint_callback, per_task_logger, lr_monitor],
        logger=[tb_logger, csv_logger],
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    print("Starting training...")
    print(f"Checkpoints will be saved to: checkpoints/exp_0_encoder_decoder/\n")
    
    # Train!
    trainer.fit(model, train_loader, val_loader)
    
    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best val_loss: {checkpoint_callback.best_model_score:.4f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
