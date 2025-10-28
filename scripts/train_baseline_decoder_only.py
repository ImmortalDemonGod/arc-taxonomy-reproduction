"""
Training script for Decoder-Only baseline (Exp -1).

Trains from scratch on 18 foundational V2 tasks to establish baseline performance.
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

from src.models.baseline_decoder_only_lightning import BaselineDecoderOnlyLightningModule
from src.data.decoder_only_data import create_decoder_only_dataloader
from src.callbacks import PerTaskMetricsLogger


def main():
    """Train Decoder-Only baseline (Exp -1)."""
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
    print(f"TRAINING: Baseline (Decoder-Only)")
    print(f"{'='*70}")
    print(f"Dataset: distributional_alignment (same as Champion)")
    print(f"Train tasks: {len(train_files)}")
    print(f"Val tasks: {len(val_files)}")
    print(f"Total tasks: {len(train_files) + len(val_files)}")
    print(f"Architecture: Decoder-Only with RoPE")
    print(f"Training config: Trial 69 hyperparameters")
    print(f"Loss function: CrossEntropyLoss (Option A)")
    print(f"{'='*70}\n")
    
    # Create data loaders with Trial 69 batch size
    train_loader = create_decoder_only_dataloader(
        train_files,
        batch_size=32,  # Trial 69 value
        shuffle=True,
        max_seq_len=512,  # Maximum sequence length
    )
    
    val_loader = create_decoder_only_dataloader(
        val_files,
        batch_size=32,
        shuffle=False,
        max_seq_len=512,
    )
    
    # Create model (PARAMETER-MATCHED to Champion: 1.74M params)
    model = BaselineDecoderOnlyLightningModule(
        vocab_size=11,
        context_length=512,
        d_model=164,  # Matched to Champion
        num_layers=4,  # Deeper to match param count
        num_heads=4,
        d_ff=656,  # Matched to Champion
        dropout=0.167,
        learning_rate=0.0018498849832733245,  # Trial 69
        weight_decay=0.0,  # Trial 69
        beta1=0.95,  # Trial 69
        beta2=0.999,
        max_epochs=100,
        sep_token=10,
        pad_token=10,
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/baseline_decoder_only",
        filename="decoder_only-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    
    # Per-task metrics logger (experiment-specific directory)
    per_task_logger = PerTaskMetricsLogger(log_dir="logs/per_task_metrics/baseline")
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # TensorBoard and CSV loggers
    tb_logger = TensorBoardLogger(
        save_dir="logs/baseline_decoder_only",
        name="baseline_decoder_only_training",
        version=None,
    )
    
    csv_logger = CSVLogger(
        save_dir="logs",
        name="baseline_decoder_only_csv",
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
        fast_dev_run=fast_dev_run if fast_dev_run else False,  # CLI override for testing
    )
    
    print("Starting training...")
    print(f"Checkpoints will be saved to: checkpoints/exp_-1_decoder_only/\n")
    
    # Train!
    trainer.fit(model, train_loader, val_loader)
    
    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best val_loss: {checkpoint_callback.best_model_score:.4f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
