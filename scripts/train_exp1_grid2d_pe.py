"""
Training script for Exp 1: Encoder-Decoder + Grid2D Positional Encoding

This experiment tests the contribution of 2D spatial bias over generic 1D encoding.

Expected result: +15-25% improvement over Exp 0 (Generic E-D baseline).
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

from src.models.exp1_grid2d_pe_lightning import Exp1Grid2DPELightningModule
from src.data.encoder_decoder_data import create_encoder_decoder_dataloader
from src.callbacks import PerTaskMetricsLogger


def main():
    """Train Exp 1: E-D + Grid2D PE."""
    # Parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast_dev_run', type=int, default=None,
                        help='Run fast_dev_run with N batches for testing')
    args, unknown = parser.parse_known_args()
    fast_dev_run = args.fast_dev_run

    
    # Set seed for reproducibility
    pl.seed_everything(307, workers=True)
    
    # Set matmul precision for Tensor Cores (A6000 optimization)
    torch.set_float32_matmul_precision('high')
    
    # Data paths
    data_dir = Path(__file__).parent.parent / "data" / "distributional_alignment"
    
    # Load split manifest
    with open(data_dir / "split_manifest.json") as f:
        split_info = json.load(f)
    
    train_files = [data_dir / fname for fname in split_info["train_files"]]
    val_files = [data_dir / fname for fname in split_info["val_files"]]
    
    print(f"\nExp 1: Encoder-Decoder + Grid2D Positional Encoding")
    print(f"=" * 70)
    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")
    print(f"Target params: 1.71M (parameter-matched to Champion)")
    print()
    
    # Create data loaders
    train_loader = create_encoder_decoder_dataloader(
        train_files,
        batch_size=32,
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
    model = Exp1Grid2DPELightningModule(
        vocab_size=11,
        d_model=168,  # Matched to Champion
        num_encoder_layers=1,  # Same structure as Champion
        num_decoder_layers=3,  # Same structure as Champion
        num_heads=4,
        d_ff=672,  # Matched to Champion
        max_grid_size=30,  # For Grid2D PE
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
        dirpath="checkpoints/exp1_grid2d_pe",
        filename="exp1-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    
    per_task_logger = PerTaskMetricsLogger(log_dir="logs/per_task_metrics")
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    tb_logger = TensorBoardLogger(
        save_dir="logs",
        name="exp1_grid2d_pe_training",
        version=None,
    )
    
    csv_logger = CSVLogger(
        save_dir="logs",
        name="exp1_grid2d_pe_csv",
        version=None,
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator='auto',
        devices=1,
        precision='16-mixed',
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback, per_task_logger, lr_monitor],
        logger=[tb_logger, csv_logger],
        deterministic=False,  # For MPS compatibility
        enable_progress_bar=True,
        log_every_n_steps=10,
    )
    
    # Print model summary
    print("\nModel Summary:")
    print(f"  Config: d_model=168, enc_layers=1, dec_layers=3, d_ff=672")
    print(f"  Architecture: Encoder-Decoder + Grid2D PE")
    print(f"  Expected params: ~1.71M")
    print()
    
    # Train
    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    
    print(f"\n{'='*70}")
    print("Training complete!")
    
    # Only print checkpoint info if not in fast_dev_run mode
    if not fast_dev_run:
        print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    else:
        print(f"Fast dev run completed (5 batches)")
    
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
