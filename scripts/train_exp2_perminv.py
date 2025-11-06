"""
Training script for Exp 2: E-D + PermInvariant Embedding ONLY (Independent Test)

This experiment tests the contribution of color-permutation equivariance INDEPENDENTLY.
- Encoder-Decoder: YES
- PermInvariant Embedding: YES  
- Grid2D PE: NO (uses standard 1D PE)
- Context System: NO

This is for the INDEPENDENT ablation design where each component is tested separately.
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
    """Train Exp 2: E-D + PermInv only (independent test)."""
    # Parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast_dev_run', type=int, default=None,
                        help='Run fast_dev_run with N batches for testing')
    parser.add_argument('--seed', type=int, default=307, help='Random seed for reproducibility')
    parser.add_argument('--max_epochs', type=int, default=100, help='Maximum training epochs')
    args, unknown = parser.parse_known_args()
    fast_dev_run = args.fast_dev_run

    
    # Set seed for reproducibility
    pl.seed_everything(args.seed, workers=True)
    
    # Set matmul precision for Tensor Cores (A6000 optimization)
    torch.set_float32_matmul_precision('high')
    
    # Data paths
    data_dir = Path(__file__).parent.parent / "data" / "distributional_alignment"
    
    # Load split manifest
    with open(data_dir / "split_manifest.json") as f:
        split_info = json.load(f)
    
    train_files = [data_dir / fname for fname in split_info["train_files"]]
    val_files = [data_dir / fname for fname in split_info["val_files"]]
    
    print(f"\nExp 2: E-D + PermInvariant Embedding ONLY (Independent Test)")
    print(f"=" * 70)
    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")
    print(f"Target params: 1.71M (parameter-matched to Champion)")
    print(f"Components: E-D + PermInv ONLY (NO Grid2D, NO Context)")
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
    # Using baseline module with use_perminv=True for independent testing
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
        use_perminv=True,  # KEY: Enable PermInv for independent testing
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/exp2_perminv",
        filename="exp2-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    checkpoint_acc = ModelCheckpoint(
        dirpath="checkpoints/exp2_perminv",
        filename="exp2-acc-{epoch:02d}-{val_grid_accuracy:.4f}",
        monitor="val_grid_accuracy",
        mode="max",
        save_top_k=3,
        save_last=False,
    )
    
    per_task_logger = PerTaskMetricsLogger(
        log_dir="logs/per_task_metrics/exp2",
        experiment_name="exp2_perminv"
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    tb_logger = TensorBoardLogger(
        save_dir="logs",
        name="exp2_perminv_training",
        version=None,
    )
    
    csv_logger = CSVLogger(
        save_dir="logs",
        name="exp2_perminv_csv",
        version=None,
    )
    # Pre-seed CSV header with validation metric keys to avoid header rewrite errors
    csv_logger.log_metrics({
        'val_grid_accuracy': 0.0,
        'val_cell_accuracy': 0.0,
        'val_change_recall': 0.0,
        'val_copy_rate': 0.0,
        'val_transformation_quality': 0.0,
    }, step=0)
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        precision='16-mixed',  # Mixed precision (Trial 69 used 16, but '16-mixed' is modern syntax)
        gradient_clip_val=1.0,  # Trial 69
        deterministic=False,  # Set to False for MPS compatibility (Mac)
        callbacks=[checkpoint_callback, checkpoint_acc, per_task_logger, lr_monitor],
        logger=[tb_logger, csv_logger],
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True,
        fast_dev_run=fast_dev_run if fast_dev_run else False,  # CLI override for testing
    )
    
    # Print model summary
    print("\nModel Summary:")
    print(f"  Config: d_model=168, enc_layers=1, dec_layers=3, d_ff=672")
    print(f"  Architecture: E-D + PermInvariant Embedding ONLY")
    print(f"  Components: NO Grid2D PE, NO Context System")
    print(f"  Tests: Independent contribution of PermInv")
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
