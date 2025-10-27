"""
Training script for Encoder-Decoder baseline (Exp 0).

Trains from scratch on 18 foundational V2 tasks to test E-D architecture contribution.
"""
import sys
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.encoder_decoder_lightning import EncoderDecoderLightningModule
from src.data.encoder_decoder_data import create_encoder_decoder_dataloader


def main():
    """Train Encoder-Decoder baseline (Exp 0)."""
    
    # Set seed for reproducibility (Trial 69 used 307)
    pl.seed_everything(307, workers=True)
    
    # Get data files
    data_dir = Path(__file__).parent.parent / "data" / "tasks"
    task_files = sorted(list(data_dir.glob("*.json")))
    
    print(f"\n{'='*70}")
    print(f"TRAINING: Exp 0 (Encoder-Decoder Baseline)")
    print(f"{'='*70}")
    print(f"Task files: {len(task_files)}")
    print(f"Architecture: Standard Encoder-Decoder Transformer")
    print(f"Training config: Trial 69 hyperparameters")
    print(f"Loss function: CrossEntropyLoss (Option A)")
    print(f"{'='*70}\n")
    
    # Split data: 80% train, 20% val
    split_idx = int(len(task_files) * 0.8)
    train_files = task_files[:split_idx]
    val_files = task_files[split_idx:]
    
    print(f"Train files: {len(train_files)}")
    print(f"Val files: {len(val_files)}\n")
    
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
    
    # Create model with Trial 69-aligned configuration
    model = EncoderDecoderLightningModule(
        vocab_size=11,
        d_model=128,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        d_ff=512,
        dropout=0.1,
        learning_rate=0.0018498849832733245,  # Trial 69
        weight_decay=0.0,  # Trial 69
        beta1=0.95,  # Trial 69
        beta2=0.999,
        max_epochs=100,
        pad_token=10,
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/exp_0_encoder_decoder",
        filename="encoder_decoder-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=7,  # Trial 69 value
        mode="min",
        verbose=True,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Create trainer with Trial 69 configuration
    # Note: Using '16-mixed' for precision on Mac/MPS compatibility
    trainer = pl.Trainer(
        max_epochs=100,
        precision='16-mixed',  # Mixed precision (Trial 69 used 16, but '16-mixed' is modern syntax)
        gradient_clip_val=1.0,  # Trial 69
        deterministic=False,  # Set to False for MPS compatibility (Mac)
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
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
