"""
Training script for Champion architecture (Exp 3).

Trains from scratch on 18 foundational V2 tasks with full champion architecture:
- Encoder-Decoder
- Grid2D Positional Encoding  
- Permutation-Invariant Embeddings
- Context Bridge

Uses CrossEntropyLoss (Option A - simple, standard approach).
"""
import sys
from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.champion_lightning import ChampionLightningModule
from src.data.champion_data import create_champion_dataloader


def main():
    """Train Champion architecture (Exp 3)."""
    
    # Performance optimizations for Tensor Cores (A6000)
    # Uses TensorFloat32 for ~2-3x faster matmuls with minimal precision loss
    torch.set_float32_matmul_precision('high')
    
    # Set seed for reproducibility (Trial 69 used 307)
    pl.seed_everything(307, workers=True)
    
    # Get data files from distributional_alignment dataset (Phase 1B - 400 tasks)
    # This is the correct pretraining dataset that produced champion_bootstrap.ckpt
    import json
    # Path: reproduction/scripts -> publications -> arc_reactor -> data/synthetic_data
    arc_reactor_root = Path(__file__).parent.parent.parent.parent.parent
    data_dir = arc_reactor_root / "data" / "synthetic_data" / "distributional_alignment"
    
    # Load split manifest to get train/val split
    with open(data_dir / "split_manifest.json") as f:
        split_info = json.load(f)
    
    train_files = [data_dir / fname for fname in split_info["train_files"]]
    val_files = [data_dir / fname for fname in split_info["val_files"]]
    
    task_files = train_files + val_files  # For info printing
    
    print(f"\n{'='*70}")
    print(f"TRAINING: Exp 3 (Champion Architecture) - Phase 1B")
    print(f"{'='*70}")
    print(f"Dataset: distributional_alignment (re-arc synthetic)")
    print(f"Total tasks: {len(task_files)}")
    print(f"Train tasks: {len(train_files)} ({len(train_files) * 15} examples)")
    print(f"Val tasks: {len(val_files)} ({len(val_files) * 15} examples)")
    print(f"Architecture: Full Champion (E-D + Grid2D PE + PermInv + Bridge)")
    print(f"Training config: Trial 69 hyperparameters")
    print(f"Loss function: CrossEntropyLoss (Option A)")
    print(f"Context pairs: 2 (fixed)")
    print(f"Expected baseline: ~2.34% grid accuracy (champion_bootstrap)")
    print(f"{'='*70}\n")
    
    # Create data loaders with Trial 69 batch size
    train_loader = create_champion_dataloader(
        train_files,
        batch_size=32,  # Trial 69 value
        shuffle=True,
        num_context_pairs=2,  # Trial 69 value
        max_grid_size=30,
    )
    
    val_loader = create_champion_dataloader(
        val_files,
        batch_size=32,
        shuffle=False,
        num_context_pairs=2,
        max_grid_size=30,
    )
    
    # Create model with Trial 69-aligned configuration
    # Using architecture parameters from Trial 69
    model = ChampionLightningModule(
        vocab_size=11,
        d_model=160,  # Trial 69 value
        num_encoder_layers=1,  # Trial 69 value
        num_decoder_layers=3,  # Trial 69 value (1.7M params)
        num_heads=4,
        d_ff=640,  # Trial 69 value (1.7M params)
        max_grid_size=30,
        dropout=0.16712351989226623,  # Trial 69 optimized dropout
        learning_rate=0.0018498849832733245,  # Trial 69
        weight_decay=0.0,  # Trial 69
        beta1=0.95,  # Trial 69
        beta2=0.999,
        max_epochs=100,
        pad_token=10,
        use_context=True,
        use_bridge=True,
    )
    
    # NOTE: torch.compile DISABLED due to hangs with dynamic shapes
    # Even with unlimited cache, it causes training to freeze mid-epoch
    # Cost: ~20-30% slower training (38s/epoch -> 50s/epoch estimated)
    # Benefit: Reliable training without hangs
    # model = torch.compile(model)  # DISABLED
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/exp_3_champion",
        filename="champion-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",  # Monitor loss (standard practice)
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",  # Stop when loss plateaus
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
    print(f"Checkpoints will be saved to: checkpoints/exp_3_champion/\n")
    
    # Train!
    trainer.fit(model, train_loader, val_loader)
    
    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best val_loss: {checkpoint_callback.best_model_score:.4f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
