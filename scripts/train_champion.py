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
    
    # Increase torch.compile cache limit to handle dynamic shapes
    # Default is 64 which causes constant recompilation with varying grid sizes
    # Dataset analysis shows 870+ unique source shapes, 855+ target shapes
    # Conservative estimate: 743,850 possible (src × tgt) combinations
    # Setting to very high value (effectively unlimited for our use case)
    import torch._dynamo as dynamo
    dynamo.config.cache_size_limit = 8192  # Should handle all realistic combinations
    
    # Set seed for reproducibility (Trial 69 used 307)
    pl.seed_everything(307, workers=True)
    
    # Get data files
    data_dir = Path(__file__).parent.parent / "data" / "tasks"
    task_files = sorted(list(data_dir.glob("*.json")))
    
    print(f"\n{'='*70}")
    print(f"TRAINING: Exp 3 (Champion Architecture)")
    print(f"{'='*70}")
    print(f"Task files: {len(task_files)}")
    print(f"Architecture: Full Champion (E-D + Grid2D PE + PermInv + Bridge)")
    print(f"Training config: Trial 69 hyperparameters")
    print(f"Loss function: CrossEntropyLoss (Option A)")
    print(f"Context pairs: 2 (fixed)")
    print(f"{'='*70}\n")
    
    # Split data: 80% train, 20% val
    split_idx = int(len(task_files) * 0.8)
    train_files = task_files[:split_idx]
    val_files = task_files[split_idx:]
    
    print(f"Train files: {len(train_files)}")
    print(f"Val files: {len(val_files)}\n")
    
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
    
    # PyTorch 2.0+ compile for ~20-30% additional speedup
    # Cache limit set to 8192 to handle all grid shape variations
    print("Compiling model with torch.compile (cache_size_limit=8192)...")
    model = torch.compile(model)
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/exp_3_champion",
        filename="champion-{epoch:02d}-{val_grid_accuracy:.4f}",
        monitor="val_grid_accuracy",  # Monitor grid-solving (headline metric)
        mode="max",
        save_top_k=3,
        save_last=True,
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_grid_accuracy",  # Stop when grid-solving plateaus
        patience=7,  # Trial 69 value
        mode="max",
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
