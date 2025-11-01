"""
Training script for Champion architecture (Exp 3).

Supports two dataset modes:
1. Re-ARC (default): Pre-training on 400 synthetic re-arc tasks
2. ARC-AGI-2: Fine-tuning/transfer learning on real ARC competition tasks

Architecture:
- Encoder-Decoder
- Grid2D Positional Encoding  
- Permutation-Invariant Embeddings
- Context Bridge and Encoder

Uses Trial 69 hyperparameters (optimized on real ARC) and CrossEntropyLoss (Option A).

USAGE EXAMPLES:

1. Pre-train on re-arc (Phase 1B - reproduce champion_bootstrap.ckpt):
   python scripts/train_exp3_champion.py --dataset rearc

2. Transfer learning: Champion → ARC-AGI-2 (Experiment 2):
   python scripts/train_exp3_champion.py --dataset arc-agi-2 \\
       --checkpoint weights/champion-epoch=36-val_loss=0.5926.ckpt

3. Transfer learning: Merged LoRA → ARC-AGI-2 (Experiment 3b):
   python scripts/train_exp3_champion.py --dataset arc-agi-2 \\
       --checkpoint weights/champion_merged_loras.ckpt

4. Quick test (5 batches):
   python scripts/train_exp3_champion.py --fast_dev_run 5
"""
import sys
import argparse
from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.exp3_champion_lightning import Exp3ChampionLightningModule
from src.data.champion_data import create_champion_dataloader
from src.callbacks import PerTaskMetricsLogger


def main():
    """Train Champion architecture (Exp 3)."""
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description='Train Champion architecture on re-arc or ARC-AGI-2')
    parser.add_argument('--dataset', type=str, default='rearc', choices=['rearc', 'arc-agi-2'],
                        help='Dataset to train on: rearc (synthetic 400 tasks) or arc-agi-2 (real competition)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint for transfer learning (e.g., champion_bootstrap.ckpt or champion_merged_loras.ckpt)')
    parser.add_argument('--fast_dev_run', type=int, default=None,
                        help='Run fast_dev_run with N batches for testing')
    args, unknown = parser.parse_known_args()
    
    dataset_mode = args.dataset
    checkpoint_path = args.checkpoint
    fast_dev_run = args.fast_dev_run

    
    # Performance optimizations for Tensor Cores (A6000)
    # Uses TensorFloat32 for ~2-3x faster matmuls with minimal precision loss
    torch.set_float32_matmul_precision('high')
    
    # Set seed for reproducibility (Trial 69 used 307)
    pl.seed_everything(307, workers=True)
    
    # Load dataset based on mode
    import json
    
    if dataset_mode == 'rearc':
        # Re-ARC synthetic dataset (Phase 1B - 400 tasks)
        # This is the pretraining dataset that produced champion_bootstrap.ckpt
        data_dir = Path(__file__).parent.parent / "data" / "distributional_alignment"
        
        # Load split manifest to get train/val split
        with open(data_dir / "split_manifest.json") as f:
            split_info = json.load(f)
        
        train_files = [data_dir / fname for fname in split_info["train_files"]]
        val_files = [data_dir / fname for fname in split_info["val_files"]]
        
        dataset_name = "Re-ARC Synthetic (distributional_alignment)"
        dataset_info = f"{len(train_files) + len(val_files)} tasks"
        experiment_phase = "Phase 1B (Pre-training)"
        
    elif dataset_mode == 'arc-agi-2':
        # ARC-AGI-2 real competition dataset
        # Located in reproduction/data/arc_prize_2025/
        arc_data_dir = Path(__file__).parent.parent / "data" / "arc_prize_2025"
        
        # Load training challenges JSON
        with open(arc_data_dir / "arc-agi_training_challenges.json") as f:
            training_data = json.load(f)
        
        # Load evaluation challenges JSON  
        with open(arc_data_dir / "arc-agi_evaluation_challenges.json") as f:
            eval_data = json.load(f)
        
        # Convert JSON dict format to individual task files format
        # ARC-AGI-2 format: {task_id: {train: [...], test: [...]}, ...}
        # We need to create temporary task files or load them differently
        # For now, we'll save them as temp files
        temp_dir = Path(__file__).parent.parent / "data" / "arc_agi_temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        train_files = []
        for task_id, task_data in training_data.items():
            task_file = temp_dir / f"{task_id}.json"
            with open(task_file, 'w') as f:
                json.dump(task_data, f)
            train_files.append(task_file)
        
        val_files = []
        for task_id, task_data in eval_data.items():
            task_file = temp_dir / f"{task_id}_eval.json"
            with open(task_file, 'w') as f:
                json.dump(task_data, f)
            val_files.append(task_file)
        
        dataset_name = "ARC-AGI-2 Competition (Real Tasks)"
        dataset_info = f"{len(train_files)} train, {len(val_files)} eval tasks"
        experiment_phase = "Transfer Learning" if checkpoint_path else "Training from Scratch"
    
    # Determine experiment name for logging (used throughout)
    if checkpoint_path:
        ckpt_name = Path(checkpoint_path).stem.split('-')[0]  # "champion" or "champion_merged_loras"
        if "merged" in ckpt_name:
            exp_name = f"exp3b_merged_lora_{dataset_mode}"  # e.g., "exp3b_merged_lora_arc-agi-2"
        else:
            exp_name = f"exp2_champion_{dataset_mode}"      # e.g., "exp2_champion_arc-agi-2"
    else:
        exp_name = f"exp3_champion_{dataset_mode}"          # e.g., "exp3_champion_rearc"
    
    # Print experiment configuration
    print(f"\n{'='*80}")
    print(f"TRAINING: Champion Architecture (Exp 3)")
    print(f"{'='*80}")
    print(f"")
    print(f"📊 DATASET:")
    print(f"   Name: {dataset_name}")
    print(f"   Tasks: {dataset_info}")
    print(f"   Phase: {experiment_phase}")
    print(f"")
    print(f"🏗️  ARCHITECTURE:")
    print(f"   Model: Full Champion (V3)")
    print(f"   Components: Encoder-Decoder + Grid2D PE + PermInv + Context Bridge")
    print(f"   Parameters: ~1.7M")
    print(f"   Bridge: 8 heads, 2 tokens (apply_to_decoder=true)")
    print(f"")
    print(f"⚙️  TRAINING CONFIG (Trial 69 - Optimized on Real ARC):")
    print(f"   Learning Rate: 0.00185")
    print(f"   Optimizer: Adam (beta1=0.95, beta2=0.999, weight_decay=0.0)")
    print(f"   Scheduler: CosineAnnealingWarmRestarts (T_0=6)")
    print(f"   Batch Size: 32")
    print(f"   Max Epochs: 100")
    print(f"   Precision: Mixed (16-bit)")
    print(f"   Gradient Clip: 1.0")
    print(f"   Context Pairs: 2 (FIXED)")
    print(f"   Max Grid Size: 30x30")
    print(f"   Seed: 307 (Trial 69)")
    print(f"")
    if checkpoint_path:
        print(f"🔄 TRANSFER LEARNING:")
        print(f"   Loading from: {checkpoint_path}")
        print(f"   Strategy: Continue training with same hyperparameters")
        print(f"")
    print(f"💾 OUTPUTS:")
    print(f"   Experiment Name: {exp_name}")
    print(f"   Checkpoints: checkpoints/{exp_name}/")
    print(f"   TensorBoard: logs/{exp_name}_tb/")
    print(f"   CSV Logs: logs/{exp_name}_csv/")
    print(f"   Per-task metrics: logs/per_task_metrics/{exp_name}/")
    print(f"")
    print(f"{'='*80}\n")
    
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
    if checkpoint_path:
        # Load from checkpoint for transfer learning
        print(f"Loading model from checkpoint: {checkpoint_path}")
        torch.serialization.add_safe_globals([
            'omegaconf.listconfig.ListConfig',
            'omegaconf.dictconfig.DictConfig'
        ])
        try:
            # Primary path: Lightning-formatted checkpoint
            model = Exp3ChampionLightningModule.load_from_checkpoint(
                checkpoint_path,
                # Override training config to continue with same hyperparameters
                learning_rate=0.0018498849832733245,  # Trial 69
                weight_decay=0.0,
                beta1=0.95,
                beta2=0.999,
                max_epochs=100,
            )
            print("✓ Checkpoint loaded successfully (Lightning format)\n")
        except Exception as e:
            # Fallback path: Non-Lightning checkpoint (e.g., merged LoRA without version field)
            print(f"⚠️  Lightning load failed ({e}). Falling back to raw state_dict load...")
            # Initialize a fresh module with Trial 69 hyperparameters
            model = Exp3ChampionLightningModule(
                vocab_size=11,
                d_model=160,
                num_encoder_layers=1,
                num_decoder_layers=3,
                num_heads=4,
                d_ff=640,
                max_grid_size=30,
                dropout=0.16712351989226623,
                learning_rate=0.0018498849832733245,
                weight_decay=0.0,
                beta1=0.95,
                beta2=0.999,
                max_epochs=100,
                pad_token=10,
                use_context=True,
                use_bridge=True,
            )
            # Load checkpoint and extract state_dict
            ckpt_obj = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            state_dict = ckpt_obj.get('state_dict', ckpt_obj)
            # Attempt to load with non-strict to allow harmless metadata/key diffs
            incompatible = model.load_state_dict(state_dict, strict=False)
            # Report any key issues for traceability
            missing = len(getattr(incompatible, 'missing_keys', []))
            unexpected = len(getattr(incompatible, 'unexpected_keys', []))
            print(f"✓ Raw state_dict loaded (strict=False). Missing keys: {missing}, Unexpected keys: {unexpected}\n")
    else:
        # Create new model from scratch
        print("Creating model from scratch...")
        model = Exp3ChampionLightningModule(
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
        print("✓ Model created successfully\n")
    
    # NOTE: torch.compile DISABLED due to hangs with dynamic shapes
    # Even with unlimited cache, it causes training to freeze mid-epoch
    # Cost: ~20-30% slower training (38s/epoch -> 50s/epoch estimated)
    # Benefit: Reliable training without hangs
    # model = torch.compile(model)  # DISABLED
    
    # Callbacks with unique directories (exp_name calculated earlier)
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{exp_name}",
        filename="champion-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",  # Monitor loss (standard practice)
        mode="min",
        save_top_k=3,
        save_last=True,
        every_n_epochs=1,  # Save checkpoint every epoch for crash recovery
    )

    checkpoint_acc = ModelCheckpoint(
        dirpath=f"checkpoints/{exp_name}",
        filename="champion-acc-{epoch:02d}-{val_grid_accuracy:.4f}",
        monitor="val_grid_accuracy",
        mode="max",
        save_top_k=3,
        save_last=False,
        every_n_epochs=1,
    )
    
    # Per-task metrics logger - writes CSV files after each epoch
    # Robust to GPU crashes - all data is on disk immediately
    per_task_logger = PerTaskMetricsLogger(
        log_dir=f"logs/per_task_metrics/{exp_name}",
        experiment_name=exp_name
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Loggers for comprehensive metric tracking
    tb_logger = TensorBoardLogger(
        save_dir="logs",
        name=f"{exp_name}_tb",
        version=None,  # Auto-increment version
    )
    
    csv_logger = CSVLogger(
        save_dir="logs",
        name=f"{exp_name}_csv",
        version=None,
    )
    
    # Create trainer with Trial 69 configuration
    # NOTE: Early stopping REMOVED for overnight run - will train full 100 epochs
    # Note: Using '16-mixed' for precision on Mac/MPS compatibility
    trainer = pl.Trainer(
        max_epochs=100,
        precision='16-mixed',  # Mixed precision (Trial 69 used 16, but '16-mixed' is modern syntax)
        gradient_clip_val=1.0,  # Trial 69
        deterministic=False,  # Set to False for MPS compatibility (Mac)
        callbacks=[checkpoint_callback, checkpoint_acc, per_task_logger, lr_monitor],
        logger=[tb_logger, csv_logger],  # Multiple loggers for comprehensive tracking
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True,
        fast_dev_run=fast_dev_run if fast_dev_run else False,  # CLI override for testing
    )
    
    print("🚀 Starting training...\n")
    
    # Train!
    trainer.fit(model, train_loader, val_loader)
    
    # Print completion summary
    print(f"\n{'='*80}")
    print(f"✅ TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"")
    print(f"Dataset: {dataset_name}")
    print(f"Phase: {experiment_phase}")
    print(f"")
    if not fast_dev_run:
        print(f"📁 OUTPUTS:")
        print(f"   Experiment: {exp_name}")
        print(f"   Best checkpoint (loss): {checkpoint_callback.best_model_path}")
        if checkpoint_callback.best_model_score is not None:
            print(f"   Best val_loss: {checkpoint_callback.best_model_score:.4f}")
        print(f"   Best checkpoint (acc): {checkpoint_acc.best_model_path}")
        if checkpoint_acc.best_model_score is not None:
            print(f"   Best val_grid_accuracy: {checkpoint_acc.best_model_score:.4f}")
        print(f"   TensorBoard logs: logs/{exp_name}_tb/")
        print(f"   CSV logs: logs/{exp_name}_csv/")
        print(f"   Per-task metrics: logs/per_task_metrics/{exp_name}/")
        print(f"")
        if dataset_mode == 'arc-agi-2':
            print(f"📊 NEXT STEPS:")
            print(f"   1. Analyze results: Check val_grid_accuracy trend")
            print(f"   2. Compare baselines: Exp 2 (champion) vs Exp 3b (merged LoRA)")
            print(f"   3. Generate figures: Per-task performance analysis")
            print(f"   4. Update paper: Transfer learning results section")
    else:
        print(f"Fast dev run completed successfully")
    print(f"")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
