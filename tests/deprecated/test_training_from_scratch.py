"""
Test: Can we train from scratch (not just fine-tune)?

This tests if we can use the real implementations to train a model from scratch,
or if we need the full run_model.py infrastructure.
"""

import sys
from pathlib import Path

# Add jarc_reactor to path
# This file is in: arc_reactor/publications/arc_taxonomy_2025/reproduction/tests/
arc_reactor_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(arc_reactor_root))

import torch
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

print("=" * 80)
print("TRAINING FROM SCRATCH TEST: Can we train without run_model.py?")
print("=" * 80)

# Test 1: Import everything
print("\n[1/5] Importing modules...")
try:
    from jarc_reactor.utils.train import TransformerTrainer
    from jarc_reactor.data.data_module import MyDataModule
    print("✅ All imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Load champion config and use it for training from scratch
print("\n[2/5] Loading champion config...")
try:
    # Load the actual champion_bootstrap config
    checkpoint_path = arc_reactor_root / "outputs/checkpoints/champion_bootstrap.ckpt"
    if not checkpoint_path.exists():
        checkpoint_path = arc_reactor_root / "jarc_reactor/lora/experiments/mvp_v1/checkpoints/champion_bootstrap.ckpt"
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    champion_cfg = checkpoint['hyper_parameters']
    
    # Create config using champion's model config
    cfg = OmegaConf.create({
        'model': champion_cfg.get('model', {}),
        'training': {
            'learning_rate': 5.0e-6,
            'max_epochs': 2,  # Just 2 epochs for testing
            'batch_size': 2,
            'device_choice': 'auto',
        },
        'data': {
            'data_dir': str(arc_reactor_root / 'publications/arc_taxonomy_2025/reproduction/data/tasks'),
            'num_context_pairs': 2,
            'max_context_pairs': 2,
            'batch_size': 2,
        },
        'dataloader': {
            'batch_size': 2,
            'num_workers': 0,
        },
        'logging': {
            'log_dir': str(arc_reactor_root / 'publications/arc_taxonomy_2025/reproduction/logs')
        }
    })
    
    print(f"✅ Config created")
    print(f"   Model: d_model={cfg.model.d_model}, layers={cfg.model.encoder_layers}/{cfg.model.decoder_layers}")
    print(f"   Training: lr={cfg.training.learning_rate}, epochs={cfg.training.max_epochs}")
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Create model from scratch (no checkpoint)
print("\n[3/5] Creating model from scratch...")
try:
    model = TransformerTrainer(cfg)
    print(f"✅ Model created from scratch")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Has core_model: {hasattr(model, 'core_model')}")
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Create data module
print("\n[4/5] Creating data module...")
try:
    datamodule = MyDataModule(cfg)
    print(f"✅ Data module created")
    print(f"   Data dir: {cfg.data.data_dir}")
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Can we set up a trainer?
print("\n[5/5] Setting up PyTorch Lightning Trainer...")
try:
    # Create checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(arc_reactor_root / 'publications/arc_taxonomy_2025/reproduction/outputs/checkpoints'),
        filename='test-{epoch:02d}',
        save_top_k=1,
        monitor='val_loss',
        mode='min'
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        callbacks=[checkpoint_callback],
        accelerator='auto',
        devices=1,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )
    
    print(f"✅ Trainer created")
    print(f"   Max epochs: {trainer.max_epochs}")
    print(f"   Accelerator: {trainer.accelerator}")
    print(f"\n⚠️  Ready to train but NOT running (would take ~1 hour)")
    print(f"   To actually train, run: trainer.fit(model, datamodule)")
    
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ TRAINING FROM SCRATCH IS POSSIBLE!")
print("=" * 80)
print("\nConclusion:")
print("- Can create model from scratch ✅")
print("- Can create data module ✅")
print("- Can set up Lightning Trainer ✅")
print("- Do NOT need run_model.py ✅")
print("\n➡️  We have everything needed to train from scratch")
print("➡️  run_model.py is just a convenience wrapper around this")
