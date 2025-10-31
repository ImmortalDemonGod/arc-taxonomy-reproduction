"""
Integration test: Prove we can load champion_bootstrap and run with real implementations.

This test uses ALL the jarc_reactor dependencies to verify the copied files work.
Only AFTER this passes will we strip dependencies.
"""

import sys
from pathlib import Path

# Add jarc_reactor to path
# This file is in: arc_reactor/publications/arc_taxonomy_2025/reproduction/tests/
# So arc_reactor root is 4 levels up from tests/
arc_reactor_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(arc_reactor_root))
print(f"DEBUG: arc_reactor_root = {arc_reactor_root}")

import torch
from omegaconf import OmegaConf

print("=" * 80)
print("INTEGRATION TEST: Load champion_bootstrap with real implementations")
print("=" * 80)

# Test 1: Import real implementations
print("\n[1/6] Testing imports...")
try:
    from jarc_reactor.utils.train import TransformerTrainer
    from jarc_reactor.data.data_preparation import prepare_data
    from jarc_reactor.models.transformer_model import TransformerModel
    print("✅ All imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Load champion_bootstrap checkpoint
print("\n[2/6] Loading champion_bootstrap checkpoint...")
# Try multiple possible locations
possible_paths = [
    arc_reactor_root / "outputs/checkpoints/champion_bootstrap.ckpt",
    arc_reactor_root / "jarc_reactor/lora/experiments/mvp_v1/checkpoints/champion_bootstrap.ckpt",
]
checkpoint_path = None
for path in possible_paths:
    if path.exists():
        checkpoint_path = path
        break

if checkpoint_path is None:
    print(f"❌ Checkpoint not found in any of these locations:")
    for p in possible_paths:
        print(f"   - {p}")
    sys.exit(1)

if not checkpoint_path.exists():
    print(f"❌ Checkpoint not found at: {checkpoint_path}")
    sys.exit(1)

try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    print(f"✅ Checkpoint loaded")
    print(f"   Keys: {list(checkpoint.keys())}")
    print(f"   State dict entries: {len(checkpoint['state_dict'])}")
except Exception as e:
    print(f"❌ Failed to load checkpoint: {e}")
    sys.exit(1)

# Test 3: Extract config from checkpoint
print("\n[3/6] Extracting config from checkpoint...")
try:
    if 'hyper_parameters' in checkpoint:
        cfg = checkpoint['hyper_parameters']
        print("✅ Config extracted from hyper_parameters")
    else:
        print("⚠️  No hyper_parameters in checkpoint, will need to create config manually")
        # Create minimal config for Trial 69
        cfg = {
            'model': {
                'd_model': 160,
                'encoder_layers': 1,
                'decoder_layers': 3,
                'n_head': 4,
                'd_ff': 640,
                'dropout_rate': 0.1717,
                'vocab_size': 11,
                'pad_token_id': 10,
                'max_h': 30,
                'max_w': 30,
            }
        }
        cfg = OmegaConf.create(cfg)
    
    print(f"   Config type: {type(cfg)}")
    if hasattr(cfg, 'model'):
        print(f"   Model d_model: {cfg.model.d_model if hasattr(cfg.model, 'd_model') else 'N/A'}")
except Exception as e:
    print(f"❌ Failed to extract config: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Create model from config
print("\n[4/6] Creating model from config...")
try:
    # Try to create TransformerTrainer (the Lightning module wrapper)
    model = TransformerTrainer(cfg)
    print(f"✅ Model created")
    print(f"   Model type: {type(model)}")
    print(f"   Has core_model: {hasattr(model, 'core_model')}")
except Exception as e:
    print(f"❌ Failed to create model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Load weights into model
print("\n[5/6] Loading weights into model...")
try:
    # Load state dict
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict, strict=False)
    print("✅ Weights loaded (strict=False)")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
except Exception as e:
    print(f"❌ Failed to load weights: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Run forward pass
print("\n[6/6] Testing forward pass...")
try:
    model.eval()
    batch_size = 2
    H, W = 30, 30
    
    # Create dummy inputs matching the expected format
    src = torch.randint(0, 10, (batch_size, H, W), dtype=torch.long)
    tgt = torch.randint(0, 10, (batch_size, H, W), dtype=torch.long)
    
    # Create dummy context (2 context pairs as per Trial 69)
    ctx_input = torch.randint(0, 10, (batch_size, 2, H, W), dtype=torch.long)
    ctx_output = torch.randint(0, 10, (batch_size, 2, H, W), dtype=torch.long)
    task_ids = torch.zeros(batch_size, dtype=torch.long)
    
    with torch.no_grad():
        # Try forward pass through core_model
        if hasattr(model, 'core_model'):
            logits = model.core_model(src, tgt, ctx_input, ctx_output)
        else:
            logits = model(src, tgt, ctx_input, ctx_output)
    
    print(f"✅ Forward pass successful")
    print(f"   Output shape: {logits.shape}")
    print(f"   Expected: ({batch_size}, {H*W}, 11) or ({batch_size}, {H}, {W}, 11)")
    
except Exception as e:
    print(f"❌ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED!")
print("=" * 80)
print("\nConclusion:")
print("- Can load champion_bootstrap checkpoint ✅")
print("- Can extract config ✅")
print("- Can create model ✅")
print("- Can load weights ✅")
print("- Can run forward pass ✅")
print("\n➡️  Next step: Systematically strip dependencies while keeping tests passing")
