"""
Test script to verify champion_bootstrap.ckpt can be loaded correctly.
"""
import torch
from model import ARCTransformer

# Path to champion_bootstrap checkpoint
CHECKPOINT_PATH = "../../../outputs/checkpoints/champion_bootstrap.ckpt"

print("Testing champion_bootstrap.ckpt loading...")
print(f"Checkpoint path: {CHECKPOINT_PATH}")

# Load checkpoint to inspect structure
print("\n1. Inspecting checkpoint structure...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
print(f"   Checkpoint keys: {list(checkpoint.keys())}")

if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
    print(f"   State dict has {len(state_dict)} entries")
    
    # Show sample keys
    sample_keys = list(state_dict.keys())[:10]
    print(f"   Sample keys:")
    for key in sample_keys:
        print(f"     - {key}: {state_dict[key].shape}")

# Attempt to load model
print("\n2. Loading model from checkpoint...")
try:
    model = ARCTransformer.load_pretrained(CHECKPOINT_PATH)
    print("   ✓ Model loaded successfully!")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Total parameters: {total_params:,}")
    
    # Test forward pass
    print("\n3. Testing forward pass...")
    batch_size = 1
    seq_len = 900  # 30x30
    src = torch.randint(0, 10, (batch_size, seq_len))
    tgt = torch.randint(0, 10, (batch_size, seq_len))
    
    with torch.no_grad():
        logits = model(src, tgt)
    
    print(f"   ✓ Forward pass successful!")
    print(f"   ✓ Output shape: {logits.shape}")
    print(f"   ✓ Expected: ({batch_size}, {seq_len}, {model.vocab_size})")
    
    print("\n✅ All tests passed! Checkpoint loading works correctly.")
    
except Exception as e:
    print(f"\n❌ Error loading model: {e}")
    import traceback
    traceback.print_exc()
