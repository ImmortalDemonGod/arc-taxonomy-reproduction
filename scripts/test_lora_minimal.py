"""
Minimal LoRA test - just verify the pipeline works.

No full training, no GPU, just proof-of-concept.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml
from src.data.champion_data import create_champion_dataloader
from src.models.exp3_champion_lightning import Exp3ChampionLightningModule
from peft import get_peft_model, LoraConfig

print("Minimal LoRA Pipeline Test")
print("="*60)

# Load config
with open('configs/atomic_lora_training.yaml') as f:
    config = yaml.safe_load(f)

# 1. Load Champion
print("\n1. Loading Champion (CPU only)...")
ckpt = torch.load('weights/champion-epoch=36-val_loss=0.5926.ckpt', map_location='cpu', weights_only=False)
model = Exp3ChampionLightningModule(**ckpt['hyper_parameters'])
model.load_state_dict(ckpt['state_dict'])
base_model = model.model
print(f"   ✅ Loaded: {sum(p.numel() for p in base_model.parameters()):,} params")

# 2. Setup LoRA
print("\n2. Setting up LoRA...")
lora_config = LoraConfig(
    inference_mode=False,
    r=config['lora_rank'],
    lora_alpha=config['lora_alpha'],
    lora_dropout=config['lora_dropout'],
    target_modules=config['target_modules'],
)
lora_model = get_peft_model(base_model, lora_config)
trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
print(f"   ✅ LoRA ready: {trainable:,} trainable params (rest frozen)")

# 3. Load data (use same dataloader as production script)
print("\n3. Loading one task...")
data_dir = Path(config['data_dir'])
test_file = list(data_dir.glob("*.json"))[0]
loader = create_champion_dataloader(
    [test_file],
    batch_size=4,
    shuffle=False,
    num_context_pairs=2,
    max_grid_size=35,
)
print(f"   ✅ Task {test_file.stem}: {len(loader)} batches")

# 4. Test forward pass
print("\n4. Testing forward pass (1 batch, no training)...")
batch = next(iter(loader))
src, tgt, ctx_in, ctx_out, src_shapes, tgt_shapes, task_ids = batch

try:
    with torch.no_grad():
        logits = lora_model(
            src=src,
            tgt=tgt[:, :-1],  # Teacher forcing
            src_grid_shape=src_shapes[0],
            tgt_grid_shape=tgt_shapes[0],
            ctx_input=ctx_in,
            ctx_output=ctx_out
        )
    print(f"   ✅ Forward pass works: output shape {logits.shape}")
except Exception as e:
    print(f"   ❌ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 5. Test backward pass
print("\n5. Testing backward pass (1 step, no optimization)...")
try:
    logits = lora_model(
        src=src,
        tgt=tgt[:, :-1],
        src_grid_shape=src_shapes[0],
        tgt_grid_shape=tgt_shapes[0],
        ctx_input=ctx_in,
        ctx_output=ctx_out
    )
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        tgt[:, 1:].reshape(-1),
        ignore_index=10
    )
    loss.backward()
    print(f"   ✅ Backward pass works: loss={loss.item():.4f}")
except Exception as e:
    print(f"   ❌ Backward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 6. Test saving
print("\n6. Testing adapter save/load...")
try:
    output_dir = Path("outputs/test_lora_minimal")
    output_dir.mkdir(parents=True, exist_ok=True)
    lora_model.save_pretrained(output_dir)
    print(f"   ✅ Saved to {output_dir}")
    
    # Test loading
    from src.lora_utils import flatten_adapter
    flat = flatten_adapter(str(output_dir))
    print(f"   ✅ Reloaded: {flat.shape[0]:,} params")
except Exception as e:
    print(f"   ❌ Save/load failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("🎉 ALL TESTS PASSED")
print("="*60)
print("\nPipeline is ready. LoRA:")
print(f"  - Correctly freezes base model")
print(f"  - Adds {trainable:,} trainable params")
print(f"  - Forward/backward work")
print(f"  - Save/load work")
print("\nNext: Run full training with `python scripts/train_atomic_loras.py`")
