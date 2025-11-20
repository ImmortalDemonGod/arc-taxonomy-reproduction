# Model Checkpoints

This directory contains PyTorch Lightning checkpoints for the Champion architecture and merged LoRA adaptations.

---

## Available Checkpoints

### 1. `champion-epoch=36-val_loss=0.5926.ckpt` (21 MB)

**Source:** Trial 69 from V3 architectural sweep (October 2025)  
**Architecture:** Champion V3 (Encoder-Decoder + Grid2D PE + PermInv + Context Bridge)  
**Parameters:** 1.7M  
**Training Data:** 400 re-arc tasks (distributional_alignment dataset)  
**Training Duration:** 36 epochs  
**Performance:** 2.34% grid accuracy on re-arc validation set

**Usage:**
```python
# Load for inference
from src.models.exp3_champion_lightning import Exp3ChampionLightningModule
import torch

ckpt = torch.load('weights/champion-epoch=36-val_loss=0.5926.ckpt', map_location='cpu')
model = Exp3ChampionLightningModule(**ckpt['hyper_parameters'])
model.load_state_dict(ckpt['state_dict'])

# Or use for transfer learning (ARC-AGI-2)
python scripts/train_exp3_champion.py \
    --dataset arc-agi-2 \
    --checkpoint weights/champion-epoch=36-val_loss=0.5926.ckpt
```

**Contents:**
- `state_dict`: Model weights
- `hyper_parameters`: Complete model configuration
- `optimizer_states`: Adam optimizer state
- `lr_schedulers`: CosineAnnealingWarmRestarts state

---

### 2. `champion_merged_loras.ckpt` (14 MB)

**Source:** Merged atomic LoRAs from 18 foundational tasks  
**Base Model:** Champion V3 (same as above)  
**Training:** Each task fine-tuned with LoRA adapters (400 examples/task)  
**Merge Method:** Weighted average of task-specific LoRA weights  

**Usage:**
```python
# Load merged LoRA checkpoint
ckpt = torch.load('weights/champion_merged_loras.ckpt', map_location='cpu')
model = Exp3ChampionLightningModule(**ckpt['hyper_parameters'])
model.load_state_dict(ckpt['state_dict'])

# Or use for ARC-AGI-2 transfer
python scripts/train_exp3_champion.py \
    --dataset arc-agi-2 \
    --checkpoint weights/champion_merged_loras.ckpt
```

**Purpose:** Tests whether task-specific adaptations can be merged for improved transfer learning on ARC-AGI-2.

---

## Checkpoint Format

All checkpoints are PyTorch Lightning `.ckpt` files containing:

```python
{
    'state_dict': {...},              # Model weights
    'hyper_parameters': {...},        # Model config
    'optimizer_states': [...],        # Optimizer state
    'lr_schedulers': [...],          # Scheduler state
    'epoch': int,                     # Training epoch
    'global_step': int,               # Total steps
    'callbacks': {...}                # Callback states
}
```

---

## Related Scripts

- **Training:** `scripts/train_exp3_champion.py` (supports both re-arc and ARC-AGI-2)
- **LoRA merging:** `scripts/merge_loras.py` (creates merged checkpoint)
- **Atomic LoRA training:** `scripts/train_atomic_loras.py` (generates task-specific adapters)
- **Testing:** `scripts/test_lora_minimal.py` (loads and validates checkpoints)

---

## Notes

- Checkpoints are compatible with PyTorch Lightning 2.0+
- Loading with `weights_only=False` is required (contains optimizer states)
- Both checkpoints use the same Champion V3 architecture
- Merged LoRA checkpoint is smaller (14MB vs 21MB) due to selective weight merging
