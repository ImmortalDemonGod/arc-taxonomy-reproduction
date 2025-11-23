# Quick Start Guide - ARC Taxonomy Reproduction

**Date:** November 2025  
**Status:** Release-Ready Reproduction Package  
**Repository:** https://github.com/ImmortalDemonGod/arc-taxonomy-reproduction

---

## Prerequisites

- **Python 3.10 or 3.11** (3.11 recommended)
- **CUDA-capable GPU** (highly recommended) or CPU
- **8GB+ RAM** (16GB recommended for larger experiments)
- **~3GB disk space** for code + data + checkpoints
- **Git** with Git LFS support (for submodules)

---

## Installation (3 Steps)

### 1. Clone/Copy Repository

```bash
# Clone repository with submodules
git clone --recursive https://github.com/ImmortalDemonGod/arc-taxonomy-reproduction.git
cd arc-taxonomy-reproduction/

# If you forgot --recursive, initialize submodules:
git submodule update --init --recursive
```

**Important:** The `re-arc` submodule is required for data generation.

### 2. Create Virtual Environment

```bash
# Create venv
python3 -m venv venv

# Activate
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
# Install package in development mode
pip install -e .

# Or just install requirements
pip install -r requirements.txt
```

---

## Verify Installation

```bash
# Option 1: Comprehensive verification (recommended)
python verify_setup.py

# Option 2: Quick training test (1 batch per model)
python scripts/test_all_training.py
```

**Expected output from test_all_training.py:**
```
✅ Decoder-Only: PASSED (530K params)
✅ Encoder-Decoder: PASSED (928K params)
✅ Champion: PASSED (1.7M params)

🎉 ALL MODELS READY FOR TRAINING!
```

**Expected output from verify_setup.py:**
```
✓ Python version: 3.11.x
✓ PyTorch installed with CUDA support
✓ All required packages present
✓ Data directories exist
✓ re-arc submodule initialized

✅ Setup verification complete!
```

---

## Training (Choose Your Experiment)

### Option 1: Champion Model (Recommended)

```bash
python scripts/train_exp3_champion.py
```

**Configuration:**
- **Architecture:** 1.7M parameters
- **Layers:** 1 encoder + 3 decoder
- **Features:** Grid2D positional encoding + Permutation-invariant embeddings + Context bridge
- **Training:** Adam optimizer, LR=0.00185, batch_size=32
- **Data:** Foundational skills dataset (14 train, 4 val tasks)
- **Loss:** CrossEntropyLoss
- **Max Grid Size:** 30 (critical for training stability)

**Checkpoints saved to:** `checkpoints/exp_3_champion/`  
**Logs:** `lightning_logs/`

### Option 2: Encoder-Decoder Baseline

```bash
python scripts/train_exp0_encoder_decoder.py
```

**Configuration:**
- Architecture: 928K parameters  
- Layers: 2 encoder + 2 decoder
- Standard transformer baseline

**Checkpoints saved to:** `checkpoints/exp_0_encoder_decoder/`

### Option 3: Decoder-Only Baseline

```bash
python scripts/train_baseline_decoder_only.py
```

**Configuration:**
- **Architecture:** 530K parameters
- **Type:** Decoder-only with RoPE (Rotary Position Embeddings)
- **Purpose:** Simplest baseline for comparison

**Checkpoints saved to:** `checkpoints/exp_-1_decoder_only/`

---

## Monitoring Training

### TensorBoard (Optional)

```bash
# Install if not already present
pip install tensorboard

# In a separate terminal
tensorboard --logdir=lightning_logs/

# Open browser to http://localhost:6006
```

**Metrics tracked:**
- Training/validation loss per epoch
- Grid-level and cell-level accuracy
- Learning rate schedule
- Per-task performance (when applicable)

### Console Output

Training scripts print:
- Train/val loss per epoch
- Validation accuracy
- Learning rate
- ETA and progress bar

---

## Expected Training Time

**On GPU (V100/A100):**
- Champion: ~2-4 hours (50 epochs typical)
- Encoder-Decoder: ~1-2 hours
- Decoder-Only: ~30-60 minutes

**On CPU (not recommended):**
- Champion: ~10-20 hours
- Encoder-Decoder: ~5-10 hours
- Decoder-Only: ~2-4 hours

---

## Cloud-Specific Setup

### AWS EC2

```bash
# Recommended: p3.2xlarge (V100 GPU)
# Or: g4dn.xlarge (T4 GPU) for budget

# Install CUDA if needed
sudo apt-get update
sudo apt-get install -y nvidia-cuda-toolkit

# Then follow standard installation above
```

### Google Colab

```python
# In first cell
!git clone --recursive https://github.com/ImmortalDemonGod/arc-taxonomy-reproduction.git
%cd arc-taxonomy-reproduction/
!pip install -e .

# Verify setup
!python verify_setup.py

# In second cell - train champion model
!python scripts/train_exp3_champion.py
```

### Paperspace / Lambda Labs

```bash
# Usually pre-configured with CUDA
# Clone with submodules
git clone --recursive https://github.com/ImmortalDemonGod/arc-taxonomy-reproduction.git
cd arc-taxonomy-reproduction/

# Install and verify
pip install -e .
python verify_setup.py

# Train
python scripts/train_exp3_champion.py
```

---

## Troubleshooting

### Issue: "CUDA out of memory"

**Solution:** Reduce batch size in training script:
```python
# In train_exp3_champion.py, around line ~150
# Find: batch_size=32
# Change to: batch_size=16
```

Or use gradient accumulation:
```python
# In trainer configuration
trainer = pl.Trainer(
    accumulate_grad_batches=2,  # Effectively doubles batch size
    # ...
)
```

### Issue: "No module named 'src'"

**Solution:** Install package properly:
```bash
pip install -e .
```

Or add to PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue: "Task files not found" or "re-arc module not found"

**Solution:** Ensure submodules are initialized:
```bash
git submodule update --init --recursive
```

**Data directory structure:**
```
arc-taxonomy-reproduction/
├── data/
│   ├── distributional_alignment/  # 400 re-arc tasks
│   ├── taxonomy/                  # Task classifications
│   ├── arc_prize_2025/           # ARC-AGI-2 evaluation data
│   └── external_validation/       # ViTARC results
├── re-arc/                        # Submodule (auto-generated)
└── ...
```

If data is missing, some scripts can regenerate it:
```bash
python scripts/generate_synthetic_arc_dataset.py
```

---

## What Gets Created

**During Training:**
```
reproduction/
├── checkpoints/          # Model checkpoints
│   ├── exp_-1_decoder_only/
│   ├── exp_0_encoder_decoder/
│   └── exp_3_champion/
├── lightning_logs/       # TensorBoard logs
│   └── version_N/
└── logs/                 # Additional logs
```

**After Training:**
- Best model checkpoint (by val_loss)
- Last checkpoint
- TensorBoard event files
- Training metrics CSV

---

## Next Steps After Training

### 1. Evaluate Model

```python
# Load best checkpoint
from pytorch_lightning import Trainer
from src.models.champion_lightning import ChampionLightningModule

model = ChampionLightningModule.load_from_checkpoint(
    "checkpoints/exp_3_champion/champion-epoch=XX-val_loss=Y.YY.ckpt"
)

# Run evaluation
trainer = Trainer()
trainer.test(model, dataloaders=val_loader)
```

### 2. Analyze Results

```bash
# View TensorBoard logs
tensorboard --logdir=lightning_logs/

# Check convergence, val_loss trends, etc.
```

### 3. Run Ablation Studies

```bash
# Train all ablation experiments
python scripts/train_baseline_decoder_only.py     # Exp -1
python scripts/train_exp0_encoder_decoder.py      # Exp 0
python scripts/train_exp1_grid2d_pe.py            # Exp 1: +Grid2D
python scripts/train_exp2_perminv.py              # Exp 2: +PermInv
python scripts/train_exp3_champion.py             # Exp 3: Full champion

# Analyze results
python scripts/test_complete_ablation.py
```

---

## Configuration (Advanced)

### Modify Hyperparameters

Edit training scripts directly:

```python
# In train_exp3_champion.py, around line ~71
model = ChampionLightningModule(
    d_model=160,                    # Model dimension
    num_encoder_layers=1,           # Encoder depth
    num_decoder_layers=3,           # Decoder depth  
    d_ff=640,                       # FFN dimension
    learning_rate=0.00185,          # Learning rate
    dropout=0.167,                  # Dropout rate
    max_grid_size=30,               # Critical: DO NOT change
    # ... etc
)
```

**Warning:** Changing `max_grid_size` from 30 to 35 causes 31% performance degradation due to training instability.

### Change Data Split

```python
# In train_exp3_champion.py, around line ~37
split_idx = int(len(task_files) * 0.8)  # Change 0.8 to desired ratio
```

### Adjust Training Duration

```python
# In train_exp3_champion.py, around line ~100
trainer = pl.Trainer(
    max_epochs=100,                    # Maximum epochs
    callbacks=[early_stop_callback],   # Early stopping with patience=7
    # ... etc
)
```

---

## Support & Documentation

**Full Documentation:**
- `README.md` - Project overview and scientific context
- `QUICKSTART.md` - This guide
- `APPLYING_TO_YOUR_MODEL.md` - Adapting to other architectures
- `docs/methodology/` - Taxonomy development and validation
- `docs/results/` - Experimental results and analysis
- `docs/archive/` - Development logs and technical details

**Testing:**
- `verify_setup.py` - Comprehensive environment check
- `scripts/test_all_training.py` - Quick model validation
- `scripts/smoke_test_all.py` - Detailed smoke tests
- `tests/` - Unit tests (`pytest` with 82% coverage)

**Issue? Check:**
1. Python version: 3.10 or 3.11 (`python --version`)
2. GPU availability: `nvidia-smi` shows GPU
3. CUDA version: 11.7+ or 12.x (`nvcc --version`)
4. Disk space: >3GB free (`df -h`)
5. RAM: >8GB available (`free -h` on Linux)
6. Submodules: `re-arc/` directory exists
7. Dependencies: `pip install -e .` completed without errors

---

## Example: Complete Cloud Workflow

```bash
# 1. SSH into cloud GPU instance
ssh user@gpu-instance

# 2. Setup environment
git clone --recursive https://github.com/ImmortalDemonGod/arc-taxonomy-reproduction.git
cd arc-taxonomy-reproduction/
python3 -m venv venv
source venv/bin/activate
pip install -e .

# 3. Verify setup
python verify_setup.py

# 4. Start training in background with nohup
nohup python scripts/train_exp3_champion.py > train.log 2>&1 &
echo $! > train.pid  # Save process ID

# 5. Monitor progress
tail -f train.log

# 6. Alternative: Use screen/tmux for interactive session
screen -S arc_training
python scripts/train_exp3_champion.py
# Press Ctrl+A, then D to detach
# Reconnect later: screen -r arc_training

# 7. Check if training is still running
ps -p $(cat train.pid) || echo "Training complete"

# 8. Download results when done
scp -r user@gpu-instance:~/arc-taxonomy-reproduction/checkpoints/ ./local_checkpoints/
scp -r user@gpu-instance:~/arc-taxonomy-reproduction/lightning_logs/ ./local_logs/
```

---

---

## Additional Resources

### Hyperparameter Optimization (HPO)

For advanced users who want to explore hyperparameter space:

```bash
# Run Optuna-based HPO sweep
python scripts/optimize.py --config configs/hpo/champion_sweep.yaml

# Analyze results
python scripts/extract_per_category_metrics.py
```

See `docs/archive/dev_logs/HPO_*.md` for HPO system details.

### Atomic LoRA Training

Train task-specific LoRA adapters:

```bash
python scripts/train_atomic_loras.py --num_workers 4
```

### Data Generation

Regenerate synthetic ARC tasks from re-arc:

```bash
python scripts/generate_synthetic_arc_dataset.py
```

---

## Citation

If you use this reproduction package, please cite:

```bibtex
@article{ingram2025neuralaffinity,
  title={A Neural Affinity Framework for Abstract Reasoning: Diagnosing the 
         Compositional Gap in Transformer Architectures via Procedural Task Taxonomy},
  author={Ingram, Miguel and Merritt, Arthur},
  year={2025}
}
```

---

**Ready to train!** 🚀

| Metric | Value |
|--------|-------|
| **Setup time** | 5-10 minutes |
| **Training time** | 2-4 hours (V100/A100 GPU) |
| **Expected result** | 1.7M parameter champion model |
| **Dataset** | 14 foundational tasks (train), 4 validation |
| **Repository** | [GitHub](https://github.com/ImmortalDemonGod/arc-taxonomy-reproduction) |

**Questions or issues?** Open an issue on GitHub or check the comprehensive documentation in `README.md` and `docs/`.
