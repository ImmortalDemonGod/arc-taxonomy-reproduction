# Quick Start Guide - ARC Taxonomy Reproduction

**Date:** October 27, 2025  
**Status:** Cloud-Ready Standalone Package

---

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM
- ~2GB disk space for code + data

---

## Installation (3 Steps)

### 1. Clone/Copy Repository

```bash
# If from git
git clone <repo_url>
cd reproduction/

# Or just copy the reproduction/ folder to your cloud instance
```

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
# Quick test (1 batch per model)
python scripts/test_all_training.py
```

**Expected output:**
```
✅ Decoder-Only: PASSED (530K params)
✅ Encoder-Decoder: PASSED (928K params)
✅ Champion: PASSED (1.7M params)

🎉 ALL MODELS READY FOR TRAINING!
```

---

## Training (Choose Your Experiment)

### Option 1: Champion Model (Recommended)

```bash
python scripts/train_champion.py
```

**Configuration:**
- Architecture: 1.7M parameters
- Layers: 1 encoder + 3 decoder
- Training: Adam, LR=0.00185, batch=32
- Data: 18 tasks (14 train, 4 val)
- Loss: CrossEntropyLoss (Option A)

**Checkpoints saved to:** `checkpoints/exp_3_champion/`

### Option 2: Encoder-Decoder Baseline

```bash
python scripts/train_encoder_decoder.py
```

**Configuration:**
- Architecture: 928K parameters  
- Layers: 2 encoder + 2 decoder
- Standard transformer baseline

**Checkpoints saved to:** `checkpoints/exp_0_encoder_decoder/`

### Option 3: Decoder-Only Baseline

```bash
python scripts/train_decoder_only.py
```

**Configuration:**
- Architecture: 530K parameters
- Decoder-only with RoPE
- Simplest baseline

**Checkpoints saved to:** `checkpoints/exp_-1_decoder_only/`

---

## Monitoring Training

### TensorBoard (Recommended)

```bash
# In a separate terminal
tensorboard --logdir=lightning_logs/

# Open browser to http://localhost:6006
```

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
!git clone <repo_url>
%cd reproduction/
!pip install -r requirements.txt

# In second cell
!python scripts/train_champion.py
```

### Paperspace / Lambda Labs

```bash
# Usually pre-configured with CUDA
# Just follow standard installation
pip install -r requirements.txt
python scripts/train_champion.py
```

---

## Troubleshooting

### Issue: "CUDA out of memory"

**Solution:** Reduce batch size in training script:
```python
# In train_champion.py, line ~48
batch_size=16,  # Reduce from 32
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

### Issue: "Task files not found"

**Solution:** Ensure data is in correct location:
```
reproduction/
├── data/
│   └── tasks/
│       ├── 05269061.json
│       ├── 1190e5a7.json
│       └── ... (18 total)
```

Copy from: `data/tasks/*.json`

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
# Train all baselines
python scripts/train_decoder_only.py
python scripts/train_encoder_decoder.py  
python scripts/train_champion.py

# Compare results
```

---

## Configuration (Advanced)

### Modify Hyperparameters

Edit training scripts directly:

```python
# In train_champion.py, line ~71
model = ChampionLightningModule(
    d_model=160,              # Model dimension
    num_decoder_layers=3,     # Decoder depth
    d_ff=640,                 # FFN dimension
    learning_rate=0.00185,    # Learning rate
    dropout=0.167,            # Dropout rate
    # ... etc
)
```

### Change Data Split

```python
# In train_champion.py, line ~37
split_idx = int(len(task_files) * 0.8)  # Change 0.8 to desired ratio
```

### Adjust Training Duration

```python
# In train_champion.py, line ~100
trainer = pl.Trainer(
    max_epochs=100,           # Maximum epochs
    patience=7,               # Early stopping patience
    # ... etc
)
```

---

## Support & Documentation

**Full Documentation:**
- `README.md` - Project overview
- `docs/TRAINING_READY_SUMMARY_OCT27.md` - Complete status
- `docs/PARAMETER_COUNT_FIX.md` - Architecture details
- `docs/DROPOUT_AND_CONFIG_FIXES.md` - Configuration details

**Testing:**
- `scripts/test_all_training.py` - Quick validation
- `scripts/smoke_test_all.py` - Detailed smoke tests
- `tests/` - Unit tests (run with `pytest`)

**Issue? Check:**
1. Python version (3.10+)
2. GPU availability (`nvidia-smi`)
3. CUDA version (11.7+)
4. Disk space (>2GB free)
5. RAM (>8GB)

---

## Example: Complete Cloud Workflow

```bash
# 1. SSH into cloud GPU instance
ssh user@gpu-instance

# 2. Setup
git clone <repo>
cd reproduction/
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Verify
python scripts/test_all_training.py

# 4. Start training in background
nohup python scripts/train_champion.py > train.log 2>&1 &

# 5. Monitor
tail -f train.log

# 6. Or use screen/tmux
screen -S training
python scripts/train_champion.py
# Ctrl+A, D to detach
# screen -r training to reattach

# 7. Download results when done
scp -r user@gpu-instance:~/reproduction/checkpoints/ ./local/
```

---

**Ready to train!** 🚀

**Estimated setup time:** 5-10 minutes  
**Estimated training time:** 2-4 hours (GPU)  
**Expected result:** 1.7M parameter champion model trained on 18 ARC tasks
