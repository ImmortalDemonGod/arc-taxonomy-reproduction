# Cloud Deployment Checklist

**Package:** ARC Taxonomy Reproduction  
**Date:** October 27, 2025, 11:55 AM  
**Status:** ✅ CLOUD-READY STANDALONE PACKAGE

---

## Pre-Deployment Verification

### ✅ Package Structure
```
reproduction/
├── requirements.txt        ✅ Updated with proper constraints
├── setup.py               ✅ Package installer
├── QUICKSTART.md          ✅ Quick start guide
├── verify_setup.py        ✅ Setup verification script
├── run_training.sh        ✅ Training launcher
├── src/                   ✅ Source code
│   ├── models/            ✅ 3 baseline architectures
│   ├── data/              ✅ Data loaders
│   └── ...
├── scripts/               ✅ Training scripts
│   ├── train_champion.py
│   ├── train_encoder_decoder.py
│   ├── train_decoder_only.py
│   └── test_all_training.py
├── data/                  ✅ Task files (18 JSON files)
└── tests/                 ✅ Unit tests
```

### ✅ Local Verification Complete
```bash
$ python verify_setup.py

✅ Python Version: v3.11.9
✅ Dependencies: All installed
✅ Data Files: 18 task files
✅ Model Imports: All working
✅ Training Scripts: All present
✅ Training Test: All models passed (1.7M params champion)
```

---

## Deployment Steps

### Step 1: Copy Package to Cloud

**Option A: Git (Recommended)**
```bash
# On cloud instance
git clone <repo_url>
cd reproduction/
```

**Option B: Direct Copy**
```bash
# From local machine
tar -czf arc-taxonomy-reproduction.tar.gz reproduction/
scp arc-taxonomy-reproduction.tar.gz user@cloud-instance:~/

# On cloud instance
tar -xzf arc-taxonomy-reproduction.tar.gz
cd reproduction/
```

**Option C: Rsync (Best for updates)**
```bash
rsync -avz --progress reproduction/ user@cloud-instance:~/reproduction/
```

### Step 2: Verify Cloud Environment

```bash
# SSH into cloud
ssh user@cloud-instance
cd reproduction/

# Run verification
python verify_setup.py
```

**Expected Output:**
- ✅ Python 3.10+
- ✅ All dependencies
- ✅ GPU detected (if GPU instance)
- ✅ Data files present
- ✅ All models ready

### Step 3: Install Dependencies

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install package
pip install -e .

# Or just requirements
pip install -r requirements.txt
```

### Step 4: Start Training

**Quick test first:**
```bash
python scripts/test_all_training.py
```

**Start champion training:**
```bash
# Using launcher script
./run_training.sh champion

# Or directly
python scripts/train_champion.py

# In background
nohup python scripts/train_champion.py > train.log 2>&1 &

# With screen/tmux (recommended)
screen -S training
python scripts/train_champion.py
# Ctrl+A, D to detach
```

---

## Cloud Platform Specifics

### AWS EC2

**Recommended Instance:** p3.2xlarge (V100, $3.06/hr)

```bash
# Launch instance with Deep Learning AMI
# SSH in
ssh -i key.pem ubuntu@ec2-XX-XX-XX-XX.compute.amazonaws.com

# Clone repo
git clone <repo_url>
cd reproduction/

# Verify CUDA
nvidia-smi

# Install & run
pip install -r requirements.txt
python verify_setup.py
./run_training.sh champion
```

### Google Cloud (GCP)

**Recommended:** n1-standard-4 with V100 GPU

```bash
# Create instance with GPU
gcloud compute instances create arc-training \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-v100,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release

# SSH and setup
gcloud compute ssh arc-training
cd ~
git clone <repo_url>
cd reproduction/
pip install -r requirements.txt
```

### Lambda Labs / Paperspace

**Usually pre-configured with PyTorch + CUDA**

```bash
# Just clone and run
git clone <repo_url>
cd reproduction/
pip install -r requirements.txt
./run_training.sh champion
```

### Google Colab (Free Option)

```python
# Cell 1: Setup
!git clone <repo_url>
%cd reproduction
!pip install -r requirements.txt

# Cell 2: Verify
!python verify_setup.py

# Cell 3: Train
!python scripts/train_champion.py
```

---

## Monitoring Training

### Option 1: TensorBoard (Recommended)

```bash
# In separate terminal/tmux pane
tensorboard --logdir=lightning_logs/ --host=0.0.0.0 --port=6006

# If on cloud, forward port
ssh -L 6006:localhost:6006 user@cloud-instance

# Open browser: http://localhost:6006
```

### Option 2: Console Logs

```bash
# If running in background
tail -f train.log

# Live updates
watch -n 5 'tail -20 train.log'
```

### Option 3: Checkpoints

```bash
# Check latest checkpoint
ls -lht checkpoints/exp_3_champion/ | head -5

# Monitor val_loss
ls checkpoints/exp_3_champion/*.ckpt | while read f; do 
    echo "$f: $(echo $f | grep -oP 'val_loss=\K[^.]+')"; 
done
```

---

## Expected Training Behavior

### Champion Model (Exp 3)

**First Epoch:**
- Duration: ~5-10 minutes (GPU)
- Train loss: ~2.5-3.0 (initial)
- Val loss: ~2.3-2.8
- Val accuracy: ~3-10%

**After 10 Epochs:**
- Train loss: ~1.5-2.0
- Val loss: ~1.8-2.2  
- Val accuracy: ~20-40%

**Convergence (30-50 epochs):**
- Train loss: ~0.8-1.2
- Val loss: ~1.5-1.8
- Val accuracy: ~40-60%

**Early stopping** typically triggers around epoch 30-50.

---

## Troubleshooting

### Issue: Out of Memory

```python
# In training script, reduce batch size
batch_size=16,  # or 8

# Or enable gradient checkpointing
trainer = pl.Trainer(
    gradient_clip_val=1.0,
    accumulate_grad_batches=2,  # Effective batch=64
)
```

### Issue: Slow Training

```bash
# Check GPU utilization
nvidia-smi -l 1

# If low (<50%), increase num_workers
# In data loader creation
num_workers=4,  # Add this
```

### Issue: Connection Lost

```bash
# Use screen/tmux (prevents interruption)
screen -S training
python scripts/train_champion.py

# Detach: Ctrl+A, D
# Reattach: screen -r training
```

### Issue: Checkpoint Loading Fails

```bash
# Check file permissions
ls -l checkpoints/

# Check disk space
df -h

# Verify checkpoint
python -c "import torch; ckpt=torch.load('checkpoints/exp_3_champion/last.ckpt', map_location='cpu'); print('OK')"
```

---

## Downloading Results

### After Training Complete

```bash
# Download checkpoints
scp -r user@cloud-instance:~/reproduction/checkpoints/ ./local/

# Download logs
scp -r user@cloud-instance:~/reproduction/lightning_logs/ ./local/

# Download specific checkpoint
scp user@cloud-instance:~/reproduction/checkpoints/exp_3_champion/best.ckpt ./
```

### Using rsync (Better for large files)

```bash
rsync -avz --progress \
  user@cloud-instance:~/reproduction/checkpoints/ \
  ./local/checkpoints/
```

---

## Cost Estimates

### Per Experiment (Champion Model)

**AWS p3.2xlarge (V100):**
- Cost: $3.06/hour
- Training time: 2-4 hours
- Total: **$6-12 per experiment**

**GCP n1-standard-4 + V100:**
- Cost: ~$2.50/hour
- Training time: 2-4 hours
- Total: **$5-10 per experiment**

**Lambda Labs (V100):**
- Cost: $0.50/hour
- Training time: 2-4 hours
- Total: **$1-2 per experiment**

**All 3 Experiments:**
- Champion: $6-12
- Encoder-Decoder: $3-6
- Decoder-Only: $1-3
- **Total: $10-21**

---

## Post-Training Checklist

- [ ] Training completed without errors
- [ ] Best checkpoint saved
- [ ] Val accuracy > 40% (champion)
- [ ] TensorBoard logs generated
- [ ] Downloaded checkpoints locally
- [ ] Downloaded training logs
- [ ] Verified checkpoint loads correctly
- [ ] Documented final metrics
- [ ] Terminated cloud instance (💰 save money!)

---

## Quick Commands Reference

```bash
# Verify setup
python verify_setup.py

# Quick test
./run_training.sh test

# Train champion
./run_training.sh champion

# Train all
./run_training.sh all

# Background training
nohup ./run_training.sh champion > train.log 2>&1 &

# Monitor
tail -f train.log

# TensorBoard
tensorboard --logdir=lightning_logs/ --host=0.0.0.0 --port=6006

# Check GPU
nvidia-smi

# Check disk
df -h

# Check processes
ps aux | grep python
```

---

## Emergency Recovery

### Training Crashed

```bash
# Check last log
tail -100 train.log

# Resume from last checkpoint
python scripts/train_champion.py \
  --resume_from_checkpoint checkpoints/exp_3_champion/last.ckpt
```

### Out of Disk Space

```bash
# Clean lightning logs
rm -rf lightning_logs/version_0/
rm -rf lightning_logs/version_1/

# Clean old checkpoints (keep best)
cd checkpoints/exp_3_champion/
ls -t | tail -n +4 | xargs rm --
```

### Instance Terminated

```bash
# Checkpoints auto-saved every epoch
# Just restart from last.ckpt
python scripts/train_champion.py \
  --resume_from_checkpoint checkpoints/exp_3_champion/last.ckpt
```

---

## Success Criteria

✅ **Ready for Production Training:**
- [x] Package verified locally
- [x] All tests passing
- [x] 1.7M parameter champion model
- [x] Separate dropout rates configured
- [x] Trial 69 hyperparameters matched
- [x] Training scripts tested
- [x] Data pipeline validated
- [x] Documentation complete

✅ **Cloud Deployment Ready:**
- [x] Standalone package (no external dependencies)
- [x] Requirements.txt complete
- [x] Setup.py for easy installation
- [x] Verification script
- [x] Launch script
- [x] Quick start guide
- [x] Troubleshooting docs

**Status: 🚀 READY FOR CLOUD TRAINING**

---

**Last Updated:** October 27, 2025, 11:55 AM  
**Verification:** All local tests passing, 18 tasks ready  
**Estimated Cloud Cost:** $10-20 for full ablation study
