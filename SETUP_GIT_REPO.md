# Setting Up Standalone Git Repository

**Date:** October 27, 2025, 12:05 PM  
**Purpose:** Create standalone git repo for cloud deployment

---

## Current Situation

The `reproduction/` folder is currently tracked in the parent `Holistic-Performance-Enhancement` repository. We need to make it a standalone repo that can be independently cloned on cloud GPU instances.

---

## Option 1: Initialize New Repo (Recommended)

### Step 1: Initialize Git in reproduction folder

```bash
cd /Users/tomriddle1/Holistic-Performance-Enhancement/cultivation/systems/arc_reactor/publications/arc_taxonomy_2025/reproduction

# Initialize new git repo
git init

# Check status
git status
```

### Step 2: Add all files

```bash
# Add all reproduction files
git add .

# Check what will be committed
git status
```

### Step 3: Create initial commit

```bash
# Commit with message
git commit -m "Initial commit: ARC Taxonomy Reproduction Package v1.0

- 3 baseline architectures (530K, 928K, 1.7M params)
- Trial 69 configuration matched
- Separate encoder/decoder dropout rates
- 18 task dataset
- Option A: CrossEntropyLoss baseline
- Cloud-ready standalone package
- Complete documentation and deployment guides"
```

### Step 4: Create GitHub repository (Optional)

```bash
# On GitHub, create new repository: arc-taxonomy-reproduction
# Then link it:

git remote add origin https://github.com/YOUR_USERNAME/arc-taxonomy-reproduction.git
git branch -M main
git push -u origin main
```

---

## Option 2: Extract from Parent Repo

If you want to preserve the git history from the parent repo:

```bash
# Clone parent repo
git clone /path/to/Holistic-Performance-Enhancement temp-extract
cd temp-extract

# Filter to only reproduction folder
git filter-branch --subdirectory-filter cultivation/systems/arc_reactor/publications/arc_taxonomy_2025/reproduction -- --all

# This creates a repo with only reproduction/ content
# Then push to new remote
```

---

## Verify Standalone Repo

### Test local clone:

```bash
# Try cloning locally
cd /tmp
git clone /Users/tomriddle1/.../reproduction test-clone
cd test-clone

# Verify
python verify_setup.py
./run_training.sh test
```

### Test remote clone (after GitHub push):

```bash
# On another machine or in /tmp
git clone https://github.com/YOUR_USERNAME/arc-taxonomy-reproduction.git
cd arc-taxonomy-reproduction

# Setup
pip install -r requirements.txt
python verify_setup.py
```

---

## For Paperspace Deployment

### Once repo is on GitHub:

```bash
# On Paperspace terminal
cd /notebooks
git clone https://github.com/YOUR_USERNAME/arc-taxonomy-reproduction.git
cd arc-taxonomy-reproduction

# Install and verify
pip install -r requirements.txt
python verify_setup.py

# Train
./run_training.sh champion
```

---

## Ignore in Parent Repo

To prevent the reproduction repo from being committed to the parent:

```bash
# In parent repo root
cd /Users/tomriddle1/Holistic-Performance-Enhancement

# Add to .gitignore
echo "cultivation/systems/arc_reactor/publications/arc_taxonomy_2025/reproduction/.git" >> .gitignore
echo "cultivation/systems/arc_reactor/publications/arc_taxonomy_2025/reproduction/__pycache__" >> .gitignore
echo "cultivation/systems/arc_reactor/publications/arc_taxonomy_2025/reproduction/checkpoints" >> .gitignore
echo "cultivation/systems/arc_reactor/publications/arc_taxonomy_2025/reproduction/lightning_logs" >> .gitignore

# Or better: ignore the whole folder as a submodule
echo "cultivation/systems/arc_reactor/publications/arc_taxonomy_2025/reproduction/" >> .gitignore
```

---

## Quick Commands

```bash
# ==============================================================================
# COPY-PASTE THIS TO CREATE REPO
# ==============================================================================

cd /Users/tomriddle1/Holistic-Performance-Enhancement/cultivation/systems/arc_reactor/publications/arc_taxonomy_2025/reproduction

# Initialize repo
git init
git add .
git commit -m "Initial commit: ARC Taxonomy Reproduction v1.0 - Cloud-ready package with 1.7M param champion model"

# Link to GitHub (optional - do this after creating repo on GitHub)
# git remote add origin https://github.com/YOUR_USERNAME/arc-taxonomy-reproduction.git
# git branch -M main
# git push -u origin main

# Verify
git log --oneline
git status

echo "✅ Git repo created!"
echo "Next: Create GitHub repo and push, or use Paperspace's git clone from local path"

# ==============================================================================
```

---

## Alternative: Use as Tarball

If you don't want to use GitHub:

```bash
# Create distributable package
cd /Users/tomriddle1/Holistic-Performance-Enhancement/cultivation/systems/arc_reactor/publications/arc_taxonomy_2025

tar -czf arc-taxonomy-reproduction-v1.0.tar.gz \
    --exclude='reproduction/checkpoints' \
    --exclude='reproduction/lightning_logs' \
    --exclude='reproduction/__pycache__' \
    --exclude='reproduction/.git' \
    --exclude='reproduction/*_backup' \
    reproduction/

# Upload to Paperspace
# Then extract:
tar -xzf arc-taxonomy-reproduction-v1.0.tar.gz
cd reproduction/
pip install -r requirements.txt
```

---

## Recommended: GitHub Approach

**Why GitHub:**
- ✅ Easy to clone on any cloud platform
- ✅ Version control for updates
- ✅ Can make it public or private
- ✅ Easy collaboration
- ✅ Free for public repos

**Steps:**
1. Initialize git in reproduction/ folder
2. Commit all files
3. Create GitHub repo
4. Push to GitHub
5. Clone on Paperspace: `git clone https://github.com/USER/arc-taxonomy-reproduction.git`

---

## File Checklist

Make sure these are in the repo:

### Core Files
- [x] `requirements.txt` - Dependencies
- [x] `setup.py` - Package installer
- [x] `README.md` - Overview
- [x] `.gitignore` - Ignore patterns

### Documentation
- [x] `QUICKSTART.md`
- [x] `CLOUD_DEPLOYMENT.md`
- [x] `PACKAGE_READY.md`
- [x] `docs/` - All documentation

### Code
- [x] `src/` - Source code
- [x] `scripts/` - Training scripts
- [x] `tests/` - Unit tests

### Data
- [x] `data/tasks/` - 18 JSON files

### Tools
- [x] `verify_setup.py`
- [x] `run_training.sh`

---

## After Setup

### Update README with clone instructions:

```markdown
## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/arc-taxonomy-reproduction.git
cd arc-taxonomy-reproduction
pip install -r requirements.txt
python verify_setup.py
./run_training.sh champion
```
```

---

**Created:** October 27, 2025, 12:05 PM  
**Status:** Ready to initialize standalone git repo
