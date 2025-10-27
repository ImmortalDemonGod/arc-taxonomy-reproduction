# Reproduction Package

**Status:** In development (Week 1-2)  
**Purpose:** Standalone package for reproducing paper results

---

## ARC Taxonomy Reproduction Package

**Built on Stanford CS336 Foundation**

This package provides a minimal, verifiable implementation for reproducing the ARC Taxonomy paper experiments. Built from scratch using the cs336_basics reference implementation, this codebase demonstrates that each architectural component is empirically necessary through incremental ablation experiments.

## Strategic Approach: Build-Up vs Tear-Down

**Why we started from cs336_basics instead of stripping down existing code:**

- **Credibility:** Built on Stanford's academically-validated reference implementation
- **Clarity:** Every component justified through empirical ablation study
- **Reproducibility:** Clean, minimal codebase (~450 lines of ARC-specific code)
- **Scientific Value:** Implementation process generates core paper evidence

## Contents

- `model.py` - Simplified Transformer implementation
- `finetune.py` - Fine-tuning script for individual tasks
- `validate_classifier.py` - Taxonomy classifier validation
- `data_utils.py` - Data loading and preprocessing utilities
- `weights/` - Pre-trained model weights
- `tests/` - Unit tests for all components
- `requirements.txt` - Pinned dependencies

---

## Quick Start

### Prerequisites

```bash
# Python 3.10+
python --version

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Pre-Trained Weights

```bash
# Download from [URL to be provided]
cd weights/
# wget [URL]/pretrained_model.pt
# Or follow instructions in weights/README.md
```

### Validate Taxonomy Classifier

```bash
# Run classifier validation
python validate_classifier.py

# Expected output:
# ✓ Classifier accuracy: 97.5% (390/400 tasks)
# ✓ Runtime: ~2 minutes
```

### Fine-Tune on Specific Task

```bash
# Fine-tune on task 137eaa0f (A2 category)
python finetune.py --task_id 137eaa0f --category A2 --epochs 100

# Expected output:
# ✓ Final accuracy: ~17% (ceiling effect for A2)
# ✓ Runtime: ~30 minutes
```

---

## Expected Results

### Taxonomy Classification
- **Accuracy:** 97.5% (390/400 tasks correctly classified)
- **Runtime:** ~2 minutes on CPU
- **Output:** Classification results JSON file

### Fine-Tuning Experiments
| Category | Example Task | Expected Accuracy | Runtime |
|----------|--------------|-------------------|---------|
| C1 (High) | [task_id] | 60%+ | ~30 min |
| S3 (Low) | [task_id] | 25-30% | ~30 min |
| A2 (Very Low) | 137eaa0f | ~17% (ceiling) | ~30 min |

---

## File Descriptions

### `model.py`
Simplified Transformer implementation extracted from the main codebase:
- Standard Transformer architecture
- Input/output tokenization for ARC grids
- Pre-training compatible

### `finetune.py`
Fine-tuning script for individual ARC tasks:
- Loads pre-trained weights
- Fine-tunes on single task
- Tracks training dynamics
- Saves results

### `validate_classifier.py`
Validates the taxonomy classifier:
- Loads all 400 task definitions
- Runs rule-based classifier
- Compares against ground truth
- Reports accuracy metrics

### `data_utils.py`
Data loading and preprocessing:
- ARC task loading
- Grid tokenization
- Data augmentation
- Batch preparation

---

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_model.py

# Run with coverage
pytest --cov=. --cov-report=html
```

---

## Troubleshooting

### Issue: Import errors
**Solution:** Ensure you're in the virtual environment and dependencies are installed

### Issue: CUDA out of memory
**Solution:** Reduce batch size or use CPU (`--device cpu`)

### Issue: Weights not found
**Solution:** Download weights following instructions in `weights/README.md`

---

## Citation

If you use this reproduction package, please cite:

```bibtex
@article{author2025arc,
  title={An Empirical Answer to re-arc: A Systematic Taxonomy, 
         Curriculum Analysis, and Neural Affinity Framework for the 
         Abstraction and Reasoning Corpus},
  author={Author Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

---

## Contact

For questions or issues with reproduction:
- Open an issue on GitHub
- Email: [author email]

---

**Last Updated:** October 23, 2025  
**Status:** Files to be created in Week 1-2
