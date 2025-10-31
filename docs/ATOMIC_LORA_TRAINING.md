# Visual Taxonomy Classifier

Grid-based visual classifier for validating ARC Taxonomy generalizability to ARC-AGI-2.

## Overview

This module implements a two-phased approach to classify ARC tasks based on visual grid patterns:

**Phase 1 (MANDATORY):** Direct Classification
- Train 400 atomic LoRA adapters (one per task)
- Merge into 9 category-specific LoRAs
- Train simple CNN to classify grid pairs → category labels
- Apply to ARC-AGI-2 for validation

**Phase 2 (ASPIRATIONAL):** Oracle-Predictor System  
- Use inference-time optimization to discover skill signatures
- Train TaskEncoder to predict 400-dim λ vectors
- Enables richer interpretability and task similarity analysis

## Directory Structure

```
visual_classification/
├── src/                    # Source modules
│   ├── vector_utils.py     # LoRA vector manipulation (unflatten_adapter)
│   ├── arc_task_dataset.py # PyTorch dataset for ARC tasks
│   └── task_encoder.py     # CNN/ContextEncoder architectures
│
├── scripts/               # Training & inference scripts
│   ├── 1_train_atomic_skills.py        # Phase 0
│   ├── 2_merge_category_skills.py      # Phase 1B
│   ├── phase1_train_direct_classifier.py   # Phase 1C
│   ├── phase1_classify_arc_agi_2.py        # Phase 1D
│   ├── phase2_generate_oracle_signatures.py # Phase 2A
│   ├── phase2_train_oracle_predictor.py    # Phase 2B
│   └── phase2_analyze_arc_agi_2.py         # Phase 2C
│
├── configs/               # YAML configuration files
│   ├── lora_config.yaml   # LoRA hyperparameters & target modules
│   └── encoder_config.yaml # TaskEncoder architecture config
│
├── skill_library/         # Trained LoRA adapters
│   ├── atomic_skills/     # 400 task-specific adapters
│   └── category_skills/   # 9 merged category adapters
│
├── models/                # Trained TaskEncoder checkpoints
│   ├── task_encoder_direct_best.ckpt
│   └── task_encoder_oracle_best.ckpt
│
├── results/               # Classification results & metrics
│   ├── atomic_training_summary.json
│   ├── phase1_training_metrics.json
│   ├── arc_agi_2_classifications_phase1.json
│   └── phase2_complete_report.md
│
└── oracle_signatures/     # Phase 2: Lambda signature vectors
    ├── lambda_signatures.npy
    └── convergence_metrics.json
```

## Quick Start

### Prerequisites

```bash
# Ensure you're in the reproduction directory
cd /path/to/arc_taxonomy_2025/reproduction

# Activate environment
source /path/to/.venv/bin/activate

# Install dependencies
pip install peft safetensors torch torchvision pyyaml tqdm
```

### Phase 0: Train Atomic Skills

```bash
# Train 400 task-specific LoRA adapters
python visual_classification/scripts/1_train_atomic_skills.py

# Expected output:
# - skill_library/atomic_skills/{task_id}/adapter_model.safetensors (400 files)
# - results/atomic_training_summary.json

# Time: ~30-60 GPU hours (parallelizable to 3-6 wall-clock hours)
```

### Phase 1B: Merge Category Skills

```bash
# Average atomic LoRAs by category
python visual_classification/scripts/2_merge_category_skills.py

# Expected output:
# - skill_library/category_skills/{S1,S2,S3,C1,C2,K1,L1,A1,A2}_adapter.safetensors
# - skill_library/category_centroids.npy

# Time: <1 hour
```

### Phase 1C: Train Direct Classifier

```bash
# Train CNN to map grids → categories
python visual_classification/scripts/phase1_train_direct_classifier.py

# Expected output:
# - models/task_encoder_direct_best.ckpt
# - results/phase1_training_metrics.json

# Time: 5-10 GPU hours
```

### Phase 1D: Classify ARC-AGI-2

```bash
# Apply classifier to ARC-AGI-2
python visual_classification/scripts/phase1_classify_arc_agi_2.py

# Expected output:
# - results/arc_agi_2_classifications_phase1.json
# - results/category_distribution_comparison.csv

# Time: <1 hour
```

**🎯 END OF PHASE 1: Paper's critical weakness solved!**

## Configuration

### LoRA Configuration (`configs/lora_config.yaml`)

Key design decisions (documented in `VISUAL_CLASSIFIER_IMPLEMENTATION.md` Decision #6):

```yaml
lora_rank: 16              # Balance of capacity and efficiency
lora_alpha: 32             # Scaling factor (2×rank)
target_modules:            # CRITICAL: Attention projections only
  - "q_proj"               # Captures task-specific reasoning patterns
  - "k_proj"
  - "v_proj"
  - "o_proj"
```

**Rationale:** Targeting attention mechanisms captures relational reasoning (what ARC tests) rather than low-level feature transformations. This choice affects interpretability and must be documented in the paper's Methods section.

## Key Implementation Details

### 1. Lightning Checkpoint Handling

The `champion_bootstrap.ckpt` is saved from `Exp3ChampionLightningModule`. Loading requires stripping the "model." prefix:

```python
checkpoint = torch.load("champion_bootstrap.ckpt")
clean_state_dict = {
    k.replace("model.", ""): v 
    for k, v in checkpoint['state_dict'].items()
    if k.startswith("model.")
}
base_model.load_state_dict(clean_state_dict)
```

### 2. unflatten_adapter() Utility

Critical for Phase 2 Oracle. Reshapes flattened LoRA vectors back to state_dict format:

```python
from visual_classification.src.vector_utils import unflatten_adapter

flat_vector = torch.randn(50000)  # From optimization
ref_state_dict = load_adapter("skill_library/atomic_skills/task_001/")
restored_adapter = unflatten_adapter(flat_vector, ref_state_dict)
```

### 3. TaskEncoder Architectures

Two versions available (see Decision #7):

- **Version A (Simple):** 3-layer CNN - good baseline, fast to train
- **Version B (Advanced):** Uses Champion's own `ContextEncoderModule` - architecturally stronger

Start with Version A. Upgrade to Version B only if validation accuracy <70%.

## Testing

```bash
# Run unit tests
pytest visual_classification/tests/

# Validate unflatten_adapter round-trip
python -c "from visual_classification.src.vector_utils import validate_unflatten; ..."
```

## Citation

If you use this visual classifier, please cite:

```bibtex
@article{taxonomy2025,
  title={A Taxonomy of Reasoning Primitives for the Abstract Reasoning Corpus},
  author={...},
  journal={...},
  year={2025}
}
```

## Related Documentation

- **Implementation Log:** `../docs/VISUAL_CLASSIFIER_IMPLEMENTATION.md`
- **Design Decisions:** See "Design Decisions Log" in implementation doc
- **MVPv1 Failure Analysis:** Why Oracle architecture fixes previous issues
- **Paper Integration Plan:** How to write §4.5 and §8.2

## Status

**Phase 0:** 🚧 In Progress (Oct 28, 2025)  
**Phase 1:** ⏳ Not Started  
**Phase 2:** ⏳ Not Started

## Contact

For questions or issues, refer to the main reproduction package README.
