# Applying Neural Affinity Framework to Your Own ARC Model

**Target Audience:** Researchers building ARC solvers who want to use our taxonomy and diagnostic framework.

---

## 🎯 What You Can Take From This Work

### 1. The Taxonomy (Immediate Use)

**What:** 9-category classification of all 400 re-arc tasks with 97.5% accuracy.

**How to use:**
```python
import json

# Load our taxonomy
with open('data/taxonomy/all_tasks_classified.json') as f:
    taxonomy = json.load(f)

# Get category for any re-arc task
task_id = "0d3d703e"
category = taxonomy[task_id]["primary_category"]
affinity = taxonomy[task_id]["neural_affinity"]  # Very Low, Low, Medium, High

print(f"Task {task_id}: {category} ({affinity} affinity)")
```

**Files you need:**
- `data/taxonomy/all_tasks_classified.json` (27KB) - Full classification
- `data/taxonomy/tasks_by_category.json` (9KB) - Grouped by category

**Categories:**
- **A1:** Movement & Translation (High affinity)
- **A2:** Alignment & Symmetry (Very Low affinity - 42.9% fail)
- **C1:** Counting (High affinity)
- **C2:** Color Manipulation (Medium affinity)
- **P1:** Pattern Completion (Medium affinity)
- **P2:** Pattern Recognition (High affinity)
- **S1:** Scaling (Medium affinity)
- **S2:** Shape Manipulation (Low affinity)
- **S3:** Spatial Reasoning (Low affinity - high variance)

---

### 2. Neural Affinity Diagnostic (Model Evaluation)

**What:** Framework to predict which task types your architecture will struggle with.

**How to evaluate your model:**

```python
import json
import numpy as np

# Step 1: Test your model on the 18 foundational tasks
foundational_tasks = [
    # A1 (should be easy): 0d3d703e, 228f6490
    # A2 (should be hard): 1e0a9b12, 25d8a9c8
    # C1 (should be easy): 6d0aefbc, 62c24649
    # ... (see data/distributional_alignment/ for full list)
]

# Step 2: Measure accuracy by category
def evaluate_by_category(model, tasks, taxonomy):
    results = {}
    for task_id in tasks:
        category = taxonomy[task_id]["primary_category"]
        accuracy = model.evaluate(task_id)  # Your evaluation code
        
        if category not in results:
            results[category] = []
        results[category].append(accuracy)
    
    return {cat: np.mean(accs) for cat, accs in results.items()}

category_performance = evaluate_by_category(your_model, foundational_tasks, taxonomy)

# Step 3: Compare to baselines
# Our Transformer baseline:
# A1: 95.2%, A2: 17.3%, C1: 96.1%, C2: 78.4%
# P1: 71.8%, P2: 89.3%, S1: 73.5%, S2: 45.7%, S3: 70-96% (heterogeneous)

for category, accuracy in category_performance.items():
    print(f"{category}: {accuracy:.1f}%")
    if category == "A2" and accuracy < 20:
        print("  ⚠️  A2 ceiling detected - see paper §7.2")
    if category in ["S2", "S3"] and accuracy < 50:
        print("  ⚠️  Low affinity category - see paper §7.1")
```

**Expected outcomes:**
- If your model shows A2 < 20%, you're hitting the architectural ceiling
- If Low/Very Low affinity categories < 50%, you're seeing the compositional gap
- If you see 35%+ of tasks in low-affinity range, reconsider training curriculum

---

### 3. Compositional Gap Diagnostic (Failure Mode Analysis)

**What:** Test if your model has cell-level understanding but fails grid-level composition.

**How to diagnose:**

```python
def check_compositional_gap(model, task_id):
    """
    Check if model shows compositional gap pattern.
    
    Pattern: >80% cell accuracy but <10% grid accuracy
    Indicates: Model understands primitives but can't compose them
    """
    results = model.evaluate_detailed(task_id)  # Your evaluation
    
    cell_accuracy = results['cell_accuracy']
    grid_accuracy = results['grid_accuracy']
    
    has_gap = (cell_accuracy > 0.80 and grid_accuracy < 0.10)
    
    return {
        'task_id': task_id,
        'cell_accuracy': cell_accuracy,
        'grid_accuracy': grid_accuracy,
        'has_compositional_gap': has_gap
    }

# Test on your trained models
gaps = [check_compositional_gap(model, task) for task in test_tasks]
gap_percentage = sum(g['has_compositional_gap'] for g in gaps) / len(gaps)

print(f"Compositional Gap: {gap_percentage:.1%}")
# Our finding: 69.5% of tasks show this pattern
# If you see >60%, composition is your main bottleneck
```

**Interpretation:**
- **Gap > 60%:** Composition is the primary failure mode (like us)
- **Gap < 30%:** Either your model composes well, or has other issues
- **Gap + High cell accuracy:** You need architectural changes, not more data

---

### 4. Curriculum Design (Training Strategy)

**What:** Use affinity distribution to design balanced curricula.

**How to design curriculum:**

```python
import json
from collections import Counter

# Load taxonomy
with open('data/taxonomy/all_tasks_classified.json') as f:
    taxonomy = json.load(f)

# Strategy 1: Balanced curriculum (all categories equal)
def create_balanced_curriculum(taxonomy, tasks_per_category=50):
    by_category = {}
    for task_id, info in taxonomy.items():
        cat = info['primary_category']
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(task_id)
    
    curriculum = []
    for cat, tasks in by_category.items():
        curriculum.extend(tasks[:tasks_per_category])
    
    return curriculum

# Strategy 2: Affinity-weighted (more hard tasks)
def create_affinity_weighted_curriculum(taxonomy, total_tasks=400):
    """
    Oversample low-affinity categories.
    
    Our finding: 35.3% are low affinity, but standard training
    treats them equally, leading to poor generalization.
    """
    by_affinity = {'Very Low': [], 'Low': [], 'Medium': [], 'High': []}
    for task_id, info in taxonomy.items():
        affinity = info['neural_affinity']
        by_affinity[affinity].append(task_id)
    
    # Weight: 3x for Very Low, 2x for Low, 1x for Medium/High
    curriculum = []
    curriculum.extend(by_affinity['Very Low'] * 3)
    curriculum.extend(by_affinity['Low'] * 2)
    curriculum.extend(by_affinity['Medium'])
    curriculum.extend(by_affinity['High'])
    
    return curriculum[:total_tasks]

# Strategy 3: Progressive difficulty
def create_progressive_curriculum(taxonomy):
    """Start with high affinity, progress to low."""
    ordered = sorted(
        taxonomy.items(),
        key=lambda x: {'High': 0, 'Medium': 1, 'Low': 2, 'Very Low': 3}[x[1]['neural_affinity']]
    )
    return [task_id for task_id, _ in ordered]

# Use the curriculum
curriculum = create_affinity_weighted_curriculum(taxonomy)
print(f"Created curriculum with {len(curriculum)} tasks")
print(f"Affinity distribution:")
for affinity in ['Very Low', 'Low', 'Medium', 'High']:
    count = sum(1 for tid in curriculum if taxonomy[tid]['neural_affinity'] == affinity)
    print(f"  {affinity}: {count} ({count/len(curriculum):.1%})")
```

**Our recommendations:**
- **DON'T:** Use re-arc uniformly (35% low-affinity tasks will dominate errors)
- **DO:** Oversample low-affinity categories or train category-specific experts
- **DO:** Test generalization on stratified splits (not random)

---

### 5. S3 Heterogeneity Analysis (Sub-category Discovery)

**What:** Method to detect when a category has hidden sub-types with vastly different difficulty.

**How to find heterogeneous categories:**

```python
import numpy as np

def analyze_category_variance(model, taxonomy, category):
    """
    Find categories with high performance variance.
    
    High variance = likely contains distinct sub-types
    """
    tasks = [tid for tid, info in taxonomy.items() 
             if info['primary_category'] == category]
    
    accuracies = [model.evaluate(task) for task in tasks]
    
    return {
        'category': category,
        'mean': np.mean(accuracies),
        'std': np.std(accuracies),
        'min': np.min(accuracies),
        'max': np.max(accuracies),
        'range': np.max(accuracies) - np.min(accuracies),
        'tasks': len(tasks)
    }

# Check all categories
for category in ['A1', 'A2', 'C1', 'C2', 'P1', 'P2', 'S1', 'S2', 'S3']:
    stats = analyze_category_variance(your_model, taxonomy, category)
    print(f"{category}: {stats['mean']:.1f}% ± {stats['std']:.1f}% "
          f"(range: {stats['min']:.1f}%-{stats['max']:.1f}%)")
    
    if stats['range'] > 20:
        print(f"  ⚠️  High variance - may need sub-classification")

# Our finding: S3 has 70-96% range → split into S3-A and S3-B
# If you see >20% range, consider subdividing the category
```

**Files with S3 sub-classification:**
- `data/s3_subclassification/s3_final_classification.json` (5KB)
- Shows how we split S3 into two sub-categories

---

### 6. External Validation (Transfer Testing)

**What:** Test if your findings transfer to other ARC datasets.

**How we validated:**

```python
# Step 1: Train visual classifier on re-arc tasks
# (We used ResNet-18 on task visualizations)

# Step 2: Apply to ARC-AGI-2 (unseen human-designed tasks)
# Result: 36.25% 9-way accuracy (vs 11% random baseline)

# Step 3: Compare affinity predictions to independent study
# We used Li et al.'s ViTARC results (400 models, different architecture)
# Result: Our affinity predicted performance (p < 0.001, d = 0.726)

# For your model:
def validate_taxonomy_transfer(taxonomy, external_dataset):
    """
    Test if affinity predictions hold on external data.
    """
    predictions = []
    actuals = []
    
    for task in external_dataset:
        # Map external task to closest re-arc category (using classifier)
        predicted_category = classify_external_task(task)
        predicted_affinity = taxonomy[predicted_category]['neural_affinity']
        
        # Measure actual performance
        actual_accuracy = your_model.evaluate(task)
        
        predictions.append({'Very Low': 0, 'Low': 1, 'Medium': 2, 'High': 3}[predicted_affinity])
        actuals.append(actual_accuracy)
    
    # Correlation test
    from scipy.stats import spearmanr
    correlation, p_value = spearmanr(predictions, actuals)
    
    return correlation, p_value

# If p < 0.05 and correlation > 0.5, your affinity framework transfers
```

---

## 📁 Key Files Reference

### Essential Files (Can Use Immediately)
```
data/taxonomy/
├── all_tasks_classified.json          # 27KB - Full taxonomy (400 tasks)
├── tasks_by_category.json             # 9KB - Grouped by category
└── category_descriptions.json         # Category definitions

data/s3_subclassification/
└── s3_final_classification.json       # 5KB - S3 sub-types

outputs/
└── atomic_lora_training_summary.json  # 241KB - Our 302 fine-tuning results
```

### Implementation Examples
```
scripts/
├── calculate_compositional_gap.py     # Compositional gap calculator
├── verify_a2_failures.py              # A2 ceiling detector
└── 0_generate_taxonomy_classification.py  # Taxonomy generator

src/models/
└── champion/                          # Our model (reference implementation)
```

### For Training Your Own
```
data/distributional_alignment/         # 18 foundational tasks
scripts/regenerate_data.sh             # Generate training data
scripts/train_atomic_loras.py          # Fine-tuning template
```

---

## 🔬 Reproducing Our Results

If you want to exactly reproduce our experiments:

### Quick Verification (5 minutes, no GPU)
```bash
pip install -r requirements.txt
python verify_setup.py
python scripts/calculate_compositional_gap.py  # 69.5%
python scripts/verify_a2_failures.py          # 42.9%
```

### Full Training (120 GPU-hours)
```bash
# See QUICKSTART.md for complete instructions
bash scripts/regenerate_data.sh 400
./run_training.sh exp3
python scripts/train_atomic_loras.py
```

---

## 📊 Comparison Benchmarks

Use these as baselines when evaluating your architecture:

| Category | Our Transformer | ViTARC (CNN) | Baseline (Random) |
|:---------|:----------------|:-------------|:------------------|
| **A1** (Movement) | 95.2% | 91.3% | 11% |
| **A2** (Alignment) | 17.3% | 25.8% | 11% |
| **C1** (Counting) | 96.1% | 93.7% | 11% |
| **C2** (Color) | 78.4% | 72.1% | 11% |
| **P1** (Completion) | 71.8% | 68.4% | 11% |
| **P2** (Recognition) | 89.3% | 85.9% | 11% |
| **S1** (Scaling) | 73.5% | 69.2% | 11% |
| **S2** (Shape) | 45.7% | 51.3% | 11% |
| **S3** (Spatial) | 70-96% | 65-89% | 11% |

**Key takeaways:**
- If your A2 < 20%, you've hit the same ceiling we did
- If your S2 < 50%, you're seeing the compositional gap
- If your S3 variance > 20%, subdivide the category

---

## 🚀 Suggested Research Directions

Based on our findings, here's where to focus:

### 1. **Architectural Solutions for A2** (High Impact)
- 42.9% of A2 tasks fail completely (<0.1% accuracy)
- Problem: Symmetry and alignment require global reasoning
- Potential solutions:
  - Graph neural networks for spatial relationships
  - Explicit symmetry detection modules
  - Coordinate-based attention mechanisms

### 2. **Compositional Gap Solutions** (High Impact)
- 69.5% show cell accuracy > 80% but grid accuracy < 10%
- Problem: Models understand primitives but can't compose
- Potential solutions:
  - Hierarchical architectures
  - Explicit composition modules
  - Test-time search/planning

### 3. **Curriculum Learning** (Medium Impact)
- 35.3% of re-arc is low-affinity for Transformers
- Problem: Uniform training doesn't match difficulty distribution
- Solutions:
  - Affinity-weighted sampling
  - Progressive difficulty curricula
  - Category-specific experts

### 4. **S3 Heterogeneity Resolution** (Research Question)
- S3 has 70-96% accuracy range (hidden sub-types)
- Question: Can we further subdivide?
- Approach: Cluster by failure patterns, create S3-C, S3-D, etc.

---

## 💬 Citation

If you use our taxonomy or framework:

```bibtex
@article{ingram2025neural,
  title={A Neural Affinity Framework for Abstract Reasoning: Diagnosing the 
         Compositional Gap in Transformer Architectures via Procedural Task Taxonomy},
  author={Ingram, Miguel and Merritt, Arthur},
  journal={[Venue]},
  year={2025}
}
```

---

## 🤝 Support

- **Questions:** Open an issue on GitHub
- **Bug reports:** Include your Python version and package versions
- **Feature requests:** Describe your use case and what you need

---

## 📚 Further Reading

- **Paper:** [Ingram & Merritt (2025)](http://arxiv.org/abs/2512.07109) (61 pages)
- **Quick Start:** `QUICKSTART.md` (training guide)
- **Main README:** `README.md` (verification guide)

**Key paper sections for researchers:**
- §5.2: Taxonomy development (how we classified tasks)
- §7.1: Compositional gap (cell vs grid dissociation)
- §7.2: A2 ceiling effect (architectural bottleneck)
- §7.5: ViTARC validation (external confirmation)
- Appendix B: Complete reproducibility details
: Complete reproducibility details
