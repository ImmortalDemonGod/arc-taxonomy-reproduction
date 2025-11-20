# Documentation Directory

This directory contains all documentation for the ARC Taxonomy Reproduction Package.

## Directory Structure

### 📚 `methodology/`
**Scientific methodology, architecture, and implementation details**

Key files:
- `taxonomy_definitions.md` - Taxonomy categories and definitions
- `ABLATION_MODEL_SPECIFICATIONS.md` - Complete model specs for ablation study
- `ABLATION_FAIRNESS_ANALYSIS.md` - Ablation study validation
- `VISUAL_CLASSIFIER_IMPLEMENTATION.md` - Classifier implementation details
- `ATOMIC_LORA_TRAINING.md` - LoRA training methodology
- `s3_*.md` - S3 subcategory analysis
- `ambiguous_tasks_analysis.md` - Ambiguous task handling

### 📊 `results/`
**Experimental results, validation, and verification**

Key files:
- `champion_baseline_results.txt` - Complete baseline results (cited in paper)
- `section7_verified_facts.md` - Verified claims from paper §7
- `classifier_final_exam.md` - Classifier validation results
- `arc_agi_2_evaluation.md` - ARC-AGI-2 transfer evaluation
- `vitarc_external_validation_summary.md` - External validation

### 🗂️ `archive/dev_logs/`
**Historical development logs (for maintainers only)**

Contains dated logs of bug fixes, config updates, and development milestones. These are preserved for reference but not needed to understand the scientific methodology.

### 📁 `internal/`
**Internal documentation**

- `APPLYING_TO_YOUR_MODEL.md` - Guide for applying methodology to new models
- `CLEANUP_SUMMARY.md` - Repository cleanup summary
- `AUDIT_VERIFICATION_REPORT.md` - Audit verification

### 📈 `progress/`
**Development progress tracking**

Historical implementation logs showing how the system was built.

---

## Quick Navigation

**Want to understand the methodology?** → Start with `methodology/taxonomy_definitions.md`

**Want to verify paper claims?** → See `results/section7_verified_facts.md`

**Want to apply this to your model?** → Read `internal/APPLYING_TO_YOUR_MODEL.md`

**Want ablation study details?** → See `methodology/ABLATION_MODEL_SPECIFICATIONS.md`

**Want classifier details?** → See `methodology/VISUAL_CLASSIFIER_IMPLEMENTATION.md`

---

## Technical Documentation

### Architecture Documentation
- Model specifications: `methodology/ABLATION_MODEL_SPECIFICATIONS.md`
- Data loading: `methodology/ARC_AGI_DATA_LOADING_EXPLAINED.md`
- Training configuration: See individual training scripts in `../scripts/`

### Checkpoint Information
- `checkpoint_keys.txt` - Full checkpoint structure analysis

---

## For Paper Reviewers

The most important files for verifying paper claims:

1. **§7.1 (A2 Task Failures)**: `results/section7_verified_facts.md`
2. **Taxonomy Definitions**: `methodology/taxonomy_definitions.md`
3. **Baseline Results**: `results/champion_baseline_results.txt`
4. **Ablation Study**: `methodology/ABLATION_MODEL_SPECIFICATIONS.md`, `methodology/ABLATION_FAIRNESS_ANALYSIS.md`
5. **Classifier Validation**: `results/classifier_final_exam.md`
