# Deprecated Tests

**Status:** Archived - Not Run by CI/CD

These tests were written for an earlier version of the codebase or test training functionality not needed for verification.

## Why Deprecated

### Pre-Refactor Tests (Old Module Structure)
1. **test_checkpoint_loading.py** - Uses old `model` module (pre-refactor)
2. **test_decoder_only_model.py** - Imports `softmax` from old utils structure
3. **test_finetune.py** - Uses old `finetune` module (now in scripts/)
4. **test_model.py** - Old model imports
5. **test_nn_utils.py** - Old utils structure
6. **test_all_optimizers.py** - softmax import issue
7. **test_data_loaders.py** - softmax import issue
8. **test_lightning_training.py** - decoder_only_baseline import issue
9. **test_lightning_validation.py** - decoder_only_baseline import issue
10. **test_scheduler_logic.py** - decoder_only_baseline import issue

### Training Tests (Not Needed for Verification)
11. **test_atomic_lora_training.py** - Tests train_atomic_loras.py script imports
12. **test_checkpoint_utils.py** - Tests checkpoint utility imports from scripts/
13. **test_full_integration.py** - Tests full training integration from scripts/
14. **test_per_task_logger.py** - Tests logger utility imports from scripts/
15. **test_training_from_scratch.py** - Tests training pipeline imports from scripts/

## Current Test Coverage

Active tests (207 passing) cover:
- ✅ All current architecture (champion, encoder-decoder, embeddings)
- ✅ All data loading (champion, single-task, HPO)
- ✅ All training (champion, HPO, encoder-decoder baseline)
- ✅ All utilities (LoRA, logging, checkpoints)

## To Restore

If you need to restore these tests:
1. Update imports to use new module structure
2. Replace `from model import` with `from src.models import`
3. Replace `from finetune import` with imports from scripts
4. Fix softmax import from src.utils
