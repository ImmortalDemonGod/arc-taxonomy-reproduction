"""
Test training on one task to validate pipeline.

Run this before full 400-task training.
"""
import sys
import logging
from pathlib import Path

import torch
import yaml

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lora_utils import flatten_adapter

# Import functions directly from the script
import importlib.util
spec = importlib.util.spec_from_file_location("train_atomic_loras", Path(__file__).parent / "train_atomic_loras.py")
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)

load_champion = train_module.load_champion
setup_lora = train_module.setup_lora
train_task = train_module.train_task

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def main():
    """Test training on one task."""
    logger.info("="*60)
    logger.info("Test: Train LoRA on Single Task")
    logger.info("="*60)
    
    config_path = Path('configs/atomic_lora_training.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Get test task
    data_dir = Path(config['data_dir'])
    test_file = list(data_dir.glob("*.json"))[0]
    logger.info(f"Task: {test_file.stem}")
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    try:
        # Load Champion
        logger.info("\n1. Loading Champion...")
        base_model = load_champion(Path(config['champion_checkpoint']), device)
        logger.info(f"   ✅ Loaded: {sum(p.numel() for p in base_model.parameters()):,} params")
        
        # Setup LoRA
        logger.info("\n2. Setting up LoRA...")
        lora_model = setup_lora(base_model, config).to(device)
        trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        logger.info(f"   ✅ LoRA ready: {trainable:,} trainable params")
        
        # Train
        logger.info("\n3. Training...")
        loss, epochs = train_task(lora_model, test_file, config, device)
        logger.info(f"   ✅ Done: loss={loss:.4f}, epochs={epochs}")
        
        # Save
        logger.info("\n4. Saving adapter...")
        output_dir = Path(f"outputs/atomic_loras/test_{test_file.stem}")
        output_dir.mkdir(parents=True, exist_ok=True)
        lora_model.save_pretrained(output_dir)
        logger.info(f"   ✅ Saved to: {output_dir}")
        
        # Verify
        logger.info("\n5. Verifying...")
        flat = flatten_adapter(str(output_dir))
        logger.info(f"   ✅ Reloaded: {flat.shape[0]:,} params")
        
        logger.info("\n" + "="*60)
        logger.info("🎉 TEST PASSED - Pipeline works!")
        logger.info("="*60)
        return True
        
    except Exception as e:
        logger.error("\n" + "="*60)
        logger.error("❌ TEST FAILED")
        logger.error("="*60)
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
