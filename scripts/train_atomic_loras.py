"""
Train LoRA adapters for 400 ARC tasks (Phase 0).

Usage:
    python scripts/train_atomic_loras.py

Follows the same infrastructure as Champion training (train_exp3_champion.py):
- Uses create_champion_dataloader (not custom dataset)
- Same CSV logging patterns
- Same checkpoint structure
- Exact 10x lower learning rate
"""
import sys
import json
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.exp3_champion_lightning import Exp3ChampionLightningModule
from src.data.champion_data import create_champion_dataloader
from src.lora_utils import is_peft_available

def load_champion(ckpt_path: Path, device: str) -> nn.Module:
    """Load Champion model from Lightning checkpoint."""
    torch.serialization.add_safe_globals([
        'omegaconf.listconfig.ListConfig',
        'omegaconf.dictconfig.DictConfig'
    ])
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Unpack hyperparameters as kwargs
    model = Exp3ChampionLightningModule(**ckpt['hyper_parameters'])
    model.load_state_dict(ckpt['state_dict'])
    
    return model.model.to(device)  # Return base model, not Lightning wrapper


def setup_lora(model: nn.Module, config: dict) -> nn.Module:
    """Wrap model with PEFT LoRA."""
    if not is_peft_available():
        raise ImportError("PEFT required: pip install peft")
    
    from peft import get_peft_model, LoraConfig
    
    lora_config = LoraConfig(
        # Don't specify task_type - Champion is custom architecture
        inference_mode=False,
        r=config['lora_rank'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        target_modules=config['target_modules'],
    )
    
    return get_peft_model(model, lora_config)


def train_task(model: nn.Module, task_file: Path, config: dict, device: str):
    """Train LoRA on one task. Returns (final_loss, epochs, training_history, metadata)."""
    # Use same data loader as Champion training
    # Note: samples_per_task controlled by data generation (150 or 400)
    # Note: Uses ALL examples from task (ARC's "train"/"test" are pattern demos, not ML splits)
    loader = create_champion_dataloader(
        [task_file],  # Single task
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_context_pairs=2,  # Match Champion
        max_grid_size=35,  # Match Champion
    )
    
    if len(loader) == 0:
        return float('inf'), 0, [], {'num_examples': 0, 'error': 'no_examples'}
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=10)
    model.train()
    
    # Training history
    best_loss = float('inf')
    epochs_no_improve = 0
    training_history = []
    start_time = time.time()
    num_examples = len(loader.dataset) if hasattr(loader, 'dataset') else len(loader) * config['training']['batch_size']
    
    for epoch in range(config['training']['num_epochs']):
        epoch_loss = 0.0
        count = 0
        
        for batch in loader:
            # champion_data returns: (src, tgt, ctx_input, ctx_output, src_grid_shape, tgt_grid_shape, task_id)
            src, tgt, ctx_in, ctx_out, src_shapes, tgt_shapes, task_ids = batch
            src, tgt = src.to(device), tgt.to(device)
            ctx_in, ctx_out = ctx_in.to(device), ctx_out.to(device)
            
            if tgt.size(1) <= 1:
                continue
            
            tgt_input, tgt_output = tgt[:, :-1], tgt[:, 1:]
            
            try:
                optimizer.zero_grad()
                logits = model(
                    src=src,
                    tgt=tgt_input,
                    src_grid_shape=src_shapes[0],
                    tgt_grid_shape=tgt_shapes[0],
                    ctx_input=ctx_in,
                    ctx_output=ctx_out
                )
                
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'])
                optimizer.step()
                
                epoch_loss += loss.item()
                count += 1
            except Exception as e:
                logger.warning(f"Step failed: {e}")
                continue
        
        if count == 0:
            continue
        
        avg_loss = epoch_loss / count
        training_history.append({'epoch': epoch + 1, 'loss': avg_loss})
        
        # Early stopping
        if avg_loss < best_loss - config['training']['min_delta']:
            best_loss = avg_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= config['training']['patience']:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break
    
    training_time = time.time() - start_time
    
    # Metadata
    metadata = {
        'num_examples': num_examples,
        'num_epochs_trained': len(training_history),
        'training_time_seconds': training_time,
        'early_stopped': epochs_no_improve >= config['training']['patience']
    }
    
    return best_loss, len(training_history), training_history, metadata


def main():
    """Main training loop - follows Champion training infrastructure."""
    # Parse CLI arguments (same as Champion training)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast_dev_run', type=int, nargs='?', const=2, default=None,
                        help='Run fast dev test with N tasks (default: 2 if flag provided)')
    args, unknown = parser.parse_known_args()
    fast_dev_run = args.fast_dev_run
    
    config_path = Path('configs/atomic_lora_training.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Get data files from distributional_alignment (same as Champion)
    data_dir = Path(config['data_dir'])
    with open(data_dir / "split_manifest.json") as f:
        split_info = json.load(f)
    
    # Use train_files for LoRA training (match Champion's training set)
    task_files = [data_dir / fname for fname in split_info["train_files"]]
    
    # Limit tasks if fast_dev_run specified
    if fast_dev_run:
        task_files = task_files[:fast_dev_run]
    
    # Print training summary (same format as Champion)
    print(f"\n{'='*70}")
    if fast_dev_run:
        print(f"FAST DEV RUN: Phase 0 - Atomic LoRA Skills ({fast_dev_run} tasks)")
    else:
        print(f"TRAINING: Phase 0 - Atomic LoRA Skills")
    print(f"{'='*70}")
    print(f"Dataset: distributional_alignment (re-arc synthetic)")
    print(f"Total tasks: {len(task_files)}")
    print(f"Base model: Champion (epoch=36, val_loss=0.5926)")
    print(f"LoRA config: rank={config['lora_rank']}, alpha={config['lora_alpha']}")
    print(f"Learning rate: {config['training']['learning_rate']:.7f} (10x lower than Champion)")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Max epochs per task: {config['training']['num_epochs']}")
    print(f"Early stopping: patience={config['training']['patience']}")
    print(f"Output: {config['output_dir']}")
    print(f"Device: {device}")
    if fast_dev_run:
        print(f"MODE: Fast dev run (testing only)")
    print(f"{'='*70}\n")
    
    # Load base model
    base_model = load_champion(Path(config['champion_checkpoint']), device)
    print(f"Champion loaded: {sum(p.numel() for p in base_model.parameters()):,} params\n")
    
    # Create CSV log file (same pattern as ablations)
    log_dir = Path("logs") / "atomic_loras"
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "lora_training_metrics.csv"
    
    # CSV headers (match Champion's per-task logging)
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=[
        'task_id', 'status', 'final_loss', 'epochs', 'num_examples',
        'training_time_seconds', 'early_stopped', 'error'
    ])
    csv_writer.writeheader()
    csv_file.flush()
    
    results = {'completed': 0, 'failed': 0, 'tasks': {}}
    
    for task_file in tqdm(task_files, desc="Training"):
        task_id = task_file.stem
        
        try:
            # Create fresh base model instance to avoid memory issues
            # (LoRA wrapping modifies the model in-place)
            import copy
            task_base_model = copy.deepcopy(base_model)
            lora_model = setup_lora(task_base_model, config).to(device)
            
            loss, epochs, history, metadata = train_task(lora_model, task_file, config, device)
            
            # Save adapter
            output_dir = Path(config['output_dir']) / task_id
            output_dir.mkdir(parents=True, exist_ok=True)
            lora_model.save_pretrained(output_dir)
            
            # Save training curve if requested
            if config['training'].get('save_training_curve', False):
                with open(output_dir / 'training_history.json', 'w') as f:
                    json.dump(history, f, indent=2)
            
            results['completed'] += 1
            results['tasks'][task_id] = {
                'status': 'success',
                'final_loss': loss,
                'epochs': epochs,
                'metadata': metadata
            }
            
            # Write to CSV (same as ablations)
            csv_writer.writerow({
                'task_id': task_id,
                'status': 'success',
                'final_loss': f"{loss:.6f}",
                'epochs': epochs,
                'num_examples': metadata['num_examples'],
                'training_time_seconds': f"{metadata['training_time_seconds']:.2f}",
                'early_stopped': metadata['early_stopped'],
                'error': ''
            })
            csv_file.flush()
            
            # Clean up to avoid memory buildup
            del lora_model, task_base_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            traceback_str = traceback.format_exc()
            
            # Print error immediately to console
            print(f"\n❌ FAILED: {task_id}")
            print(f"   Error: {error_msg}")
            if len(traceback_str) < 500:  # Print full traceback if short
                print(f"   Traceback:\n{traceback_str}")
            
            results['failed'] += 1
            results['tasks'][task_id] = {
                'status': 'failed',
                'error': error_msg,
                'traceback': traceback_str
            }
            
            # Write failure to CSV
            csv_writer.writerow({
                'task_id': task_id,
                'status': 'failed',
                'final_loss': '',
                'epochs': '',
                'num_examples': '',
                'training_time_seconds': '',
                'early_stopped': '',
                'error': error_msg[:100]  # Truncate long errors
            })
            csv_file.flush()
    
    csv_file.close()
    
    # Save detailed JSON summary
    Path('outputs').mkdir(exist_ok=True)
    with open('outputs/atomic_lora_training_summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print final summary (same format as Champion)
    print(f"\n{'='*70}")
    if fast_dev_run:
        print(f"Fast Dev Run Complete!")
    else:
        print(f"Training Complete!")
    print(f"{'='*70}")
    print(f"Successful: {results['completed']}/{len(task_files)}")
    print(f"Failed: {results['failed']}/{len(task_files)}")
    if not fast_dev_run:
        print(f"Adapters saved to: {config['output_dir']}")
    else:
        print(f"Test adapters saved to: {config['output_dir']}")
    print(f"CSV metrics saved to: {csv_path}")
    print(f"Detailed summary: outputs/atomic_lora_training_summary.json")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
