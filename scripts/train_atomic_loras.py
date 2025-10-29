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
import fcntl
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
from src.evaluation.metrics import compute_grid_accuracy, compute_copy_metrics_on_batch


def write_json_safely(json_path: Path, task_id: str, task_data: dict):
    """Write task results to JSON with file locking for parallel safety."""
    # Acquire exclusive lock
    with open(json_path, 'a+') as f:  # a+ allows read and creates if missing
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        
        try:
            # Read current state
            f.seek(0)
            content = f.read()
            if content.strip():
                results = json.loads(content)
            else:
                results = {'completed': 0, 'failed': 0, 'tasks': {}}
            
            # Update with new task
            results['tasks'][task_id] = task_data
            if task_data['status'] == 'success':
                results['completed'] = sum(1 for t in results['tasks'].values() if t['status'] == 'success')
            else:
                results['failed'] = sum(1 for t in results['tasks'].values() if t['status'] == 'failed')
            
            # Write back
            f.seek(0)
            f.truncate()
            json.dump(results, f, indent=2)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


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


def evaluate_model(model: nn.Module, loader, device: str) -> dict:
    """Evaluate model using standard metrics (matching ablations)."""
    model.eval()
    all_grid_correct = []
    all_cell_correct = []
    all_cell_total = []
    all_copy_rate = []
    all_change_recall = []
    all_trans_quality = []
    
    with torch.no_grad():
        for batch in loader:
            src, tgt, ctx_in, ctx_out, src_shapes, tgt_shapes, task_ids = batch
            src, tgt = src.to(device), tgt.to(device)
            ctx_in, ctx_out = ctx_in.to(device), ctx_out.to(device)
            
            if tgt.size(1) <= 1:
                continue
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            try:
                logits = model(
                    src=src,
                    tgt=tgt_input,
                    src_grid_shape=src_shapes[0],
                    tgt_grid_shape=tgt_shapes[0],
                    ctx_input=ctx_in,
                    ctx_output=ctx_out
                )
                
                preds = torch.argmax(logits, dim=-1)
                
                # Grid and cell accuracy (matching ablations)
                grid_metrics = compute_grid_accuracy(preds, tgt_output, pad_token=10)
                all_grid_correct.append(grid_metrics['grid_correct'])
                
                # Cell-level accuracy
                valid_mask = (tgt_output != 10)
                correct_cells = ((preds == tgt_output) & valid_mask).sum().item()
                total_cells = valid_mask.sum().item()
                all_cell_correct.append(correct_cells)
                all_cell_total.append(total_cells)
                
                # Copy metrics (matching ablations)
                try:
                    copy_metrics = compute_copy_metrics_on_batch(src, tgt_output, preds)
                    all_copy_rate.append(copy_metrics['copy_rate'].item())
                    all_change_recall.append(copy_metrics['change_recall'].item())
                    all_trans_quality.append(copy_metrics['transformation_f1'].item())
                except:
                    # If copy metrics fail, use defaults
                    all_copy_rate.append(0.0)
                    all_change_recall.append(0.0)
                    all_trans_quality.append(0.0)
            except:
                continue
    
    if len(all_grid_correct) == 0:
        return {
            'grid_accuracy': 0.0, 'cell_accuracy': 0.0,
            'grid_correct': 0, 'grid_total': 0,
            'cell_correct': 0, 'cell_total': 0,
            'copy_rate': 0.0, 'change_recall': 0.0, 'transformation_quality': 0.0
        }
    
    # Aggregate results (matching ablation pattern)
    grid_correct = torch.cat(all_grid_correct)
    grid_accuracy = (grid_correct.sum().item() / len(grid_correct) * 100)
    
    cell_correct_total = sum(all_cell_correct)
    cell_total_total = sum(all_cell_total)
    cell_accuracy = (cell_correct_total / cell_total_total * 100) if cell_total_total > 0 else 0.0
    
    return {
        'grid_accuracy': grid_accuracy,
        'cell_accuracy': cell_accuracy,
        'grid_correct': grid_correct.sum().item(),
        'grid_total': len(grid_correct),
        'cell_correct': cell_correct_total,
        'cell_total': cell_total_total,
        'copy_rate': sum(all_copy_rate) / len(all_copy_rate),
        'change_recall': sum(all_change_recall) / len(all_change_recall),
        'transformation_quality': sum(all_trans_quality) / len(all_trans_quality),
    }


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
        num_workers=config.get('num_workers', 4),  # Parallel data loading
        pin_memory=config.get('pin_memory', True),  # Faster GPU transfer
    )
    
    if len(loader) == 0:
        return float('inf'), 0, [], {'num_examples': 0, 'error': 'no_examples'}
    
    # Evaluate base model BEFORE training
    base_metrics = evaluate_model(model, loader, device)
    
    # Optimizer (same as Champion but 10x lower LR)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    # Mixed precision for 1.5-2x speedup without quality loss
    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
    model.train()
    
    # Training history
    best_loss = float('inf')
    best_model_state = None  # Store best model weights
    epochs_no_improve = 0
    training_history = []
    start_time = time.time()
    num_examples = len(loader.dataset) if hasattr(loader, 'dataset') else len(loader) * config['training']['batch_size']
    
    # Remove epoch cap - train until early stopping
    epoch = 0
    while True:
        epoch_loss = 0.0
        count = 0
        
        for batch in loader:
            # champion_data returns: (src, tgt, ctx_input, ctx_output, src_grid_shape, tgt_grid_shape, task_id)
            src, tgt, ctx_in, ctx_out, src_shapes, tgt_shapes, task_ids = batch
            src, tgt = src.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
            ctx_in, ctx_out = ctx_in.to(device, non_blocking=True), ctx_out.to(device, non_blocking=True)
            
            if tgt.size(1) <= 1:
                continue
            
            tgt_input, tgt_output = tgt[:, :-1], tgt[:, 1:]
            
            try:
                optimizer.zero_grad()
                
                # Mixed precision training
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        logits = model(
                            src=src,
                            tgt=tgt_input,
                            src_grid_shape=src_shapes[0],
                            tgt_grid_shape=tgt_shapes[0],
                            ctx_input=ctx_in,
                            ctx_output=ctx_out
                        )
                        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'])
                    scaler.step(optimizer)
                    scaler.update()
                else:
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
                print(f"⚠️  Step failed: {e}")
                continue
        
        if count == 0:
            continue
        
        avg_loss = epoch_loss / count
        training_history.append({'epoch': epoch + 1, 'loss': avg_loss})
        
        # Early stopping - save best model
        if avg_loss < best_loss - config['training']['min_delta']:
            best_loss = avg_loss
            epochs_no_improve = 0
            # Save best model state (deep copy to avoid reference issues)
            import copy
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= config['training']['patience']:
            # Early stopped - restore best model
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            break
        
        epoch += 1
    
    training_time = time.time() - start_time
    
    # Evaluate AFTER training
    final_metrics = evaluate_model(model, loader, device)
    grid_improvement = final_metrics['grid_accuracy'] - base_metrics['grid_accuracy']
    cell_improvement = final_metrics['cell_accuracy'] - base_metrics['cell_accuracy']
    
    # Metadata (matching ablation scripts + LoRA-specific)
    metadata = {
        'num_examples': num_examples,
        'num_epochs_trained': len(training_history),
        'training_time_seconds': training_time,
        'early_stopped': epochs_no_improve >= config['training']['patience'],
        # Grid accuracy
        'base_grid_accuracy': base_metrics['grid_accuracy'],
        'final_grid_accuracy': final_metrics['grid_accuracy'],
        'grid_improvement': grid_improvement,
        # Cell accuracy
        'base_cell_accuracy': base_metrics['cell_accuracy'],
        'final_cell_accuracy': final_metrics['cell_accuracy'],
        'cell_improvement': cell_improvement,
        # Counts
        'grid_correct': final_metrics['grid_correct'],
        'grid_total': final_metrics['grid_total'],
        'cell_correct': final_metrics['cell_correct'],
        'cell_total': final_metrics['cell_total'],
        # Copy metrics
        'copy_rate': final_metrics['copy_rate'],
        'change_recall': final_metrics['change_recall'],
        'transformation_quality': final_metrics['transformation_quality'],
    }
    
    return best_loss, len(training_history), training_history, metadata


def main():
    """Main training loop - follows Champion training infrastructure."""
    # Parse CLI arguments (same as Champion training)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast_dev_run', type=int, nargs='?', const=2, default=None,
                        help='Run fast dev test with N tasks (default: 2 if flag provided)')
    parser.add_argument('--task-range', type=str, default=None,
                        help='Train specific task range, e.g., "0:200" or "200:400"')
    args, unknown = parser.parse_known_args()
    fast_dev_run = args.fast_dev_run
    task_range = args.task_range
    
    config_path = Path('configs/atomic_lora_training.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Get data files from distributional_alignment (same as Champion)
    data_dir = Path(config['data_dir'])
    with open(data_dir / "split_manifest.json") as f:
        split_info = json.load(f)
    
    # Load task categories (matching ablations)
    task_categories = {}
    categories_file = data_dir / "task_categories.json"
    if categories_file.exists():
        with open(categories_file) as f:
            task_categories = json.load(f)
        print(f"Loaded categories for {len(task_categories)} tasks\n")
    
    # Use train_files for LoRA training (match Champion's training set)
    task_files = [data_dir / fname for fname in split_info["train_files"]]
    
    # Apply task range if specified (for parallel runs)
    if task_range:
        start, end = map(int, task_range.split(':'))
        task_files = task_files[start:end]
        print(f"Task range: {start}:{end} ({len(task_files)} tasks)\n")
    
    # Limit tasks if fast_dev_run specified
    if fast_dev_run:
        task_files = task_files[:fast_dev_run]
    
    # RESUME CAPABILITY: Check for existing adapters
    # Use separate dir for fast_dev_run to avoid interference
    output_base_dir = Path(config['output_dir'])
    if fast_dev_run:
        output_base_dir = output_base_dir.parent / f"{output_base_dir.name}_fast_dev"
    
    existing_adapters = set()
    if output_base_dir.exists():
        for adapter_dir in output_base_dir.iterdir():
            if adapter_dir.is_dir() and (adapter_dir / "adapter_model.safetensors").exists():
                existing_adapters.add(adapter_dir.name)
    
    # Filter out already-trained tasks
    remaining_tasks = [f for f in task_files if f.stem not in existing_adapters]
    
    if len(existing_adapters) > 0:
        print(f"\n⚠️  RESUME MODE: Found {len(existing_adapters)} existing adapters")
        print(f"   Skipping completed tasks, will train {len(remaining_tasks)}/{len(task_files)} remaining\n")
        task_files = remaining_tasks
    
    if len(task_files) == 0:
        print("✅ All tasks already trained! Nothing to do.")
        return
    
    # Print training summary (same format as Champion)
    print(f"\n{'='*70}")
    if fast_dev_run:
        print(f"FAST DEV RUN: Phase 0 - Atomic LoRA Skills ({fast_dev_run} tasks)")
    elif task_range:
        print(f"TRAINING: Phase 0 - Atomic LoRA Skills (tasks {task_range})")
    else:
        print(f"TRAINING: Phase 0 - Atomic LoRA Skills")
    print(f"{'='*70}")
    print(f"Dataset: distributional_alignment (re-arc synthetic)")
    print(f"Total tasks: {len(task_files)}")
    print(f"Base model: Champion (epoch=36, val_loss=0.5926)")
    print(f"LoRA config: rank={config['lora_rank']}, alpha={config['lora_alpha']}")
    print(f"Learning rate: {config['training']['learning_rate']:.7f} (10x lower than Champion)")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Training: Until early stopping (patience={config['training']['patience']} epochs, min_delta={config['training']['min_delta']})")
    print(f"Best model checkpoint: Automatically saved and restored")
    print(f"Output: {output_base_dir}")
    print(f"Device: {device}")
    if fast_dev_run:
        print(f"MODE: Fast dev run (testing only - separate output dir)")
    print(f"{'='*70}\n")
    
    # Load base model
    base_model = load_champion(Path(config['champion_checkpoint']), device)
    print(f"Champion loaded: {sum(p.numel() for p in base_model.parameters()):,} params\n")
    
    # Create log directories (for both CSV and process logs)
    # Use separate CSV for fast_dev_run to avoid overwriting main results
    log_dir = Path("logs") / "atomic_loras"
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_filename = "lora_training_metrics_fast_dev.csv" if fast_dev_run else "lora_training_metrics.csv"
    csv_path = log_dir / csv_filename
    
    # CSV headers (matching ablations + LoRA-specific)
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=[
        'task_id', 'category', 'status',
        # Grid metrics
        'base_grid_accuracy', 'final_grid_accuracy', 'grid_improvement',
        # Cell metrics  
        'base_cell_accuracy', 'final_cell_accuracy', 'cell_improvement',
        # Copy metrics
        'copy_rate', 'change_recall', 'transformation_quality',
        # Training info
        'final_loss', 'epochs', 'num_examples', 'training_time_seconds',
        'early_stopped', 'error'
    ])
    csv_writer.writeheader()
    csv_file.flush()
    
    # JSON summary path (will update after each task for crash resilience with file locking)
    json_filename = 'atomic_lora_training_summary_fast_dev.json' if fast_dev_run else 'atomic_lora_training_summary.json'
    json_path = Path('outputs') / json_filename
    Path('outputs').mkdir(exist_ok=True)
    
    # Initialize JSON file if doesn't exist
    if not json_path.exists():
        with open(json_path, 'w') as f:
            json.dump({'completed': 0, 'failed': 0, 'tasks': {}}, f, indent=2)
    
    for task_file in tqdm(task_files, desc="Training"):
        task_id = task_file.stem
        
        try:
            # Create fresh base model instance to avoid memory issues
            # (LoRA wrapping modifies the model in-place)
            import copy
            task_base_model = copy.deepcopy(base_model)
            lora_model = setup_lora(task_base_model, config).to(device)
            
            loss, epochs, history, metadata = train_task(lora_model, task_file, config, device)
            
            # Save adapter (use output_base_dir which handles fast_dev_run separation)
            task_output_dir = output_base_dir / task_id
            task_output_dir.mkdir(parents=True, exist_ok=True)
            lora_model.save_pretrained(task_output_dir)
            
            # Save training curve if requested
            if config['training'].get('save_training_curve', False):
                with open(task_output_dir / 'training_history.json', 'w') as f:
                    json.dump(history, f, indent=2)
            
            task_data = {
                'status': 'success',
                'final_loss': loss,
                'epochs': epochs,
                'metadata': metadata
            }
            
            # Write to CSV (matching ablations)
            category = task_categories.get(task_id, 'unknown')
            csv_writer.writerow({
                'task_id': task_id,
                'category': category,
                'status': 'success',
                # Grid metrics
                'base_grid_accuracy': f"{metadata['base_grid_accuracy']:.2f}",
                'final_grid_accuracy': f"{metadata['final_grid_accuracy']:.2f}",
                'grid_improvement': f"{metadata['grid_improvement']:.2f}",
                # Cell metrics
                'base_cell_accuracy': f"{metadata['base_cell_accuracy']:.2f}",
                'final_cell_accuracy': f"{metadata['final_cell_accuracy']:.2f}",
                'cell_improvement': f"{metadata['cell_improvement']:.2f}",
                # Copy metrics
                'copy_rate': f"{metadata['copy_rate']:.4f}",
                'change_recall': f"{metadata['change_recall']:.4f}",
                'transformation_quality': f"{metadata['transformation_quality']:.4f}",
                # Training info
                'final_loss': f"{loss:.6f}",
                'epochs': epochs,
                'num_examples': metadata['num_examples'],
                'training_time_seconds': f"{metadata['training_time_seconds']:.2f}",
                'early_stopped': metadata['early_stopped'],
                'error': ''
            })
            csv_file.flush()
            
            # Write JSON summary safely (with file locking for parallel runs)
            write_json_safely(json_path, task_id, task_data)
            
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
            
            task_data = {
                'status': 'failed',
                'error': error_msg,
                'traceback': traceback_str
            }
            
            # Write failure to CSV
            category = task_categories.get(task_id, 'unknown')
            csv_writer.writerow({
                'task_id': task_id,
                'category': category,
                'status': 'failed',
                # Empty metrics
                'base_grid_accuracy': '',
                'final_grid_accuracy': '',
                'grid_improvement': '',
                'base_cell_accuracy': '',
                'final_cell_accuracy': '',
                'cell_improvement': '',
                'copy_rate': '',
                'change_recall': '',
                'transformation_quality': '',
                # Training info
                'final_loss': '',
                'epochs': '',
                'num_examples': '',
                'training_time_seconds': '',
                'early_stopped': '',
                'error': error_msg[:100]  # Truncate long errors
            })
            csv_file.flush()
            
            # Write JSON summary safely (with file locking for parallel runs)
            write_json_safely(json_path, task_id, task_data)
    
    csv_file.close()
    
    # Read final results from JSON
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    # Print final summary (same format as Champion)
    print(f"\n{'='*70}")
    if fast_dev_run:
        print(f"Fast Dev Run Complete!")
    else:
        print(f"Training Complete!")
    print(f"{'='*70}")
    print(f"Successful: {results['completed']}")
    print(f"Failed: {results['failed']}")
    print(f"Adapters saved to: {output_base_dir}")
    print(f"CSV metrics saved to: {csv_path}")
    print(f"Detailed summary: {json_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
