"""Fix checkpoint printing to handle fast_dev_run mode."""
import re
from pathlib import Path

def fix_script(script_path):
    content = script_path.read_text()
    
    # Find and replace the checkpoint printing section
    old_pattern = r'    # Train!\n    trainer\.fit\(model, train_loader, val_loader\)\n    \n    print\(f"\\n\{\'=\'\*70\}"\)\n    print\(f"Training complete!"\)\n    print\(f"Best checkpoint: \{checkpoint_callback\.best_model_path\}"\)\n    print\(f"Best val_loss: \{checkpoint_callback\.best_model_score:\.4f\}"\)\n    print\(f"\{\'=\'\*70\}\\n"\)'
    
    new_code = '''    # Train!
    trainer.fit(model, train_loader, val_loader)
    
    print(f"\\n{'='*70}")
    print(f"Training complete!")
    
    # Only print checkpoint info if not in fast_dev_run mode
    if not fast_dev_run:
        print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
        if checkpoint_callback.best_model_score is not None:
            print(f"Best val_loss: {checkpoint_callback.best_model_score:.4f}")
    else:
        print(f"Fast dev run completed (5 batches)")
    
    print(f"{'='*70}\\n")'''
    
    # Try simpler pattern
    if "print(f\"Best val_loss: {checkpoint_callback.best_model_score:.4f}\")" in content:
        # Replace the section
        content = re.sub(
            r'print\(f"Best checkpoint: \{checkpoint_callback\.best_model_path\}"\)\n    print\(f"Best val_loss: \{checkpoint_callback\.best_model_score:\.4f\}"\)',
            '''# Only print checkpoint info if not in fast_dev_run mode
    if not fast_dev_run:
        print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
        if checkpoint_callback.best_model_score is not None:
            print(f"Best val_loss: {checkpoint_callback.best_model_score:.4f}")
    else:
        print(f"Fast dev run completed (5 batches)")''',
            content
        )
        script_path.write_text(content)
        return True
    return False

scripts_dir = Path('scripts')
for script in ['train_exp0_encoder_decoder.py', 'train_exp1_grid2d_pe.py', 
               'train_exp2_perminv.py', 'train_exp3_champion.py']:
    script_path = scripts_dir / script
    if script_path.exists():
        if fix_script(script_path):
            print(f"✓ Fixed {script}")
        else:
            print(f"⚠ {script} - pattern not found or already fixed")
    else:
        print(f"✗ {script} not found")

print("\n✅ All scripts updated")
