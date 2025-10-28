"""Fix all training scripts to support --fast_dev_run and separate log directories."""
import re
from pathlib import Path

def fix_training_script(script_path: Path, exp_name: str, per_task_subdir: str):
    """Fix a training script."""
    content = script_path.read_text()
    
    # 1. Add torch import if missing
    if 'import torch\n' not in content and 'import torch ' not in content:
        content = content.replace(
            'import pytorch_lightning as pl',
            'import torch\nimport pytorch_lightning as pl'
        )
    
    # 2. Add argparse import after sys import
    if 'import argparse' not in content:
        content = content.replace(
            'import sys\nfrom pathlib import Path',
            'import sys\nimport argparse\nfrom pathlib import Path'
        )
    
    # 2. Add CLI argument parsing in main()
    if '--fast_dev_run' not in content:
        # Find main() function and add argument parsing
        main_pattern = r'(def main\(\):[\s\S]*?"""[\s\S]*?""")'
        def add_arg_parse(match):
            return match.group(1) + '\n    # Parse CLI arguments\n    parser = argparse.ArgumentParser()\n    parser.add_argument(\'--fast_dev_run\', type=int, default=None,\n                        help=\'Run fast_dev_run with N batches for testing\')\n    args, unknown = parser.parse_known_args()\n    fast_dev_run = args.fast_dev_run\n'
        content = re.sub(main_pattern, add_arg_parse, content)
    
    # 3. Add Tensor Core optimization after seed_everything
    if "torch.set_float32_matmul_precision('high')" not in content:
        content = content.replace(
            'pl.seed_everything(307, workers=True)',
            "pl.seed_everything(307, workers=True)\n    \n    # Set matmul precision for Tensor Cores (A6000 optimization)\n    torch.set_float32_matmul_precision('high')"
        )
    
    # 4. Fix PerTaskMetricsLogger to use experiment-specific subdirectory
    content = re.sub(
        r'per_task_logger = PerTaskMetricsLogger\(\)',
        f'per_task_logger = PerTaskMetricsLogger(log_dir="logs/per_task_metrics/{per_task_subdir}")',
        content
    )
    
    # 4. Add fast_dev_run to Trainer
    if 'fast_dev_run=' not in content:
        content = re.sub(
            r'(trainer = pl\.Trainer\([^)]+enable_model_summary=True,)',
            r'\1\n        fast_dev_run=fast_dev_run if fast_dev_run else False,  # CLI override for testing',
            content
        )
    
    script_path.write_text(content)
    print(f"✓ Fixed {script_path.name}")

# Fix all scripts
scripts_dir = Path(__file__).parent
fixes = [
    ('train_exp0_encoder_decoder.py', 'Exp 0', 'exp0'),
    ('train_exp1_grid2d_pe.py', 'Exp 1', 'exp1'),
    ('train_exp2_perminv.py', 'Exp 2', 'exp2'),
    ('train_exp3_champion.py', 'Exp 3', 'exp3'),
]

for script_name, exp_name, subdir in fixes:
    script_path = scripts_dir / script_name
    if script_path.exists():
        fix_training_script(script_path, exp_name, subdir)
    else:
        print(f"⚠ {script_name} not found")

print("\n✅ All training scripts fixed!")
