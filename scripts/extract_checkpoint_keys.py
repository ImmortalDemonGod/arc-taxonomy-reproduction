#!/usr/bin/env python3
"""
Extract and document all keys from champion_bootstrap.ckpt.

This script provides the indispensable map for Phase 2 integration.
"""
import torch
from pathlib import Path
from collections import defaultdict


def analyze_checkpoint(ckpt_path: str, output_path: str):
    """Load checkpoint and extract all keys with structure analysis."""
    
    print(f"Loading checkpoint from: {ckpt_path}")
    # Need weights_only=False due to OmegaConf objects in hyper_parameters
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    # Top-level keys
    top_keys = list(checkpoint.keys())
    
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CHAMPION_BOOTSTRAP.CKPT - COMPLETE KEY STRUCTURE\n")
        f.write("=" * 80 + "\n\n")
        
        # Section 1: Top-level keys
        f.write("## TOP-LEVEL KEYS\n")
        f.write("-" * 80 + "\n")
        for key in top_keys:
            f.write(f"- {key}\n")
        f.write("\n\n")
        
        # Section 2: State dict keys (the critical ones)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            f.write("## STATE_DICT KEYS (Total: {})\n".format(len(state_dict)))
            f.write("-" * 80 + "\n\n")
            
            # Group by prefix (model component)
            grouped = defaultdict(list)
            for key in sorted(state_dict.keys()):
                prefix = key.split('.')[0] if '.' in key else key
                grouped[prefix].append(key)
            
            for prefix in sorted(grouped.keys()):
                f.write(f"### Component: {prefix} ({len(grouped[prefix])} keys)\n")
                for key in grouped[prefix]:
                    shape = tuple(state_dict[key].shape) if hasattr(state_dict[key], 'shape') else 'scalar'
                    dtype = state_dict[key].dtype if hasattr(state_dict[key], 'dtype') else type(state_dict[key])
                    f.write(f"  - {key}\n")
                    f.write(f"      Shape: {shape}, Dtype: {dtype}\n")
                f.write("\n")
        
        # Section 3: Hyper parameters
        if 'hyper_parameters' in checkpoint:
            f.write("## HYPER_PARAMETERS\n")
            f.write("-" * 80 + "\n")
            hyper_params = checkpoint['hyper_parameters']
            
            def print_nested_dict(d, indent=0, file_handle=f):
                """Recursively print nested dict structure."""
                for key, value in sorted(d.items()):
                    if isinstance(value, dict):
                        file_handle.write("  " * indent + f"- {key}:\n")
                        print_nested_dict(value, indent + 1, file_handle)
                    else:
                        file_handle.write("  " * indent + f"- {key}: {value}\n")
            
            print_nested_dict(hyper_params)
            f.write("\n")
        
        # Section 4: Other metadata
        f.write("## METADATA\n")
        f.write("-" * 80 + "\n")
        for key in top_keys:
            if key not in ['state_dict', 'hyper_parameters']:
                value = checkpoint[key]
                if isinstance(value, (int, float, str, bool)):
                    f.write(f"- {key}: {value}\n")
                else:
                    f.write(f"- {key}: {type(value).__name__}\n")
        f.write("\n")
        
        # Section 5: Summary statistics
        f.write("## SUMMARY STATISTICS\n")
        f.write("-" * 80 + "\n")
        if 'state_dict' in checkpoint:
            total_params = sum(p.numel() for p in state_dict.values() if hasattr(p, 'numel'))
            f.write(f"Total parameters: {total_params:,}\n")
            f.write(f"Total state dict keys: {len(state_dict)}\n")
        f.write(f"Checkpoint size (MB): {Path(ckpt_path).stat().st_size / 1024 / 1024:.2f}\n")
    
    print(f"✅ Checkpoint analysis saved to: {output_path}")
    print(f"   Total state_dict keys: {len(state_dict) if 'state_dict' in checkpoint else 0}")


if __name__ == "__main__":
    # Paths
    ckpt_path = "/Users/tomriddle1/Holistic-Performance-Enhancement/cultivation/systems/arc_reactor/outputs/checkpoints/champion_bootstrap.ckpt"
    output_path = "/Users/tomriddle1/Holistic-Performance-Enhancement/cultivation/systems/arc_reactor/publications/arc_taxonomy_2025/reproduction/docs/checkpoint_keys.txt"
    
    analyze_checkpoint(ckpt_path, output_path)
