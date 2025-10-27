#!/usr/bin/env python3
"""Debug script to explore checkpoint config structure."""
import torch
from omegaconf import OmegaConf

ckpt_path = "/Users/tomriddle1/Holistic-Performance-Enhancement/cultivation/systems/arc_reactor/outputs/checkpoints/champion_bootstrap.ckpt"

checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
hparams = checkpoint['hyper_parameters']

# Print first-level keys
print("=== TOP LEVEL KEYS ===")
for key in sorted(hparams.keys()):
    print(f"- {key}")

print("\n=== LOOKING FOR MODEL PARAMS ===")

# Check if there's a 'model' key
if 'model' in hparams:
    print("Found 'model' key")
    model_params = hparams['model']
    if isinstance(model_params, dict):
        for key in sorted(model_params.keys()):
            print(f"  - model.{key}")

# Check cumoe.model
if 'cumoe' in hparams and 'model' in hparams['cumoe']:
    print("\nFound 'cumoe.model' key")
    cumoe_model = hparams['cumoe']['model']
    if isinstance(cumoe_model, dict):
        for key in sorted(cumoe_model.keys()):
            print(f"  - cumoe.model.{key}")
            
# Look for d_model
print("\n=== SEARCHING FOR d_model ===")
def find_key_recursive(d, target_key, path=""):
    """Recursively find all occurrences of a key."""
    results = []
    if isinstance(d, dict):
        for key, value in d.items():
            new_path = f"{path}.{key}" if path else key
            if key == target_key:
                results.append((new_path, value))
            if isinstance(value, dict):
                results.extend(find_key_recursive(value, target_key, new_path))
    return results

d_model_locations = find_key_recursive(hparams, 'd_model')
for path, value in d_model_locations:
    print(f"  {path} = {value}")

n_encoder_locations = find_key_recursive(hparams, 'n_encoder_layers')
if n_encoder_locations:
    print("\n=== n_encoder_layers ===")
    for path, value in n_encoder_locations:
        print(f"  {path} = {value}")
else:
    print("\n⚠️  n_encoder_layers not found, searching for alternatives...")
    num_layers_locations = find_key_recursive(hparams, 'num_layers')
    n_layers_locations = find_key_recursive(hparams, 'n_layers')
    print("num_layers:", num_layers_locations)
    print("n_layers:", n_layers_locations)
