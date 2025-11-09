"""
Verify actual parameter counts for all ablation models.
"""
import torch
from src.models.baseline_decoder_only_lightning import BaselineDecoderOnlyLightningModule
from src.models.exp0_encoder_decoder_lightning import Exp0EncoderDecoderLightningModule
from src.models.exp1_grid2d_pe_lightning import Exp1Grid2DPELightningModule
from src.models.exp2_perminv_lightning import Exp2PermInvLightningModule
from src.models.exp3_champion_lightning import Exp3ChampionLightningModule

print('='*70)
print('CURRENT PARAMETER COUNTS (ACTUAL)')
print('='*70)

# Baseline current config
baseline_current = BaselineDecoderOnlyLightningModule(
    d_model=164, num_layers=4, d_ff=656
)
baseline_params = sum(p.numel() for p in baseline_current.parameters())
print(f'Baseline (d_model=164, layers=4, d_ff=656): {baseline_params:,}')

# Exp0 current config
exp0_current = Exp0EncoderDecoderLightningModule(
    d_model=168, num_encoder_layers=1, num_decoder_layers=3, d_ff=672
)
exp0_params = sum(p.numel() for p in exp0_current.parameters())
print(f'Exp0 (d_model=168, 1+3, d_ff=672): {exp0_params:,}')

# Exp1 current config
exp1_current = Exp1Grid2DPELightningModule(
    d_model=168, num_encoder_layers=1, num_decoder_layers=3, d_ff=672
)
exp1_params = sum(p.numel() for p in exp1_current.parameters())
print(f'Exp1 (d_model=168, 1+3, d_ff=672): {exp1_params:,}')

# Exp2 current config
exp2_current = Exp2PermInvLightningModule(
    d_model=168, num_encoder_layers=1, num_decoder_layers=3, d_ff=672
)
exp2_params = sum(p.numel() for p in exp2_current.parameters())
print(f'Exp2 (d_model=168, 1+3, d_ff=672): {exp2_params:,}')

# Champion current config
champion_current = Exp3ChampionLightningModule(
    d_model=160, num_encoder_layers=1, num_decoder_layers=3, d_ff=640
)
champion_params = sum(p.numel() for p in champion_current.parameters())
print(f'Champion (d_model=160, 1+3, d_ff=640): {champion_params:,}')

print()
print('='*70)
print('PROPOSED CONFIGS (ACTUAL)')
print('='*70)

# Baseline proposed: d_model=160, layers=5
baseline_new = BaselineDecoderOnlyLightningModule(
    d_model=160, num_layers=5, d_ff=640
)
baseline_new_params = sum(p.numel() for p in baseline_new.parameters())
print(f'Baseline (d_model=160, layers=5, d_ff=640): {baseline_new_params:,}')

# Exp0 proposed
exp0_new = Exp0EncoderDecoderLightningModule(
    d_model=160, num_encoder_layers=1, num_decoder_layers=3, d_ff=640
)
exp0_new_params = sum(p.numel() for p in exp0_new.parameters())
print(f'Exp0 (d_model=160, 1+3, d_ff=640): {exp0_new_params:,}')

# Exp1 proposed
exp1_new = Exp1Grid2DPELightningModule(
    d_model=160, num_encoder_layers=1, num_decoder_layers=3, d_ff=640
)
exp1_new_params = sum(p.numel() for p in exp1_new.parameters())
print(f'Exp1 (d_model=160, 1+3, d_ff=640): {exp1_new_params:,}')

# Exp2 proposed
exp2_new = Exp2PermInvLightningModule(
    d_model=160, num_encoder_layers=1, num_decoder_layers=3, d_ff=640
)
exp2_new_params = sum(p.numel() for p in exp2_new.parameters())
print(f'Exp2 (d_model=160, 1+3, d_ff=640): {exp2_new_params:,}')

print(f'Champion (d_model=160, 1+3, d_ff=640): {champion_params:,}')

print()
print('='*70)
print('VERIFICATION')
print('='*70)
if baseline_new_params == exp0_new_params == exp1_new_params == exp2_new_params == champion_params:
    print('✅ ALL MODELS HAVE IDENTICAL PARAMETER COUNTS')
    print(f'   Exact count: {champion_params:,} parameters')
else:
    print('❌ PARAMETER COUNTS DO NOT MATCH:')
    print(f'   Baseline: {baseline_new_params:,}')
    print(f'   Exp0: {exp0_new_params:,}')
    print(f'   Exp1: {exp1_new_params:,}')
    print(f'   Exp2: {exp2_new_params:,}')
    print(f'   Champion: {champion_params:,}')
    print()
    print('Differences from Champion:')
    print(f'   Baseline: {baseline_new_params - champion_params:+,}')
    print(f'   Exp0: {exp0_new_params - champion_params:+,}')
    print(f'   Exp1: {exp1_new_params - champion_params:+,}')
    print(f'   Exp2: {exp2_new_params - champion_params:+,}')
