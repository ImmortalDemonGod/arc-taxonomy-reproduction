"""
Calculate parameter counts for different model configurations.
Used to design parameter-matched ablation experiments.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.decoder_only_lightning import DecoderOnlyLightningModule
from src.models.encoder_decoder_lightning import EncoderDecoderLightningModule
from src.models.champion_lightning import ChampionLightningModule


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_params(n):
    """Format parameter count."""
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def test_config(name, model):
    """Test a model configuration."""
    params = count_parameters(model)
    print(f"{name:40s}: {format_params(params):>10s} ({params:,} params)")
    return params


print("="*70)
print("PARAMETER COUNT ANALYSIS")
print("="*70)
print()

# Champion baseline (our target)
print("CHAMPION (Target = 1.7M):")
print("-" * 70)
champion = ChampionLightningModule(
    vocab_size=11,
    d_model=160,
    num_encoder_layers=1,
    num_decoder_layers=3,
    num_heads=4,
    d_ff=640,
    max_grid_size=30,
    dropout=0.167,
)
target_params = test_config("Champion (E-D + Grid2D + PermInv + Bridge)", champion)

print("\n" + "="*70)
print("TESTING ABLATION CONFIGURATIONS")
print("="*70)
print()

# Test Encoder-Decoder (no Grid2D, no PermInv, no Bridge)
print("Exp 0: Generic Encoder-Decoder (Goal: ~1.7M)")
print("-" * 70)

# Try matching Champion's base dimensions first
ed_1 = EncoderDecoderLightningModule(
    vocab_size=11,
    d_model=160,
    num_encoder_layers=1,
    num_decoder_layers=3,
    num_heads=4,
    d_ff=640,
    dropout=0.167,
)
test_config("E-D (same dims as Champion)", ed_1)

# Try increasing layers to compensate
ed_2 = EncoderDecoderLightningModule(
    vocab_size=11,
    d_model=160,
    num_encoder_layers=2,
    num_decoder_layers=4,
    num_heads=4,
    d_ff=640,
    dropout=0.167,
)
test_config("E-D (enc=2, dec=4)", ed_2)

# Try increasing d_model
ed_3 = EncoderDecoderLightningModule(
    vocab_size=11,
    d_model=192,
    num_encoder_layers=1,
    num_decoder_layers=3,
    num_heads=4,
    d_ff=768,
    dropout=0.167,
)
test_config("E-D (d_model=192, d_ff=768)", ed_3)

print()
print("Exp -1: Decoder-Only (Goal: ~1.7M)")
print("-" * 70)

# Decoder-Only needs more capacity since it has no encoder
dec_1 = DecoderOnlyLightningModule(
    vocab_size=11,
    context_length=512,
    d_model=160,
    num_layers=6,  # Double decoder layers
    num_heads=4,
    d_ff=640,
    dropout=0.167,
)
test_config("Decoder-Only (layers=6)", dec_1)

dec_2 = DecoderOnlyLightningModule(
    vocab_size=11,
    context_length=512,
    d_model=200,  # Increase d_model
    num_layers=5,
    num_heads=4,
    d_ff=800,
    dropout=0.167,
)
test_config("Decoder-Only (d_model=200, layers=5)", dec_2)

dec_3 = DecoderOnlyLightningModule(
    vocab_size=11,
    context_length=512,
    d_model=192,
    num_layers=5,
    num_heads=4,
    d_ff=768,
    dropout=0.167,
)
test_config("Decoder-Only (d_model=192, layers=5)", dec_3)

print()
print("="*70)
print(f"Target: {format_params(target_params)} ({target_params:,} params)")
print("="*70)
