"""
Systematic search for parameter-matched configurations.
Target: 1.72M params (matching Champion)
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


TARGET = 1_724_619  # Champion's param count
TOLERANCE = 0.05  # 5% tolerance


def test_and_score(name, model):
    """Test config and return how close to target."""
    params = count_parameters(model)
    diff = abs(params - TARGET)
    pct_diff = diff / TARGET * 100
    
    status = "✅" if pct_diff <= TOLERANCE * 100 else "  "
    
    print(f"{status} {name:45s}: {format_params(params):>10s} ({pct_diff:>5.1f}% diff)")
    return params, pct_diff


print("="*75)
print(f"SYSTEMATIC PARAMETER MATCHING (Target: {format_params(TARGET)})")
print("="*75)
print()

# Champion (baseline)
print("BASELINE:")
print("-" * 75)
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
test_and_score("Champion (Exp 3)", champion)

print()
print("="*75)
print("ENCODER-DECODER SEARCH (Exp 0)")
print("="*75)
print()

# Strategy: E-D with same dims is 1.55M, need ~170K more params
# Try slightly larger dimensions

configs = [
    # Increase d_model slightly (must be divisible by 4)
    (172, 1, 3, 688),
    (168, 1, 3, 672),
    (164, 1, 3, 656),
    # Try more layers
    (160, 1, 4, 640),
    (160, 2, 3, 640),
    # Mix approaches
    (164, 1, 4, 656),
    (168, 1, 4, 672),
]

best_ed = None
best_ed_diff = float('inf')

for d_model, enc_layers, dec_layers, d_ff in configs:
    model = EncoderDecoderLightningModule(
        vocab_size=11,
        d_model=d_model,
        num_encoder_layers=enc_layers,
        num_decoder_layers=dec_layers,
        num_heads=4,
        d_ff=d_ff,
        dropout=0.167,
    )
    params, diff = test_and_score(
        f"E-D (d={d_model}, enc={enc_layers}, dec={dec_layers}, ff={d_ff})",
        model
    )
    
    if diff < best_ed_diff:
        best_ed_diff = diff
        best_ed = (d_model, enc_layers, dec_layers, d_ff, params)

print()
print("="*75)
print("DECODER-ONLY SEARCH (Exp -1)")
print("="*75)
print()

# Strategy: Decoder-only needs fewer layers since no encoder
# Start smaller and work up

configs = [
    # Try 4 layers with varying d_model (must be divisible by 4)
    (160, 4, 640),
    (168, 4, 672),
    (164, 4, 656),
    # Try 5 layers with smaller d_model
    (148, 5, 592),
    (144, 5, 576),
    (140, 5, 560),
    (136, 5, 544),
]

best_dec = None
best_dec_diff = float('inf')

for d_model, layers, d_ff in configs:
    model = DecoderOnlyLightningModule(
        vocab_size=11,
        context_length=512,
        d_model=d_model,
        num_layers=layers,
        num_heads=4,
        d_ff=d_ff,
        dropout=0.167,
    )
    params, diff = test_and_score(
        f"Decoder (d={d_model}, layers={layers}, ff={d_ff})",
        model
    )
    
    if diff < best_dec_diff:
        best_dec_diff = diff
        best_dec = (d_model, layers, d_ff, params)

print()
print("="*75)
print("BEST MATCHES")
print("="*75)
print()

if best_ed:
    d_model, enc_layers, dec_layers, d_ff, params = best_ed
    print(f"✅ Encoder-Decoder: d_model={d_model}, enc_layers={enc_layers}, "
          f"dec_layers={dec_layers}, d_ff={d_ff}")
    print(f"   Params: {format_params(params)} ({best_ed_diff:.2f}% diff)")
    print()

if best_dec:
    d_model, layers, d_ff, params = best_dec
    print(f"✅ Decoder-Only: d_model={d_model}, num_layers={layers}, d_ff={d_ff}")
    print(f"   Params: {format_params(params)} ({best_dec_diff:.2f}% diff)")
    print()

print("="*75)
