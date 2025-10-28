"""
Complete ablation test: Verify all 5 experiments work and are parameter-matched.

Tests the full ablation series:
- Exp -1: Decoder-Only (1.74M params)
- Exp 0: Generic Encoder-Decoder (1.71M params)
- Exp 1: E-D + Grid2D PE (1.71M params)
- Exp 2: Exp 1 + PermInv (1.71M params)
- Exp 3: Champion (all components) (1.72M params)

All models within 2% of 1.72M target for valid ablation.
"""
import sys
from pathlib import Path
import pytorch_lightning as pl

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.baseline_decoder_only_lightning import BaselineDecoderOnlyLightningModule
from src.models.exp0_encoder_decoder_lightning import Exp0EncoderDecoderLightningModule
from src.models.exp1_grid2d_pe_lightning import Exp1Grid2DPELightningModule
from src.models.exp2_perminv_lightning import Exp2PermInvLightningModule
from src.models.exp3_champion_lightning import Exp3ChampionLightningModule
from src.data.decoder_only_data import create_decoder_only_dataloader
from src.data.encoder_decoder_data import create_encoder_decoder_dataloader
from src.data.champion_data import create_champion_dataloader


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


def test_model(name, model, train_loader, val_loader):
    """Test a model with fast_dev_run."""
    print(f"\n{'='*70}")
    print(f"TESTING: {name}")
    print(f"{'='*70}")
    
    # Count parameters
    params = count_parameters(model)
    print(f"Parameters: {format_params(params)} ({params:,})")
    
    try:
        trainer = pl.Trainer(
            fast_dev_run=True,  # Run 1 batch train + 1 batch val
            precision='16-mixed',
            gradient_clip_val=1.0,
            enable_progress_bar=True,
        )
        
        trainer.fit(model, train_loader, val_loader)
        
        print(f"✅ {name}: PASSED")
        return True, params
        
    except Exception as e:
        print(f"❌ {name}: FAILED")
        print(f"Error: {e}")
        return False, params


def main():
    """Test all models in ablation series."""
    pl.seed_everything(307, workers=True)
    
    # Get data
    import json
    data_dir = Path(__file__).parent.parent / "data" / "distributional_alignment"
    
    with open(data_dir / "split_manifest.json") as f:
        split_info = json.load(f)
    
    train_files = [data_dir / fname for fname in split_info["train_files"][:5]]
    val_files = [data_dir / fname for fname in split_info["val_files"][:2]]
    
    print(f"\n{'='*70}")
    print(f"COMPLETE ABLATION TEST - BASELINE + 4 EXPERIMENTS")
    print(f"{'='*70}")
    print(f"Data: {len(train_files)} train, {len(val_files)} val files")
    print(f"Target: 1.72M params (±2% tolerance)")
    print(f"Running 1 train batch + 1 val batch per model\n")
    
    results = {}
    param_counts = {}
    
    # Baseline: Decoder-Only
    print("\n[1/5] Testing Baseline: Decoder-Only (1.74M params)...")
    train_loader = create_decoder_only_dataloader(train_files, batch_size=4, shuffle=True)
    val_loader = create_decoder_only_dataloader(val_files, batch_size=4, shuffle=False)
    model = BaselineDecoderOnlyLightningModule(
        vocab_size=11,
        context_length=512,
        d_model=164,
        num_layers=4,
        num_heads=4,
        d_ff=656,
        dropout=0.167,
    )
    results['Baseline: Decoder-Only'], param_counts['Baseline'] = test_model(
        'Baseline: Decoder-Only', model, train_loader, val_loader
    )
    
    # Exp 0: Generic Encoder-Decoder
    print("\n[2/5] Testing Exp 0: Generic E-D (1.71M params)...")
    train_loader = create_encoder_decoder_dataloader(train_files, batch_size=4, shuffle=True)
    val_loader = create_encoder_decoder_dataloader(val_files, batch_size=4, shuffle=False)
    model = Exp0EncoderDecoderLightningModule(
        vocab_size=11,
        d_model=168,
        num_encoder_layers=1,
        num_decoder_layers=3,
        num_heads=4,
        d_ff=672,
        dropout=0.167,
    )
    results['Exp 0: Generic E-D'], param_counts['Exp 0'] = test_model(
        'Exp 0: Generic E-D', model, train_loader, val_loader
    )
    
    # Exp 1: E-D + Grid2D PE
    print("\n[3/5] Testing Exp 1: E-D + Grid2D PE (1.71M params)...")
    train_loader = create_encoder_decoder_dataloader(train_files, batch_size=4, shuffle=True)
    val_loader = create_encoder_decoder_dataloader(val_files, batch_size=4, shuffle=False)
    model = Exp1Grid2DPELightningModule(
        vocab_size=11,
        d_model=168,
        num_encoder_layers=1,
        num_decoder_layers=3,
        num_heads=4,
        d_ff=672,
        max_grid_size=30,
        dropout=0.167,
    )
    results['Exp 1: E-D + Grid2D PE'], param_counts['Exp 1'] = test_model(
        'Exp 1: E-D + Grid2D PE', model, train_loader, val_loader
    )
    
    # Exp 2: E-D + Grid2D PE + PermInv
    print("\n[4/5] Testing Exp 2: E-D + Grid2D PE + PermInv (1.71M params)...")
    train_loader = create_encoder_decoder_dataloader(train_files, batch_size=4, shuffle=True)
    val_loader = create_encoder_decoder_dataloader(val_files, batch_size=4, shuffle=False)
    model = Exp2PermInvLightningModule(
        vocab_size=11,
        d_model=168,
        num_encoder_layers=1,
        num_decoder_layers=3,
        num_heads=4,
        d_ff=672,
        max_grid_size=30,
        dropout=0.167,
    )
    results['Exp 2: E-D + Grid2D + PermInv'], param_counts['Exp 2'] = test_model(
        'Exp 2: E-D + Grid2D + PermInv', model, train_loader, val_loader
    )
    
    # Exp 3: Champion
    print("\n[5/5] Testing Exp 3: Champion (1.72M params)...")
    train_loader = create_champion_dataloader(train_files, batch_size=4, shuffle=True, num_context_pairs=2)
    val_loader = create_champion_dataloader(val_files, batch_size=4, shuffle=False, num_context_pairs=2)
    model = Exp3ChampionLightningModule(
        vocab_size=11,
        d_model=160,
        num_encoder_layers=1,
        num_decoder_layers=3,
        num_heads=4,
        d_ff=640,
        max_grid_size=30,
        dropout=0.167,
    )
    results['Exp 3: Champion'], param_counts['Exp 3'] = test_model(
        'Exp 3: Champion', model, train_loader, val_loader
    )
    
    # Summary
    print(f"\n{'='*70}")
    print(f"PARAMETER MATCHING ANALYSIS")
    print(f"{'='*70}")
    
    TARGET = 1_724_619  # Champion's exact count
    
    for name in ['Baseline', 'Exp 0', 'Exp 1', 'Exp 2', 'Exp 3']:
        params = param_counts[name]
        diff_pct = abs(params - TARGET) / TARGET * 100
        status = "✅" if diff_pct <= 2.0 else "⚠️"
        print(f"{status} {name:10s}: {format_params(params):>10s} ({diff_pct:>4.1f}% diff from target)")
    
    print(f"\n{'='*70}")
    print(f"FUNCTIONAL TEST RESULTS")
    print(f"{'='*70}")
    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:35s}: {status}")
    
    all_passed = all(results.values())
    print(f"\n{'='*70}")
    if all_passed:
        print("🎉 ALL 5 EXPERIMENTS READY FOR TRAINING!")
        print("   Complete ablation series validated:")
        print("   Baseline → Exp 0 → Exp 1 → Exp 2 → Exp 3")
    else:
        print("⚠️  SOME EXPERIMENTS FAILED - CHECK ERRORS ABOVE")
    print(f"{'='*70}\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
