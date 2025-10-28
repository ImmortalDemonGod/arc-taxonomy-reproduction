"""
Quick test script to verify all models can train.

Runs fast_dev_run (1 batch train + 1 batch val) for each model.
"""
import sys
from pathlib import Path
import pytorch_lightning as pl

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.decoder_only_lightning import DecoderOnlyLightningModule
from src.models.encoder_decoder_lightning import EncoderDecoderLightningModule
from src.models.champion_lightning import ChampionLightningModule
from src.data.decoder_only_data import create_decoder_only_dataloader
from src.data.encoder_decoder_data import create_encoder_decoder_dataloader
from src.data.champion_data import create_champion_dataloader


def test_model(name, model, train_loader, val_loader):
    """Test a model with fast_dev_run."""
    print(f"\n{'='*70}")
    print(f"TESTING: {name}")
    print(f"{'='*70}")
    
    try:
        trainer = pl.Trainer(
            fast_dev_run=True,  # Run 1 batch train + 1 batch val
            precision='16-mixed',
            gradient_clip_val=1.0,
            enable_progress_bar=True,
        )
        
        trainer.fit(model, train_loader, val_loader)
        
        print(f"✅ {name}: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ {name}: FAILED")
        print(f"Error: {e}")
        return False


def main():
    """Test all models."""
    pl.seed_everything(307, workers=True)
    
    # Get data from distributional_alignment dataset
    import json
    data_dir = Path(__file__).parent.parent / "data" / "distributional_alignment"
    
    # Load split manifest
    with open(data_dir / "split_manifest.json") as f:
        split_info = json.load(f)
    
    # Use just first 5 files for quick test
    train_files = [data_dir / fname for fname in split_info["train_files"][:5]]
    val_files = [data_dir / fname for fname in split_info["val_files"][:2]]
    
    print(f"\n{'='*70}")
    print(f"FAST DEV RUN TEST - ALL MODELS")
    print(f"{'='*70}")
    print(f"Data: {len(train_files)} train, {len(val_files)} val files")
    print(f"Running 1 train batch + 1 val batch per model\n")
    
    results = {}
    
    # Test 1: Decoder-Only (PARAMETER-MATCHED: 1.74M)
    print("\n[1/3] Testing Decoder-Only (1.74M params)...")
    train_loader = create_decoder_only_dataloader(train_files, batch_size=4, shuffle=True)
    val_loader = create_decoder_only_dataloader(val_files, batch_size=4, shuffle=False)
    model = DecoderOnlyLightningModule(
        vocab_size=11,
        context_length=512,
        d_model=164,  # Matched
        num_layers=4,  # Matched
        num_heads=4,
        d_ff=656,  # Matched
        dropout=0.167,
    )
    results['Decoder-Only'] = test_model('Decoder-Only', model, train_loader, val_loader)
    
    # Test 2: Encoder-Decoder (PARAMETER-MATCHED: 1.71M)
    print("\n[2/3] Testing Encoder-Decoder (1.71M params)...")
    train_loader = create_encoder_decoder_dataloader(train_files, batch_size=4, shuffle=True)
    val_loader = create_encoder_decoder_dataloader(val_files, batch_size=4, shuffle=False)
    model = EncoderDecoderLightningModule(
        vocab_size=11,
        d_model=168,  # Matched
        num_encoder_layers=1,  # Matched
        num_decoder_layers=3,  # Matched
        num_heads=4,
        d_ff=672,  # Matched
        dropout=0.167,
    )
    results['Encoder-Decoder'] = test_model('Encoder-Decoder', model, train_loader, val_loader)
    
    # Test 3: Champion
    print("\n[3/3] Testing Champion...")
    train_loader = create_champion_dataloader(train_files, batch_size=4, shuffle=True, num_context_pairs=2)
    val_loader = create_champion_dataloader(val_files, batch_size=4, shuffle=False, num_context_pairs=2)
    model = ChampionLightningModule(
        vocab_size=11,
        d_model=160,
        num_encoder_layers=1,
        num_decoder_layers=3,  # Trial 69: 1.7M params
        num_heads=4,
        d_ff=640,  # Trial 69: 1.7M params
        max_grid_size=30,
        dropout=0.167,
    )
    results['Champion'] = test_model('Champion', model, train_loader, val_loader)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:20s}: {status}")
    
    all_passed = all(results.values())
    print(f"\n{'='*70}")
    if all_passed:
        print("🎉 ALL MODELS READY FOR TRAINING!")
    else:
        print("⚠️  SOME MODELS FAILED - CHECK ERRORS ABOVE")
    print(f"{'='*70}\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
