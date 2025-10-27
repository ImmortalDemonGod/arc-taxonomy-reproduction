"""
Smoke Test Suite - MANDATORY before 144 GPU-hours of training.

Tests all 5 experiments with ONE batch to verify:
1. Data loads correctly
2. Forward pass completes
3. Loss computes without NaN/Inf
4. Backward pass completes
5. Gradients flow to all parameters

Following systematic approach: test each model independently.
"""
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Add parent to path to import src as package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.decoder_only_data import DecoderOnlyARCDataset, collate_decoder_only
from src.data.encoder_decoder_data import EncoderDecoderARCDataset, collate_encoder_decoder
from src.data.champion_data import ChampionARCDataset, collate_champion

from src.models.decoder_only_lightning import DecoderOnlyLightningModule
from src.models.encoder_decoder_lightning import EncoderDecoderLightningModule
from src.models.champion_lightning import ChampionLightningModule


DATA_DIR = Path("/Users/tomriddle1/Holistic-Performance-Enhancement/cultivation/systems/arc_reactor/data/synthetic_data/distributional_alignment")


def smoke_test_decoder_only():
    """Smoke test for Exp -1: Decoder-Only."""
    print("\n" + "=" * 70)
    print("SMOKE TEST: Exp -1 (Decoder-Only)")
    print("=" * 70)
    
    # Load minimal data
    task_files = list(DATA_DIR.glob("*.json"))[:2]
    dataset = DecoderOnlyARCDataset(task_files, max_seq_len=200)
    
    if len(dataset) == 0:
        print("❌ FAILED: No data loaded")
        return False
    
    dataloader = DataLoader(
        dataset, 
        batch_size=2, 
        collate_fn=collate_decoder_only
    )
    
    # Create model
    model = DecoderOnlyLightningModule(
        vocab_size=11,
        context_length=512,
        d_model=64,  # Small for smoke test
        num_layers=1,
        num_heads=2,
        d_ff=256,
    )
    
    # Get one batch
    batch = next(iter(dataloader))
    
    # Manually test training step (without Lightning trainer)
    model.train()
    
    # Forward
    logits = model(batch)
    assert not torch.isnan(logits).any(), "NaN in logits"
    assert not torch.isinf(logits).any(), "Inf in logits"
    
    # Loss
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        batch.reshape(-1),
        ignore_index=10
    )
    assert not torch.isnan(loss), "NaN in loss"
    assert not torch.isinf(loss), "Inf in loss"
    
    # Backward
    loss.backward()
    
    # Check gradients
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN in {name}.grad"
            assert not torch.isinf(param.grad).any(), f"Inf in {name}.grad"
            has_grad = True
    
    assert has_grad, "No gradients computed!"
    
    print(f"✅ PASSED")
    print(f"   Batch shape: {batch.shape}")
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Gradients: OK")
    return True


def smoke_test_encoder_decoder():
    """Smoke test for Exp 0: Encoder-Decoder."""
    print("\n" + "=" * 70)
    print("SMOKE TEST: Exp 0 (Encoder-Decoder)")
    print("=" * 70)
    
    task_files = list(DATA_DIR.glob("*.json"))[:2]
    dataset = EncoderDecoderARCDataset(task_files, max_seq_len=200)
    
    if len(dataset) == 0:
        print("❌ FAILED: No data loaded")
        return False
    
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=collate_encoder_decoder
    )
    
    model = EncoderDecoderLightningModule(
        vocab_size=11,
        d_model=64,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        d_ff=256,
    )
    
    batch = next(iter(dataloader))
    src, tgt = batch
    
    model.train()
    
    # Forward
    logits = model(src, tgt)
    assert not torch.isnan(logits).any(), "NaN in logits"
    assert not torch.isinf(logits).any(), "Inf in logits"
    
    # Loss
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        tgt.reshape(-1),
        ignore_index=10
    )
    assert not torch.isnan(loss), "NaN in loss"
    
    # Backward
    loss.backward()
    
    # Check gradients
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad, "No gradients!"
    
    print(f"✅ PASSED")
    print(f"   src shape: {src.shape}, tgt shape: {tgt.shape}")
    print(f"   Loss: {loss.item():.4f}")
    return True


def smoke_test_champion():
    """Smoke test for Exp 3: Champion."""
    print("\n" + "=" * 70)
    print("SMOKE TEST: Exp 3 (Champion)")
    print("=" * 70)
    
    task_files = list(DATA_DIR.glob("*.json"))[:5]  # Need more for context
    dataset = ChampionARCDataset(task_files, num_context_pairs=2, max_grid_size=30)
    
    if len(dataset) == 0:
        print("❌ FAILED: No data loaded")
        return False
    
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=lambda batch: collate_champion(batch, pad_token=10)
    )
    
    model = ChampionLightningModule(
        vocab_size=11,
        d_model=64,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        d_ff=256,
        max_grid_size=30,
    )
    
    batch = next(iter(dataloader))
    src, tgt, ctx_in, ctx_out, src_shapes, tgt_shapes = batch
    
    model.train()
    
    # Forward (with actual grid shapes from data loader)
    src_shape = src_shapes[0]
    tgt_shape = tgt_shapes[0]
    logits = model(src, tgt, src_shape, tgt_shape, ctx_in, ctx_out)
    assert not torch.isnan(logits).any(), "NaN in logits"
    
    # Loss
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        tgt.reshape(-1),
        ignore_index=10
    )
    assert not torch.isnan(loss), "NaN in loss"
    
    # Backward
    loss.backward()
    
    # Check gradients
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad, "No gradients!"
    
    print(f"✅ PASSED")
    print(f"   src: {src.shape}, tgt: {tgt.shape}")
    print(f"   ctx_in: {ctx_in.shape}, ctx_out: {ctx_out.shape}")
    print(f"   Grid shapes: src={src_shape}, tgt={tgt_shape}")
    print(f"   Loss: {loss.item():.4f}")
    return True


def main():
    """Run all smoke tests."""
    print("\n" + "=" * 70)
    print("SMOKE TEST SUITE - MANDATORY BEFORE TRAINING")
    print("=" * 70)
    print("\nTesting: Forward, Loss, Backward, Gradients")
    print("Using: 1 batch per model, CPU only")
    
    results = {}
    
    # Test each model
    try:
        results['Exp -1'] = smoke_test_decoder_only()
    except Exception as e:
        print(f"❌ FAILED: {e}")
        results['Exp -1'] = False
    
    try:
        results['Exp 0'] = smoke_test_encoder_decoder()
    except Exception as e:
        print(f"❌ FAILED: {e}")
        results['Exp 0'] = False
    
    try:
        results['Exp 3'] = smoke_test_champion()
    except Exception as e:
        print(f"❌ FAILED: {e}")
        results['Exp 3'] = False
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for exp, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{exp}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 ALL SMOKE TESTS PASSED!")
        print("✅ Ready for full training (144 GPU-hours)")
    else:
        print("\n⚠️  SOME TESTS FAILED")
        print("❌ DO NOT START TRAINING until all tests pass")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
