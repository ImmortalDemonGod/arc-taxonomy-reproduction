"""
Tests for Generic Encoder-Decoder baseline model (Exp 0).

Following cs336 pedagogical style and TDD principles.
"""
import pytest
import torch
from src.models.encoder_decoder_baseline import (
    create_encoder_decoder_model,
    prepare_grid_batch,
)


def test_model_creation():
    """Test model instantiation with correct parameters."""
    vocab_size = 11
    d_model = 128
    num_encoder_layers = 2
    num_decoder_layers = 2
    num_heads = 4
    d_ff = 512
    
    model = create_encoder_decoder_model(
        vocab_size=vocab_size,
        d_model=d_model,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        num_heads=num_heads,
        d_ff=d_ff,
    )
    
    # Test forward pass
    batch_size = 2
    src_len = 20
    tgt_len = 15
    
    src = torch.randint(0, vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_len))
    
    logits = model(src, tgt)
    
    assert logits.shape == (batch_size, tgt_len, vocab_size), \
        f"Expected shape ({batch_size}, {tgt_len}, {vocab_size}), got {logits.shape}"


def test_prepare_grid_batch_flatten():
    """Test grid flattening for encoder-decoder format."""
    # 3x3 input grid, 2x2 output grid
    input_grids = [
        torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    ]
    output_grids = [
        torch.tensor([[9, 8], [7, 6]]),
        torch.tensor([[5, 4], [3, 2]])
    ]
    
    src, tgt = prepare_grid_batch(input_grids, output_grids, pad_token=10)
    
    # Check shapes
    assert src.shape[0] == 2, "Batch size should be 2"
    assert tgt.shape[0] == 2, "Batch size should be 2"
    
    # Check content - first grid
    assert src[0, 0].item() == 1, "First token should match input grid"
    assert tgt[0, 0].item() == 9, "First target token should match output grid"


def test_encoder_decoder_separate_sequences():
    """Test that encoder and decoder handle separate sequences correctly."""
    vocab_size = 11
    d_model = 64
    
    model = create_encoder_decoder_model(
        vocab_size=vocab_size,
        d_model=d_model,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        d_ff=256,
    )
    
    batch_size = 1
    src_len = 10
    tgt_len = 8
    
    # Different length sequences
    src = torch.randint(0, vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_len))
    
    logits = model(src, tgt)
    
    # Output length should match target length, not source
    assert logits.shape[1] == tgt_len, \
        f"Output length should match target ({tgt_len}), got {logits.shape[1]}"


def test_model_with_padding():
    """Test model handles padding correctly."""
    vocab_size = 11
    pad_token = 10
    
    model = create_encoder_decoder_model(
        vocab_size=vocab_size,
        d_model=64,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        d_ff=256,
    )
    
    # Create batch with padding
    src = torch.tensor([
        [1, 2, 3, 4, 5, pad_token, pad_token],  # Shorter sequence
        [1, 2, 3, 4, 5, 6, 7]  # Full sequence
    ])
    tgt = torch.tensor([
        [5, 6, 7, pad_token],
        [8, 9, 1, 2]
    ])
    
    logits = model(src, tgt)
    
    assert logits.shape == (2, 4, vocab_size)
    assert not torch.isnan(logits).any(), "Model should handle padding without NaN"


def test_gradient_flow():
    """Test that gradients flow through the model."""
    vocab_size = 11
    model = create_encoder_decoder_model(
        vocab_size=vocab_size,
        d_model=32,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        d_ff=128,
    )
    
    src = torch.randint(0, vocab_size, (2, 10))
    tgt = torch.randint(0, vocab_size, (2, 8))
    
    logits = model(src, tgt)
    loss = logits.sum()
    loss.backward()
    
    # Check that gradients exist
    has_grad = False
    for param in model.parameters():
        if param.grad is not None:
            has_grad = True
            break
    
    assert has_grad, "Gradients should flow through model"


if __name__ == "__main__":
    print("Running encoder-decoder baseline tests...")
    test_model_creation()
    print("✅ test_model_creation")
    test_prepare_grid_batch_flatten()
    print("✅ test_prepare_grid_batch_flatten")
    test_encoder_decoder_separate_sequences()
    print("✅ test_encoder_decoder_separate_sequences")
    test_model_with_padding()
    print("✅ test_model_with_padding")
    test_gradient_flow()
    print("✅ test_gradient_flow")
    print("\n✅ All encoder-decoder baseline tests passed!")
