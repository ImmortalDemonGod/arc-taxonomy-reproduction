"""
Tests for Encoder-Decoder with Grid2D PE (Experiment 1).

Following cs336 pedagogical style and TDD principles.
"""
import pytest
import torch
from src.models.ed_with_grid2d_pe import (
    create_ed_with_grid2d_pe,
    EncoderDecoderWithGrid2DPE,
)


def test_model_creation():
    """Test model instantiation."""
    model = create_ed_with_grid2d_pe(
        vocab_size=11,
        d_model=64,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        d_ff=256,
        max_grid_size=30,
    )
    
    assert isinstance(model, EncoderDecoderWithGrid2DPE)
    assert model.d_model == 64
    assert model.vocab_size == 11


def test_forward_pass_with_grid_shapes():
    """Test forward pass with grid shape information."""
    vocab_size = 11
    model = create_ed_with_grid2d_pe(
        vocab_size=vocab_size,
        d_model=32,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        d_ff=128,
    )
    
    batch_size = 2
    # 3x3 input grid (9 tokens), 2x2 output grid (4 tokens)
    src_h, src_w = 3, 3
    tgt_h, tgt_w = 2, 2
    
    src = torch.randint(0, vocab_size, (batch_size, src_h * src_w))
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_h * tgt_w))
    
    logits = model(src, tgt, (src_h, src_w), (tgt_h, tgt_w))
    
    assert logits.shape == (batch_size, tgt_h * tgt_w, vocab_size)


def test_grid2d_pe_integration():
    """Test that Grid2D PE is properly integrated."""
    model = create_ed_with_grid2d_pe(
        vocab_size=11,
        d_model=64,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        d_ff=256,
    )
    
    # Verify Grid2D PE is used
    assert hasattr(model, 'pos_encoder')
    assert model.pos_encoder.__class__.__name__ == 'Grid2DPositionalEncoding'
    
    # Verify it has the expected Grid2D PE attributes
    assert hasattr(model.pos_encoder, 'pe')  # PE tensor


def test_different_grid_sizes():
    """Test with various grid sizes."""
    model = create_ed_with_grid2d_pe(
        vocab_size=11,
        d_model=32,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        d_ff=128,
        max_grid_size=30,
    )
    
    test_cases = [
        ((2, 2), (1, 1)),  # Small
        ((5, 5), (3, 3)),  # Medium
        ((10, 10), (8, 8)),  # Large
    ]
    
    for src_shape, tgt_shape in test_cases:
        src = torch.randint(0, 11, (1, src_shape[0] * src_shape[1]))
        tgt = torch.randint(0, 11, (1, tgt_shape[0] * tgt_shape[1]))
        
        logits = model(src, tgt, src_shape, tgt_shape)
        
        expected_shape = (1, tgt_shape[0] * tgt_shape[1], 11)
        assert logits.shape == expected_shape, \
            f"For shapes {src_shape}, {tgt_shape}: expected {expected_shape}, got {logits.shape}"


def test_gradient_flow():
    """Test that gradients flow through model with Grid2D PE."""
    model = create_ed_with_grid2d_pe(
        vocab_size=11,
        d_model=32,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        d_ff=128,
    )
    
    src = torch.randint(0, 11, (2, 9))  # 3x3 grids
    tgt = torch.randint(0, 11, (2, 4))  # 2x2 grids
    
    logits = model(src, tgt, (3, 3), (2, 2))
    loss = logits.sum()
    loss.backward()
    
    # Check gradients exist
    has_grad = False
    for param in model.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    
    assert has_grad, "Gradients should flow through model"


def test_non_square_grids():
    """Test with non-square grids (H != W)."""
    model = create_ed_with_grid2d_pe(
        vocab_size=11,
        d_model=32,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        d_ff=128,
    )
    
    # Non-square input: 3x4, output: 2x5
    src_h, src_w = 3, 4
    tgt_h, tgt_w = 2, 5
    
    src = torch.randint(0, 11, (1, src_h * src_w))
    tgt = torch.randint(0, 11, (1, tgt_h * tgt_w))
    
    logits = model(src, tgt, (src_h, src_w), (tgt_h, tgt_w))
    
    assert logits.shape == (1, tgt_h * tgt_w, 11)
    assert not torch.isnan(logits).any(), "Model should handle non-square grids"


if __name__ == "__main__":
    print("Running Exp 1 (Grid2D PE) tests...")
    test_model_creation()
    print("✅ test_model_creation")
    test_forward_pass_with_grid_shapes()
    print("✅ test_forward_pass_with_grid_shapes")
    test_grid2d_pe_integration()
    print("✅ test_grid2d_pe_integration")
    test_different_grid_sizes()
    print("✅ test_different_grid_sizes")
    test_gradient_flow()
    print("✅ test_gradient_flow")
    test_non_square_grids()
    print("✅ test_non_square_grids")
    print("\n✅ All Exp 1 (Grid2D PE) tests passed!")
