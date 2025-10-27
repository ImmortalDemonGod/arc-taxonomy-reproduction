"""
Tests for E-D + Grid2D PE + PermInvariant (Experiment 2).

Following cs336 pedagogical style and TDD principles.
"""
import pytest
import torch
from src.models.ed_with_grid2d_pe_and_perminv import (
    create_ed_with_grid2d_pe_and_perminv,
    EDWithGrid2DPEAndPermInv,
)


def test_model_creation():
    """Test model instantiation with PermInvariant."""
    model = create_ed_with_grid2d_pe_and_perminv(
        vocab_size=11,
        d_model=64,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        d_ff=256,
    )
    
    assert isinstance(model, EDWithGrid2DPEAndPermInv)
    assert model.d_model == 64
    assert model.vocab_size == 11


def test_perminvariant_integration():
    """Test that PermInvariant embedding is properly integrated."""
    model = create_ed_with_grid2d_pe_and_perminv(
        vocab_size=11,
        d_model=64,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        d_ff=256,
    )
    
    # Verify PermInvariant is used
    assert hasattr(model, 'embedding')
    assert model.embedding.__class__.__name__ == 'PermInvariantEmbedding'
    
    # Verify it has the G matrix
    assert hasattr(model.embedding, 'G')
    assert model.embedding.G.shape == (11, 64)


def test_forward_pass():
    """Test forward pass with PermInvariant embedding."""
    vocab_size = 11
    model = create_ed_with_grid2d_pe_and_perminv(
        vocab_size=vocab_size,
        d_model=32,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        d_ff=128,
    )
    
    batch_size = 2
    src = torch.randint(0, vocab_size, (batch_size, 9))  # 3x3
    tgt = torch.randint(0, vocab_size, (batch_size, 4))  # 2x2
    
    logits = model(src, tgt, (3, 3), (2, 2))
    
    assert logits.shape == (batch_size, 4, vocab_size)


def test_color_permutation_property():
    """Test that embedding preserves color permutation structure."""
    model = create_ed_with_grid2d_pe_and_perminv(
        vocab_size=11,
        d_model=32,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        d_ff=128,
    )
    
    # Create a simple color sequence
    colors = torch.tensor([[1, 2, 3, 4]])
    
    # Get embeddings
    emb1 = model.embedding(colors)
    
    # Apply permutation (swap colors 1 and 2)
    colors_perm = torch.tensor([[2, 1, 3, 4]])
    emb2 = model.embedding(colors_perm)
    
    # Embeddings should be different (we swapped colors)
    assert not torch.allclose(emb1, emb2)
    
    # But the embedding structure should be consistent
    assert emb1.shape == emb2.shape


def test_gradient_flow():
    """Test that gradients flow through PermInvariant embedding."""
    model = create_ed_with_grid2d_pe_and_perminv(
        vocab_size=11,
        d_model=32,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        d_ff=128,
    )
    
    src = torch.randint(0, 11, (2, 9))
    tgt = torch.randint(0, 11, (2, 4))
    
    logits = model(src, tgt, (3, 3), (2, 2))
    loss = logits.sum()
    loss.backward()
    
    # Check gradients flow to PermInvariant G matrix
    assert model.embedding.G.grad is not None
    assert model.embedding.G.grad.abs().sum() > 0, "Gradients should flow to PermInvariant"


def test_different_grid_sizes():
    """Test with various grid sizes."""
    model = create_ed_with_grid2d_pe_and_perminv(
        vocab_size=11,
        d_model=32,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        d_ff=128,
    )
    
    test_cases = [
        ((2, 2), (1, 1)),
        ((5, 5), (3, 3)),
        ((10, 10), (8, 8)),
    ]
    
    for src_shape, tgt_shape in test_cases:
        src = torch.randint(0, 11, (1, src_shape[0] * src_shape[1]))
        tgt = torch.randint(0, 11, (1, tgt_shape[0] * tgt_shape[1]))
        
        logits = model(src, tgt, src_shape, tgt_shape)
        
        expected_shape = (1, tgt_shape[0] * tgt_shape[1], 11)
        assert logits.shape == expected_shape


if __name__ == "__main__":
    print("Running Exp 2 (Grid2D PE + PermInvariant) tests...")
    test_model_creation()
    print("✅ test_model_creation")
    test_perminvariant_integration()
    print("✅ test_perminvariant_integration")
    test_forward_pass()
    print("✅ test_forward_pass")
    test_color_permutation_property()
    print("✅ test_color_permutation_property")
    test_gradient_flow()
    print("✅ test_gradient_flow")
    test_different_grid_sizes()
    print("✅ test_different_grid_sizes")
    print("\n✅ All Exp 2 tests passed!")
