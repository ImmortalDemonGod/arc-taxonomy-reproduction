"""
Unit tests for PermInvariantEmbedding.

Tests verify:
1. Correct output shapes
2. Permutation equivariance property
3. Edge cases and parameter handling
"""
import pytest
import torch
from src.embedding import PermInvariantEmbedding


class TestPermInvariantEmbedding:
    """Test suite for PermInvariantEmbedding module."""
    
    def test_initialization(self):
        """Test that module initializes correctly."""
        emb = PermInvariantEmbedding(d_model=256, vocab_size=11, pad_idx=10)
        assert emb.d_model == 256
        assert emb.vocab_size == 11
        assert emb.pad_idx == 10
        assert emb.padding_idx == 10  # Legacy attribute
        assert emb.G.shape == (11, 256)
    
    def test_forward_1d(self):
        """Test forward pass with 1D input."""
        emb = PermInvariantEmbedding(d_model=128, vocab_size=11)
        x = torch.randint(0, 11, (100,))
        output = emb(x)
        assert output.shape == (100, 128)
    
    def test_forward_2d(self):
        """Test forward pass with 2D input (batch of sequences)."""
        emb = PermInvariantEmbedding(d_model=256, vocab_size=11)
        batch_size = 4
        seq_len = 900
        x = torch.randint(0, 11, (batch_size, seq_len))
        output = emb(x)
        assert output.shape == (batch_size, seq_len, 256)
    
    def test_forward_3d(self):
        """Test forward pass with 3D input (batch, height, width)."""
        emb = PermInvariantEmbedding(d_model=128, vocab_size=11)
        batch_size = 2
        height = 30
        width = 30
        x = torch.randint(0, 11, (batch_size, height, width))
        output = emb(x)
        assert output.shape == (batch_size, height, width, 128)
    
    def test_permutation_equivariance(self):
        """Test that embedding is equivariant to color permutations."""
        emb = PermInvariantEmbedding(d_model=64, vocab_size=11)
        
        # Original input
        x = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        emb_x = emb(x)
        
        # Permuted input (swap colors 0 and 1)
        x_perm = torch.tensor([1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        emb_x_perm = emb(x_perm)
        
        # Check that embeddings are swapped accordingly
        assert torch.allclose(emb_x[0], emb_x_perm[1])
        assert torch.allclose(emb_x[1], emb_x_perm[0])
        assert torch.allclose(emb_x[2], emb_x_perm[2])  # Unchanged colors
    
    def test_same_color_same_embedding(self):
        """Test that same color always gets same embedding."""
        emb = PermInvariantEmbedding(d_model=128, vocab_size=11)
        
        x = torch.tensor([5, 5, 5, 5])
        output = emb(x)
        
        # All positions should have identical embeddings
        for i in range(3):
            assert torch.allclose(output[i], output[i+1])
    
    def test_different_colors_different_embeddings(self):
        """Test that different colors have different embeddings."""
        emb = PermInvariantEmbedding(d_model=128, vocab_size=11)
        
        x = torch.tensor([0, 1, 2])
        output = emb(x)
        
        # Different colors should have different embeddings
        assert not torch.allclose(output[0], output[1])
        assert not torch.allclose(output[1], output[2])
        assert not torch.allclose(output[0], output[2])
    
    def test_pad_token_embedding(self):
        """Test that pad token has its own embedding."""
        emb = PermInvariantEmbedding(d_model=64, vocab_size=11, pad_idx=10)
        
        x = torch.tensor([0, 10])  # Regular color and pad
        output = emb(x)
        
        # Pad should have different embedding from color 0
        assert not torch.allclose(output[0], output[1])
    
    def test_gradient_flow(self):
        """Test that gradients flow through embedding."""
        emb = PermInvariantEmbedding(d_model=128, vocab_size=11)
        
        x = torch.randint(0, 11, (10,))
        output = emb(x)
        loss = output.sum()
        loss.backward()
        
        # Weight matrix should have gradients
        assert emb.G.grad is not None
        assert not torch.allclose(emb.G.grad, torch.zeros_like(emb.G.grad))
    
    def test_weight_sharing(self):
        """Test that all color indices use the same weight matrix."""
        emb = PermInvariantEmbedding(d_model=64, vocab_size=11)
        
        # Get embeddings for different colors
        x0 = torch.tensor([0])
        x1 = torch.tensor([1])
        
        emb0 = emb(x0)
        emb1 = emb(x1)
        
        # Both should be rows from the same weight matrix
        assert torch.allclose(emb0[0], emb.G[0])
        assert torch.allclose(emb1[0], emb.G[1])
    
    def test_batch_processing(self):
        """Test that batched inputs are processed correctly."""
        emb = PermInvariantEmbedding(d_model=128, vocab_size=11)
        
        # Single sample
        x_single = torch.randint(0, 11, (100,))
        out_single = emb(x_single)
        
        # Batched samples
        x_batch = x_single.unsqueeze(0).repeat(4, 1)
        out_batch = emb(x_batch)
        
        # Each batch element should be identical
        for i in range(4):
            assert torch.allclose(out_batch[i], out_single)
    
    def test_deterministic(self):
        """Test that embedding is deterministic."""
        emb = PermInvariantEmbedding(d_model=128, vocab_size=11)
        
        x = torch.randint(0, 11, (50,))
        out1 = emb(x)
        out2 = emb(x)
        
        assert torch.allclose(out1, out2)
    
    def test_parameter_count(self):
        """Test that parameter count is vocab_size × d_model."""
        d_model = 256
        vocab_size = 11
        emb = PermInvariantEmbedding(d_model=d_model, vocab_size=vocab_size)
        
        param_count = sum(p.numel() for p in emb.parameters())
        expected_count = vocab_size * d_model
        
        assert param_count == expected_count
    
    def test_various_vocab_sizes(self):
        """Test with different vocab sizes."""
        d_model = 128
        for vocab_size in [5, 11, 20, 100]:
            emb = PermInvariantEmbedding(d_model=d_model, vocab_size=vocab_size)
            x = torch.randint(0, vocab_size, (50,))
            output = emb(x)
            assert output.shape == (50, d_model)
            assert emb.G.shape == (vocab_size, d_model)
    
    def test_various_d_models(self):
        """Test with different d_model values."""
        vocab_size = 11
        for d_model in [64, 128, 256, 512]:
            emb = PermInvariantEmbedding(d_model=d_model, vocab_size=vocab_size)
            x = torch.randint(0, vocab_size, (50,))
            output = emb(x)
            assert output.shape == (50, d_model)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
