"""
Unit tests for Grid2DPositionalEncoding.

Tests verify:
1. Correct output shapes
2. Positional encoding properties
3. Edge cases and error handling
"""
import pytest
import torch
from src.positional_encoding import Grid2DPositionalEncoding


class TestGrid2DPositionalEncoding:
    """Test suite for Grid2DPositionalEncoding module."""
    
    def test_initialization(self):
        """Test that module initializes correctly with valid parameters."""
        pe = Grid2DPositionalEncoding(d_model=256, max_height=30, max_width=30)
        assert pe.d_model == 256
        assert pe.pe.shape == (1, 900, 256)  # 1 batch, 30*30 positions, 256 dims
    
    def test_invalid_d_model_odd(self):
        """Test that odd d_model raises ValueError."""
        with pytest.raises(ValueError, match="d_model must be even"):
            Grid2DPositionalEncoding(d_model=257)
    
    def test_invalid_d_model_negative(self):
        """Test that negative d_model raises ValueError."""
        with pytest.raises(ValueError, match="d_model must be positive"):
            Grid2DPositionalEncoding(d_model=-1)
    
    def test_forward_shape(self):
        """Test that forward pass produces correct output shape."""
        pe = Grid2DPositionalEncoding(d_model=128, max_height=30, max_width=30)
        batch_size = 4
        seq_len = 900  # 30 × 30
        d_model = 128
        
        x = torch.randn(batch_size, seq_len, d_model)
        output = pe(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_forward_smaller_sequence(self):
        """Test forward with sequence length smaller than max."""
        pe = Grid2DPositionalEncoding(d_model=256, max_height=30, max_width=30)
        batch_size = 2
        seq_len = 400  # Smaller than 30*30 = 900
        d_model = 256
        
        x = torch.randn(batch_size, seq_len, d_model)
        output = pe(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_positional_encoding_added(self):
        """Test that positional encoding is actually added to input."""
        pe = Grid2DPositionalEncoding(d_model=128, max_height=10, max_width=10)
        
        # Create zero input
        x = torch.zeros(1, 100, 128)
        output = pe(x)
        
        # Output should equal the positional encoding (since input was zeros)
        assert torch.allclose(output, pe.pe[:, :100])
        
        # Positional encoding should be non-zero
        assert not torch.allclose(output, torch.zeros_like(output))
    
    def test_batch_independence(self):
        """Test that positional encoding is the same across batch dimension."""
        pe = Grid2DPositionalEncoding(d_model=128, max_height=10, max_width=10)
        
        batch_size = 4
        x = torch.randn(batch_size, 100, 128)
        output = pe(x)
        
        # Positional encoding should be broadcast, so difference between
        # batches should only come from input
        diff_input = x[0] - x[1]
        diff_output = output[0] - output[1]
        
        # Differences should be the same (positional encoding is the same for all batches)
        assert torch.allclose(diff_input, diff_output, rtol=1e-4, atol=1e-6)
    
    def test_sinusoidal_properties(self):
        """Test that encoding has expected sinusoidal properties."""
        pe = Grid2DPositionalEncoding(d_model=256, max_height=10, max_width=10)
        
        # Get positional encoding for a zero input
        x = torch.zeros(1, 100, 256)
        output = pe(x)
        
        # Check that values are bounded (sinusoidal functions are bounded by [-1, 1])
        assert output.abs().max() <= 2.0  # Allow some margin for initialization
    
    def test_different_positions_different_encodings(self):
        """Test that different positions have different encodings."""
        pe = Grid2DPositionalEncoding(d_model=128, max_height=10, max_width=10)
        
        # Position 0 and position 1 should have different encodings
        enc_0 = pe.pe[:, 0, :]
        enc_1 = pe.pe[:, 1, :]
        
        assert not torch.allclose(enc_0, enc_1)
    
    def test_gradient_flow(self):
        """Test that gradients flow through the module."""
        pe = Grid2DPositionalEncoding(d_model=128, max_height=10, max_width=10)
        
        x = torch.randn(2, 100, 128, requires_grad=True)
        output = pe(x)
        loss = output.sum()
        loss.backward()
        
        # Input should have gradients
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))
    
    def test_deterministic(self):
        """Test that encoding is deterministic (same input → same output)."""
        pe = Grid2DPositionalEncoding(d_model=128, max_height=10, max_width=10)
        
        x = torch.randn(2, 100, 128)
        output1 = pe(x)
        output2 = pe(x)
        
        assert torch.allclose(output1, output2)
    
    def test_small_grids(self):
        """Test with smaller grid sizes."""
        pe = Grid2DPositionalEncoding(d_model=64, max_height=5, max_width=5)
        
        x = torch.randn(1, 25, 64)  # 5×5 grid
        output = pe(x)
        
        assert output.shape == (1, 25, 64)
    
    def test_various_d_models(self):
        """Test with various even d_model values."""
        for d_model in [64, 128, 256, 512]:
            pe = Grid2DPositionalEncoding(d_model=d_model, max_height=10, max_width=10)
            x = torch.randn(1, 100, d_model)
            output = pe(x)
            assert output.shape == (1, 100, d_model)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
