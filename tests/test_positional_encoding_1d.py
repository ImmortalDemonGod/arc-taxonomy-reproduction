"""
Tests for 1D Sinusoidal Positional Encoding.

Following cs336 pedagogical style and TDD principles.
"""
import pytest
import torch
import math
from src.positional_encoding_1d import (
    PositionalEncoding1D,
    create_1d_positional_encoding,
    get_sinusoidal_embeddings,
)


def test_pe_shape():
    """Test that PE preserves input shape."""
    d_model = 128
    batch_size = 4
    seq_len = 10
    
    pe = PositionalEncoding1D(d_model)
    x = torch.randn(batch_size, seq_len, d_model)
    
    output = pe(x)
    
    assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"


def test_pe_values_first_position():
    """Test PE values at position 0."""
    d_model = 4
    pe = PositionalEncoding1D(d_model, dropout=0.0)  # No dropout for testing
    
    # Create input at position 0
    x = torch.zeros(1, 1, d_model)
    output = pe(x)
    
    # At position 0, sin(0) = 0 and cos(0) = 1
    # PE(0, 2i) = sin(0) = 0
    # PE(0, 2i+1) = cos(0) = 1
    expected = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
    
    assert torch.allclose(output[0, 0, :], expected[0], atol=1e-5), \
        f"Expected {expected}, got {output[0, 0, :]}"


def test_pe_different_positions():
    """Test that different positions get different encodings."""
    d_model = 128
    pe = PositionalEncoding1D(d_model, dropout=0.0)
    
    # Create input for positions 0, 1, 2
    x = torch.zeros(1, 3, d_model)
    output = pe(x)
    
    # Encodings should be different for each position
    assert not torch.allclose(output[0, 0, :], output[0, 1, :]), \
        "Position 0 and 1 should have different encodings"
    assert not torch.allclose(output[0, 1, :], output[0, 2, :]), \
        "Position 1 and 2 should have different encodings"


def test_pe_batch_independence():
    """Test that PE is applied independently to each batch element."""
    d_model = 64
    batch_size = 3
    seq_len = 5
    
    pe = PositionalEncoding1D(d_model, dropout=0.0)
    
    # Create different inputs for each batch element
    x = torch.randn(batch_size, seq_len, d_model)
    output = pe(x)
    
    # The positional encoding added should be the same for all batch elements
    # (only the base x differs)
    # So output[i] - x[i] should be the same for all i
    pe_added_0 = output[0] - x[0]
    pe_added_1 = output[1] - x[1]
    
    assert torch.allclose(pe_added_0, pe_added_1, atol=1e-5), \
        "PE should be applied identically across batch"


def test_pe_max_len_constraint():
    """Test that PE can handle sequences up to max_len."""
    d_model = 64
    max_len = 100
    
    pe = PositionalEncoding1D(d_model, max_len=max_len)
    
    # Should work for sequence of length max_len
    x = torch.zeros(1, max_len, d_model)
    output = pe(x)
    assert output.shape == x.shape
    
    # Should also work for shorter sequences
    x_short = torch.zeros(1, 50, d_model)
    output_short = pe(x_short)
    assert output_short.shape == x_short.shape


def test_pe_odd_d_model():
    """Test PE with odd d_model."""
    d_model = 127  # Odd number
    pe = PositionalEncoding1D(d_model, dropout=0.0)
    
    x = torch.zeros(1, 5, d_model)
    output = pe(x)
    
    assert output.shape == x.shape
    assert not torch.isnan(output).any(), "PE should not produce NaN values"


def test_factory_function():
    """Test the factory function."""
    d_model = 128
    pe = create_1d_positional_encoding(d_model, max_len=1000, dropout=0.1)
    
    assert isinstance(pe, PositionalEncoding1D)
    assert pe.pe.shape == (1, 1000, d_model)


def test_functional_interface():
    """Test the functional get_sinusoidal_embeddings."""
    batch_size = 2
    seq_len = 5
    d_model = 64
    
    positions = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
    embeddings = get_sinusoidal_embeddings(positions, d_model)
    
    assert embeddings.shape == (batch_size, seq_len, d_model)
    
    # Verify position 0 properties
    assert torch.allclose(embeddings[0, 0, 0::2], torch.zeros(d_model // 2), atol=1e-5), \
        "Even indices at position 0 should be sin(0) = 0"


def test_pe_deterministic():
    """Test that PE is deterministic (same input -> same output)."""
    d_model = 128
    pe = PositionalEncoding1D(d_model, dropout=0.0)
    
    x = torch.randn(2, 10, d_model)
    
    # Run twice
    output1 = pe(x.clone())
    output2 = pe(x.clone())
    
    assert torch.allclose(output1, output2), "PE should be deterministic when dropout=0"


if __name__ == "__main__":
    print("Running 1D PE tests...")
    test_pe_shape()
    print("✅ test_pe_shape")
    test_pe_values_first_position()
    print("✅ test_pe_values_first_position")
    test_pe_different_positions()
    print("✅ test_pe_different_positions")
    test_pe_batch_independence()
    print("✅ test_pe_batch_independence")
    test_pe_max_len_constraint()
    print("✅ test_pe_max_len_constraint")
    test_pe_odd_d_model()
    print("✅ test_pe_odd_d_model")
    test_factory_function()
    print("✅ test_factory_function")
    test_functional_interface()
    print("✅ test_functional_interface")
    test_pe_deterministic()
    print("✅ test_pe_deterministic")
    print("\n✅ All 1D PE tests passed!")
