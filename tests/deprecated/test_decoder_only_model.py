"""
Tests for Decoder-Only baseline model (Exp -1).

Following cs336 pedagogical style and TDD principles.
"""
import pytest
import torch
from src.models.decoder_only_baseline import (
    flatten_grid_to_sequence,
    create_decoder_only_model,
    compute_decoder_only_loss,
)


def test_flatten_grid_to_sequence_basic():
    """Test basic grid flattening to sequence format."""
    # 3x3 input grid, 2x2 output grid
    input_grid = torch.tensor([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    output_grid = torch.tensor([
        [1, 2],
        [3, 4]
    ])
    sep_token = 10
    
    sequence = flatten_grid_to_sequence(input_grid, output_grid, sep_token)
    
    # Expected: [1,2,3,4,5,6,7,8,9, SEP, 1,2,3,4]
    expected_length = 9 + 1 + 4  # input + sep + output
    assert sequence.shape == (expected_length,), f"Expected length {expected_length}, got {sequence.shape}"
    assert sequence[9].item() == sep_token, "SEP token should be at position 9"
    assert torch.equal(sequence[:9], input_grid.flatten()), "Input portion should match flattened input"
    assert torch.equal(sequence[10:], output_grid.flatten()), "Output portion should match flattened output"


def test_flatten_grid_to_sequence_batch():
    """Test batch processing of grids."""
    # Batch of 2 grids
    batch_size = 2
    input_grids = torch.randint(0, 10, (batch_size, 3, 3))
    output_grids = torch.randint(0, 10, (batch_size, 2, 2))
    sep_token = 10
    
    sequences = torch.stack([
        flatten_grid_to_sequence(input_grids[i], output_grids[i], sep_token)
        for i in range(batch_size)
    ])
    
    assert sequences.shape == (batch_size, 14), "Batch shape should be preserved"
    assert (sequences[:, 9] == sep_token).all(), "SEP tokens should be at position 9 for all"


def test_create_decoder_only_model():
    """Test model instantiation with correct parameters."""
    vocab_size = 11  # 0-9 colors + PAD
    context_length = 100  # Max sequence length
    d_model = 128
    num_layers = 2
    num_heads = 4
    d_ff = 512
    
    model = create_decoder_only_model(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
    )
    
    # Model should accept sequences and produce logits
    batch_size = 2
    seq_len = 20
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    logits = model(input_ids)
    
    assert logits.shape == (batch_size, seq_len, vocab_size), \
        f"Expected shape ({batch_size}, {seq_len}, {vocab_size}), got {logits.shape}"
    

def test_compute_decoder_only_loss():
    """Test loss computation with masking of input tokens."""
    batch_size = 2
    seq_len = 14  # 9 input + 1 sep + 4 output
    vocab_size = 11
    
    # Simulate logits and targets
    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Input length is 9, SEP at position 9, output starts at position 10
    input_length = 9
    
    loss = compute_decoder_only_loss(logits, targets, input_length, ignore_index=-100)
    
    # Loss should be a scalar
    assert loss.ndim == 0, "Loss should be a scalar"
    assert loss.item() >= 0, "Loss should be non-negative"
    
    # Test with all-correct predictions (loss should be near 0)
    correct_logits = torch.zeros(batch_size, seq_len, vocab_size)
    for b in range(batch_size):
        for t in range(seq_len):
            correct_logits[b, t, targets[b, t]] = 100.0  # High logit for correct class
    
    loss_correct = compute_decoder_only_loss(correct_logits, targets, input_length, ignore_index=-100)
    assert loss_correct < 0.1, f"Loss with correct predictions should be near 0, got {loss_correct.item()}"


def test_loss_ignores_input_tokens():
    """Verify that loss computation ignores input tokens."""
    batch_size = 1
    seq_len = 10  # 5 input + 1 sep + 4 output
    vocab_size = 11
    input_length = 5
    
    # Create logits and targets
    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Compute loss
    loss = compute_decoder_only_loss(logits, targets, input_length, ignore_index=-100)
    
    # Manually verify: only positions 6-9 (output tokens) should contribute
    # Positions 0-5 (input + SEP) should be masked
    from torch.nn import functional as F
    
    # Manual loss calculation for output tokens only
    output_logits = logits[:, input_length+1:, :]  # Skip input + SEP
    output_targets = targets[:, input_length+1:]
    manual_loss = F.cross_entropy(
        output_logits.reshape(-1, vocab_size),
        output_targets.reshape(-1)
    )
    
    # Should match within floating point tolerance
    assert torch.allclose(loss, manual_loss, atol=1e-5), \
        f"Loss mismatch: computed={loss.item()}, manual={manual_loss.item()}"


def test_variable_grid_sizes():
    """Test handling of variable-sized grids."""
    # Different sized grids
    test_cases = [
        ((2, 2), (1, 1)),  # Small
        ((5, 5), (3, 3)),  # Medium
        ((10, 10), (5, 5)), # Large
    ]
    sep_token = 10
    
    for input_shape, output_shape in test_cases:
        input_grid = torch.randint(0, 10, input_shape)
        output_grid = torch.randint(0, 10, output_shape)
        
        sequence = flatten_grid_to_sequence(input_grid, output_grid, sep_token)
        
        expected_len = input_shape[0] * input_shape[1] + 1 + output_shape[0] * output_shape[1]
        assert sequence.shape[0] == expected_len, \
            f"For shapes {input_shape}, {output_shape}: expected {expected_len}, got {sequence.shape[0]}"


if __name__ == "__main__":
    # Run tests manually for quick validation
    print("Running decoder-only model tests...")
    test_flatten_grid_to_sequence_basic()
    print("✅ test_flatten_grid_to_sequence_basic")
    test_flatten_grid_to_sequence_batch()
    print("✅ test_flatten_grid_to_sequence_batch")
    test_variable_grid_sizes()
    print("✅ test_variable_grid_sizes")
    print("\nNote: Model tests require implementation of decoder_only_baseline.py")
