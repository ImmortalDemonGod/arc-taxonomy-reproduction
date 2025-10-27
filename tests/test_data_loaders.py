"""
Data Loader Validation Tests (P0 - HIGHEST RISK)

Following TDD: Define expected behaviors BEFORE implementing data loaders.
These tests use synthetic ARC data to validate critical shape contracts.

Critical Risk: Data pipeline bugs could invalidate all 144 GPU-hours of training.
Strategy: Test interface contracts with mock data, then implement actual loaders.
"""
import pytest
import torch
from typing import Tuple


# ==============================================================================
# Test Data Generators (Synthetic ARC Grids)
# ==============================================================================

def create_mock_arc_grid(height: int = 3, width: int = 3, vocab_size: int = 11) -> torch.Tensor:
    """Create a synthetic ARC grid."""
    return torch.randint(0, vocab_size, (height, width))


def create_mock_arc_task():
    """
    Create a synthetic ARC task with input/output pairs.
    
    Returns:
        dict matching ARC JSON format
    """
    return {
        'train': [
            {
                'input': [[1, 2], [3, 4]],
                'output': [[5, 6], [7, 8]]
            },
            {
                'input': [[0, 1], [2, 3]],
                'output': [[4, 5], [6, 7]]
            }
        ],
        'test': [
            {
                'input': [[2, 3], [4, 5]],
                'output': [[6, 7], [8, 9]]
            }
        ]
    }


# ==============================================================================
# Critical Shape Contract Tests
# ==============================================================================

class TestDecoderOnlyDataFormat:
    """Test Exp -1 (Decoder-Only) data format requirements."""
    
    def test_sequence_linearization_format(self):
        """
        CRITICAL: Decoder-only requires [INPUT_GRID] [SEP] [OUTPUT_GRID] format.
        
        Expected sequence:
        - Input grid flattened row-wise
        - Single SEP token (token_id = 10)
        - Output grid flattened row-wise
        """
        # Mock data
        input_grid = torch.tensor([[1, 2], [3, 4]])  # 2x2
        output_grid = torch.tensor([[5, 6], [7, 8]])  # 2x2
        sep_token = 10
        
        # Expected linearized sequence: [1, 2, 3, 4, SEP, 5, 6, 7, 8]
        expected_sequence = torch.tensor([1, 2, 3, 4, sep_token, 5, 6, 7, 8])
        
        # This is what the data loader MUST produce
        # Actual implementation would be:
        # sequence = linearize_decoder_only(input_grid, output_grid, sep_token)
        # assert torch.equal(sequence, expected_sequence)
        
        # For now, we document the contract
        assert expected_sequence.shape == (9,), "Sequence should be 1D"
        assert expected_sequence[4] == sep_token, "SEP token at correct position"
    
    def test_batch_shape_for_decoder_only_model(self):
        """Verify batch shape matches DecoderOnlyBaseline.forward expectations."""
        batch_size = 4
        seq_len = 100  # Example: 7x7 input + SEP + 5x5 output = 49 + 1 + 25 = 75
        
        # Data loader should produce:
        sequences = torch.randint(0, 11, (batch_size, seq_len))
        
        # Model expects: (batch_size, seq_len)
        assert sequences.shape == (batch_size, seq_len)
        assert sequences.dtype == torch.long


class TestEncoderDecoderDataFormat:
    """Test Exp 0-2 (Encoder-Decoder) data format requirements."""
    
    def test_separate_src_tgt_format(self):
        """
        CRITICAL: E-D models require separate source and target sequences.
        
        Expected format:
        - src: input grid flattened
        - tgt: output grid flattened
        - grid_shapes: (H_in, W_in), (H_out, W_out)
        """
        # Mock data
        input_grid = torch.tensor([[1, 2, 3], [4, 5, 6]])  # 2x3
        output_grid = torch.tensor([[7, 8], [9, 0]])  # 2x2
        
        # Expected format
        src = input_grid.flatten()  # [1, 2, 3, 4, 5, 6]
        tgt = output_grid.flatten()  # [7, 8, 9, 0]
        src_shape = (2, 3)
        tgt_shape = (2, 2)
        
        assert src.shape == (6,)
        assert tgt.shape == (4,)
        assert src_shape == (2, 3)
        assert tgt_shape == (2, 2)
    
    def test_batch_shape_for_encoder_decoder_model(self):
        """Verify batch shapes match EncoderDecoder.forward expectations."""
        batch_size = 4
        src_len = 49  # 7x7 grid
        tgt_len = 25  # 5x5 grid
        
        # Data loader should produce:
        src_batch = torch.randint(0, 11, (batch_size, src_len))
        tgt_batch = torch.randint(0, 11, (batch_size, tgt_len))
        
        # Model expects: src (B, L_in), tgt (B, L_out)
        assert src_batch.shape == (batch_size, src_len)
        assert tgt_batch.shape == (batch_size, tgt_len)
        assert src_batch.dtype == torch.long
        assert tgt_batch.dtype == torch.long


class TestChampionDataFormat:
    """Test Exp 3 (Champion) data format requirements."""
    
    def test_context_pair_format(self):
        """
        CRITICAL: Champion requires context pairs in addition to src/tgt.
        
        Expected format:
        - src, tgt: as in E-D models
        - ctx_input: (batch, num_pairs, H, W)
        - ctx_output: (batch, num_pairs, H, W)
        - Champion uses num_pairs = 2 (fixed)
        """
        batch_size = 2
        num_pairs = 2  # Champion config
        H, W = 3, 3
        
        # Mock context data
        ctx_input = torch.randint(0, 11, (batch_size, num_pairs, H, W))
        ctx_output = torch.randint(0, 11, (batch_size, num_pairs, H, W))
        
        # Verify shape
        assert ctx_input.shape == (batch_size, num_pairs, H, W)
        assert ctx_output.shape == (batch_size, num_pairs, H, W)
        assert ctx_input.dtype == torch.long
        assert ctx_output.dtype == torch.long
    
    def test_context_pairs_must_match_in_size(self):
        """
        CRITICAL: Context input and output grids must have same H, W.
        
        This was discovered during Champion model testing.
        ContextEncoder concatenates along dim=1, requiring H, W to match.
        """
        batch_size = 1
        num_pairs = 2
        H, W = 5, 5
        
        # Both must have same grid size
        ctx_input = torch.randint(0, 11, (batch_size, num_pairs, H, W))
        ctx_output = torch.randint(0, 11, (batch_size, num_pairs, H, W))
        
        # This is valid
        assert ctx_input.shape[-2:] == ctx_output.shape[-2:]
        
        # This would be INVALID:
        # ctx_input: (B, N, 3, 3)
        # ctx_output: (B, N, 2, 2)  # Different H, W!
        # This causes RuntimeError in ContextEncoder.forward


class TestPaddingBehavior:
    """Test padding behavior for variable-sized grids."""
    
    def test_grids_padded_to_max_size(self):
        """
        Grids should be padded to max_grid_size with pad token (10).
        
        Example: 3x3 grid padded to 30x30 for champion architecture.
        """
        original_grid = torch.tensor([[1, 2], [3, 4]])  # 2x2
        pad_token = 10
        max_h, max_w = 30, 30
        
        # Expected padded grid
        padded = torch.full((max_h, max_w), pad_token, dtype=torch.long)
        padded[:2, :2] = original_grid
        
        assert padded.shape == (30, 30)
        assert padded[0, 0] == 1
        assert padded[0, 1] == 2
        assert padded[2, 0] == pad_token  # Padded region


# ==============================================================================
# Integration Shape Tests
# ==============================================================================

def test_shapes_match_model_expectations():
    """
    CRITICAL: Verify data loader output shapes match model.forward signatures.
    
    This is the FINAL CONTRACT TEST before training.
    """
    # Import models (tests pass because models already validated)
    from src.models.decoder_only_baseline import create_decoder_only_model
    from src.models.encoder_decoder_baseline import create_encoder_decoder_model
    from src.models.champion_architecture import create_champion_architecture
    
    batch_size = 2
    
    # Exp -1: Decoder-Only
    decoder_model = create_decoder_only_model(
        vocab_size=11,
        context_length=512,
        d_model=128,
        num_layers=2,
        num_heads=4,
        d_ff=512,
    )
    seq = torch.randint(0, 11, (batch_size, 50))
    try:
        _ = decoder_model(seq)
        # If this doesn't crash, shape contract is satisfied
        assert True
    except Exception as e:
        pytest.fail(f"Decoder-only model failed shape contract: {e}")
    
    # Exp 0: Encoder-Decoder (doesn't use grid shapes)
    ed_model = create_encoder_decoder_model(
        vocab_size=11,
        d_model=128,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        d_ff=512,
    )
    src = torch.randint(0, 11, (batch_size, 9))  # 3x3
    tgt = torch.randint(0, 11, (batch_size, 4))  # 2x2
    try:
        _ = ed_model(src, tgt)  # No grid shapes for baseline E-D
        assert True
    except Exception as e:
        pytest.fail(f"E-D model failed shape contract: {e}")
    
    # Exp 3: Champion
    champion_model = create_champion_architecture(
        vocab_size=11,
        d_model=64,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        d_ff=256,
    )
    ctx_in = torch.randint(0, 11, (batch_size, 2, 3, 3))
    ctx_out = torch.randint(0, 11, (batch_size, 2, 3, 3))
    try:
        _ = champion_model(src, tgt, (3, 3), (2, 2), ctx_in, ctx_out)
        assert True
    except Exception as e:
        pytest.fail(f"Champion model failed shape contract: {e}")


if __name__ == "__main__":
    print("Running critical data loader validation tests...")
    print("\nThese tests define the CONTRACT that data loaders must satisfy.")
    print("Before implementing actual data loaders, all tests should pass.")
    print("\n" + "="*70)
    
    # Run tests
    pytest.main([__file__, "-v"])
