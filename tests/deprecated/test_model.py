"""
Unit tests for TransformerModel (reproduction/src/).

Tests the local TransformerModel implementation with proper imports
from the src/ package structure.
"""
import pytest
import sys
from pathlib import Path

# Setup paths
tests_dir = Path(__file__).parent
reproduction_dir = tests_dir.parent
sys.path.insert(0, str(reproduction_dir))

import torch
from omegaconf import OmegaConf

# Import from reproduction/src/
from src.model.transformer_model import TransformerModel
from src.config_schema import ModelConfigSchema, ContextEncoderConfig


@pytest.fixture
def minimal_config():
    """Create minimal model config for testing."""
    cfg = OmegaConf.create({
        'model': {
            'max_h': 30,
            'max_w': 30,
            'd_model': 128,
            'vocab_size': 11,
            'pad_token_id': 10,
            'encoder_layers': 2,
            'decoder_layers': 2,
            'n_head': 4,
            'd_ff': 512,
            'dropout_rate': 0.1,
            'norm_first': True,
            'output_dim': 11,
            'context_encoder': {
                'grid_height': 30,
                'grid_width': 30,
                'vocab_size': 11,
                'pad_token_id': 10,
                'd_model': 128,
                'n_head': 4,
                'pixel_layers': 2,
                'grid_layers': 1,
                'pe_type': 'rotary',
                'pool_type': 'attn',
                'dynamic_pairs': False,
                'attn_dropout': 0.1,
                'ffn_dropout': 0.1,
            }
        }
    })
    return cfg.model


def test_model_initialization(minimal_config):
    """Test TransformerModel can be initialized with minimal config."""
    try:
        model = TransformerModel(minimal_config)
        assert model is not None
        assert isinstance(model, torch.nn.Module)
        
        # Check key attributes
        assert model.d_model == 128
        assert model.vocab_size == 11
        assert model.pad_token_id == 10
        
        # Check submodules exist
        assert hasattr(model, 'encoder')
        assert hasattr(model, 'decoder')
        assert hasattr(model, 'context_encoder')
        assert hasattr(model, 'output_fc')
        
        print("✅ test_model_initialization passed")
        
    except Exception as e:
        pytest.fail(f"Model initialization failed: {e}")


def test_forward_pass(minimal_config):
    """Test forward pass with dummy data."""
    model = TransformerModel(minimal_config)
    model.eval()
    
    batch_size = 2
    H, W = 30, 30
    
    # Create dummy inputs
    src = torch.randint(0, 10, (batch_size, H, W), dtype=torch.long)
    tgt = torch.randint(0, 10, (batch_size, H, W), dtype=torch.long)
    ctx_input = torch.randint(0, 10, (batch_size, 2, H, W), dtype=torch.long)
    ctx_output = torch.randint(0, 10, (batch_size, 2, H, W), dtype=torch.long)
    
    try:
        with torch.no_grad():
            logits = model(src, tgt, ctx_input, ctx_output)
        
        # Check output shape
        assert logits.shape == (batch_size, H, W, 11), \
            f"Expected shape ({batch_size}, {H}, {W}, 11), got {logits.shape}"
        
        # Check logits are finite
        assert torch.isfinite(logits).all(), "Logits contain NaN or Inf"
        
        print(f"✅ test_forward_pass passed (output shape: {logits.shape})")
        
    except Exception as e:
        pytest.fail(f"Forward pass failed: {e}")


def test_context_encoder(minimal_config):
    """Test context encoder processes context pairs."""
    model = TransformerModel(minimal_config)
    model.eval()
    
    batch_size = 2
    num_pairs = 2
    H, W = 30, 30
    
    ctx_input = torch.randint(0, 10, (batch_size, num_pairs, H, W), dtype=torch.long)
    ctx_output = torch.randint(0, 10, (batch_size, num_pairs, H, W), dtype=torch.long)
    
    try:
        with torch.no_grad():
            # Access context encoder directly
            context_embedding = model.context_encoder(ctx_input, ctx_output)
        
        # Check embedding shape [B, d_model]
        assert context_embedding.shape == (batch_size, 128), \
            f"Expected shape ({batch_size}, 128), got {context_embedding.shape}"
        
        assert torch.isfinite(context_embedding).all()
        
        print(f"✅ test_context_encoder passed (embedding shape: {context_embedding.shape})")
        
    except Exception as e:
        pytest.fail(f"Context encoder failed: {e}")


def test_parameter_count(minimal_config):
    """Test model has reasonable parameter count."""
    model = TransformerModel(minimal_config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Sanity checks
    assert total_params > 0, "Model has no parameters"
    assert trainable_params > 0, "Model has no trainable parameters"
    assert total_params == trainable_params, "Some parameters are frozen unexpectedly"
    
    # Reasonable range for minimal config (rough estimate)
    assert 10_000 < total_params < 10_000_000, \
        f"Parameter count seems unusual: {total_params:,}"
    
    print(f"✅ test_parameter_count passed ({total_params:,} parameters)")


def test_batch_independence(minimal_config):
    """Test that different batch items don't interfere."""
    model = TransformerModel(minimal_config)
    model.eval()
    
    H, W = 30, 30
    
    # Create two identical inputs in batch
    src1 = torch.randint(0, 10, (1, H, W), dtype=torch.long)
    src2 = src1.clone()
    src_batch = torch.cat([src1, src2], dim=0)
    
    tgt1 = torch.randint(0, 10, (1, H, W), dtype=torch.long)
    tgt2 = tgt1.clone()
    tgt_batch = torch.cat([tgt1, tgt2], dim=0)
    
    ctx_input1 = torch.randint(0, 10, (1, 2, H, W), dtype=torch.long)
    ctx_input2 = ctx_input1.clone()
    ctx_input_batch = torch.cat([ctx_input1, ctx_input2], dim=0)
    
    ctx_output1 = torch.randint(0, 10, (1, 2, H, W), dtype=torch.long)
    ctx_output2 = ctx_output1.clone()
    ctx_output_batch = torch.cat([ctx_output1, ctx_output2], dim=0)
    
    try:
        with torch.no_grad():
            logits_batch = model(src_batch, tgt_batch, ctx_input_batch, ctx_output_batch)
        
        # Check that identical inputs produce identical outputs
        diff = (logits_batch[0] - logits_batch[1]).abs().max().item()
        assert diff < 1e-5, f"Batch items differ: max diff = {diff}"
        
        print(f"✅ test_batch_independence passed (max diff: {diff:.2e})")
        
    except Exception as e:
        pytest.fail(f"Batch independence test failed: {e}")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
