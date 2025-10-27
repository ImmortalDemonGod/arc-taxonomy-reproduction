"""
Tests for Champion Architecture (Experiment 3).

Following cs336 pedagogical style and TDD principles.
"""
import pytest
import torch
from src.models.champion_architecture import (
    create_champion_architecture,
    ChampionArchitecture,
)


def test_model_creation_with_defaults():
    """Test champion model with default champion parameters."""
    model = create_champion_architecture(
        vocab_size=11,
        d_model=160,
        num_encoder_layers=1,
        num_decoder_layers=3,
        num_heads=4,
        d_ff=640,
    )
    
    assert isinstance(model, ChampionArchitecture)
    assert model.d_model == 160
    assert model.has_context == True
    assert model.has_bridge == True


def test_forward_without_context():
    """Test forward pass without providing context (should work)."""
    model = create_champion_architecture(
        vocab_size=11,
        d_model=64,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        d_ff=256,
    )
    
    batch_size = 2
    src = torch.randint(0, 11, (batch_size, 9))  # 3x3
    tgt = torch.randint(0, 11, (batch_size, 4))  # 2x2
    
    logits = model(src, tgt, (3, 3), (2, 2))
    
    assert logits.shape == (batch_size, 4, 11)


def test_forward_with_context():
    """Test forward pass with context pairs."""
    model = create_champion_architecture(
        vocab_size=11,
        d_model=64,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        d_ff=256,
    )
    
    batch_size = 2
    num_pairs = 2  # Champion uses 2 context pairs
    
    src = torch.randint(0, 11, (batch_size, 9))  # 3x3
    tgt = torch.randint(0, 11, (batch_size, 4))  # 2x2
    # Context pairs should have matching grid sizes
    ctx_input = torch.randint(0, 11, (batch_size, num_pairs, 3, 3))
    ctx_output = torch.randint(0, 11, (batch_size, num_pairs, 3, 3))
    
    logits = model(src, tgt, (3, 3), (2, 2), ctx_input, ctx_output)
    
    assert logits.shape == (batch_size, 4, 11)


def test_context_encoder_integration():
    """Test that context encoder is properly integrated."""
    model = create_champion_architecture(use_context=True, use_bridge=True)
    
    assert hasattr(model, 'context_encoder')
    assert model.context_encoder is not None
    assert model.context_encoder.__class__.__name__ == 'ContextEncoderModule'


def test_bridge_integration():
    """Test that bridge is properly integrated."""
    model = create_champion_architecture(use_context=True, use_bridge=True)
    
    assert hasattr(model, 'bridge')
    assert model.bridge is not None
    assert model.bridge.__class__.__name__ == 'ConcatMLPBridge'


def test_gradient_flow_with_context():
    """Test gradients flow through full model including context and bridge."""
    model = create_champion_architecture(
        vocab_size=11,
        d_model=64,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        d_ff=256,
    )
    
    batch_size = 1
    src = torch.randint(0, 11, (batch_size, 9))
    tgt = torch.randint(0, 11, (batch_size, 4))
    # Context pairs with matching grid sizes
    ctx_input = torch.randint(0, 11, (batch_size, 2, 3, 3))
    ctx_output = torch.randint(0, 11, (batch_size, 2, 3, 3))
    
    logits = model(src, tgt, (3, 3), (2, 2), ctx_input, ctx_output)
    loss = logits.sum()
    loss.backward()
    
    # Check gradients flow to key components
    assert model.embedding.G.grad is not None
    assert model.context_encoder.embedding.G.grad is not None
    
    # Check bridge has gradients
    bridge_has_grad = False
    for name, param in model.bridge.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            bridge_has_grad = True
            break
    assert bridge_has_grad, "Bridge should receive gradients"


def test_model_without_context_or_bridge():
    """Test model can be created without context/bridge (like Exp 2)."""
    model = create_champion_architecture(
        use_context=False,
        use_bridge=False,
        d_model=64,
    )
    
    assert model.has_context == False
    assert model.has_bridge == False
    assert model.context_encoder is None
    assert model.bridge is None
    
    # Should still work
    src = torch.randint(0, 11, (1, 9))
    tgt = torch.randint(0, 11, (1, 4))
    logits = model(src, tgt, (3, 3), (2, 2))
    assert logits.shape == (1, 4, 11)


if __name__ == "__main__":
    print("Running Champion Architecture tests...")
    test_model_creation_with_defaults()
    print("✅ test_model_creation_with_defaults")
    test_forward_without_context()
    print("✅ test_forward_without_context")
    test_forward_with_context()
    print("✅ test_forward_with_context")
    test_context_encoder_integration()
    print("✅ test_context_encoder_integration")
    test_bridge_integration()
    print("✅ test_bridge_integration")
    test_gradient_flow_with_context()
    print("✅ test_gradient_flow_with_context")
    test_model_without_context_or_bridge()
    print("✅ test_model_without_context_or_bridge")
    print("\n✅ All Champion Architecture tests passed!")
