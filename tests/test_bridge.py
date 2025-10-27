"""
Unit tests for Bridge modules.

Tests verify:
1. IdentityBridge returns input unchanged
2. ConcatMLPBridge properly integrates context
3. Both bridges handle None context gracefully
4. Ablation-ready: can swap between bridge types
"""
import pytest
import torch
from src.bridge import IdentityBridge, ConcatMLPBridge


class TestIdentityBridge:
    """Test suite for IdentityBridge."""
    
    def test_initialization(self):
        """Test that IdentityBridge initializes correctly."""
        bridge = IdentityBridge(d_model=256, d_ctx=512)
        assert bridge.d_model == 256
        assert bridge.d_ctx == 512
    
    def test_forward_ignores_context(self):
        """Test that IdentityBridge returns input unchanged."""
        bridge = IdentityBridge(d_model=256, d_ctx=512)
        
        x = torch.randn(4, 100, 256)
        context = torch.randn(4, 512)
        
        output = bridge(x, context)
        
        assert torch.allclose(output, x)
    
    def test_forward_with_none_context(self):
        """Test that IdentityBridge handles None context."""
        bridge = IdentityBridge(d_model=256, d_ctx=512)
        
        x = torch.randn(4, 100, 256)
        output = bridge(x, None)
        
        assert torch.allclose(output, x)
    
    def test_no_parameters(self):
        """Test that IdentityBridge has no trainable parameters."""
        bridge = IdentityBridge(d_model=256, d_ctx=512)
        
        param_count = sum(p.numel() for p in bridge.parameters())
        assert param_count == 0


class TestConcatMLPBridge:
    """Test suite for ConcatMLPBridge."""
    
    def test_initialization(self):
        """Test that ConcatMLPBridge initializes correctly."""
        bridge = ConcatMLPBridge(d_model=256, d_ctx=512)
        
        assert bridge.d_model == 256
        assert bridge.d_ctx == 512
        assert hasattr(bridge, 'proj')
        assert hasattr(bridge, 'act')
        assert hasattr(bridge, 'ln')
    
    def test_forward_shape(self):
        """Test that ConcatMLPBridge produces correct output shape."""
        bridge = ConcatMLPBridge(d_model=256, d_ctx=512)
        
        batch_size = 4
        seq_len = 100
        x = torch.randn(batch_size, seq_len, 256)
        context = torch.randn(batch_size, 512)
        
        output = bridge(x, context)
        
        assert output.shape == (batch_size, seq_len, 256)
    
    def test_forward_with_none_context(self):
        """Test that ConcatMLPBridge returns input unchanged when context is None."""
        bridge = ConcatMLPBridge(d_model=256, d_ctx=512)
        
        x = torch.randn(4, 100, 256)
        output = bridge(x, None)
        
        assert torch.allclose(output, x)
    
    def test_context_integration(self):
        """Test that context actually affects the output."""
        bridge = ConcatMLPBridge(d_model=256, d_ctx=512)
        
        x = torch.randn(4, 100, 256)
        context1 = torch.randn(4, 512)
        context2 = torch.randn(4, 512)
        
        output1 = bridge(x, context1)
        output2 = bridge(x, context2)
        
        # Different contexts should produce different outputs
        assert not torch.allclose(output1, output2)
    
    def test_gradient_flow(self):
        """Test that gradients flow through bridge."""
        bridge = ConcatMLPBridge(d_model=256, d_ctx=512)
        
        x = torch.randn(2, 50, 256, requires_grad=True)
        context = torch.randn(2, 512, requires_grad=True)
        
        output = bridge(x, context)
        loss = output.sum()
        loss.backward()
        
        # Both inputs should have gradients
        assert x.grad is not None
        assert context.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))
        assert not torch.allclose(context.grad, torch.zeros_like(context.grad))
    
    def test_parameter_count(self):
        """Test that parameter count is as expected."""
        d_model = 256
        d_ctx = 512
        bridge = ConcatMLPBridge(d_model=d_model, d_ctx=d_ctx)
        
        # Parameters: Linear(d_model + d_ctx → d_model) + LayerNorm(d_model)
        # Linear: (d_model + d_ctx) * d_model + d_model (bias)
        # LayerNorm: d_model * 2 (weight + bias)
        expected = (d_model + d_ctx) * d_model + d_model + d_model * 2
        
        actual = sum(p.numel() for p in bridge.parameters())
        assert actual == expected
    
    def test_deterministic(self):
        """Test that bridge is deterministic."""
        bridge = ConcatMLPBridge(d_model=128, d_ctx=256)
        
        x = torch.randn(2, 50, 128)
        context = torch.randn(2, 256)
        
        output1 = bridge(x, context)
        output2 = bridge(x, context)
        
        assert torch.allclose(output1, output2)
    
    def test_batch_processing(self):
        """Test that batches are processed independently."""
        bridge = ConcatMLPBridge(d_model=128, d_ctx=256)
        
        # Single sample
        x_single = torch.randn(1, 50, 128)
        ctx_single = torch.randn(1, 256)
        out_single = bridge(x_single, ctx_single)
        
        # Batched (same sample repeated)
        x_batch = x_single.repeat(4, 1, 1)
        ctx_batch = ctx_single.repeat(4, 1)
        out_batch = bridge(x_batch, ctx_batch)
        
        # Each batch element should match single result
        for i in range(4):
            assert torch.allclose(out_batch[i], out_single[0])
    
    def test_external_modules(self):
        """Test initialization with external modules."""
        import torch.nn as nn
        
        d_model = 128
        d_ctx = 256
        
        # Create external modules
        external = {
            'proj': nn.Linear(d_model + d_ctx, d_model),
            'act': nn.ReLU(),
            'ln': nn.LayerNorm(d_model)
        }
        
        bridge = ConcatMLPBridge(d_model=d_model, d_ctx=d_ctx, external_modules=external)
        
        # Should use external modules
        assert bridge._external is not None
        
        # Test forward pass works
        x = torch.randn(2, 50, d_model)
        context = torch.randn(2, d_ctx)
        output = bridge(x, context)
        
        assert output.shape == (2, 50, d_model)


class TestBridgeAblation:
    """Test that bridges are swappable for ablation study."""
    
    def test_api_compatibility(self):
        """Test that both bridges have compatible APIs."""
        d_model = 256
        d_ctx = 512
        
        identity = IdentityBridge(d_model=d_model, d_ctx=d_ctx)
        concat_mlp = ConcatMLPBridge(d_model=d_model, d_ctx=d_ctx)
        
        x = torch.randn(4, 100, d_model)
        context = torch.randn(4, d_ctx)
        
        # Both should accept same inputs
        out_identity = identity(x, context)
        out_concat = concat_mlp(x, context)
        
        # Both should produce correct shape
        assert out_identity.shape == (4, 100, d_model)
        assert out_concat.shape == (4, 100, d_model)
    
    def test_swap_bridges(self):
        """Test swapping between bridge types."""
        d_model = 128
        d_ctx = 256
        
        x = torch.randn(2, 50, d_model)
        context = torch.randn(2, d_ctx)
        
        # Use identity bridge (Exp 0-2)
        bridge = IdentityBridge(d_model=d_model, d_ctx=d_ctx)
        out1 = bridge(x, context)
        assert torch.allclose(out1, x)  # No conditioning
        
        # Swap to concat MLP bridge (Exp 3)
        bridge = ConcatMLPBridge(d_model=d_model, d_ctx=d_ctx)
        out2 = bridge(x, context)
        assert not torch.allclose(out2, x)  # With conditioning


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
