"""
Tests for checkpoint loading and config sanitization utilities.

Critical test: Ensures we can load champion_bootstrap.ckpt and instantiate models.
"""
import pytest
import torch
from pathlib import Path
from src.checkpoint_utils import (
    load_and_sanitize_config_from_checkpoint,
    load_champion_checkpoint,
)
from src.config import ModelConfig


# Path to champion checkpoint
CHAMPION_CKPT = Path("/Users/tomriddle1/Holistic-Performance-Enhancement/cultivation/systems/arc_reactor/outputs/checkpoints/champion_bootstrap.ckpt")


@pytest.mark.skipif(not CHAMPION_CKPT.exists(), reason="Champion checkpoint not found")
def test_load_config_from_champion():
    """Test loading and sanitizing config from actual champion_bootstrap.ckpt."""
    config = load_and_sanitize_config_from_checkpoint(CHAMPION_CKPT)
    
    # Verify it's a ModelConfig instance
    assert isinstance(config, ModelConfig)
    
    # Verify core architecture parameters
    assert config.d_model > 0
    assert config.encoder_layers > 0
    assert config.decoder_layers > 0
    assert config.n_head > 0
    assert config.d_ff > 0
    
    # Verify grid parameters
    assert config.max_h > 0
    assert config.max_w > 0
    assert config.vocab_size > 0
    
    # Champion should have context encoder and conditioning (bridge)
    assert config.context_encoder is not None
    assert config.conditioning is not None
    assert config.conditioning.bridge is not None
    
    print(f"✅ Loaded config: {config}")


@pytest.mark.skipif(not CHAMPION_CKPT.exists(), reason="Champion checkpoint not found")
def test_load_champion_checkpoint_full():
    """Test loading full checkpoint (config + state_dict)."""
    config, state_dict = load_champion_checkpoint(CHAMPION_CKPT)
    
    # Verify config
    assert isinstance(config, ModelConfig)
    
    # Verify state_dict
    assert isinstance(state_dict, dict)
    assert len(state_dict) > 0
    
    # Check that Lightning prefixes are removed
    for key in state_dict.keys():
        assert not key.startswith('model.'), f"Key still has 'model.' prefix: {key}"
        assert not key.startswith('core_model.'), f"Key still has 'core_model.' prefix: {key}"
    
    # Check for expected component keys
    encoder_keys = [k for k in state_dict.keys() if k.startswith('encoder.')]
    decoder_keys = [k for k in state_dict.keys() if k.startswith('decoder.')]
    context_keys = [k for k in state_dict.keys() if k.startswith('context_encoder.')]
    bridge_keys = [k for k in state_dict.keys() if k.startswith('context_integration.')]
    
    assert len(encoder_keys) > 0, "No encoder keys found"
    assert len(decoder_keys) > 0, "No decoder keys found"
    assert len(context_keys) > 0, "No context encoder keys found"
    assert len(bridge_keys) > 0, "No bridge keys found"
    
    print(f"✅ State dict components: "
          f"encoder={len(encoder_keys)}, "
          f"decoder={len(decoder_keys)}, "
          f"context={len(context_keys)}, "
          f"bridge={len(bridge_keys)}")


@pytest.mark.skipif(not CHAMPION_CKPT.exists(), reason="Champion checkpoint not found")
def test_config_values_reasonable():
    """Test that loaded config values are in reasonable ranges."""
    config = load_and_sanitize_config_from_checkpoint(CHAMPION_CKPT)
    
    # Architecture should be reasonably sized
    assert 32 <= config.d_model <= 1024, f"Unexpected d_model: {config.d_model}"
    assert 1 <= config.encoder_layers <= 12, f"Unexpected encoder_layers: {config.encoder_layers}"
    assert 1 <= config.decoder_layers <= 12, f"Unexpected decoder_layers: {config.decoder_layers}"
    assert 1 <= config.n_head <= 16, f"Unexpected n_head: {config.n_head}"
    assert 64 <= config.d_ff <= 4096, f"Unexpected d_ff: {config.d_ff}"
    
    # Grid parameters
    assert 10 <= config.max_h <= 100, f"Unexpected max_h: {config.max_h}"
    assert 10 <= config.max_w <= 100, f"Unexpected max_w: {config.max_w}"
    assert config.vocab_size == 11, f"Expected vocab_size=11, got {config.vocab_size}"
    
    # Dropout should be reasonable
    assert 0.0 <= config.dropout_rate <= 0.5, f"Unexpected dropout_rate: {config.dropout_rate}"


@pytest.mark.skipif(not CHAMPION_CKPT.exists(), reason="Champion checkpoint not found")
def test_context_encoder_config():
    """Test that context encoder config is loaded correctly."""
    config = load_and_sanitize_config_from_checkpoint(CHAMPION_CKPT)
    
    assert config.context_encoder is not None, "Context encoder should be present in champion"
    
    ctx_config = config.context_encoder
    assert ctx_config.d_model > 0
    assert ctx_config.pixel_layers > 0
    assert ctx_config.n_head > 0
    # Note: dynamic_pairs in checkpoint but champion was trained with fixed 2 pairs
    
    print(f"✅ Context encoder config: {ctx_config}")


@pytest.mark.skipif(not CHAMPION_CKPT.exists(), reason="Champion checkpoint not found")
def test_bridge_config():
    """Test that bridge config is loaded correctly."""
    config = load_and_sanitize_config_from_checkpoint(CHAMPION_CKPT)
    
    assert config.conditioning is not None, "Conditioning should be present in champion"
    assert config.conditioning.bridge is not None, "Bridge should be present in champion"
    
    bridge_config = config.conditioning.bridge
    assert bridge_config.type in ['identity', 'concat_mlp']
    
    if bridge_config.type == 'concat_mlp':
        assert bridge_config.hidden_factor > 0
        assert bridge_config.tokens > 0
        assert bridge_config.heads > 0
    
    print(f"✅ Bridge config: {bridge_config}")


def test_nonexistent_checkpoint():
    """Test error handling for missing checkpoint."""
    with pytest.raises(FileNotFoundError):
        load_and_sanitize_config_from_checkpoint("/nonexistent/path/to/checkpoint.ckpt")


if __name__ == "__main__":
    # Run tests manually
    if CHAMPION_CKPT.exists():
        print("Running checkpoint utils tests...\n")
        test_load_config_from_champion()
        test_load_champion_checkpoint_full()
        test_config_values_reasonable()
        test_context_encoder_config()
        test_bridge_config()
        print("\n✅ All tests passed!")
    else:
        print(f"⚠️  Champion checkpoint not found at: {CHAMPION_CKPT}")
