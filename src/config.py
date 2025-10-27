"""
Minimal configuration schema for champion_bootstrap reproduction.

Ported from jarc_reactor/config_schema.py.

These are lightweight type hint containers. The actual configuration is loaded
from the checkpoint as an OmegaConf DictConfig object, which provides dict-like
access with attribute notation.
"""
from dataclasses import dataclass, field
from typing import Any, Optional

__all__ = ["ContextEncoderConfig", "BridgeConfig", "ConditioningConfig", "ModelConfig"]


@dataclass
class ContextEncoderConfig:
    """Configuration for context encoder.
    
    Champion uses:
    - grid_height: 30
    - grid_width: 30
    - vocab_size: 11
    - pad_token_id: 10
    - d_model: 512
    - pixel_layers: 4
    - grid_layers: 2
    - dynamic_pairs: false
    """
    grid_height: int = 30
    grid_width: int = 30
    vocab_size: int = 11
    pad_token_id: int = 10
    d_model: int = 512
    n_head: int = 8
    pixel_layers: int = 4
    grid_layers: int = 2
    pe_type: str = "rotary"
    pool_type: str = "attn"
    dynamic_pairs: bool = False
    attn_dropout: float = 0.1
    ffn_dropout: float = 0.1
    pe_dropout: float = 0.0
    dropout_rate: float = 0.12
    use_positional_encoding: bool = True
    order_sensitive: bool = False
    comp_mode: str = "cross_out_diff_prod"
    order_comp_use_layernorm: bool = True


@dataclass
class BridgeConfig:
    """Configuration for context bridge (conditioning integration).
    
    Champion uses ConcatMLPBridge with:
    - type: "concat_mlp"
    - apply_to_encoder: true
    - apply_to_decoder: true
    """
    type: str = "concat_mlp"
    apply_to_encoder: bool = True
    apply_to_decoder: bool = True
    tokens: int = 2
    heads: int = 8
    hidden_factor: float = 2.0


@dataclass
class ConditioningConfig:
    """Container for conditioning/bridge configuration."""
    bridge: BridgeConfig = field(default_factory=BridgeConfig)


@dataclass
class ModelConfig:
    """Main model configuration schema.
    
    Champion (champion_bootstrap.ckpt) uses:
    - max_h/max_w: 30
    - d_model: 256
    - encoder_layers: 6
    - decoder_layers: 6
    - n_head: 8
    - d_ff: 1024
    - Standard TransformerEncoder/Decoder (no experimental features)
    """
    # Grid dimensions
    max_h: int = 30
    max_w: int = 30
    
    # Model dimensions
    d_model: int = 256
    vocab_size: int = 11
    pad_token_id: int = 10
    
    # Architecture
    encoder_layers: int = 6
    decoder_layers: int = 6
    n_head: int = 8
    d_ff: int = 1024
    norm_first: bool = True
    
    # Dropout
    dropout_rate: float = 0.15
    encoder_dropout_rate: float = 0.61
    decoder_dropout_rate: float = 0.12
    
    # Output
    output_dim: int = 11
    
    # Context encoder
    context_encoder: Optional[ContextEncoderConfig] = None
    
    # Conditioning (bridge)
    conditioning: Optional[ConditioningConfig] = None
    
    # Decoder mode
    use_single_decoder: bool = False
    eval_stabilization: bool = False
    eval_final_clamp: bool = False
    
    # Optional fields for compatibility
    checkpoint_path: Optional[str] = None
    amp_enabled: bool = False
    context_scaling_factor: float = 2.0
    context_effect_gain: float = 4.0
    
    # Experimental feature flags (all disabled for champion)
    use_tri_temporal: bool = False
    attention_type: str = "vanilla"
    
    def __getattr__(self, name: str) -> Any:
        """Provide flexible attribute access for OmegaConf compatibility.
        
        When config is loaded from checkpoint, it becomes an OmegaConf DictConfig.
        This allows graceful handling of missing attributes by returning None.
        """
        return None
