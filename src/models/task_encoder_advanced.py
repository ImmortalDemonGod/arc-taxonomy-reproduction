"""
TaskEncoder Version B: Advanced architecture using Champion's ContextEncoder.

Uses the same ContextEncoderModule that Champion uses for processing demonstration pairs.
Architecturally stronger and creates direct link to Champion model.
"""
import torch
import torch.nn as nn

from ..context import ContextEncoderModule
from ..config import ContextEncoderConfig


class TaskEncoderAdvanced(nn.Module):
    """
    Version B: Uses Champion's own ContextEncoderModule.
    
    Scientifically stronger narrative: "We used Champion's own encoder to extract
    task-level features from demonstrations."
    
    This is the same module Champion uses during training, so it's proven to work
    for understanding input/output relationships.
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_demos: int = 3,
        context_d_model: int = 256,
        n_head: int = 8,
        pixel_layers: int = 3,
        grid_layers: int = 2,
        dropout_rate: float = 0.1,
    ):
        """
        Args:
            embed_dim: Output embedding dimension (256 for Phase 1, 400 for Phase 2)
            num_demos: Number of demonstration pairs (default: 3)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_demos = num_demos
        
        # Configurable ContextEncoder (processes all demo pairs jointly and pools across pairs)
        context_config = ContextEncoderConfig(
            grid_height=30,
            grid_width=30,
            vocab_size=11,
            pad_token_id=10,
            d_model=context_d_model,
            n_head=n_head,
            pixel_layers=pixel_layers,
            grid_layers=grid_layers,
            pe_type="rotary",
            pool_type="attn",
            dynamic_pairs=False,
            attn_dropout=dropout_rate,
            ffn_dropout=dropout_rate,
            dropout_rate=dropout_rate,
        )
        
        # Champion's ContextEncoderModule
        # This processes a pair of grids (input, output) into a context vector
        self.context_encoder = ContextEncoderModule(context_config)
        
        # Project pooled context embedding to target embed_dim if needed
        if context_d_model == embed_dim:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Sequential(
                nn.Linear(context_d_model, max(256, embed_dim)),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(max(256, embed_dim), embed_dim),
            )
    
    def forward(self, demo_input: torch.Tensor, demo_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            demo_input: (batch, num_demos, H, W) - input grids
            demo_output: (batch, num_demos, H, W) - output grids
        
        Returns:
            embeddings: (batch, embed_dim) - task embeddings
        """
        batch_size, num_demos, H, W = demo_input.shape
        
        # Process all demonstration pairs jointly
        # ContextEncoderModule pools across pairs internally to a single [B, d_model] vector
        context_vec = self.context_encoder(demo_input, demo_output)  # (batch, context_d_model)
        embeddings = self.proj(context_vec)  # (batch, embed_dim)
        
        return embeddings
