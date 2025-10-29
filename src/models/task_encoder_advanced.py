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
    
    def __init__(self, embed_dim: int = 256, num_demos: int = 3):
        """
        Args:
            embed_dim: Output embedding dimension (256 for Phase 1, 400 for Phase 2)
            num_demos: Number of demonstration pairs (default: 3)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_demos = num_demos
        
        # Use Champion's ContextEncoder config (actual parameters from config.py)
        context_config = ContextEncoderConfig(
            grid_height=30,
            grid_width=30,
            vocab_size=11,
            pad_token_id=10,
            d_model=160,  # Smaller than Champion's 512 for efficiency
            n_head=4,
            pixel_layers=2,  # Fewer than Champion's 4 for efficiency
            grid_layers=1,   # Fewer than Champion's 2
            pe_type="rotary",
            pool_type="attn",
            dynamic_pairs=False,
            attn_dropout=0.1,
            ffn_dropout=0.1
        )
        
        # Champion's ContextEncoderModule
        # This processes a pair of grids (input, output) into a context vector
        self.context_encoder = ContextEncoderModule(context_config)
        
        # Aggregate context vectors from multiple demonstrations
        # Each demo pair → 160-dim context vector
        # Stack 3 demos → 480-dim → MLP → embed_dim
        self.aggregator = nn.Sequential(
            nn.Linear(160 * num_demos, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, embed_dim)
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
        
        # Process each demonstration pair through Champion's ContextEncoder
        context_vectors = []
        
        for i in range(num_demos):
            # Get i-th demo pair
            input_grid = demo_input[:, i, :, :]  # (batch, H, W)
            output_grid = demo_output[:, i, :, :]
            
            # ContextEncoder expects: (batch, num_grids, height, width)
            # Add num_grids dimension (1 pair = 1 input grid)
            input_4d = input_grid.unsqueeze(1)  # (batch, 1, H, W)
            output_4d = output_grid.unsqueeze(1)
            
            # Get context vector
            context = self.context_encoder(input_4d, output_4d)  # (batch, d_model)
            context_vectors.append(context)
        
        # Stack context vectors: (batch, num_demos, 160)
        stacked_contexts = torch.stack(context_vectors, dim=1)
        
        # Flatten: (batch, num_demos * 160)
        flattened = stacked_contexts.reshape(batch_size, -1)
        
        # Aggregate to final embedding
        embeddings = self.aggregator(flattened)  # (batch, embed_dim)
        
        return embeddings
