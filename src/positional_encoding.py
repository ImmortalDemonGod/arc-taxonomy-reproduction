"""
2D Positional Encoding for ARC Grid Tasks.

Adapted from jarc_reactor for standalone reproduction package.
"""
import math
import torch
import torch.nn as nn

__all__ = ["Grid2DPositionalEncoding"]


class Grid2DPositionalEncoding(nn.Module):
    """
    Sinusoidal 2D positional encoding for grid-structured inputs.
    
    - Encodes height and width dimensions separately using sinusoidal functions
    - First half of d_model encodes y-coordinates, second half encodes x-coordinates  
    - Pre-computed table registered as buffer (not trainable)
    - Input shape: [batch, seq_len, d_model] where seq_len = height × width
    - Output shape: [batch, seq_len, d_model] (input + positional encoding)
    """
    
    def __init__(self, d_model: int, max_height: int = 30, max_width: int = 30):
        super().__init__()
        self.d_model = d_model
        
        # Validation
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {d_model}")
            
        # Pre-compute positional encoding table
        pe = torch.zeros(max_height * max_width, d_model)
        
        # Split d_model in half: first half for y-coordinates, second half for x-coordinates
        d_half = d_model // 2
        if d_half % 2 != 0:
            raise ValueError(f"Half of d_model ({d_half}) must be even for sinusoidal pairs")
        
        # Generate grid coordinates
        pos_y = torch.arange(max_height, dtype=torch.float).unsqueeze(1)
        pos_x = torch.arange(max_width, dtype=torch.float).unsqueeze(1)
        
        # Create 2D coordinate grids
        grid_y = pos_y.repeat(1, max_width).view(-1, 1)  # [H*W, 1]
        grid_x = pos_x.repeat(max_height, 1).view(-1, 1)  # [H*W, 1]
        
        # Compute division term for sinusoidal encoding
        div_term = torch.exp(
            torch.arange(0, d_half, 2).float() * (-math.log(10000.0) / d_half)
        )
        
        # Apply sinusoidal encoding to y-coordinates (first half of d_model)
        pe[:, 0:d_half:2] = torch.sin(grid_y * div_term)  # Even indices
        pe[:, 1:d_half:2] = torch.cos(grid_y * div_term)  # Odd indices
        
        # Apply sinusoidal encoding to x-coordinates (second half of d_model)
        pe[:, d_half::2] = torch.sin(grid_x * div_term)   # Even indices
        pe[:, d_half + 1::2] = torch.cos(grid_x * div_term)  # Odd indices
        
        # Add batch dimension and register as buffer (not trainable)
        pe = pe.unsqueeze(0)  # [1, H*W, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]
