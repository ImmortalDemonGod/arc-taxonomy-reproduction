"""
1D Sinusoidal Positional Encoding for baseline experiments.

Standard implementation following Vaswani et al. (2017) "Attention is All You Need".
Used for Exp 0 (Generic Encoder-Decoder baseline).
"""
import torch
import torch.nn as nn
from torch import Tensor
import math


class PositionalEncoding1D(nn.Module):
    """
    1D sinusoidal positional encoding.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Following cs336 pedagogical style: clear, well-documented, standard implementation.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.0):
        """
        Initialize 1D positional encoding.
        
        Args:
            d_model: Model embedding dimension
            max_len: Maximum sequence length
            dropout: Dropout probability (applied after adding PE)
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute div_term: 10000^(2i/d_model) = exp(2i * -log(10000) / d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cos to odd indices
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            # Handle odd d_model
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        
        # Register as buffer (not a parameter, but part of state_dict)
        self.register_buffer('pe', pe.unsqueeze(0))  # Shape: (1, max_len, d_model)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added, same shape as input
        """
        # x shape: (B, L, D)
        # pe shape: (1, max_len, D)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


def create_1d_positional_encoding(
    d_model: int,
    max_len: int = 5000,
    dropout: float = 0.0,
) -> PositionalEncoding1D:
    """
    Factory function for creating 1D positional encoding.
    
    Provides a clean interface following cs336 style.
    
    Args:
        d_model: Model embedding dimension
        max_len: Maximum sequence length
        dropout: Dropout probability
        
    Returns:
        PositionalEncoding1D module
    """
    return PositionalEncoding1D(d_model, max_len, dropout)


def get_sinusoidal_embeddings(
    positions: Tensor,
    d_model: int,
) -> Tensor:
    """
    Compute sinusoidal positional embeddings for given positions.
    
    Functional interface (doesn't require creating a module).
    Useful for testing and visualization.
    
    Args:
        positions: Position indices of shape (batch_size, seq_len)
        d_model: Embedding dimension
        
    Returns:
        Positional embeddings of shape (batch_size, seq_len, d_model)
    """
    batch_size, seq_len = positions.shape
    
    # Create embedding matrix
    embeddings = torch.zeros(batch_size, seq_len, d_model, device=positions.device)
    
    # Compute div_term
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=positions.device).float() 
        * (-math.log(10000.0) / d_model)
    )
    
    # Apply sin/cos
    # positions shape: (B, L) -> unsqueeze to (B, L, 1)
    pos_unsqueezed = positions.unsqueeze(-1).float()
    
    embeddings[:, :, 0::2] = torch.sin(pos_unsqueezed * div_term)
    
    if d_model % 2 == 0:
        embeddings[:, :, 1::2] = torch.cos(pos_unsqueezed * div_term)
    else:
        embeddings[:, :, 1::2] = torch.cos(pos_unsqueezed * div_term[:-1])
    
    return embeddings
