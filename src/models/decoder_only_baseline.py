"""
Decoder-Only Baseline Model (Experiment -1)

This module implements a simple decoder-only transformer baseline for ARC tasks,
following the pedagogical style of cs336_basics.

The model treats ARC tasks as sequence-to-sequence problems by:
1. Flattening 2D grids into 1D sequences
2. Concatenating [INPUT] [SEP] [OUTPUT] 
3. Training with causal masking (standard language modeling objective)

This baseline is expected to perform poorly (~0-1% accuracy) because:
- Causal masking prevents holistic 2D reasoning
- Loses spatial structure information
- Cannot attend to full input when generating output

Purpose: Establish empirical floor for ablation experiments.
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from ..layers import transformer_lm


def flatten_grid_to_sequence(
    input_grid: Tensor,
    output_grid: Tensor,
    sep_token: int = 10
) -> Tensor:
    """
    Convert a pair of 2D grids into a single 1D sequence.
    
    Format: [INPUT_PIXELS] [SEP] [OUTPUT_PIXELS]
    
    Args:
        input_grid: 2D tensor of shape (H_in, W_in) with color values
        output_grid: 2D tensor of shape (H_out, W_out) with color values
        sep_token: Special token to separate input and output (default: 10)
        
    Returns:
        1D tensor of shape (H_in * W_in + 1 + H_out * W_out)
        
    Example:
        >>> input_grid = torch.tensor([[1, 2], [3, 4]])
        >>> output_grid = torch.tensor([[5, 6]])
        >>> seq = flatten_grid_to_sequence(input_grid, output_grid, sep_token=10)
        >>> seq
        tensor([1, 2, 3, 4, 10, 5, 6])
    """
    # Flatten grids to 1D (row-major order)
    input_flat = input_grid.flatten()
    output_flat = output_grid.flatten()
    
    # Create SEP token
    sep = torch.tensor([sep_token], dtype=input_flat.dtype, device=input_flat.device)
    
    # Concatenate: [INPUT] [SEP] [OUTPUT]
    sequence = torch.cat([input_flat, sep, output_flat])
    
    return sequence


def create_decoder_only_model(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float = 10000.0,
    dropout: float = 0.1,
) -> nn.Module:
    """
    Create a decoder-only transformer language model using cs336 components.
    
    This is a thin wrapper around cs336_basics.layers.transformer_lm that
    provides ARC-specific defaults and documentation.
    
    Args:
        vocab_size: Number of unique tokens (11 for ARC: 0-9 colors + PAD)
        context_length: Maximum sequence length
        d_model: Model embedding dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads (must divide d_model)
        d_ff: Feedforward layer dimension
        rope_theta: RoPE (Rotary Positional Encoding) base frequency
        dropout: Dropout probability
        
    Returns:
        Decoder-only transformer model that maps indices to logits
        
    Notes:
        - Uses RoPE for positional encoding (1D, suitable for sequences)
        - Causal self-attention mask prevents future token visibility
        - Output shape: (batch_size, seq_len, vocab_size)
    """
    # Weight initialization will be handled by transformer_lm
    # cs336 uses standard PyTorch initialization
    
    # We need to create a weight dict for the cs336 interface
    # Since we're building from scratch, we'll let transformer_lm handle initialization
    # But for the functional interface, we need to provide weights
    
    # For now, create a simple nn.Module wrapper
    class DecoderOnlyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.vocab_size = vocab_size
            self.context_length = context_length
            self.d_model = d_model
            self.num_layers = num_layers
            self.num_heads = num_heads
            self.d_ff = d_ff
            self.rope_theta = rope_theta
            
            # Token embedding
            self.token_embedding = nn.Embedding(vocab_size, d_model)
            
            # Transformer layers (using standard PyTorch for simplicity)
            # cs336's transformer_lm is functional, so we'll use PyTorch's built-in
            from torch.nn import TransformerDecoder, TransformerDecoderLayer
            
            decoder_layer = TransformerDecoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,  # Pre-norm like cs336
            )
            self.transformer = TransformerDecoder(decoder_layer, num_layers)
            
            # Output projection
            self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
            
            # Tie weights (common practice in language models)
            self.output_proj.weight = self.token_embedding.weight
            
        def forward(self, input_ids: Tensor) -> Tensor:
            """
            Forward pass.
            
            Args:
                input_ids: Token indices of shape (batch_size, seq_len)
                
            Returns:
                Logits of shape (batch_size, seq_len, vocab_size)
            """
            batch_size, seq_len = input_ids.shape
            
            # Embed tokens
            x = self.token_embedding(input_ids)  # (B, L, D)
            
            # Create causal mask
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                seq_len, device=input_ids.device
            )
            
            # Transformer decoder (self-attention only, no cross-attention)
            # We pass x as both tgt and memory since it's self-attention
            x = self.transformer(x, x, tgt_mask=causal_mask)
            
            # Project to vocabulary
            logits = self.output_proj(x)
            
            return logits
    
    return DecoderOnlyModel()


def compute_decoder_only_loss(
    logits: Tensor,
    targets: Tensor,
    input_length: int,
    ignore_index: int = -100,
) -> Tensor:
    """
    Compute cross-entropy loss for decoder-only model, masking input tokens.
    
    The model is trained to predict output tokens only. Input tokens and the
    SEP token are masked out from the loss computation.
    
    Args:
        logits: Model predictions of shape (batch_size, seq_len, vocab_size)
        targets: Ground truth indices of shape (batch_size, seq_len)
        input_length: Number of input tokens (before SEP token)
        ignore_index: Index to ignore in loss calculation (default: -100)
        
    Returns:
        Scalar loss tensor
        
    Example:
        >>> # Sequence: [1,2,3, SEP, 5,6,7]  (input_length=3)
        >>> # Loss only computed on positions 4,5,6 (output tokens)
        >>> logits = torch.randn(1, 7, 11)
        >>> targets = torch.tensor([[1, 2, 3, 10, 5, 6, 7]])
        >>> loss = compute_decoder_only_loss(logits, targets, input_length=3)
    """
    batch_size, seq_len, vocab_size = logits.shape
    
    # Create mask: True for positions to KEEP in loss
    # We want to mask out: [0..input_length] (input + SEP)
    # Keep: [input_length+1..seq_len] (output tokens)
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=targets.device)
    mask[:, input_length+1:] = True  # Only keep output tokens
    
    # Apply mask by setting masked positions to ignore_index
    masked_targets = targets.clone()
    masked_targets[~mask] = ignore_index
    
    # Compute cross-entropy loss
    loss = nn.functional.cross_entropy(
        logits.reshape(-1, vocab_size),
        masked_targets.reshape(-1),
        ignore_index=ignore_index,
    )
    
    return loss


def create_sequence_batch(
    input_grids: list[Tensor],
    output_grids: list[Tensor],
    sep_token: int = 10,
    pad_token: int = 10,
    max_length: Optional[int] = None,
) -> tuple[Tensor, Tensor, list[int]]:
    """
    Create a padded batch of sequences from lists of grids.
    
    Args:
        input_grids: List of 2D input grids
        output_grids: List of 2D output grids
        sep_token: SEP token value
        pad_token: PAD token value
        max_length: Maximum sequence length (None = use longest in batch)
        
    Returns:
        - input_ids: Padded sequences of shape (batch_size, max_len)
        - targets: Same as input_ids (for language modeling)
        - input_lengths: List of input grid lengths (for loss masking)
    """
    sequences = []
    input_lengths = []
    
    for input_grid, output_grid in zip(input_grids, output_grids):
        seq = flatten_grid_to_sequence(input_grid, output_grid, sep_token)
        sequences.append(seq)
        input_lengths.append(input_grid.numel())
    
    # Determine max length
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    
    # Pad sequences
    batch_size = len(sequences)
    input_ids = torch.full((batch_size, max_length), pad_token, dtype=torch.long)
    
    for i, seq in enumerate(sequences):
        seq_len = min(len(seq), max_length)
        input_ids[i, :seq_len] = seq[:seq_len]
    
    # Targets are same as inputs (language modeling objective)
    targets = input_ids.clone()
    
    return input_ids, targets, input_lengths
