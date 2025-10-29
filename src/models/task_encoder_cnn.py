"""
TaskEncoder Version A: Simple CNN baseline.

Lightweight 3-layer CNN that maps demonstration pairs to category embeddings.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskEncoderCNN(nn.Module):
    """
    Version A: Lightweight CNN baseline for TaskEncoder.
    
    Maps 3 demonstration pairs (input/output grids) to a category embedding.
    Simple, fast, interpretable.
    
    Architecture:
        - Conv blocks: 11 → 32 → 64 → 128 channels
        - Global average pooling
        - MLP to embed_dim
    """
    
    def __init__(self, embed_dim: int = 256, num_demos: int = 3, vocab_size: int = 11):
        """
        Args:
            embed_dim: Output embedding dimension (256 for Phase 1, 400 for Phase 2)
            num_demos: Number of demonstration pairs (default: 3)
            vocab_size: ARC color vocabulary size (0-10, so 11 total)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_demos = num_demos
        
        # Embedding layer for grid values (0-10)
        self.value_embed = nn.Embedding(vocab_size, 16)
        
        # Process each demo pair (input + output concatenated)
        # Input: (batch, num_demos, 2, H, W) after embedding → (batch, num_demos, 32, H, W)
        self.demo_conv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 32 = 16*2 (input+output)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # Global average pooling
        )
        
        # Aggregate across demonstrations
        self.demo_agg = nn.Sequential(
            nn.Linear(128 * num_demos, 512),
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
        
        # Embed grid values
        input_embed = self.value_embed(demo_input)  # (batch, num_demos, H, W, 16)
        output_embed = self.value_embed(demo_output)
        
        # Reshape to (batch * num_demos, H, W, 16)
        input_embed = input_embed.reshape(batch_size * num_demos, H, W, 16)
        output_embed = output_embed.reshape(batch_size * num_demos, H, W, 16)
        
        # Concatenate input/output and permute to channels-first
        # (batch * num_demos, H, W, 32) → (batch * num_demos, 32, H, W)
        pair_embed = torch.cat([input_embed, output_embed], dim=-1)
        pair_embed = pair_embed.permute(0, 3, 1, 2)
        
        # Process through conv layers
        pair_features = self.demo_conv(pair_embed)  # (batch * num_demos, 128, 1, 1)
        pair_features = pair_features.squeeze(-1).squeeze(-1)  # (batch * num_demos, 128)
        
        # Reshape back to (batch, num_demos, 128)
        pair_features = pair_features.reshape(batch_size, num_demos, 128)
        
        # Flatten across demos
        pair_features = pair_features.reshape(batch_size, -1)  # (batch, num_demos * 128)
        
        # Final MLP to embedding
        embeddings = self.demo_agg(pair_features)  # (batch, embed_dim)
        
        return embeddings
