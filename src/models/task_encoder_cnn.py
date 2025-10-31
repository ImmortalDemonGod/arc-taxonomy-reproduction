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
    
    def __init__(self, embed_dim: int = 256, num_demos: int = 3, vocab_size: int = 11, width_mult: float = 1.0, depth: int = 3, mlp_hidden: int = 512, demo_agg: str = 'flatten', use_coords: bool = False):
        """
        Args:
            embed_dim: Output embedding dimension (256 for Phase 1, 400 for Phase 2)
            num_demos: Number of demonstration pairs (default: 3)
            vocab_size: ARC color vocabulary size (0-10, so 11 total)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_demos = num_demos
        self.demo_agg_mode = demo_agg
        self.use_coords = use_coords
        
        # Embedding layer for grid values (0-10)
        self.value_embed = nn.Embedding(vocab_size, 16)
        
        in_ch = 32 + (2 if self.use_coords else 0)
        c1 = int(64 * width_mult)
        c2 = int(128 * width_mult)
        channels = [c1, c2]
        if depth >= 4:
            channels.append(int(256 * width_mult))
        conv_layers = []
        cur_in = in_ch
        for c in channels:
            conv_layers.append(nn.Conv2d(cur_in, c, kernel_size=3, padding=1))
            conv_layers.append(nn.ReLU())
            cur_in = c
        conv_layers.append(nn.AdaptiveAvgPool2d(1))
        self.demo_conv = nn.Sequential(*conv_layers)
        feat_c = channels[-1]
        self.feat_c = feat_c
        
        if self.demo_agg_mode == 'flatten':
            in_dim = feat_c * num_demos
        else:
            in_dim = feat_c
        self.demo_agg = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(mlp_hidden, embed_dim)
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
        if self.use_coords:
            ys = torch.linspace(0, 1, H, device=pair_embed.device).view(1, 1, H, 1).expand(pair_embed.size(0), -1, -1, W)
            xs = torch.linspace(0, 1, W, device=pair_embed.device).view(1, 1, 1, W).expand(pair_embed.size(0), -1, H, -1)
            coords = torch.cat([ys, xs], dim=1)
            pair_embed = torch.cat([pair_embed, coords], dim=1)
        
        # Process through conv layers
        pair_features = self.demo_conv(pair_embed)
        pair_features = pair_features.squeeze(-1).squeeze(-1)
        
        # Reshape back to (batch, num_demos, feat_c)
        pair_features = pair_features.reshape(batch_size, num_demos, self.feat_c)
        
        if self.demo_agg_mode == 'flatten':
            pair_features = pair_features.reshape(batch_size, -1)
        else:
            pair_features = pair_features.reshape(batch_size, num_demos, -1).mean(dim=1)
        embeddings = self.demo_agg(pair_features)
        
        return embeddings
