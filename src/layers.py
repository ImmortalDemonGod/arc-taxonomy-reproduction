import math

import torch
from torch import Tensor, nn
from jaxtyping import Float
from .utils import softmax as _softmax


class Linear(nn.Module):
    """
    Minimal from-scratch Linear layer implementing y = x @ W^T (+ b).

    - Weight shape: (out_features, in_features)
    - Bias is optional and disabled by default for parity with tests that only provide weights.
    - Parameter initialization follows torch.nn.Linear: uniform(-1/sqrt(fan_in), 1/sqrt(fan_in)).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty((out_features,)))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        fan_in = max(1, self.in_features)
        bound = 1.0 / math.sqrt(fan_in)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, in_features: Float[Tensor, " ... in_features"]) -> Float[Tensor, " ... out_features"]:
        y = in_features.matmul(self.weight.t())
        if self.bias is not None:
            y = y + self.bias
        return y


class Embedding(nn.Module):
    """
    Simple token embedding layer.

    - Weight shape: (num_embeddings, embedding_dim)
    - Forward performs table lookup: weight[token_ids]
    - Initialization: truncated normal N(0, 1) clipped to [-3, 3]
    """

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)

        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Truncated normal with mean=0, std=1, a=-3, b=3
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: Tensor) -> Float[Tensor, " ... embedding_dim"]:
        return self.weight[token_ids]


def silu(in_features: Tensor) -> Tensor:
    """SiLU activation: x * sigmoid(x)."""
    return in_features * torch.sigmoid(in_features)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization without bias.

    y = (x / sqrt(mean(x^2) + eps)) * weight
    where reduction is over the last dimension (feature dimension).
    """

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.d_model))

    def forward(self, in_features: Tensor) -> Tensor:
        x = in_features
        # Compute rms over last dim
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        y = x / rms
        return y * self.weight


class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward block:
      out = W2( SiLU(W1(x)) * W3(x) )
    Shapes:
      - W1: (d_ff, d_model)
      - W3: (d_ff, d_model)
      - W2: (d_model, d_ff)
    """

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.d_ff = int(d_ff)
        # Use our minimal Linear layers without bias
        self.w1 = Linear(in_features=self.d_model, out_features=self.d_ff, bias=False)
        self.w2 = Linear(in_features=self.d_ff, out_features=self.d_model, bias=False)
        self.w3 = Linear(in_features=self.d_model, out_features=self.d_ff, bias=False)

    def forward(self, in_features: Tensor) -> Tensor:
        a = self.w1(in_features)
        b = self.w3(in_features)
        gated = silu(a) * b
        return self.w2(gated)


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: torch.Tensor | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Compute scaled dot-product attention.

    Args:
        Q: Query tensor of shape (..., queries, d_k)
        K: Key tensor of shape   (..., keys,   d_k)
        V: Value tensor of shape (..., values, d_v) with values == keys
        mask: Optional boolean mask of shape (..., queries, keys). True indicates keep; False indicates mask out.

    Returns:
        Tensor of shape (..., queries, d_v)
    """
    q = Q.float()
    k = K.float()
    v = V.float()
    d_k = q.shape[-1]
    scale = 1.0 / math.sqrt(max(1, d_k))
    # (..., queries, keys)
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    attn = _softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    return out.to(V.dtype)


def multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    in_features: Tensor,
) -> Tensor:
    """
    Optimized batched Multi-Head Self-Attention (no RoPE).

    Args:
        d_model: embedding dimension
        num_heads: number of attention heads (divides d_model)
        *_proj_weight: concatenated projection weights for all heads, shape (d_model, d_model)
        in_features: tensor of shape (..., seq_len, d_model)

    Returns:
        Tensor of shape (..., seq_len, d_model)
    """
    head_dim = d_model // num_heads
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

    orig_leading = in_features.shape[:-2]
    seq_len = in_features.shape[-2]

    x = in_features.reshape(-1, seq_len, d_model)

    # Projections for all heads in single matmuls
    q_all = x.matmul(q_proj_weight.t())  # (B, S, d_model)
    k_all = x.matmul(k_proj_weight.t())  # (B, S, d_model)
    v_all = x.matmul(v_proj_weight.t())  # (B, S, d_model)

    # Reshape to heads: (B, H, S, D)
    def to_heads(t: Tensor) -> Tensor:
        # Weights are stacked by heads along the output rows: (H*D, d_model).
        # After projecting to (B, S, H*D), split into heads then move heads before seq.
        return t.view(t.shape[0], seq_len, num_heads, head_dim).transpose(1, 2).contiguous()

    q = to_heads(q_all)
    k = to_heads(k_all)
    v = to_heads(v_all)

    # Scaled dot-product attention per head with causal mask
    causal = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device)).view(1, 1, seq_len, seq_len)
    context = scaled_dot_product_attention(q, k, v, mask=causal)  # (B, H, S, D)

    # Combine heads: (B, S, H*D=d_model)
    context = context.transpose(1, 2).contiguous().view(context.shape[0], seq_len, d_model)

    # Output projection
    out = context.matmul(o_proj_weight.t())  # (B, S, d_model)

    return out.reshape(*orig_leading, seq_len, d_model)


def transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict,
    in_features: Tensor,
) -> Tensor:
    """
    Pre-norm Transformer block with RoPE:
      x = x + MHA(RMSNorm(x))
      x = x + FFN(RMSNorm(x))
    """
    x = in_features
    seq_len = x.shape[-2]
    device = x.device

    # First RMSNorm
    ln1 = RMSNorm(d_model=d_model, eps=1e-5)
    with torch.no_grad():
        ln1.weight.copy_(weights["ln1.weight"])  # (d_model,)
    x_norm = ln1(x)

    # Positions for RoPE
    pos = torch.arange(seq_len, device=device).view(1, -1)

    # Multi-head self-attention with RoPE
    attn_out = multihead_self_attention_with_rope(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        theta=theta,
        q_proj_weight=weights["attn.q_proj.weight"],
        k_proj_weight=weights["attn.k_proj.weight"],
        v_proj_weight=weights["attn.v_proj.weight"],
        o_proj_weight=weights["attn.output_proj.weight"],
        in_features=x_norm,
        token_positions=pos,
    )
    x = x + attn_out

    # Second RMSNorm
    ln2 = RMSNorm(d_model=d_model, eps=1e-5)
    with torch.no_grad():
        ln2.weight.copy_(weights["ln2.weight"])  # (d_model,)
    x_norm2 = ln2(x)

    # SwiGLU FFN using provided weights
    W1 = weights["ffn.w1.weight"]  # (d_ff, d_model)
    W2 = weights["ffn.w2.weight"]  # (d_model, d_ff)
    W3 = weights["ffn.w3.weight"]  # (d_ff, d_model)

    a = x_norm2.matmul(W1.t())
    b = x_norm2.matmul(W3.t())
    gated = silu(a) * b
    ffn_out = gated.matmul(W2.t())

    x = x + ffn_out
    return x


def rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Tensor,
    token_positions: Tensor,
) -> Tensor:
    """
    Apply Rotary Positional Embedding (RoPE) to queries or keys.

    Args:
        d_k: embedding dimension (must be even)
        theta: RoPE base (e.g., 10000.0)
        max_seq_len: maximum sequence length (unused at runtime; kept for API parity)
        in_query_or_key: tensor of shape (..., seq_len, d_k)
        token_positions: tensor of positions of shape (..., seq_len) or (seq_len,)

    Returns:
        Tensor of same shape as in_query_or_key with RoPE applied along the last dim.
    """
    x = in_query_or_key
    assert d_k % 2 == 0, "RoPE requires even embedding dimension"
    half = d_k // 2

    # Prepare inverse frequencies
    i = torch.arange(0, half, device=x.device, dtype=x.dtype)
    inv_freq = 1.0 / (torch.tensor(theta, dtype=x.dtype, device=x.device) ** (i / half))

    # Broadcast positions
    pos = token_positions.to(device=x.device, dtype=x.dtype)
    if pos.dim() == 1:
        pos = pos.view(1, -1)
    # Align pos to have trailing seq_len dimension
    while pos.dim() < x.dim() - 1:
        pos = pos.unsqueeze(0)

    # Compute angles and trig
    angles = pos.unsqueeze(-1) * inv_freq  # (..., seq_len, half)
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    # Split even/odd components
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    x_rot_even = x_even * cos - x_odd * sin
    x_rot_odd = x_odd * cos + x_even * sin

    out = torch.empty_like(x)
    out[..., 0::2] = x_rot_even
    out[..., 1::2] = x_rot_odd
    return out


def transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict,
    in_indices: Tensor,
) -> Tensor:
    """
    Transformer language model forward pass using provided state dict weights.

    Pipeline:
      - Token embedding lookup
      - N x [pre-norm block with RoPE-attn + SwiGLU FFN]
      - Final RMSNorm
      - LM head projection to vocab size
    """
    # Embedding lookup
    token_emb = weights["token_embeddings.weight"]  # (vocab_size, d_model)
    x = token_emb[in_indices]  # (B, S, d_model)

    # Apply Transformer blocks
    for i in range(int(num_layers)):
        block_weights = {
            "ln1.weight": weights[f"layers.{i}.ln1.weight"],
            "attn.q_proj.weight": weights[f"layers.{i}.attn.q_proj.weight"],
            "attn.k_proj.weight": weights[f"layers.{i}.attn.k_proj.weight"],
            "attn.v_proj.weight": weights[f"layers.{i}.attn.v_proj.weight"],
            "attn.output_proj.weight": weights[f"layers.{i}.attn.output_proj.weight"],
            "ln2.weight": weights[f"layers.{i}.ln2.weight"],
            "ffn.w1.weight": weights[f"layers.{i}.ffn.w1.weight"],
            "ffn.w2.weight": weights[f"layers.{i}.ffn.w2.weight"],
            "ffn.w3.weight": weights[f"layers.{i}.ffn.w3.weight"],
        }
        x = transformer_block(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            max_seq_len=context_length,
            theta=rope_theta,
            weights=block_weights,
            in_features=x,
        )

    # Final RMSNorm
    ln_final = RMSNorm(d_model=d_model, eps=1e-5)
    with torch.no_grad():
        ln_final.weight.copy_(weights["ln_final.weight"])  # (d_model,)
    x = ln_final(x)

    # LM head projection
    lm_head_w = weights["lm_head.weight"]  # (vocab_size, d_model)
    logits = x.matmul(lm_head_w.t())  # (B, S, vocab_size)
    return logits


def multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    in_features: Tensor,
    token_positions: Tensor,
) -> Tensor:
    """
    Batched Multi-Head Self-Attention with RoPE (causal, no dropout).
    """
    head_dim = d_model // num_heads
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

    orig_leading = in_features.shape[:-2]
    seq_len = in_features.shape[-2]

    x = in_features.reshape(-1, seq_len, d_model)

    # Projections
    q_all = x.matmul(q_proj_weight.t())
    k_all = x.matmul(k_proj_weight.t())
    v_all = x.matmul(v_proj_weight.t())

    def to_heads(t: Tensor) -> Tensor:
        return t.view(t.shape[0], seq_len, num_heads, head_dim).transpose(1, 2).contiguous()

    q = to_heads(q_all)
    k = to_heads(k_all)
    v = to_heads(v_all)

    # Apply RoPE to Q and K per head
    q = rope(d_k=head_dim, theta=theta, max_seq_len=max_seq_len, in_query_or_key=q, token_positions=token_positions)
    k = rope(d_k=head_dim, theta=theta, max_seq_len=max_seq_len, in_query_or_key=k, token_positions=token_positions)

    # Causal mask and attention
    causal = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device)).view(1, 1, seq_len, seq_len)
    context = scaled_dot_product_attention(q, k, v, mask=causal)

    # Merge heads
    context = context.transpose(1, 2).contiguous().view(context.shape[0], seq_len, d_model)
    out = context.matmul(o_proj_weight.t())
    return out.reshape(*orig_leading, seq_len, d_model)
