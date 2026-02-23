import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

class Linear(nn.Module):
    """
    Linear transformation layer: y = xW^T
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.T

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model))
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        # Learnable scale parameter (gamma)
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight

def softmax(x: Tensor, dim: int = -1) -> Tensor:
    x_max = x.max(dim=dim, keepdim=True).values
    x_shifted = x - x_max
    exp_x = torch.exp(x_shifted)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.theta = theta

        inv_freq = 1.0 / (theta ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

        self._precompute_cache(max_seq_len)

    def _precompute_cache(self, seq_len: int):
        positions = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", torch.cos(emb), persistent=False)
        self.register_buffer("sin_cached", torch.sin(emb), persistent=False)

    def _rotate_half(self, x: Tensor) -> Tensor:
        d = x.shape[-1] // 2
        x1 = x[..., :d]
        x2 = x[..., d:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        if x.dim() == 4:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        return x * cos + self._rotate_half(x) * sin


def apply_rope(x: Tensor, d_model: int, theta: float, max_seq_len: int, token_positions: Tensor) -> Tensor:
    """
    Functional interface for applying RoPE.
    """
    rope = RotaryPositionEmbedding(d_model, max_seq_len, theta)
    rope = rope.to(x.device)
    return rope(x, token_positions)

def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Optional[Tensor] = None,
) -> Tensor:
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))

    weights = softmax(scores, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)
    return weights @ V

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> Tensor:
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        return mask

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        mask = self._create_causal_mask(seq_len, x.device)
        attn_out = scaled_dot_product_attention(Q, K, V, mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.output_proj(attn_out)

class MultiHeadSelfAttentionWithRoPE(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
        self.rope = RotaryPositionEmbedding(self.d_k, max_seq_len, theta)

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> Tensor:
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        return mask

    def forward(self, x: Tensor, token_positions: Optional[Tensor] = None) -> Tensor:
        batch_size, seq_len, _ = x.shape
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)
        mask = self._create_causal_mask(seq_len, x.device)
        attn_out = scaled_dot_product_attention(Q, K, V, mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.output_proj(attn_out)

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float = 10000.0,
        eps: float = 1e-5,
    ):
        super().__init__()

        self.ln1 = RMSNorm(d_model, eps)
        self.ln2 = RMSNorm(d_model, eps)
        self.attn = MultiHeadSelfAttentionWithRoPE(d_model, num_heads, max_seq_len, theta)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: Tensor, token_positions: Optional[Tensor] = None) -> Tensor:
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        eps: float = 1e-5,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers

        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, eps)
            for _ in range(num_layers)
        ])
        self.final_ln = RMSNorm(d_model, eps)
        self.output = Linear(d_model, vocab_size)

    def forward(self, token_ids: Tensor, token_positions: Optional[Tensor] = None) -> Tensor:
        batch_size, seq_len = token_ids.shape
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x, token_positions)
        x = self.final_ln(x)
        return self.output(x)

    def load_weights(self, state_dict: dict):
        if "token_embeddings.weight" in state_dict:
            self.token_embeddings.weight.data.copy_(state_dict["token_embeddings.weight"])

        if "output.weight" in state_dict:
            self.output.weight.data.copy_(state_dict["output.weight"])

        if "final_ln.weight" in state_dict:
            self.final_ln.weight.data.copy_(state_dict["final_ln.weight"])

        for layer_idx, layer in enumerate(self.layers):
            prefix = f"layers.{layer_idx}"

            if f"{prefix}.ln1.weight" in state_dict:
                layer.ln1.weight.data.copy_(state_dict[f"{prefix}.ln1.weight"])
            if f"{prefix}.ln2.weight" in state_dict:
                layer.ln2.weight.data.copy_(state_dict[f"{prefix}.ln2.weight"])

            if f"{prefix}.attn.q_proj.weight" in state_dict:
                layer.attn.q_proj.weight.data.copy_(state_dict[f"{prefix}.attn.q_proj.weight"])
            if f"{prefix}.attn.k_proj.weight" in state_dict:
                layer.attn.k_proj.weight.data.copy_(state_dict[f"{prefix}.attn.k_proj.weight"])
            if f"{prefix}.attn.v_proj.weight" in state_dict:
                layer.attn.v_proj.weight.data.copy_(state_dict[f"{prefix}.attn.v_proj.weight"])
            if f"{prefix}.attn.output_proj.weight" in state_dict:
                layer.attn.output_proj.weight.data.copy_(state_dict[f"{prefix}.attn.output_proj.weight"])

            if f"{prefix}.ffn.w1.weight" in state_dict:
                layer.ffn.w1.weight.data.copy_(state_dict[f"{prefix}.ffn.w1.weight"])
            if f"{prefix}.ffn.w2.weight" in state_dict:
                layer.ffn.w2.weight.data.copy_(state_dict[f"{prefix}.ffn.w2.weight"])
            if f"{prefix}.ffn.w3.weight" in state_dict:
                layer.ffn.w3.weight.data.copy_(state_dict[f"{prefix}.ffn.w3.weight"])

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_flops_per_token(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
) -> int:
    attn_macs = (
        3 * d_model * d_model        # Q, K, V projections
        + 2 * context_length * d_model  # QK^T and AV
        + d_model * d_model           # output projection
    )
    ffn_macs = 3 * d_model * d_ff
    per_layer_macs = attn_macs + ffn_macs
    output_macs = d_model * vocab_size
    total_macs = num_layers * per_layer_macs + output_macs
    return 2 * total_macs


def estimate_memory_bytes(
    vocab_size: int,
    d_model: int,
    num_layers: int,
    d_ff: int,
    dtype_bytes: int = 4,  # float32 = 4 bytes
) -> int:
    params = (
        vocab_size * d_model  # token_embeddings
        + vocab_size * d_model  # output projection
        + d_model  # final layer norm
    )
    per_layer_params = (
        2 * d_model              # ln1, ln2
        + 4 * d_model * d_model  # Q, K, V, O projections
        + 3 * d_model * d_ff     # w1, w2, w3
    )

    params += num_layers * per_layer_params

    return params * dtype_bytes
