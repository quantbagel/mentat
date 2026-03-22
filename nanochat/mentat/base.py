"""Base classes for Mentat compiled algorithmic modules.

A CompiledModule defines an algorithm as:
  1. A set of typed tokens for the execution trace
  2. Attention weights that select the right tokens by type
  3. FFN weights that implement the step function (truth table / lookup)
  4. Output dimensions that hold the exact transition result

The current Mentat scope is standalone compiled kernels: a CompiledBlock is a
frozen transformer block with type-selective 2D attention and a windowed causal
mask. Integration into NanoChat's main residual stream is deferred for now.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CompiledWeights:
    """All weight tensors needed for a compiled block."""
    qkv: torch.Tensor          # (3*d_model, d_model) — attention Q/K/V
    out_proj: torch.Tensor      # (d_model, d_model)   — attention output
    ffn_up_w: torch.Tensor      # (hidden, d_model)    — FFN up weight
    ffn_up_b: torch.Tensor      # (hidden,)            — FFN up bias
    ffn_down_w: torch.Tensor    # (d_model, hidden)    — FFN down weight
    ffn_down_b: torch.Tensor    # (d_model,)           — FFN down bias
    embeddings: dict[int, torch.Tensor]  # token_id → embedding vector (compiled dims only)
    head_weights: dict[int, tuple[torch.Tensor, float]]  # token_id → (weight_row, bias)


class CompiledModule(ABC):
    """Base class for compiling an algorithm into transformer weights.

    Subclass this and implement the abstract methods to define your algorithm.
    The framework handles the rest: building the standalone frozen block and
    exposing the token embeddings needed to drive it.

    Example (binary full adder):
        class BinaryAdder(CompiledModule):
            compiled_dims = 10
            window_size = 3
            ...
    """

    @property
    @abstractmethod
    def compiled_dims(self) -> int:
        """Number of dimensions used by the compiled subspace."""
        ...

    @property
    @abstractmethod
    def window_size(self) -> int:
        """Attention window size (how many positions back to look)."""
        ...

    @property
    @abstractmethod
    def n_active_heads(self) -> int:
        """Number of attention heads with compiled weights (rest are zeroed)."""
        ...

    @property
    @abstractmethod
    def ffn_hidden(self) -> int:
        """Hidden dimension of the compiled FFN."""
        ...

    @abstractmethod
    def compile(self, d_model: int, n_heads: int) -> CompiledWeights:
        """Produce all weight tensors for the compiled block.

        Args:
            d_model: full model dimension (compiled_dims is a subset)
            n_heads: total number of 2D attention heads (d_model // 2)

        Returns:
            CompiledWeights with all tensors sized for (d_model, n_heads).
        """
        ...

    @abstractmethod
    def format_trace(self, *args) -> list[int]:
        """Generate execution trace tokens for training data."""
        ...


class CompiledBlock(nn.Module):
    """A frozen transformer block with compiled weights.

    No LayerNorm (compiled weights are calibrated for raw values).
    Uses a windowed causal mask for constant-lookback attention.
    All parameters are frozen after compilation.
    """

    def __init__(self, module: CompiledModule, d_model: int):
        super().__init__()
        n_heads = d_model // 2
        self.n_heads = n_heads
        self.window = module.window_size

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.ffn_up = nn.Linear(d_model, module.ffn_hidden, bias=True)
        self.ffn_down = nn.Linear(module.ffn_hidden, d_model, bias=True)

        # Compile and freeze
        weights = module.compile(d_model, n_heads)
        self.qkv.weight.data.copy_(weights.qkv)
        self.out_proj.weight.data.copy_(weights.out_proj)
        self.ffn_up.weight.data.copy_(weights.ffn_up_w)
        self.ffn_up.bias.data.copy_(weights.ffn_up_b)
        self.ffn_down.weight.data.copy_(weights.ffn_down_w)
        self.ffn_down.bias.data.copy_(weights.ffn_down_b)

        for p in self.parameters():
            p.requires_grad_(False)

        # Store embedding/head info for standalone experiments or future integration.
        self.compiled_embeddings = weights.embeddings
        self.compiled_head = weights.head_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        device = x.device

        # 2D-head attention
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, 2).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(2)

        # Windowed causal mask
        causal = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), 1)
        window = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool), -self.window)
        scores = scores.masked_fill(causal | window, float("-inf"))

        attn_out = (F.softmax(scores, -1) @ v).transpose(1, 2).contiguous().reshape(B, T, C)
        attn_out = self.out_proj(attn_out)
        x = x + attn_out

        # Truth-table FFN
        h = F.relu(self.ffn_up(x))
        x = x + self.ffn_down(h)
        return x


def apply_residual_bypass(
    x: torch.Tensor, raw_emb: torch.Tensor, compiled_dims: int
) -> torch.Tensor:
    """Replace the compiled subspace with raw token embeddings.

    This helper is kept for future integration experiments. It is not part of
    the current standalone Mentat path.
    """
    return torch.cat([raw_emb[:, :, :compiled_dims], x[:, :, compiled_dims:]], dim=-1)
