"""
src/models/transformer.py

Khối Transformer Block và các thành phần cốt lõi:
- RMSNorm: Chuẩn hóa Root Mean Square tối ưu chi phí CPU.
- FeedForward: Khối MLP 2 tầng với hàm kích hoạt GELU.
- TransformerBlock: Cấu trúc Pre-Norm (Pre-RMSNorm) kết hợp Residual Connections.
"""

from typing import Optional, Any, Tuple, Union
import torch
import torch.nn as nn
from src.models.attention import CausalSelfAttention


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (Zhang & Sennrich, 2019):
      RMSNorm(x) = (x / sqrt(mean(x^2) + eps)) * gamma
    Tiết kiệm 30-50% chi phí tính toán so với LayerNorm do không cần tính mean.
    """
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_model)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.eps)
        return x_norm * self.weight


class FeedForward(nn.Module):
    """
    Khối Mạng Nơ-ron Truyền thẳng (FFN / MLP) trong Transformer Block:
      FFN(x) = Linear_2(GELU(Linear_1(x)))
    Thường mở rộng chiều ẩn lên d_ff = 4 * d_model.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, bias: bool = False):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=bias)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    """
    Một khối Transformer Decoder hoàn chỉnh theo kiến trúc Pre-Norm (Pre-RMSNorm):
      x = x + Attention(RMSNorm(x))
      x = x + FeedForward(RMSNorm(x))
    Đường tắt Residual Stream giữ nguyên vẹn luồng thông tin và gradient xuyên suốt mạng.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        context_length: int,
        d_ff: int,
        dropout: float = 0.1,
        bias: bool = False
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            context_length=context_length,
            dropout=dropout,
            bias=bias
        )
        self.norm2 = RMSNorm(d_model)
        self.ffn = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            bias=bias
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Any] = None,
        layer_idx: Optional[int] = None
    ) -> torch.Tensor:
        # 1. Attention với Pre-Norm và Residual Connection (kèm hỗ trợ KV Cache)
        x = x + self.attn(self.norm1(x), kv_cache=kv_cache, layer_idx=layer_idx)
        # 2. Feed-Forward với Pre-Norm và Residual Connection
        x = x + self.ffn(self.norm2(x))
        return x
