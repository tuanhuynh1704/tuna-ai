"""
src/models/attention.py

Causal Multi-Head Self-Attention (MHA) kèm Causal Masking:
- Chiếu tuyến tính đầu vào thành Q (Query), K (Key), V (Value).
- Phân tách thành n_heads đầu chú ý độc lập.
- Scaled Dot-Product Attention: softmax((Q K^T) / sqrt(d_k) + CausalMask) V.
- Bảo đảm tính tự hồi quy (Autoregressive): mỗi token chỉ được tương tác với quá khứ.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, context_length: int, dropout: float = 0.1, bias: bool = False):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) phải chia hết cho n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)

        # 3 ma trận chiếu Q, K, V gộp chung thành 1 Linear duy nhất để tối ưu tính toán
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=bias)
        # Ma trận chiếu đầu ra (Output Projection)
        self.c_proj = nn.Linear(d_model, d_model, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Tạo Causal Mask tam giác trên và đăng ký vào buffer (không phải tham số học)
        mask = torch.triu(torch.full((context_length, context_length), float('-inf')), diagonal=1)
        self.register_buffer("mask", mask.view(1, 1, context_length, context_length))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # Batch_Size, Context_Length, d_model

        # 1. Tính Q, K, V đồng thời: (B, T, 3 * d_model)
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # 2. Chuyển đổi sang Multi-Head: (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # 3. Scaled Dot-Product Attention: S = (Q @ K^T) * scale
        # q: (B, H, T, d_k), k.transpose(-2, -1): (B, H, d_k, T) -> att: (B, H, T, T)
        att = (q @ k.transpose(-2, -1)) * self.scale

        # 4. Áp dụng Causal Mask
        att = att + self.mask[:, :, :T, :T]

        # 5. Softmax và Dropout
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # 6. Nhân với Value V: (B, H, T, T) @ (B, H, T, d_head) -> (B, H, T, d_head)
        y = att @ v

        # 7. Ghép các heads trở lại hình dạng chuẩn: (B, T, d_model)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 8. Chiếu đầu ra (Output Projection)
        return self.resid_dropout(self.c_proj(y))
