"""
src/models/embeddings.py

Thực hành Milestone M5:
Các tầng Embedding cho Kiến trúc Causal Transformer Decoder:
- TokenEmbedding: Chiếu Token ID (số nguyên) thành vector đặc trưng liên tục R^{d_model}.
- PositionalEmbedding: Mã hóa vị trí thứ tự của token trong chuỗi (0 đến context_length - 1).
- TransformerEmbedding: Kết hợp Token Embedding và Positional Embedding kèm Dropout.
"""

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Ma trận Embedding Từ vựng W_E: shape (vocab_size, d_model).
    Mỗi hàng đại diện cho một token trong từ vựng.
    """
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.lut = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch_Size, Context_Length) -> (Batch_Size, Context_Length, d_model)
        return self.lut(x)


class PositionalEmbedding(nn.Module):
    """
    Mã hóa vị trí học được (Learned Positional Embedding) tương tự GPT-2:
    W_P: shape (context_length, d_model).
    Cung cấp thông tin về vị trí tương đối và tuyệt đối của mỗi từ trong câu.
    """
    def __init__(self, context_length: int, d_model: int):
        super().__init__()
        self.lut = nn.Embedding(context_length, d_model)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(seq_len, dtype=torch.long, device=device)
        # positions: (seq_len,) -> (1, seq_len, d_model)
        return self.lut(positions).unsqueeze(0)


class TransformerEmbedding(nn.Module):
    """Kết hợp Token Embedding và Positional Embedding."""
    def __init__(self, vocab_size: int, d_model: int, context_length: int, dropout: float = 0.1):
        super().__init__()
        self.token_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEmbedding(context_length, d_model)
        self.drop = nn.Dropout(dropout)
        self.context_length = context_length

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (Batch_Size, Seq_Len)
        B, T = input_ids.shape
        assert T <= self.context_length, f"Độ dài chuỗi {T} vượt quá context_length {self.context_length}"

        tok_embeddings = self.token_emb(input_ids)
        pos_embeddings = self.pos_emb(T, input_ids.device)

        # Tổng hợp embedding: X = Token_Emb + Pos_Emb
        x = tok_embeddings + pos_embeddings
        return self.drop(x)
