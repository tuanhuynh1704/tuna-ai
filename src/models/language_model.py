"""
src/models/language_model.py

Kiến trúc Mô hình Ngôn ngữ Hoàn chỉnh (TunaLM / Mini GPT):
- Autoregressive Decoder-only Transformer.
- Kết hợp Embedding, N khối Transformer Blocks, Final RMSNorm và LM Head.
- Tính toán Cross-Entropy Loss trực tiếp cho Next-Token Prediction.
- Tích hợp Hàm sinh văn bản tự hồi quy (Autoregressive Text Generation).
"""

from typing import Tuple, Optional, Dict, Any
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.embeddings import TransformerEmbedding
from src.models.transformer import TransformerBlock, RMSNorm


class TunaLMConfig:
    """Cấu hình tham số cho mô hình ngôn ngữ Tuna-AI."""
    def __init__(
        self,
        vocab_size: int = 4096,
        context_length: int = 128,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        bias: bool = False,
        tie_weights: bool = True
    ):
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.bias = bias
        self.tie_weights = tie_weights

    @classmethod
    def from_json(cls, json_path: str) -> "TunaLMConfig":
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            vocab_size=data.get("vocab_size", 4096),
            context_length=data.get("context_length", 128),
            d_model=data.get("d_model", 64),
            n_layers=data.get("n_layers", 2),
            n_heads=data.get("n_heads", 2),
            d_ff=data.get("d_ff", 256),
            dropout=data.get("dropout", 0.1),
            bias=data.get("bias", False),
            tie_weights=data.get("tie_weights", True)
        )


class TunaLM(nn.Module):
    """Mô hình Ngôn ngữ Lớn Song ngữ Cốt lõi của Dự án Tuna-AI."""

    def __init__(self, config: TunaLMConfig):
        super().__init__()
        self.config = config

        # 1. Tầng Embedding (Token + Learned Position)
        self.embedding = TransformerEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            context_length=config.context_length,
            dropout=config.dropout
        )

        # 2. Chuỗi các khối Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                context_length=config.context_length,
                d_ff=config.d_ff,
                dropout=config.dropout,
                bias=config.bias
            )
            for _ in range(config.n_layers)
        ])

        # 3. Tầng Chuẩn hóa Cuối cùng (Final RMSNorm)
        self.final_norm = RMSNorm(config.d_model)

        # 4. Đầu chiếu Ngôn ngữ (Language Modeling Head: d_model -> vocab_size)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight Tying (Chia sẻ trọng số giữa Token Embedding và LM Head)
        if config.tie_weights:
            self.lm_head.weight = self.embedding.token_emb.lut.weight

        # Khởi tạo trọng số ngẫu nhiên chuẩn
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_parameters(self) -> int:
        """Đếm tổng số tham số học được của mô hình."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass của mô hình:
        input_ids: (Batch_Size, Seq_Len)
        targets: (Batch_Size, Seq_Len) (tùy chọn)
        
        Trả về:
          - logits: (Batch_Size, Seq_Len, vocab_size)
          - loss: Cross-Entropy loss (nếu có targets)
        """
        B, T = input_ids.shape

        # 1. Embedding
        x = self.embedding(input_ids)

        # 2. Truyền qua N khối Transformer Block
        for block in self.blocks:
            x = block(x)

        # 3. Chuẩn hóa cuối
        x = self.final_norm(x)

        # 4. Tạo Logits cho toàn bộ từ điển
        logits = self.lm_head(x)

        # 5. Tính Loss nếu có targets
        loss = None
        if targets is not None:
            # Next-token prediction: Logits ở vị trí t dự đoán target ở vị trí t+1
            shift_logits = logits[..., :-1, :].contiguous()
            shift_targets = targets[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_targets.view(-1),
                ignore_index=-100
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: Optional[int] = 50
    ) -> torch.Tensor:
        """
        Thuật toán sinh văn bản tự hồi quy (Autoregressive Text Generation):
        Lặp lại: Nhận chuỗi token -> Dự đoán token tiếp theo -> Nối vào chuỗi -> Lặp lại.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Nếu chuỗi dài hơn context_length, cắt bớt chỉ lấy context_length token gần nhất
            idx_cond = input_ids if input_ids.size(1) <= self.config.context_length else input_ids[:, -self.config.context_length:]

            # Forward qua mô hình
            logits, _ = self(idx_cond)

            # Lấy logits của token ở vị trí cuối cùng: (B, vocab_size)
            logits_last = logits[:, -1, :]

            if temperature == 0.0:
                # Greedy decoding (chọn xác suất cao nhất)
                idx_next = torch.argmax(logits_last, dim=-1, keepdim=True)
            else:
                # Chia cho temperature để điều chỉnh độ ngẫu nhiên
                logits_last = logits_last / temperature

                # Top-K filtering
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits_last, min(top_k, logits_last.size(-1)))
                    logits_last[logits_last < v[:, [-1]]] = float('-inf')

                probs = F.softmax(logits_last, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            # Nối token mới sinh vào chuỗi đầu vào
            input_ids = torch.cat((input_ids, idx_next), dim=1)

        return input_ids
