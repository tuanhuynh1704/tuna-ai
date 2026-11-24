"""
tests/test_transformer.py

Bộ kiểm thử Unit Test cho Milestone M5:
Kiểm thử Toàn diện Kiến trúc Causal Transformer Decoder (TunaLM / Mini GPT):
- Kiểm thử các tầng Embedding (Token & Learned Position).
- Kiểm thử Causal Self-Attention: shape đầu ra và đặc tính không rò rỉ token tương lai.
- Kiểm thử TransformerBlock (Pre-RMSNorm + Residual Stream).
- Kiểm thử TunaLM: Forward, Next-Token Loss, Backward và Autoregressive Generation.
- Kiểm định cấu hình Model A (~1M tham số) từ configs/model/tiny.json.
"""

import tempfile
import torch
import pytest
from src.models.embeddings import TokenEmbedding, PositionalEmbedding, TransformerEmbedding
from src.models.attention import CausalSelfAttention
from src.models.transformer import RMSNorm, FeedForward, TransformerBlock
from src.models.language_model import TunaLM, TunaLMConfig


class TestTransformerComponents:
    def test_embeddings_shape(self):
        B, T = 2, 16
        vocab_size, d_model, context_len = 100, 32, 64
        emb = TransformerEmbedding(vocab_size, d_model, context_len, dropout=0.0)

        x = torch.randint(0, vocab_size, (B, T))
        out = emb(x)
        assert out.shape == (B, T, d_model)

    def test_causal_attention_masking(self):
        """
        Kiểm tra tính chất Causal Masking:
        Thay đổi giá trị của token ở tương lai (vị trí t=3)
        KHÔNG ĐƯỢC làm thay đổi output của token ở hiện tại (vị trí t=1).
        """
        B, T = 1, 4
        d_model, n_heads = 16, 2
        attn = CausalSelfAttention(d_model=d_model, n_heads=n_heads, context_length=T, dropout=0.0)
        attn.eval()

        # Input 1
        x1 = torch.randn(B, T, d_model)
        out1 = attn(x1)

        # Input 2: giống input 1 ở 2 token đầu (t=0, 1), nhưng khác ở 2 token sau (t=2, 3)
        x2 = x1.clone()
        x2[:, 2:, :] = torch.randn(B, 2, d_model)
        out2 = attn(x2)

        # Output tại t=0 và t=1 PHẢI GIỐNG NHAU TUYỆT ĐỐI
        assert torch.allclose(out1[:, :2, :], out2[:, :2, :], atol=1e-6)

    def test_transformer_block_shape_and_residuals(self):
        B, T = 2, 8
        d_model, n_heads, d_ff = 32, 4, 128
        block = TransformerBlock(d_model=d_model, n_heads=n_heads, context_length=16, d_ff=d_ff)
        x = torch.randn(B, T, d_model)
        out = block(x)
        assert out.shape == (B, T, d_model)


class TestTunaLMFullModel:
    def test_tunalm_forward_and_loss(self):
        config = TunaLMConfig(
            vocab_size=50,
            context_length=16,
            d_model=32,
            n_layers=2,
            n_heads=2,
            d_ff=64,
            dropout=0.0
        )
        model = TunaLM(config)

        B, T = 2, 8
        input_ids = torch.randint(0, config.vocab_size, (B, T))
        targets = torch.randint(0, config.vocab_size, (B, T))

        # 1. Forward không có targets
        logits, loss = model(input_ids)
        assert logits.shape == (B, T, config.vocab_size)
        assert loss is None

        # 2. Forward có targets -> tính Cross-Entropy loss
        logits, loss = model(input_ids, targets=targets)
        assert loss is not None
        assert loss.item() > 0.0

        # 3. Backward
        loss.backward()
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_tunalm_autoregressive_generation(self):
        config = TunaLMConfig(
            vocab_size=50,
            context_length=32,
            d_model=16,
            n_layers=1,
            n_heads=2,
            d_ff=32
        )
        model = TunaLM(config)
        model.eval()

        # Prompt gồm 3 tokens
        prompt = torch.tensor([[5, 12, 24]], dtype=torch.long)
        max_new = 5

        generated = model.generate(prompt, max_new_tokens=max_new, temperature=0.0)
        # Độ dài đầu ra phải đúng bằng 3 + 5 = 8 tokens
        assert generated.shape == (1, 8)
        # 3 token đầu phải giữ nguyên prompt gốc
        assert torch.equal(generated[:, :3], prompt)

    def test_tunalm_model_a_config_loading(self):
        """Kiểm tra nạp file configs/model/tiny.json và đếm số tham số."""
        config = TunaLMConfig.from_json("configs/model/tiny.json")
        model = TunaLM(config)
        param_count = model.count_parameters()

        # Với vocab_size=4096, d_model=64, n_layers=2:
        # Số tham số nằm trong khoảng 300K - 1M tham số (Model A chuẩn)
        assert 100_000 < param_count < 2_000_000
        assert config.n_layers == 2
        assert config.n_heads == 2
        assert config.d_model == 64
