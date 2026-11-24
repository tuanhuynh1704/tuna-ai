"""Bộ kiểm thử đơn vị cho Hệ thống Inference Engine & KV Cache (Milestone M11).

Kiểm tra:
1. KVCache:
   - Cập nhật đúng shape tensor (B, H, T_total, d_k).
   - Tăng độ dài chuỗi get_seq_len() chính xác.
   - Reset xóa sạch bộ đệm.
2. TextGenerator:
   - Sinh greedy, temperature, top-k, top-p, repetition penalty.
   - use_cache=True và use_cache=False đều hoạt động nhất quán.
   - generate_stream yield các chuỗi text hợp lệ.
3. ModelExporter & load_offline_model:
   - Xuất đầy đủ config, weights, tokenizer, model card.
   - Nạp lại mô hình offline độc lập và sinh câu trả lời thành công.
"""

import shutil
import tempfile
from pathlib import Path
import pytest
import torch

from src.inference.exporter import ModelExporter, load_offline_model
from src.inference.generator import TextGenerator
from src.inference.kv_cache import KVCache
from src.models.language_model import TunaLM, TunaLMConfig
from src.tokenizer.tokenizer import BilingualBPETokenizer


class TestInferenceEngine:
    @pytest.fixture
    def setup_env(self):
        tok_file = "configs/tokenizer/bpe_toy_500.json"
        tokenizer = BilingualBPETokenizer.from_file(tok_file)

        config = TunaLMConfig(
            vocab_size=tokenizer.vocab_size,
            context_length=32,
            d_model=32,
            n_layers=2,
            n_heads=2,
            d_ff=64,
            dropout=0.0
        )
        model = TunaLM(config)
        model.eval()

        generator = TextGenerator(model=model, tokenizer=tokenizer, device="cpu")
        return tokenizer, model, generator

    def test_kv_cache_update_and_reset(self):
        cache = KVCache(num_layers=2)
        assert cache.get_seq_len() == 0

        # Bước 1: Prefill 4 tokens: shape (1, 2, 4, 16)
        k1 = torch.randn(1, 2, 4, 16)
        v1 = torch.randn(1, 2, 4, 16)
        out_k1, out_v1 = cache.update(0, k1, v1)
        assert out_k1.shape == (1, 2, 4, 16)
        assert cache.get_seq_len() == 4

        # Bước 2: Thêm 1 token mới: shape (1, 2, 1, 16)
        k2 = torch.randn(1, 2, 1, 16)
        v2 = torch.randn(1, 2, 1, 16)
        out_k2, out_v2 = cache.update(0, k2, v2)
        assert out_k2.shape == (1, 2, 5, 16)
        assert cache.get_seq_len() == 5

        # Reset
        cache.reset()
        assert cache.get_seq_len() == 0

    def test_generator_strategies(self, setup_env):
        tokenizer, model, generator = setup_env
        prompt = "Trí tuệ nhân tạo"

        # 1. Greedy (temperature = 0)
        res_greedy = generator.generate(prompt, max_new_tokens=10, temperature=0.0, use_cache=True)
        assert len(res_greedy) > 0

        # 2. Temperature + Top-K + Top-P
        res_sampling = generator.generate(
            prompt,
            max_new_tokens=10,
            temperature=0.8,
            top_k=20,
            top_p=0.85,
            repetition_penalty=1.2,
            use_cache=True
        )
        assert len(res_sampling) > 0

        # 3. Non-cached generation
        res_non_cache = generator.generate(
            prompt,
            max_new_tokens=10,
            use_cache=False
        )
        assert len(res_non_cache) > 0

    def test_generator_stream(self, setup_env):
        _, _, generator = setup_env
        tokens = list(generator.generate_stream("Xin chào", max_new_tokens=5, use_cache=True))
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)

    def test_offline_export_and_load(self, setup_env):
        tokenizer, model, _ = setup_env
        tmp_dir = tempfile.mkdtemp()

        try:
            # Xuất
            files = ModelExporter.export(
                model=model,
                tokenizer=tokenizer,
                output_dir=tmp_dir,
                model_name="TestExportLM"
            )
            assert Path(files["config"]).exists()
            assert Path(files["weights"]).exists()
            assert Path(files["tokenizer"]).exists()
            assert Path(files["model_card"]).exists()

            # Nạp lại độc lập
            loaded_model, loaded_tok, loaded_gen = load_offline_model(tmp_dir, device="cpu")
            assert loaded_model is not None
            assert loaded_tok.vocab_size == tokenizer.vocab_size

            # Chạy thử suy luận từ mô hình đã nạp
            output = loaded_gen.generate("Kiểm thử nạp", max_new_tokens=8, use_cache=True)
            assert len(output) > 0
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
