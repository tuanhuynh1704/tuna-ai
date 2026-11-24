"""Bộ kiểm thử đơn vị cho Hệ thống Đánh giá Evaluation Suite (Milestone M9).

Kiểm tra:
1. ClozeBenchmark: Đánh giá log-likelihood và tính đúng đắn của Cloze test.
2. ModelEvaluator:
   - evaluate_loss_and_perplexity trên Hold-out set.
   - evaluate_next_token_accuracy (Top-1, Top-5).
   - benchmark_inference_latency (TTFT, ITL, throughput).
   - run_full_evaluation tạo EvaluationReport hoàn chỉnh.
"""

import shutil
import tempfile
from pathlib import Path
import pytest
import torch

from src.data.loader import ShardDataLoader
from src.data.packer import DatasetPacker
from src.evaluation.benchmarks import ClozeBenchmark
from src.evaluation.evaluator import EvaluationReport, ModelEvaluator
from src.models.language_model import TunaLM, TunaLMConfig
from src.tokenizer.tokenizer import BilingualBPETokenizer


class TestEvaluationSuite:
    @pytest.fixture
    def setup_eval_env(self):
        tmp_dir = tempfile.mkdtemp()
        tok_file = "configs/tokenizer/bpe_toy_500.json"
        tokenizer = BilingualBPETokenizer.from_file(tok_file)

        config = TunaLMConfig(
            vocab_size=tokenizer.vocab_size,
            context_length=32,
            d_model=32,
            n_layers=1,
            n_heads=2,
            d_ff=64,
            dropout=0.0
        )
        model = TunaLM(config)

        # Tạo shards dữ liệu
        docs = [
            "Trí tuệ nhân tạo phát triển mạnh mẽ.",
            "Artificial intelligence with transformer models.",
            "Xử lý ngôn ngữ tự nhiên tiếng Việt và tiếng Anh.",
        ] * 4

        packer = DatasetPacker(
            tokenizer=tokenizer,
            output_dir=tmp_dir,
            max_tokens_per_shard=80,
            shard_prefix="test_eval_",
        )
        records = packer.pack(docs)
        shard_paths = [r["path"] for r in records]

        loader = ShardDataLoader(
            shard_paths=shard_paths,
            batch_size=2,
            context_length=16,
            val_split=0.2,
            device="cpu"
        )

        yield tmp_dir, tokenizer, model, loader
        shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_cloze_benchmark(self, setup_eval_env):
        _, tokenizer, model, _ = setup_eval_env
        cloze = ClozeBenchmark()
        results = cloze.evaluate(model, tokenizer, device="cpu")

        assert results["total_items"] == len(cloze.items)
        assert 0.0 <= results["accuracy_percent"] <= 100.0
        assert "vi_knowledge" in results["category_accuracy_percent"]
        assert "en_knowledge" in results["category_accuracy_percent"]
        assert len(results["details"]) == len(cloze.items)

    def test_perplexity_and_loss_eval(self, setup_eval_env):
        _, _, model, loader = setup_eval_env
        evaluator = ModelEvaluator(device="cpu")
        loss, ppl = evaluator.evaluate_loss_and_perplexity(model, loader, eval_iters=5)

        assert loss > 0.0
        assert ppl >= 1.0

    def test_next_token_accuracy(self, setup_eval_env):
        _, _, model, loader = setup_eval_env
        evaluator = ModelEvaluator(device="cpu")
        top1, top5 = evaluator.evaluate_next_token_accuracy(model, loader, eval_iters=5)

        assert 0.0 <= top1 <= 100.0
        assert 0.0 <= top5 <= 100.0
        # Top-5 accuracy luôn phải >= Top-1 accuracy
        assert top5 >= top1

    def test_benchmark_inference_latency(self, setup_eval_env):
        _, tokenizer, model, _ = setup_eval_env
        evaluator = ModelEvaluator(device="cpu")
        perf = evaluator.benchmark_inference_latency(
            model=model,
            tokenizer=tokenizer,
            prompt="Trí tuệ",
            max_new_tokens=10,
            warmup_runs=1,
            test_runs=2
        )

        assert perf["time_to_first_token_ms"] > 0.0
        assert perf["inter_token_latency_ms"] > 0.0
        assert perf["generation_throughput_tok_per_sec"] > 0.0
        assert perf["ram_usage_mb"] > 0.0

    def test_run_full_evaluation(self, setup_eval_env):
        _, tokenizer, model, loader = setup_eval_env
        evaluator = ModelEvaluator(device="cpu")
        report = evaluator.run_full_evaluation(
            model=model,
            tokenizer=tokenizer,
            data_loader=loader,
            model_name="TestLM",
            eval_iters=3
        )

        assert isinstance(report, EvaluationReport)
        assert report.model_name == "TestLM"
        assert report.parameters_count > 0
        assert report.top5_accuracy_percent >= report.top1_accuracy_percent
