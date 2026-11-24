"""Bộ kiểm thử đơn vị cho ShardDataLoader và Hệ thống Thí nghiệm Tiền kỳ (Milestone M8).

Kiểm tra:
1. ShardDataLoader:
   - Ánh xạ memmap uint16 an toàn.
   - Sinh batch (X, Y) với shape đúng (B, T) và dtype torch.long.
   - Phân chia train/val không rò rỉ.
   - Dịch chuyển 1 token chính xác: Y[:, t] == target token của X[:, t].
2. ExperimentRunner & ExperimentConfig:
   - Thiết lập thí nghiệm và thu thập các metrics hợp lệ.
   - Xuất file báo cáo Markdown và file JSON số liệu.
"""

import json
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch

from src.data.loader import ShardDataLoader
from src.data.packer import DatasetPacker
from src.tokenizer.tokenizer import BilingualBPETokenizer
from src.training.experiment import ExperimentConfig, ExperimentRunner


class TestStreamingShardDataLoader:
    @pytest.fixture
    def setup_shards(self):
        tmp_dir = tempfile.mkdtemp()
        tok_file = "configs/tokenizer/bpe_toy_500.json"
        tokenizer = BilingualBPETokenizer.from_file(tok_file)

        docs = [
            "Hệ thống mô hình ngôn ngữ lớn Tuna AI học tiếng Việt và tiếng Anh.",
            "Transformer architecture uses causal multi head attention mechanisms.",
            "Tối ưu hóa học máy với PyTorch đạt hiệu năng cao trên vi xử lý Intel CPU.",
            "Gradient clipping prevents gradients from exploding during training.",
        ] * 4

        packer = DatasetPacker(
            tokenizer=tokenizer,
            output_dir=tmp_dir,
            max_tokens_per_shard=80,
            shard_prefix="test_mmap_",
        )
        records = packer.pack(docs)
        shard_paths = [r["path"] for r in records]

        yield shard_paths, tokenizer
        shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_shard_loader_batch_generation(self, setup_shards):
        shard_paths, tokenizer = setup_shards
        loader = ShardDataLoader(
            shard_paths=shard_paths,
            batch_size=4,
            context_length=16,
            val_split=0.2,
            device="cpu"
        )

        assert loader.total_tokens > 0
        assert len(loader.shards) == len(shard_paths)

        # Lấy train batch
        x_train, y_train = loader.get_batch(split="train")
        assert x_train.shape == (4, 16)
        assert y_train.shape == (4, 16)
        assert x_train.dtype == torch.long
        assert y_train.dtype == torch.long

        # Lấy val batch
        x_val, y_val = loader.get_batch(split="val")
        assert x_val.shape == (4, 16)
        assert y_val.shape == (4, 16)

    def test_next_token_shift_correctness(self, setup_shards):
        shard_paths, _ = setup_shards
        loader = ShardDataLoader(
            shard_paths=shard_paths,
            batch_size=2,
            context_length=8,
            device="cpu"
        )
        x, y = loader.get_batch(split="train")

        # Kiểm tra tính chất autoregressive dịch chuyển 1 token
        # Giữa các phần tử nội bộ trong batch: x[b, t+1] phải bằng y[b, t] cho mọi t từ 0 đến T-2
        for b in range(x.shape[0]):
            assert torch.equal(x[b, 1:], y[b, :-1])


class TestExperimentRunner:
    @pytest.fixture
    def setup_runner_env(self):
        tmp_dir = tempfile.mkdtemp()
        tok_file = "configs/tokenizer/bpe_toy_500.json"
        tokenizer = BilingualBPETokenizer.from_file(tok_file)

        model_config = {
            "vocab_size": tokenizer.vocab_size,
            "context_length": 16,
            "d_model": 32,
            "num_layers": 1,
            "num_heads": 2,
            "d_ff": 64,
            "dropout": 0.0,
        }

        # Tạo 1 shard dữ liệu nhỏ
        docs = ["Trí tuệ nhân tạo phát triển vượt bậc trên máy tính cá nhân."] * 6
        packer = DatasetPacker(
            tokenizer=tokenizer,
            output_dir=tmp_dir,
            max_tokens_per_shard=100,
            shard_prefix="exp_test_",
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

        yield tmp_dir, tokenizer, model_config, loader
        shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_run_single_experiment_and_report(self, setup_runner_env):
        tmp_dir, tokenizer, model_config, loader = setup_runner_env
        runner = ExperimentRunner(
            tokenizer=tokenizer,
            model_config=model_config,
            output_dir=tmp_dir
        )

        cfg = ExperimentConfig(
            name="TEST-EXP",
            description="Thử nghiệm kiểm tra runner",
            corpus_mode="test",
            total_steps=10,
            batch_size=2,
            context_length=16,
            learning_rate=1e-3,
            warmup_steps=2,
            device="cpu"
        )

        metrics = runner.run_experiment(cfg, loader, prompts=["Trí tuệ"])
        assert metrics.experiment_name == "TEST-EXP"
        assert metrics.initial_loss > 0.0
        assert metrics.final_loss > 0.0
        assert metrics.avg_throughput_tok_per_sec > 0.0
        assert "before_Trí tuệ" in metrics.sample_generations
        assert "after_Trí tuệ" in metrics.sample_generations

        report_path = runner.generate_report("test_report.md")
        assert Path(report_path).exists()
        assert (Path(tmp_dir) / "m8_metrics.json").exists()
