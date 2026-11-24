"""
tests/test_training_engine.py

Bộ kiểm thử Unit Test cho Milestone M6:
Kiểm thử Động cơ Huấn luyện (Training Engine, Checkpoint Manager & Scheduler):
- Kiểm thử CheckpointManager: Lưu trữ, xoay vòng file, và nạp lại chính xác state_dict.
- Kiểm thử CosineWarmupScheduler: Tăng tuyến tính trong giai đoạn Warmup và giảm dần Cosine.
- Kiểm thử Data Loader: get_batch trả về tensor X và Y dịch chuyển đúng 1 token.
- Kiểm thử Vòng lặp Huấn luyện: Trainer step làm suy giảm hàm mất mát qua từng bước tối ưu.
"""

import tempfile
import torch
import torch.nn as nn
import torch.optim as optim
import pytest

from src.models.language_model import TunaLM, TunaLMConfig
from src.training.checkpoint import CheckpointManager
from src.training.scheduler import CosineWarmupScheduler
from src.training.trainer import Trainer
from src.data.loader import get_batch


class TestTrainingComponents:
    def test_checkpoint_save_and_resume(self):
        config = TunaLMConfig(vocab_size=30, context_length=16, d_model=16, n_layers=1, n_heads=1, d_ff=32)
        model = TunaLM(config)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = CosineWarmupScheduler(optimizer, warmup_steps=5, total_steps=20)

        with tempfile.TemporaryDirectory() as tmp_dir:
            mgr = CheckpointManager(checkpoint_dir=tmp_dir, max_to_keep=2)

            # Lưu checkpoint tại step 10
            ckpt_path = mgr.save(model, optimizer, step=10, loss=2.345, scheduler=scheduler, config=config)

            # Khởi tạo model mới và nạp lại
            model_new = TunaLM(config)
            optimizer_new = optim.AdamW(model_new.parameters(), lr=1e-3)
            scheduler_new = CosineWarmupScheduler(optimizer_new, warmup_steps=5, total_steps=20)

            state = mgr.load(ckpt_path, model_new, optimizer_new, scheduler_new)

            assert state["step"] == 10
            assert abs(state["loss"] - 2.345) < 1e-4

            # Trọng số phải khớp tuyệt đối 100%
            for p1, p2 in zip(model.parameters(), model_new.parameters()):
                assert torch.equal(p1, p2)

    def test_cosine_warmup_scheduler_schedule(self):
        model = nn.Linear(10, 2)
        base_lr = 1e-2
        min_lr = 1e-4
        optimizer = optim.SGD(model.parameters(), lr=base_lr)
        scheduler = CosineWarmupScheduler(optimizer, warmup_steps=10, total_steps=50, min_lr=min_lr)

        # Step 0: lr = 0
        assert scheduler.get_lr() == 0.0

        # Step 5: giữa warmup -> lr = base_lr * 0.5
        for _ in range(5):
            scheduler.step()
        assert abs(scheduler.get_lr() - (base_lr * 0.5)) < 1e-5

        # Step 10: đỉnh warmup -> lr = base_lr
        for _ in range(5):
            scheduler.step()
        assert abs(scheduler.get_lr() - base_lr) < 1e-5

        # Step 50: kết thúc decay -> lr = min_lr
        for _ in range(40):
            scheduler.step()
        assert abs(scheduler.get_lr() - min_lr) < 1e-5

    def test_get_batch_shifted_targets(self):
        # Dữ liệu tuần tự: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
        tokens = list(range(20))
        B, T = 2, 5
        x, y = get_batch(tokens, batch_size=B, context_length=T)

        assert x.shape == (B, T)
        assert y.shape == (B, T)
        # Mục tiêu Y phải dịch chuyển đúng 1 vị trí so với X: y[b, t] == x[b, t] + 1
        assert torch.equal(y, x + 1)

    def test_trainer_step_decreases_loss(self):
        config = TunaLMConfig(vocab_size=20, context_length=8, d_model=16, n_layers=1, n_heads=1, d_ff=32)
        model = TunaLM(config)
        optimizer = optim.AdamW(model.parameters(), lr=0.01)

        trainer = Trainer(model, optimizer, grad_clip=1.0)
        # Cố định một batch lặp đi lặp lại để kiểm tra mô hình overfitting được trên batch đó
        x = torch.randint(0, config.vocab_size, (2, config.context_length))
        y = torch.randint(0, config.vocab_size, (2, config.context_length))

        loss_0 = trainer.train_step(x, y)
        for _ in range(15):
            loss_final = trainer.train_step(x, y)

        assert loss_final < loss_0 * 0.5, f"Loss không giảm đủ: {loss_0} -> {loss_final}"
