"""
src/training/scheduler.py

Bộ điều phối Tốc độ Học (Learning Rate Scheduler):
- Linear Warmup kết hợp Cosine Decay (Chuẩn mực huấn luyện Transformer / LLM của Chinchilla & LLaMA):
  1. Giai đoạn Warmup: Tăng dần tốc độ học tuyến tính từ 0 đến base_lr trong warmup_steps đầu tiên.
     Giúp các ma trận trọng số ngẫu nhiên ổn định phương sai trước khi tiếp nhận gradient lớn.
  2. Giai đoạn Cosine Decay: Giảm dần tốc độ học theo đường cong cosin tiệm cận min_lr.
"""

import math
from typing import List
import torch.optim as optim


class CosineWarmupScheduler:
    """Điều phối Learning Rate theo chu kỳ Linear Warmup và Cosine Decay."""

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-5
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self) -> float:
        base_lr = self.base_lrs[0]

        # 1. Giai đoạn Linear Warmup
        if self.current_step < self.warmup_steps:
            return base_lr * (float(self.current_step) / float(max(1, self.warmup_steps)))

        # 2. Sau khi vượt quá total_steps: giữ ở min_lr
        if self.current_step > self.total_steps:
            return self.min_lr

        # 3. Giai đoạn Cosine Decay
        decay_ratio = float(self.current_step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (base_lr - self.min_lr)

    def state_dict(self) -> dict:
        return {
            "current_step": self.current_step,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr": self.min_lr,
            "base_lrs": self.base_lrs,
        }

    def load_state_dict(self, state: dict):
        self.current_step = state["current_step"]
        self.warmup_steps = state["warmup_steps"]
        self.total_steps = state["total_steps"]
        self.min_lr = state["min_lr"]
        self.base_lrs = state["base_lrs"]
