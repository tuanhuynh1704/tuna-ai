"""
src/training/trainer.py

Động cơ Huấn luyện (Training Engine) cho TunaLM / Mini GPT:
- Tích hợp vòng lặp lan truyền thuận, tính Next-Token Loss và lan truyền ngược.
- Gradient Clipping (cắt tỉa gradient toàn cục bảo vệ mô hình khỏi bùng nổ loss NaN).
- Hỗ trợ cả nạp dữ liệu từ RAM mảng token và ShardDataLoader streaming memmap uint16.
- Tự động đánh giá (Evaluation), đo lường tài nguyên và lưu Checkpoint định kỳ.
"""

from typing import Dict, Any, Optional, List, Tuple, Union
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.training.checkpoint import CheckpointManager
from src.training.scheduler import CosineWarmupScheduler
from src.data.loader import get_batch, ShardDataLoader


class Trainer:
    """Động cơ Huấn luyện Mô hình Ngôn ngữ TunaLM."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[CosineWarmupScheduler] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        grad_clip: float = 1.0,
        grad_accum_steps: int = 1,
        device: str = "cpu"
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_manager = checkpoint_manager
        self.grad_clip = grad_clip
        self.grad_accum_steps = grad_accum_steps
        self.device = device

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Thực thi 1 bước tối ưu hóa hoàn chỉnh."""
        self.model.train()
        self.optimizer.zero_grad()

        # 1. Forward pass
        logits, loss = self.model(x, targets=y)

        # 2. Backward pass
        loss.backward()

        # 3. Gradient Clipping (chống Exploding Gradients)
        if self.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)

        # 4. Cập nhật Optimizer & Scheduler
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        return loss.item()

    @torch.no_grad()
    def evaluate(
        self,
        val_data: Optional[Union[List[int], np.ndarray]] = None,
        data_loader: Optional[ShardDataLoader] = None,
        batch_size: int = 4,
        context_length: int = 32,
        eval_iters: int = 5
    ) -> float:
        """Đo lường hàm mất mát trung bình trên tập dữ liệu kiểm thử (Validation)."""
        self.model.eval()
        total_loss = 0.0

        for _ in range(eval_iters):
            if data_loader is not None:
                x, y = data_loader.get_batch(split="val")
            elif val_data is not None:
                x, y = get_batch(val_data, batch_size=batch_size, context_length=context_length, device=self.device)
            else:
                raise ValueError("Cần cung cấp val_data hoặc data_loader để thực hiện evaluate()")

            _, loss = self.model(x, targets=y)
            total_loss += loss.item()

        return total_loss / eval_iters

    def fit(
        self,
        train_data: Optional[Union[List[int], np.ndarray]] = None,
        val_data: Optional[Union[List[int], np.ndarray]] = None,
        data_loader: Optional[ShardDataLoader] = None,
        total_steps: int = 100,
        batch_size: int = 4,
        context_length: int = 32,
        eval_interval: int = 20,
        save_interval: int = 50,
        log_interval: int = 10
    ) -> Dict[str, Any]:
        """
        Vòng lặp huấn luyện chính:
        Hỗ trợ cả nạp dữ liệu từ train_data mảng RAM và data_loader streaming memmap.
        """
        train_losses = []
        val_losses = []
        step_times = []

        start_time = time.time()
        tokens_per_step = batch_size * context_length

        for step in range(1, total_steps + 1):
            t0 = time.perf_counter()

            # Lấy 1 batch
            if data_loader is not None:
                x, y = data_loader.get_batch(split="train")
            elif train_data is not None:
                x, y = get_batch(train_data, batch_size=batch_size, context_length=context_length, device=self.device)
            else:
                raise ValueError("Cần cung cấp train_data hoặc data_loader cho quá trình huấn luyện")

            loss = self.train_step(x, y)
            train_losses.append(loss)

            dt = time.perf_counter() - t0
            step_times.append(dt)

            # In thông tin tiến độ
            if step % log_interval == 0 or step == 1:
                throughput = tokens_per_step / max(dt, 1e-6)
                lr = self.scheduler.get_lr() if self.scheduler else self.optimizer.param_groups[0]["lr"]
                print(f"Step {step:4d}/{total_steps} | Loss: {loss:.4f} | LR: {lr:.2e} | Tốc độ: {throughput:6.1f} tok/s | Thời gian/step: {dt*1000:5.1f} ms")

            # Đánh giá validation
            has_val = (val_data is not None) or (data_loader is not None)
            if has_val and (step % eval_interval == 0 or step == total_steps):
                v_loss = self.evaluate(val_data=val_data, data_loader=data_loader, batch_size=batch_size, context_length=context_length)
                val_losses.append((step, v_loss))
                print(f"  ---> [Validation] Step {step}: Val Loss = {v_loss:.4f} (Perplexity = {math.exp(v_loss):.2f})")

            # Lưu Checkpoint
            if self.checkpoint_manager is not None and (step % save_interval == 0 or step == total_steps):
                ckpt_path = self.checkpoint_manager.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    step=step,
                    loss=loss,
                    scheduler=self.scheduler,
                    config=getattr(self.model, "config", None)
                )
                print(f"  ---> [Checkpoint] Đã lưu tại: {ckpt_path}")

        total_duration = time.time() - start_time
        avg_throughput = (total_steps * tokens_per_step) / max(total_duration, 1e-6)

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "total_duration": total_duration,
            "avg_throughput": avg_throughput,
        }
