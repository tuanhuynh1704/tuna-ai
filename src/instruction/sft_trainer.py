"""
src/instruction/sft_trainer.py

Động cơ Huấn luyện Tinh chỉnh Chỉ dẫn (SFT Trainer) cho Tuna-AI:
- Tối ưu hóa mô hình với hàm mất mát Cross-Entropy chỉ tính trên các token của Assistant (Response Masking).
- Theo dõi sự suy giảm của Response Loss và Perplexity của phản hồi.
- Cắt tỉa Gradient toàn cục và điều chỉnh tốc độ học Cosine Warmup.
"""

import math
import time
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.language_model import TunaLM
from src.training.scheduler import CosineWarmupScheduler


class SFTTrainer:
    """Động cơ Huấn luyện Supervised Fine-Tuning."""

    def __init__(
        self,
        model: TunaLM,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[CosineWarmupScheduler] = None,
        grad_clip: float = 1.0,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        self.device = device

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Thực thi 1 bước gradient descent với Response Masking."""
        self.model.train()
        self.optimizer.zero_grad()

        x = x.to(self.device)
        y = y.to(self.device)

        _, loss = self.model(x, targets=y)
        loss.backward()

        if self.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        return loss.item()

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> float:
        """Tính Response Loss trung bình trên tập kiểm thử hội thoại."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for x, y in val_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            _, loss = self.model(x, targets=y)
            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(1, num_batches)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        log_interval: int = 5,
    ) -> Dict[str, Any]:
        """Vòng lặp huấn luyện SFT qua các epochs."""
        train_losses: List[float] = []
        val_losses: List[Tuple[int, float]] = []
        step = 0
        t_start = time.time()

        for epoch in range(1, epochs + 1):
            for x, y in train_loader:
                step += 1
                loss = self.train_step(x, y)
                train_losses.append(loss)

                if step % log_interval == 0 or step == 1:
                    lr = self.scheduler.get_lr() if self.scheduler else self.optimizer.param_groups[0]["lr"]
                    print(f"Epoch {epoch:2d} | Step {step:3d} | SFT Response Loss: {loss:.4f} | LR: {lr:.2e}")

            if val_loader is not None:
                v_loss = self.evaluate(val_loader)
                val_losses.append((step, v_loss))
                ppl = math.exp(v_loss) if v_loss < 20 else float("inf")
                print(f"  ---> [SFT Validation] Epoch {epoch}: Val Loss = {v_loss:.4f} (Perplexity = {ppl:.2f})")

        total_time = time.time() - t_start
        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "total_steps": step,
            "total_time_sec": total_time,
            "final_train_loss": train_losses[-1] if train_losses else 0.0,
        }
