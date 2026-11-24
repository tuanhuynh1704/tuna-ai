"""
src/training/checkpoint.py

Hệ thống Quản lý Điểm lưu (Checkpoint Manager) cho Tuna-AI:
- Lưu trữ an toàn trạng thái huấn luyện: model weights, optimizer state, scheduler state, global step, loss.
- Tự động xoay vòng checkpoint (giới hạn max_to_keep) để tiết kiệm dung lượng ổ cứng SSD 512GB.
- Khôi phục trạng thái (Resume Training) đảm bảo việc tiếp tục huấn luyện diễn ra liền mạch.
"""

from typing import Dict, Any, Optional, List
import os
import glob
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim


class CheckpointManager:
    """Quản lý việc lưu và nạp checkpoint huấn luyện mô hình."""

    def __init__(self, checkpoint_dir: str = "checkpoints", max_to_keep: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = max_to_keep

    def save(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        step: int,
        loss: float,
        scheduler: Optional[Any] = None,
        config: Optional[Any] = None,
        filename: Optional[str] = None
    ) -> str:
        """Lưu trạng thái huấn luyện vào file .pt."""
        if filename is None:
            filename = f"checkpoint_step_{step:06d}.pt"
        filepath = self.checkpoint_dir / filename

        state = {
            "step": step,
            "loss": loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "config": vars(config) if hasattr(config, "__dict__") else config,
        }

        # Lưu checkpoint an toàn
        torch.save(state, filepath)

        # Xoay vòng file: xóa checkpoint cũ nhất nếu vượt quá max_to_keep
        self._cleanup_old_checkpoints()
        return str(filepath)

    def load(
        self,
        filepath: str,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Nạp lại trạng thái từ checkpoint."""
        checkpoint = torch.load(filepath, map_location="cpu")

        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"] is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return {
            "step": checkpoint.get("step", 0),
            "loss": checkpoint.get("loss", 0.0),
            "config": checkpoint.get("config", None)
        }

    def _cleanup_old_checkpoints(self):
        """Xóa các checkpoint cũ hơn, chỉ giữ lại max_to_keep checkpoint mới nhất."""
        pattern = str(self.checkpoint_dir / "checkpoint_step_*.pt")
        ckpts = sorted(glob.glob(pattern))
        if len(ckpts) > self.max_to_keep:
            for old_ckpt in ckpts[:-self.max_to_keep]:
                try:
                    os.remove(old_ckpt)
                except OSError:
                    pass
