"""
src/core/loss.py

Các hàm Mất mát (Loss Functions) xây dựng trên Tensor Autograd:
- MSELoss: Sai số bình phương trung bình cho bài toán hồi quy.
- BinaryCrossEntropyWithLogitsLoss: Hàm mất mát phân loại nhị phân ổn định số học.
- CrossEntropyLoss: Phân loại đa lớp (Multi-class Cross Entropy) cho Next-Token Prediction.
"""

from typing import Optional
import numpy as np
from src.core.tensor import Tensor, unbroadcast


def mse_loss(y_pred: Tensor, y_true: Tensor) -> Tensor:
    """Mean Squared Error Loss: MSE = mean((y_pred - y_true)^2)"""
    diff = y_pred - y_true
    return (diff * diff).mean()


class MSELoss:
    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return mse_loss(y_pred, y_true)


def binary_cross_entropy_with_logits(logits: Tensor, targets: Tensor) -> Tensor:
    """
    BCEWithLogits:
      Loss = max(z, 0) - z * y + log(1 + exp(-|z|))
    Tính toán trực tiếp từ Logits z (chưa qua Sigmoid) để tránh tràn số exp.
    """
    z = logits.data
    y = targets.data

    # Công thức ổn định số học
    loss_data = np.maximum(z, 0) - z * y + np.log(1.0 + np.exp(-np.abs(z)))
    mean_loss = float(np.mean(loss_data))

    req_grad = logits.requires_grad
    out = Tensor(mean_loss, requires_grad=req_grad, _children=(logits,), _op='BCEWithLogits')

    def _backward():
        if logits.requires_grad:
            # d(Loss)/dz = (sigmoid(z) - y) / N
            p = np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))
            grad_z = (p - y) / float(z.size)
            logits.grad = grad_z if logits.grad is None else logits.grad + grad_z

    out._backward = _backward
    return out
