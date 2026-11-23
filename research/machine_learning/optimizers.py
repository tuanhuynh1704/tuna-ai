"""
research/machine_learning/optimizers.py

Thực hành Ngày 6 - Tuần 1:
Nghiên cứu và triển khai các thuật toán tối ưu hóa (Optimizers) từ đầu:
- Stochastic Gradient Descent (SGD) with Momentum và Weight Decay.
- AdamW (Decoupled Weight Decay) - Thuật toán tối ưu hóa tiêu chuẩn cho LLM và Transformer.
"""

from typing import List, Union
import numpy as np


class Optimizer:
    """Lớp cơ sở cho các thuật toán tối ưu hóa."""
    def __init__(self, params: List[np.ndarray], lr: float = 1e-3, weight_decay: float = 0.0):
        self.params = params
        self.lr = lr
        self.weight_decay = weight_decay

    def zero_grad(self, grads: List[np.ndarray]):
        """Xóa trắng gradient sau mỗi bước cập nhật."""
        for g in grads:
            if g is not None:
                g.fill(0.0)

    def step(self, grads: List[np.ndarray]):
        raise NotImplementedError


class SGDMomentum(Optimizer):
    """
    Thuật toán Gradient Descent tích hợp Động lượng (Momentum) và Tiêu hao Trọng số (Weight Decay):
      v = beta * v + (grad + weight_decay * param)
      param = param - lr * v
    """
    def __init__(
        self, 
        params: List[np.ndarray], 
        lr: float = 1e-2, 
        momentum: float = 0.9, 
        weight_decay: float = 0.0
    ):
        super().__init__(params, lr, weight_decay)
        self.momentum = momentum
        # Vận tốc (velocities) cho từng tham số
        self.velocities = [np.zeros_like(p) for p in self.params]

    def step(self, grads: List[np.ndarray]):
        for p, g, v in zip(self.params, grads, self.velocities):
            if g is None:
                continue

            # Áp dụng L2 regularization (weight decay)
            grad_with_decay = g + self.weight_decay * p

            # Cập nhật động lượng: v = beta * v + grad
            v[:] = self.momentum * v + grad_with_decay

            # Cập nhật trọng số
            p -= self.lr * v


class AdamW(Optimizer):
    """
    Thuật toán AdamW (Loshchilov & Hutter, 2017) - Tiêu chuẩn vàng trong huấn luyện Transformer / LLM:
    Tách rời Weight Decay khỏi phép ước lượng moment bậc 1 và bậc 2 (Decoupled Weight Decay):
      m = beta1 * m + (1 - beta1) * grad
      v = beta2 * v + (1 - beta2) * (grad^2)
      m_hat = m / (1 - beta1^t)
      v_hat = v / (1 - beta2^t)
      param = param - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * param)
    """
    def __init__(
        self, 
        params: List[np.ndarray], 
        lr: float = 1e-3, 
        betas: tuple = (0.9, 0.999), 
        eps: float = 1e-8, 
        weight_decay: float = 1e-2
    ):
        super().__init__(params, lr, weight_decay)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0  # Timestep

        # Khởi tạo moment bậc 1 (m) và moment bậc 2 (v)
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]

    def step(self, grads: List[np.ndarray]):
        self.t += 1
        bias_correction1 = 1.0 - (self.beta1 ** self.t)
        bias_correction2 = 1.0 - (self.beta2 ** self.t)

        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            if g is None:
                continue

            # 1. Cập nhật moment bậc 1 (Moving Average of Gradients)
            m[:] = self.beta1 * m + (1.0 - self.beta1) * g

            # 2. Cập nhật moment bậc 2 (Moving Average of Squared Gradients)
            v[:] = self.beta2 * v + (1.0 - self.beta2) * (g ** 2)

            # 3. Hiệu chỉnh độ lệch khởi tạo (Bias Correction)
            m_hat = m / bias_correction1
            v_hat = v / bias_correction2

            # 4. Bước cập nhật trọng số Adam + Decoupled Weight Decay
            step_direction = m_hat / (np.sqrt(v_hat) + self.eps)
            p -= self.lr * (step_direction + self.weight_decay * p)
