"""
tests/test_calculus_derivatives.py

Bộ kiểm thử Unit Test cho Ngày 4 - Tuần 1:
Kiểm chứng đạo hàm giải tích (Analytical Gradients) bằng Sai phân Hữu hạn (Finite Differences)
và đối chiếu Softmax Cross-Entropy Gradient với PyTorch Autograd.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from research.mathematics.calculus_derivatives import (
    numerical_gradient_1d,
    Sigmoid,
    Tanh,
    ReLU,
    GELU,
    SiLU,
    cross_entropy_loss_and_grad,
    softmax,
)


class TestActivationGradientsWithFiniteDifferences:
    """So sánh đạo hàm giải tích của từng hàm kích hoạt với đạo hàm số trị."""

    @pytest.mark.parametrize("act_cls", [Sigmoid, Tanh, ReLU, GELU, SiLU])
    def test_activation_derivative(self, act_cls):
        # Tránh kiểm tra ngay tại điểm kỳ dị x=0 của ReLU
        x = np.array([-2.5, -1.0, -0.2, 0.3, 1.5, 3.0], dtype=np.float64)

        analytical_grad = act_cls.backward(x)
        numerical_grad = numerical_gradient_1d(act_cls.forward, x, eps=1e-6)

        # Sai số tương đối (Relative error) < 1e-5
        diff = np.abs(analytical_grad - numerical_grad)
        denom = np.maximum(np.abs(analytical_grad), np.abs(numerical_grad)) + 1e-8
        rel_error = diff / denom

        assert np.all(rel_error < 1e-4), f"Lỗi đạo hàm tại {act_cls.__name__}: max rel_error = {np.max(rel_error)}"


class TestCrossEntropyLossAndGradient:
    def test_cross_entropy_matches_pytorch(self):
        """Đối chiếu CrossEntropy Loss & Gradient tự viết với torch.nn.CrossEntropyLoss."""
        np.random.seed(42)
        N, V = 4, 10  # 4 tokens, vocab size 10
        logits_np = np.random.randn(N, V).astype(np.float64)
        targets_np = np.random.randint(0, V, size=(N,)).astype(np.int64)

        # 1. Triển khai thủ công
        loss_manual, grad_manual = cross_entropy_loss_and_grad(logits_np, targets_np)

        # 2. PyTorch Autograd
        logits_pt = torch.tensor(logits_np, requires_grad=True, dtype=torch.float64)
        targets_pt = torch.tensor(targets_np, dtype=torch.long)

        criterion = nn.CrossEntropyLoss()
        loss_pt = criterion(logits_pt, targets_pt)
        loss_pt.backward()

        grad_pt = logits_pt.grad.numpy()

        # Đối chiếu loss
        assert np.isclose(loss_manual, loss_pt.item(), atol=1e-6)
        # Đối chiếu gradient
        assert np.allclose(grad_manual, grad_pt, atol=1e-6)

    def test_gradient_with_numerical_check(self):
        """Kiểm chứng gradient của CrossEntropy đối với một logit đầu vào bằng sai phân hữu hạn."""
        np.random.seed(123)
        N, V = 2, 5
        logits = np.random.randn(N, V).astype(np.float64)
        targets = np.array([1, 4], dtype=np.int64)

        loss, analytical_grad = cross_entropy_loss_and_grad(logits, targets)

        # Định nghĩa hàm loss phụ thuộc logits
        def loss_fn(l):
            val, _ = cross_entropy_loss_and_grad(l, targets)
            return val

        # Tính đạo hàm số trị
        eps = 1e-6
        numerical_grad = np.zeros_like(logits)
        for i in range(N):
            for j in range(V):
                l_plus = logits.copy()
                l_plus[i, j] += eps
                l_minus = logits.copy()
                l_minus[i, j] -= eps
                numerical_grad[i, j] = (loss_fn(l_plus) - loss_fn(l_minus)) / (2.0 * eps)

        assert np.allclose(analytical_grad, numerical_grad, atol=1e-5)
