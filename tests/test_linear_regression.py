"""
tests/test_linear_regression.py

Bộ kiểm thử Unit Test cho Ngày 6 - Tuần 1:
- Kiểm thử thuật toán SGDMomentum và AdamW đối chiếu PyTorch.
- Kiểm thử mô hình Hồi quy Tuyến tính (Linear Regression) huấn luyện từ đầu hội tụ
  sát với nghiệm giải tích chính xác (Normal Equation Closed-form).
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from research.machine_learning.optimizers import SGDMomentum, AdamW
from research.machine_learning.linear_regression import LinearRegressionModel, solve_normal_equation


class TestOptimizersMatchesPyTorch:
    def test_sgd_momentum_step_matches_pytorch(self):
        """Đối chiếu 1 bước cập nhật của SGDMomentum tự viết với torch.optim.SGD."""
        # Trọng số và gradient khởi tạo
        p_np = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        g_np = np.array([0.5, -0.2, 0.1], dtype=np.float32)

        lr = 0.05
        momentum = 0.9
        weight_decay = 0.01

        # 1. Tự viết
        opt_manual = SGDMomentum([p_np.copy()], lr=lr, momentum=momentum, weight_decay=weight_decay)
        opt_manual.step([g_np.copy()])

        # 2. PyTorch
        t_p = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float32, requires_grad=True)
        t_p.grad = torch.tensor([0.5, -0.2, 0.1], dtype=torch.float32)
        opt_pt = torch.optim.SGD([t_p], lr=lr, momentum=momentum, weight_decay=weight_decay)
        opt_pt.step()

        assert np.allclose(opt_manual.params[0], t_p.detach().numpy(), atol=1e-5)

    def test_adamw_step_matches_pytorch(self):
        """Đối chiếu 1 bước cập nhật của AdamW tự viết với torch.optim.AdamW."""
        p_np = np.array([1.5, -0.8, 2.2], dtype=np.float32)
        g_np = np.array([0.1, -0.4, 0.3], dtype=np.float32)

        lr = 1e-3
        weight_decay = 1e-2

        # 1. Tự viết
        opt_manual = AdamW([p_np.copy()], lr=lr, weight_decay=weight_decay)
        opt_manual.step([g_np.copy()])

        # 2. PyTorch
        t_p = torch.tensor([1.5, -0.8, 2.2], dtype=torch.float32, requires_grad=True)
        t_p.grad = torch.tensor([0.1, -0.4, 0.3], dtype=torch.float32)
        opt_pt = torch.optim.AdamW([t_p], lr=lr, weight_decay=weight_decay)
        opt_pt.step()

        assert np.allclose(opt_manual.params[0], t_p.detach().numpy(), atol=1e-5)


class TestLinearRegressionTraining:
    def test_linear_regression_convergence_and_normal_equation(self):
        """
        Huấn luyện Linear Regression trên tập dữ liệu tổng hợp:
          y = 3.5 * x1 - 2.0 * x2 + 1.2
        Kiểm tra mô hình hội tụ và trọng số xấp xỉ nghiệm Closed-form.
        """
        np.random.seed(42)
        N = 200
        D = 2
        X = np.random.randn(N, D).astype(np.float32)
        true_W = np.array([[3.5], [-2.0]], dtype=np.float32)
        true_b = 1.2
        noise = np.random.randn(N, 1).astype(np.float32) * 0.05
        y = np.matmul(X, true_W) + true_b + noise

        # 1. Tính nghiệm giải tích chính xác
        W_star, b_star = solve_normal_equation(X, y)

        # 2. Huấn luyện bằng AdamW
        model = LinearRegressionModel(input_dim=D)
        optimizer = AdamW(model.params, lr=0.1, weight_decay=0.0)
        loss_history = model.fit(X, y, optimizer=optimizer, epochs=150, batch_size=32)

        # Kiểm tra loss giảm mạnh
        assert loss_history[-1] < loss_history[0] * 0.01
        assert loss_history[-1] < 0.01  # Loss cuối rất nhỏ

        # Trọng số tìm được qua Gradient Descent phải cực kỳ sát với nghiệm giải tích
        assert np.allclose(model.W, W_star, atol=0.05)
        assert np.isclose(model.b[0], b_star, atol=0.05)

        # Đánh giá R^2 score
        y_pred = model.forward(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot)
        assert r2 > 0.98, f"R^2 score quá thấp: {r2}"
