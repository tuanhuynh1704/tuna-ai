"""
tests/test_autograd.py

Bộ kiểm thử Unit Test cho Ngày 5 - Tuần 1:
Kiểm thử Động cơ Tự động Vi phân (Autograd Engine - lớp Scalar):
- Kiểm thử các phép toán số học (+, -, *, /, pow).
- Kiểm thử tích lũy gradient khi một biến xuất hiện nhiều lần (Multi-path Accumulation: z = x + x).
- Kiểm thử mạng nơ-ron đơn tầng (Perceptron) đối chiếu với PyTorch Autograd.
- Kiểm thử các hàm kích hoạt phi tuyến (ReLU, Sigmoid, Tanh).
"""

import math
import numpy as np
import pytest
import torch
from src.core.autograd import Scalar


class TestScalarArithmeticAndGrad:
    def test_basic_addition_and_multiplication(self):
        # f = (a + b) * c
        a = Scalar(2.0, label='a')
        b = Scalar(3.0, label='b')
        c = Scalar(-4.0, label='c')

        d = a + b  # d = 5.0
        f = d * c  # f = -20.0
        f.backward()

        assert f.data == -20.0
        assert f.grad == 1.0
        assert d.grad == -4.0  # df/dd = c = -4.0
        assert c.grad == 5.0   # df/dc = d = 5.0
        assert a.grad == -4.0  # df/da = df/dd * dd/da = -4.0 * 1 = -4.0
        assert b.grad == -4.0  # df/db = df/dd * dd/db = -4.0 * 1 = -4.0

    def test_multi_path_gradient_accumulation(self):
        """
        Kiểm tra trường hợp một biến rẽ nhiều nhánh và hợp lưu:
        f = x + x -> df/dx = 2.0 (grad phải được CỘNG DỒN += chứ không bị ghi đè =)
        g = x * x -> dg/dx = 2 * x = 6.0 (khi x = 3.0)
        """
        x = Scalar(3.0)
        f = x + x
        f.backward()
        assert f.data == 6.0
        assert x.grad == 2.0

        x2 = Scalar(3.0)
        g = x2 * x2
        g.backward()
        assert g.data == 9.0
        assert x2.grad == 6.0

    def test_power_and_division(self):
        # f = x^3 / y
        x = Scalar(2.0)
        y = Scalar(4.0)

        f = (x ** 3) / y  # f = 8 / 4 = 2.0
        f.backward()

        # df/dx = 3 * x^2 / y = 3 * 4 / 4 = 3.0
        # df/dy = - x^3 / y^2 = - 8 / 16 = -0.5
        assert np.isclose(f.data, 2.0)
        assert np.isclose(x.grad, 3.0)
        assert np.isclose(y.grad, -0.5)

    def test_activations_backward(self):
        # ReLU
        x_pos = Scalar(2.0)
        y_pos = x_pos.relu()
        y_pos.backward()
        assert y_pos.data == 2.0
        assert x_pos.grad == 1.0

        x_neg = Scalar(-3.0)
        y_neg = x_neg.relu()
        y_neg.backward()
        assert y_neg.data == 0.0
        assert x_neg.grad == 0.0


class TestAutogradMatchesPyTorch:
    def test_neuron_forward_backward_matches_pytorch(self):
        """
        Mô phỏng 1 nơ-ron nhân tạo:
          z = x1*w1 + x2*w2 + b
          a = tanh(z)
          loss = (a - target)^2
        Đối chiếu gradient của w1, w2, b giữa Scalar Autograd và PyTorch.
        """
        # Giá trị khởi tạo
        x1_val, x2_val = 1.5, -2.0
        w1_val, w2_val = 0.8, -0.5
        b_val = 0.1
        target_val = 0.5

        # 1. Chạy với Scalar Autograd tự viết
        x1 = Scalar(x1_val)
        x2 = Scalar(x2_val)
        w1 = Scalar(w1_val)
        w2 = Scalar(w2_val)
        b = Scalar(b_val)

        z = x1 * w1 + x2 * w2 + b
        a = z.tanh()
        loss = (a - target_val) ** 2
        loss.backward()

        # 2. Chạy với PyTorch
        t_x1 = torch.tensor(x1_val, dtype=torch.float64)
        t_x2 = torch.tensor(x2_val, dtype=torch.float64)
        t_w1 = torch.tensor(w1_val, dtype=torch.float64, requires_grad=True)
        t_w2 = torch.tensor(w2_val, dtype=torch.float64, requires_grad=True)
        t_b = torch.tensor(b_val, dtype=torch.float64, requires_grad=True)

        t_z = t_x1 * t_w1 + t_x2 * t_w2 + t_b
        t_a = torch.tanh(t_z)
        t_loss = (t_a - target_val) ** 2
        t_loss.backward()

        # Đối chiếu kết quả
        assert np.isclose(loss.data, t_loss.item(), atol=1e-6)
        assert np.isclose(w1.grad, t_w1.grad.item(), atol=1e-6)
        assert np.isclose(w2.grad, t_w2.grad.item(), atol=1e-6)
        assert np.isclose(b.grad, t_b.grad.item(), atol=1e-6)
