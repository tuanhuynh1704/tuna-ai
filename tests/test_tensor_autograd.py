"""
tests/test_tensor_autograd.py

Bộ kiểm thử Unit Test cho Tuần 2 - Ngày 8:
Kiểm thử Động cơ Tự động Vi phân Đa chiều (Tensor Autograd Engine):
- Kiểm thử phép cộng Tensor có tự động Unbroadcasting: (B, T, D) + (D,).
- Kiểm thử phép nhân ma trận Tensor (Matmul @) đối chiếu với PyTorch.
- Kiểm thử hàm kích hoạt GELU và ReLU trên Tensor.
- Kiểm thử chuỗi tính toán mạng Nơ-ron đa tầng (MLP Forward & Backward).
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from src.core.tensor import Tensor, unbroadcast


class TestTensorBroadcastingAndOps:
    def test_unbroadcast_helper(self):
        # Grad shape (2, 3, 4) unbroadcast ve target (4,) -> sum over axis 0, 1
        grad = np.ones((2, 3, 4), dtype=np.float32)
        target_shape = (4,)
        res = unbroadcast(grad, target_shape)
        assert res.shape == (4,)
        assert np.allclose(res, np.full((4,), 6.0))

    def test_tensor_addition_with_broadcasting(self):
        """
        Đối chiếu phép cộng có broadcasting:
          X: (2, 3, 4)
          b: (4,)
          Z = X + b
          Loss = Z.sum()
        """
        np.random.seed(42)
        X_np = np.random.randn(2, 3, 4).astype(np.float32)
        b_np = np.random.randn(4).astype(np.float32)

        # 1. Tự viết
        X = Tensor(X_np, requires_grad=True)
        b = Tensor(b_np, requires_grad=True)
        Z = X + b
        loss = Z.sum()
        loss.backward()

        # 2. PyTorch
        t_X = torch.tensor(X_np, requires_grad=True)
        t_b = torch.tensor(b_np, requires_grad=True)
        t_Z = t_X + t_b
        t_loss = t_Z.sum()
        t_loss.backward()

        assert np.allclose(Z.data, t_Z.detach().numpy())
        assert np.allclose(X.grad, t_X.grad.numpy(), atol=1e-5)
        assert np.allclose(b.grad, t_b.grad.numpy(), atol=1e-5)


class TestTensorMatmulAndGradients:
    def test_matmul_backward_matches_pytorch(self):
        """
        Kiểm tra phép nhân ma trận C = A @ B:
          A: (3, 5)
          B: (5, 4)
          C = A @ B
          loss = C.sum()
        """
        np.random.seed(42)
        A_np = np.random.randn(3, 5).astype(np.float32)
        B_np = np.random.randn(5, 4).astype(np.float32)

        # 1. Tự viết
        A = Tensor(A_np, requires_grad=True)
        B = Tensor(B_np, requires_grad=True)
        C = A @ B
        loss = C.sum()
        loss.backward()

        # 2. PyTorch
        t_A = torch.tensor(A_np, requires_grad=True)
        t_B = torch.tensor(B_np, requires_grad=True)
        t_C = torch.matmul(t_A, t_B)
        t_loss = t_C.sum()
        t_loss.backward()

        assert np.allclose(C.data, t_C.detach().numpy(), atol=1e-5)
        assert np.allclose(A.grad, t_A.grad.numpy(), atol=1e-5)
        assert np.allclose(B.grad, t_B.grad.numpy(), atol=1e-5)

    def test_mlp_block_forward_backward_matches_pytorch(self):
        """
        Kiểm tra 1 khối MLP hoàn chỉnh:
          Z1 = X @ W1 + b1
          A1 = Z1.gelu()
          Z2 = A1 @ W2 + b2
          loss = ((Z2 - target) ** 2).mean()
        """
        np.random.seed(123)
        B, D_in, D_hidden, D_out = 2, 4, 8, 2
        X_np = np.random.randn(B, D_in).astype(np.float32)
        W1_np = np.random.randn(D_in, D_hidden).astype(np.float32) * 0.1
        b1_np = np.zeros(D_hidden, dtype=np.float32)
        W2_np = np.random.randn(D_hidden, D_out).astype(np.float32) * 0.1
        b2_np = np.zeros(D_out, dtype=np.float32)
        target_np = np.random.randn(B, D_out).astype(np.float32)

        # 1. Tự viết
        X = Tensor(X_np, requires_grad=False)
        W1 = Tensor(W1_np, requires_grad=True)
        b1 = Tensor(b1_np, requires_grad=True)
        W2 = Tensor(W2_np, requires_grad=True)
        b2 = Tensor(b2_np, requires_grad=True)
        target = Tensor(target_np, requires_grad=False)

        Z1 = (X @ W1) + b1
        A1 = Z1.gelu()
        Z2 = (A1 @ W2) + b2
        diff = Z2 - target
        loss = (diff * diff).mean()
        loss.backward()

        # 2. PyTorch
        t_X = torch.tensor(X_np, requires_grad=False)
        t_W1 = torch.tensor(W1_np, requires_grad=True)
        t_b1 = torch.tensor(b1_np, requires_grad=True)
        t_W2 = torch.tensor(W2_np, requires_grad=True)
        t_b2 = torch.tensor(b2_np, requires_grad=True)
        t_target = torch.tensor(target_np, requires_grad=False)

        t_Z1 = torch.matmul(t_X, t_W1) + t_b1
        t_A1 = nn.functional.gelu(t_Z1, approximate='tanh')
        t_Z2 = torch.matmul(t_A1, t_W2) + t_b2
        t_loss = torch.mean((t_Z2 - t_target) ** 2)
        t_loss.backward()

        # Đối chiếu kết quả Loss và Gradients
        assert np.isclose(loss.data.item(), t_loss.item(), atol=1e-5)
        assert np.allclose(W1.grad, t_W1.grad.numpy(), atol=1e-4)
        assert np.allclose(b1.grad, t_b1.grad.numpy(), atol=1e-4)
        assert np.allclose(W2.grad, t_W2.grad.numpy(), atol=1e-4)
        assert np.allclose(b2.grad, t_b2.grad.numpy(), atol=1e-4)
