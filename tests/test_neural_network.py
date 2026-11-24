"""
tests/test_neural_network.py

Bộ kiểm thử Unit Test cho Milestone M3:
Kiểm thử Hệ thống Module Mạng Nơ-ron (Module, Linear, Sequential, MLP, Loss):
- Kiểm thử thu thập tham số đệ quy và zero_grad().
- Kiểm thử tầng Linear đối chiếu với PyTorch torch.nn.Linear (Forward & Gradients).
- Kiểm thử hàm mất mát BinaryCrossEntropyWithLogits đối chiếu với PyTorch.
- Kiểm thử khả năng huấn luyện mạng MLP hội tụ trên bài toán phi tuyến (Accuracy > 95%).
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from src.core.tensor import Tensor
from src.core.modules import Module, Linear, Sequential, ReLU, GELU, MLP
from src.core.loss import binary_cross_entropy_with_logits, MSELoss
from research.deep_learning.mlp_classification import generate_circles_dataset, train_classifier


class TestModulesAndLayers:
    def test_parameter_collection_and_zero_grad(self):
        mlp = MLP(in_features=4, hidden_dim=8, out_features=2)
        params = mlp.parameters()
        # fc1.weight (4, 8), fc1.bias (8,), fc2.weight (8, 2), fc2.bias (2,) -> 4 tensors
        assert len(params) == 4
        assert params[0].shape == (4, 8)
        assert params[1].shape == (8,)
        assert params[2].shape == (8, 2)
        assert params[3].shape == (2,)

        # Giả lập gán gradient
        for p in params:
            p.grad = np.ones_like(p.data)

        # zero_grad() phải xóa về 0
        mlp.zero_grad()
        for p in params:
            assert np.all(p.grad == 0.0)

    def test_linear_matches_pytorch(self):
        """Đối chiếu Forward và Gradients của tầng Linear tự viết với torch.nn.Linear."""
        np.random.seed(42)
        B, in_feat, out_feat = 3, 5, 4
        X_np = np.random.randn(B, in_feat).astype(np.float32)

        # 1. Tự viết
        linear_custom = Linear(in_features=in_feat, out_features=out_feat, bias=True)
        X_custom = Tensor(X_np, requires_grad=True)
        out_custom = linear_custom(X_custom)
        loss_custom = out_custom.sum()
        loss_custom.backward()

        # 2. PyTorch
        linear_pt = nn.Linear(in_features=in_feat, out_features=out_feat, bias=True)
        with torch.no_grad():
            # Chú ý: PyTorch weight có shape (out_features, in_features), còn custom là (in_features, out_features)
            linear_pt.weight.copy_(torch.from_numpy(linear_custom.weight.data.T))
            linear_pt.bias.copy_(torch.from_numpy(linear_custom.bias.data))

        X_pt = torch.tensor(X_np, requires_grad=True)
        out_pt = linear_pt(X_pt)
        loss_pt = out_pt.sum()
        loss_pt.backward()

        # Đối chiếu kết quả Forward
        assert np.allclose(out_custom.data, out_pt.detach().numpy(), atol=1e-5)
        # Đối chiếu Gradients
        assert np.allclose(linear_custom.weight.grad, linear_pt.weight.grad.numpy().T, atol=1e-5)
        assert np.allclose(linear_custom.bias.grad, linear_pt.bias.grad.numpy(), atol=1e-5)
        assert np.allclose(X_custom.grad, X_pt.grad.numpy(), atol=1e-5)

    def test_bce_with_logits_matches_pytorch(self):
        """Đối chiếu BCEWithLogitsLoss tự viết với torch.nn.BCEWithLogitsLoss."""
        np.random.seed(123)
        logits_np = np.array([[-2.0], [0.5], [1.5], [-0.8]], dtype=np.float32)
        targets_np = np.array([[0.0], [1.0], [1.0], [0.0]], dtype=np.float32)

        # 1. Tự viết
        logits_custom = Tensor(logits_np, requires_grad=True)
        targets_custom = Tensor(targets_np, requires_grad=False)
        loss_custom = binary_cross_entropy_with_logits(logits_custom, targets_custom)
        loss_custom.backward()

        # 2. PyTorch
        logits_pt = torch.tensor(logits_np, requires_grad=True)
        targets_pt = torch.tensor(targets_np, requires_grad=False)
        criterion = nn.BCEWithLogitsLoss()
        loss_pt = criterion(logits_pt, targets_pt)
        loss_pt.backward()

        assert np.isclose(loss_custom.data, loss_pt.item(), atol=1e-5)
        assert np.allclose(logits_custom.grad, logits_pt.grad.numpy(), atol=1e-5)


class TestNonlinearLearningMilestoneM3:
    def test_mlp_solves_concentric_circles(self):
        """
        Kiểm định điều kiện nghiệm thu Milestone M3:
        Mạng Nơ-ron Đa tầng tự viết giải quyết được bài toán phi tuyến với độ chính xác > 95%.
        """
        X, y = generate_circles_dataset(n_samples=200, noise=0.05)
        mlp_model = Sequential(
            Linear(2, 16),
            GELU(),
            Linear(16, 8),
            GELU(),
            Linear(8, 1)
        )
        losses, accuracy = train_classifier(mlp_model, X, y, epochs=120, lr=0.08)

        # Kiểm tra loss giảm mạnh
        assert losses[-1] < 0.15
        # Độ chính xác phải đạt trên 95%
        assert accuracy >= 95.0, f"Độ chính xác chưa đạt yêu cầu: {accuracy:.2f}%"
