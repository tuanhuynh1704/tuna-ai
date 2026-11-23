"""
tests/test_tensor_operations.py

Bộ kiểm thử Unit Test cho Ngày 3 - Tuần 1:
Kiểm thử độ chính xác số học của:
- Norms (L1, L2, Linf, Frobenius, RMS)
- Layer Normalization đối chiếu với PyTorch nn.LayerNorm
- RMS Normalization đối chiếu công thức chuẩn
- Cắt tỉa Gradient (Gradient Clipping) đối chiếu với PyTorch clip_grad_norm_
- Phép biến đổi Tensor Multi-Head Attention (Split & Merge Heads)
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from research.mathematics.tensor_operations import (
    vector_norm,
    frobenius_norm,
    root_mean_square,
    layer_norm_manual,
    rms_norm_manual,
    clip_grad_norm_manual,
    split_heads,
    merge_heads,
)


class TestNorms:
    def test_vector_norms(self):
        v = np.array([3.0, -4.0], dtype=np.float32)
        assert np.isclose(vector_norm(v, ord=1), 7.0)
        assert np.isclose(vector_norm(v, ord=2), 5.0)
        assert np.isclose(vector_norm(v, ord="inf"), 4.0)

    def test_frobenius_norm(self):
        A = np.array([
            [1.0, 2.0],
            [3.0, 4.0]
        ], dtype=np.float32)
        # sqrt(1 + 4 + 9 + 16) = sqrt(30) ~ 5.477225575
        expected = float(np.sqrt(30.0))
        assert np.isclose(frobenius_norm(A), expected)


class TestNormalizationLayers:
    def test_layer_norm_matches_pytorch(self):
        """Đối chiếu LayerNorm tự viết với torch.nn.LayerNorm chuẩn của PyTorch."""
        np.random.seed(42)
        B, T, D = 2, 4, 8
        x_np = np.random.randn(B, T, D).astype(np.float32)
        gamma = np.random.randn(D).astype(np.float32)
        beta = np.random.randn(D).astype(np.float32)

        # 1. Chạy hàm tự viết
        out_manual = layer_norm_manual(x_np, gamma=gamma, beta=beta, eps=1e-5)

        # 2. Chạy PyTorch LayerNorm
        ln_pt = nn.LayerNorm(D, eps=1e-5)
        with torch.no_grad():
            ln_pt.weight.copy_(torch.from_numpy(gamma))
            ln_pt.bias.copy_(torch.from_numpy(beta))
            out_pt = ln_pt(torch.from_numpy(x_np)).numpy()

        assert np.allclose(out_manual, out_pt, atol=1e-5)

    def test_rms_norm_properties(self):
        """Kiểm tra đặc tính RMSNorm: đầu ra sau chuẩn hóa có RMS xấp xỉ 1.0 (khi gamma=1)."""
        np.random.seed(42)
        x = np.random.randn(3, 5, 16).astype(np.float32) * 10.0  # dữ liệu có biên độ lớn
        out = rms_norm_manual(x, gamma=None, eps=1e-6)

        # Tính RMS của từng token dọc theo chiều D
        rms_vals = np.sqrt(np.mean(out ** 2, axis=-1))
        # Kỳ vọng các giá trị RMS đều xấp xỉ 1.0
        assert np.allclose(rms_vals, np.ones_like(rms_vals), atol=1e-3)


class TestGradientClipping:
    def test_clip_grad_norm_matches_pytorch(self):
        """Đối chiếu Gradient Clipping tự viết với torch.nn.utils.clip_grad_norm_."""
        # Giả lập 3 ma trận gradient tham số
        g1 = np.array([[10.0, -20.0], [30.0, 40.0]], dtype=np.float32)
        g2 = np.array([5.0, -5.0, 15.0], dtype=np.float32)
        grads_np = [g1, g2]

        max_norm = 10.0

        # 1. Triển khai thủ công
        clipped_manual, total_norm_manual = clip_grad_norm_manual(grads_np, max_norm=max_norm)

        # 2. PyTorch
        t_p1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        t_p2 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        t_p1.grad = torch.from_numpy(g1.copy())
        t_p2.grad = torch.from_numpy(g2.copy())

        total_norm_pt = torch.nn.utils.clip_grad_norm_([t_p1, t_p2], max_norm=max_norm)

        # So sánh chuẩn toàn cục
        assert np.isclose(total_norm_manual, total_norm_pt.item(), atol=1e-4)

        # So sánh gradient sau cắt tỉa
        assert np.allclose(clipped_manual[0], t_p1.grad.numpy(), atol=1e-4)
        assert np.allclose(clipped_manual[1], t_p2.grad.numpy(), atol=1e-4)


class TestTensorShapeTransformations:
    def test_split_and_merge_heads(self):
        """
        Kiểm tra tính bảo toàn dữ liệu qua chu trình:
        Linear Proj (B, T, D) -> Split Heads (B, H, T, d_head) -> Merge Heads (B, T, D)
        """
        B, T = 2, 8
        D = 64
        n_heads = 4
        d_head = D // n_heads

        x = np.random.randn(B, T, D).astype(np.float32)

        # Phân tách thành 4 heads
        heads = split_heads(x, n_heads=n_heads)
        assert heads.shape == (B, n_heads, T, d_head)

        # Gộp lại
        x_recovered = merge_heads(heads)
        assert x_recovered.shape == (B, T, D)

        # Dữ liệu phục hồi phải chính xác 100% không bị xáo trộn vị trí
        assert np.allclose(x, x_recovered)
