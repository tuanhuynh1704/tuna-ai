"""
tests/test_matrix_multiplication.py

Unit Test cho Ngày 2 - Tuần 1:
Kiểm thử tính tương đương số học giữa các thuật toán nhân ma trận (Naive, Cache-aware, Tiled),
và kiểm thử cơ chế Scaled Dot-Product Attention với Causal Mask.
"""

import numpy as np
import pytest
from research.mathematics.matrix_multiplication import (
    matmul_naive_ijk,
    matmul_cache_friendly_ikj,
    matmul_tiled,
    scaled_dot_product_attention_simple,
    make_causal_mask,
)


class TestMatrixMultiplicationAlgorithms:
    def test_algorithms_equivalence(self):
        """Kiểm tra cả 3 thuật toán tự viết đều cho kết quả chính xác tuyệt đối như NumPy."""
        np.random.seed(42)
        M, K, N = 32, 48, 24
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)

        C_expected = np.matmul(A, B)
        C_ijk = matmul_naive_ijk(A, B)
        C_ikj = matmul_cache_friendly_ikj(A, B)
        C_tiled = matmul_tiled(A, B, block_size=16)

        assert np.allclose(C_ijk, C_expected, atol=1e-5)
        assert np.allclose(C_ikj, C_expected, atol=1e-5)
        assert np.allclose(C_tiled, C_expected, atol=1e-5)

    def test_incompatible_shapes(self):
        """Bắt lỗi khi số cột của A khác số hàng của B."""
        A = np.zeros((3, 4), dtype=np.float32)
        B = np.zeros((5, 2), dtype=np.float32)
        with pytest.raises(AssertionError):
            matmul_cache_friendly_ikj(A, B)


class TestAttentionMechanism:
    def test_causal_mask_structure(self):
        """Kiểm tra cấu trúc Causal Mask tam giác trên."""
        T = 4
        mask = make_causal_mask(T)
        # Đường chéo chính và tam giác dưới phải là 0
        for i in range(T):
            for j in range(T):
                if j <= i:
                    assert mask[i, j] == 0.0
                else:
                    assert np.isneginf(mask[i, j])

    def test_scaled_dot_product_attention_causal_property(self):
        """
        Kiểm tra tính chất Causal của Attention:
        Trọng số Attention tại vị trí t không được rò rỉ thông tin về t + k (các token tương lai).
        """
        T = 5
        d_k = 8
        d_v = 8
        np.random.seed(42)
        Q = np.random.randn(T, d_k).astype(np.float32)
        K = np.random.randn(T, d_k).astype(np.float32)
        V = np.random.randn(T, d_v).astype(np.float32)

        mask = make_causal_mask(T)
        output, weights = scaled_dot_product_attention_simple(Q, K, V, mask=mask)

        assert output.shape == (T, d_v)
        assert weights.shape == (T, T)

        # Tổng xác suất mỗi hàng trong ma trận weights phải bằng 1.0
        row_sums = np.sum(weights, axis=-1)
        assert np.allclose(row_sums, np.ones(T), atol=1e-5)

        # Mọi vị trí tương lai (j > i) phải có trọng số attention đúng bằng 0.0
        for i in range(T):
            for j in range(i + 1, T):
                assert weights[i, j] == 0.0

    def test_attention_without_mask(self):
        """Kiểm tra Attention thông thường không mask (phân phối đều cho mọi vị trí)."""
        T, d = 3, 4
        Q = np.zeros((T, d), dtype=np.float32)
        K = np.zeros((T, d), dtype=np.float32)
        V = np.ones((T, d), dtype=np.float32)

        output, weights = scaled_dot_product_attention_simple(Q, K, V, mask=None)
        # Q.K^T = 0 -> Softmax([0, 0, 0]) = [1/3, 1/3, 1/3]
        expected_weights = np.full((T, T), 1.0 / T, dtype=np.float32)
        assert np.allclose(weights, expected_weights, atol=1e-5)
        # Output = V (vì V toàn là 1)
        assert np.allclose(output, np.ones((T, d), dtype=np.float32), atol=1e-5)
