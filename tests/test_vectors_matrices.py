"""
tests/test_vectors_matrices.py

Bộ kiểm thử Unit Test cho Ngày 1 - Tuần 1:
Xác thực độ chính xác số học của các phép toán Vector, Matrix, Broadcasting và Embedding Lookup.
"""

import numpy as np
import pytest
from research.mathematics.vectors_matrices import (
    create_scalar,
    create_vector,
    create_matrix,
    create_tensor_3d,
    vector_add_manual,
    vector_scalar_mul_manual,
    vector_dot_product_manual,
    vector_l2_norm,
    vector_cosine_similarity,
    matrix_transpose_manual,
    matrix_vector_mul_manual,
    matrix_matrix_mul_manual,
    apply_bias_with_broadcasting,
    embedding_lookup,
)


class TestVectorOperations:
    def test_vector_creation_and_shapes(self):
        s = create_scalar(3.14)
        v = create_vector([1.0, 2.0, 3.0])
        m = create_matrix([[1.0, 2.0], [3.0, 4.0]])
        t = create_tensor_3d([[[1.0, 2.0]], [[3.0, 4.0]]])

        assert s.ndim == 0
        assert v.shape == (3,)
        assert m.shape == (2, 2)
        assert t.shape == (2, 1, 2)

    def test_vector_add_manual(self):
        u = np.array([1.0, 2.5, -3.0], dtype=np.float32)
        v = np.array([4.0, -1.0, 2.0], dtype=np.float32)
        res_manual = vector_add_manual(u, v)
        res_numpy = u + v
        assert np.allclose(res_manual, res_numpy)

    def test_vector_scalar_mul_manual(self):
        v = np.array([2.0, -4.0, 6.0], dtype=np.float32)
        alpha = 2.5
        res_manual = vector_scalar_mul_manual(v, alpha)
        res_numpy = v * alpha
        assert np.allclose(res_manual, res_numpy)

    def test_vector_dot_product(self):
        u = np.array([1.0, 3.0, -5.0], dtype=np.float32)
        v = np.array([4.0, -2.0, -1.0], dtype=np.float32)
        # 1*4 + 3*(-2) + (-5)*(-1) = 4 - 6 + 5 = 3.0
        dot_manual = vector_dot_product_manual(u, v)
        dot_numpy = float(np.dot(u, v))
        assert np.isclose(dot_manual, 3.0)
        assert np.isclose(dot_manual, dot_numpy)

    def test_norm_and_cosine_similarity(self):
        # Hai vector trực giao (vuông góc nhau) -> dot product = 0 -> cosine sim = 0
        u = np.array([1.0, 0.0], dtype=np.float32)
        v = np.array([0.0, 5.0], dtype=np.float32)
        assert np.isclose(vector_l2_norm(u), 1.0)
        assert np.isclose(vector_l2_norm(v), 5.0)
        assert np.isclose(vector_cosine_similarity(u, v), 0.0)

        # Hai vector cùng hướng -> cosine sim = 1.0
        w = np.array([2.0, 0.0], dtype=np.float32)
        assert np.isclose(vector_cosine_similarity(u, w), 1.0)


class TestMatrixOperations:
    def test_matrix_transpose_manual(self):
        A = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ], dtype=np.float32)
        A_t_manual = matrix_transpose_manual(A)
        assert A_t_manual.shape == (3, 2)
        assert np.allclose(A_t_manual, A.T)
        # Tính chất (A^T)^T == A
        assert np.allclose(matrix_transpose_manual(A_t_manual), A)

    def test_matrix_vector_mul_manual(self):
        A = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ], dtype=np.float32)  # shape (3, 2)
        v = np.array([0.5, -1.0], dtype=np.float32)  # shape (2,)

        # [1*0.5 + 2*(-1), 3*0.5 + 4*(-1), 5*0.5 + 6*(-1)] = [-1.5, -2.5, -3.5]
        res_manual = matrix_vector_mul_manual(A, v)
        res_numpy = np.dot(A, v)
        assert np.allclose(res_manual, [-1.5, -2.5, -3.5])
        assert np.allclose(res_manual, res_numpy)

    def test_matrix_matrix_mul_manual(self):
        A = np.array([
            [1.0, 2.0],
            [3.0, 4.0]
        ], dtype=np.float32)
        B = np.array([
            [5.0, 6.0],
            [7.0, 8.0]
        ], dtype=np.float32)

        C_manual = matrix_matrix_mul_manual(A, B)
        C_numpy = np.matmul(A, B)
        # C[0,0] = 1*5 + 2*7 = 19; C[0,1] = 1*6 + 2*8 = 22
        # C[1,0] = 3*5 + 4*7 = 43; C[1,1] = 3*6 + 4*8 = 50
        assert np.allclose(C_manual, [[19.0, 22.0], [43.0, 50.0]])
        assert np.allclose(C_manual, C_numpy)

    def test_transpose_product_identity(self):
        # Tính chất: (A B)^T = B^T A^T
        A = np.random.randn(3, 4).astype(np.float32)
        B = np.random.randn(4, 2).astype(np.float32)

        AB_t = matrix_transpose_manual(matrix_matrix_mul_manual(A, B))
        Bt_At = matrix_matrix_mul_manual(matrix_transpose_manual(B), matrix_transpose_manual(A))
        assert np.allclose(AB_t, Bt_At, atol=1e-5)


class TestBroadcastingAndEmbeddings:
    def test_bias_broadcasting(self):
        # Giả lập hidden_states của 1 batch trong Transformer: (Batch=2, Seq=3, Dim=4)
        hidden_states = np.zeros((2, 3, 4), dtype=np.float32)
        bias = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        output = apply_bias_with_broadcasting(hidden_states, bias)
        assert output.shape == (2, 3, 4)
        # Mọi token ở mọi batch đều được cộng đúng vector bias
        for b in range(2):
            for t in range(3):
                assert np.allclose(output[b, t], bias)

    def test_embedding_lookup(self):
        vocab_size = 100
        d_model = 16
        np.random.seed(42)
        embedding_matrix = np.random.randn(vocab_size, d_model).astype(np.float32)

        token_ids = [5, 42, 99, 0]
        tokens_embedded = embedding_lookup(token_ids, embedding_matrix)

        assert tokens_embedded.shape == (4, d_model)
        for idx, tid in enumerate(token_ids):
            assert np.allclose(tokens_embedded[idx], embedding_matrix[tid])

    def test_embedding_out_of_vocab_error(self):
        vocab_size = 10
        d_model = 4
        embedding_matrix = np.zeros((vocab_size, d_model), dtype=np.float32)
        with pytest.raises(AssertionError):
            embedding_lookup([15], embedding_matrix)
