"""
research/mathematics/vectors_matrices.py

Thực hành Ngày 1 - Tuần 1:
Nghiên cứu và triển khai các phép toán nền tảng trên Scalar, Vector, Matrix và Tensor.
Mục tiêu: Hiểu rõ bản chất toán học và cách thức tính toán số (Numerical Computing) với NumPy,
đặt nền móng cho Token Embedding, Linear Projection và Attention trong Transformer.
"""

from typing import Tuple, List, Union
import numpy as np


# ==============================================================================
# 1. Khái niệm cơ bản & Khởi tạo (Scalars, Vectors, Matrices, Tensors)
# ==============================================================================

def create_scalar(value: float) -> np.ndarray:
    """Tạo một Scalar (Tensor bậc 0 - 0D Tensor)."""
    return np.array(value, dtype=np.float32)


def create_vector(elements: List[float]) -> np.ndarray:
    """
    Tạo một Vector (Tensor bậc 1 - 1D Tensor biểu diễn trong không gian R^n).
    Trong Transformer, vector đại diện cho 1 từ/token đơn lẻ sau khi embed (d_model).
    """
    return np.array(elements, dtype=np.float32)


def create_matrix(rows: List[List[float]]) -> np.ndarray:
    """
    Tạo một Ma trận (Tensor bậc 2 - 2D Tensor có kích thước R^{m x n}).
    Trong Transformer, ma trận đại diện cho:
      - Ma trận trọng số (Weight Matrix W) của Linear Layer.
      - Chuỗi token sau embedding: shape (context_length, d_model).
    """
    return np.array(rows, dtype=np.float32)


def create_tensor_3d(data: List[List[List[float]]]) -> np.ndarray:
    """
    Tạo Tensor bậc 3 (3D Tensor) có kích thước (Batch, Context_Length, D_Model).
    Đây là hình dạng chuẩn của dữ liệu đi qua các block Attention trong Transformer.
    """
    return np.array(data, dtype=np.float32)


# ==============================================================================
# 2. Các phép toán trên Vector (Vector Operations)
# ==============================================================================

def vector_add_manual(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Cộng 2 vector thủ công theo từng phần tử (Element-wise Addition)."""
    assert u.ndim == 1 and v.ndim == 1, "Cả hai đầu vào phải là 1D Vector"
    assert u.shape[0] == v.shape[0], "Hai vector phải cùng số chiều"
    
    result = np.zeros_like(u)
    for i in range(len(u)):
        result[i] = u[i] + v[i]
    return result


def vector_scalar_mul_manual(v: np.ndarray, alpha: float) -> np.ndarray:
    """Nhân vector với một đại lượng vô hướng alpha."""
    assert v.ndim == 1
    result = np.zeros_like(v)
    for i in range(len(v)):
        result[i] = v[i] * alpha
    return result


def vector_dot_product_manual(u: np.ndarray, v: np.ndarray) -> float:
    """
    Tích vô hướng giữa 2 vector: u . v = sum(u_i * v_i)
    Đo lường mức độ tương đồng định hướng giữa 2 vector.
    Ứng dụng cốt lõi: Scaled Dot-Product Attention trong Transformer: Q . K^T
    """
    assert u.ndim == 1 and v.ndim == 1
    assert u.shape[0] == v.shape[0]
    
    dot_sum = 0.0
    for i in range(len(u)):
        dot_sum += float(u[i] * v[i])
    return dot_sum


def vector_l2_norm(v: np.ndarray) -> float:
    """
    Chuẩn L2 (Euclidean Norm): ||v||_2 = sqrt(sum(v_i^2))
    Đo độ dài của vector trong không gian vector.
    """
    return float(np.sqrt(np.sum(v ** 2)))


def vector_cosine_similarity(u: np.ndarray, v: np.ndarray, eps: float = 1e-9) -> float:
    """
    Độ tương đồng Cosine giữa 2 vector: cos(theta) = (u . v) / (||u|| * ||v||)
    Giá trị trong khoảng [-1, 1]. Dùng đo lường sự tương đồng ngữ nghĩa giữa token embeddings.
    """
    dot = vector_dot_product_manual(u, v)
    norm_u = vector_l2_norm(u)
    norm_v = vector_l2_norm(v)
    return dot / (norm_u * norm_v + eps)


# ==============================================================================
# 3. Các phép toán trên Ma trận (Matrix Operations)
# ==============================================================================

def matrix_transpose_manual(A: np.ndarray) -> np.ndarray:
    """
    Chuyển vị ma trận: (A^T)_{i, j} = A_{j, i}
    Biến ma trận kích thước (m, n) thành (n, m).
    """
    assert A.ndim == 2
    m, n = A.shape
    A_t = np.zeros((n, m), dtype=A.dtype)
    for i in range(m):
        for j in range(n):
            A_t[j, i] = A[i, j]
    return A_t


def matrix_vector_mul_manual(A: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Phép nhân Ma trận với Vector: y = A v
    Ma trận A (m, n) nhân vector v (n,) tạo vector y (m,).
    Đây chính là phép chiếu tuyến tính (Linear Projection) không có bias: y = x W^T.
    """
    assert A.ndim == 2 and v.ndim == 1
    m, n = A.shape
    assert n == v.shape[0], f"Kích thước không tương thích: {A.shape} vs {v.shape}"
    
    result = np.zeros(m, dtype=A.dtype)
    for i in range(m):
        for j in range(n):
            result[i] += A[i, j] * v[j]
    return result


def matrix_matrix_mul_manual(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Nhân hai ma trận: C = A B với C_{i, j} = sum_k(A_{i, k} * B_{k, j})
    A: (m, k), B: (k, n) -> C: (m, n).
    """
    assert A.ndim == 2 and B.ndim == 2
    m, k1 = A.shape
    k2, n = B.shape
    assert k1 == k2, f"Không thể nhân ma trận {A.shape} với {B.shape}"
    
    C = np.zeros((m, n), dtype=A.dtype)
    for i in range(m):
        for j in range(n):
            for k in range(k1):
                C[i, j] += A[i, k] * B[k, j]
    return C


# ==============================================================================
# 4. Cơ chế Lan truyền (Broadcasting) trong Tính toán số
# ==============================================================================

def apply_bias_with_broadcasting(hidden_states: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """
    Mô phỏng phép cộng Bias trong Linear Layer của Transformer:
    hidden_states có shape: (Batch_Size, Context_Length, D_Model) ví dụ (2, 4, 8)
    bias có shape: (D_Model,) ví dụ (8,)
    NumPy tự động broadcast bias qua chiều Batch và Context_Length mà không cần sao chép bộ nhớ.
    """
    assert hidden_states.shape[-1] == bias.shape[0], "Chiều d_model phải khớp nhau"
    return hidden_states + bias


# ==============================================================================
# 5. Ứng dụng thực tế: Token Embedding Lookup
# ==============================================================================

def embedding_lookup(token_ids: List[int], embedding_matrix: np.ndarray) -> np.ndarray:
    """
    Tra cứu ma trận Embedding (Token Embedding Layer):
    token_ids: danh sách các ID số nguyên của token, ví dụ [12, 45, 9] (Context_Length,)
    embedding_matrix: Ma trận W_E có shape (Vocab_Size, D_Model)
    Trả về: Ma trận biểu diễn vector của chuỗi token có shape (Context_Length, D_Model).
    """
    assert embedding_matrix.ndim == 2, "Ma trận embedding phải có 2 chiều (vocab_size, d_model)"
    vocab_size, d_model = embedding_matrix.shape
    
    embeddings = []
    for tid in token_ids:
        assert 0 <= tid < vocab_size, f"Token ID {tid} vượt ngoài vocab_size {vocab_size}"
        # Trích xuất hàng thứ tid
        embeddings.append(embedding_matrix[tid])
        
    return np.array(embeddings, dtype=embedding_matrix.dtype)
