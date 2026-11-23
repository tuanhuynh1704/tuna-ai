"""
research/mathematics/matrix_multiplication.py

Thực hành Ngày 2 - Tuần 1:
Nghiên cứu chuyên sâu về Tích vô hướng (Dot Product) và Nhân ma trận (Matrix Multiplication - GEMM).
Tìm hiểu cơ chế Cache Locality, Tiling (Blocked Matrix Multiplication), và đo lường sự khác biệt
giữa thuật toán tuần tự O(N^3) và giải thuật tối ưu hóa phần cứng trên CPU Intel i7-1165G7.
"""

from typing import Tuple, List, Optional
import numpy as np


# ==============================================================================
# 1. Các giải thuật Nhân ma trận thủ công (Algorithmic Implementations)
# ==============================================================================

def matmul_naive_ijk(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Thuật toán nhân ma trận nguyên thủy với thứ tự vòng lặp i-j-k.
    A: (M, K), B: (K, N) -> C: (M, N).
    
    Nhược điểm phần cứng:
      - Trong vòng lặp trong cùng (k), B[k, j] truy xuất theo CỘT.
      - Trong bộ nhớ row-major (C-style), các phần tử cùng cột nằm cách nhau N phần tử (stride lớn).
      - Hậu quả: Gây ra liên tục CPU Cache Misses, CPU phải đợi nạp dữ liệu từ RAM.
    """
    assert A.ndim == 2 and B.ndim == 2
    M, K1 = A.shape
    K2, N = B.shape
    assert K1 == K2, f"Không thể nhân ma trận: A={A.shape}, B={B.shape}"

    C = np.zeros((M, N), dtype=A.dtype)
    for i in range(M):
        for j in range(N):
            total = 0.0
            for k in range(K1):
                total += A[i, k] * B[k, j]
            C[i, j] = total
    return C


def matmul_cache_friendly_ikj(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Thuật toán nhân ma trận tối ưu bộ nhớ đệm (Cache-aware) với thứ tự i-k-j.
    
    Cải tiến phần cứng:
      - Cố định A[i, k] ở vòng lặp giữa.
      - Trong vòng lặp trong cùng (j), B[k, j] và C[i, j] đều được truy cập TUẦN TỰ theo HÀNG (stride = 1).
      - Tận dụng triệt để nạp đường truyền L1 Cache Line (64 bytes = 16 số float32).
      - Giảm thiểu tối đa Cache Misses.
    """
    assert A.ndim == 2 and B.ndim == 2
    M, K1 = A.shape
    K2, N = B.shape
    assert K1 == K2, f"Không thể nhân ma trận: A={A.shape}, B={B.shape}"

    C = np.zeros((M, N), dtype=A.dtype)
    for i in range(M):
        for k in range(K1):
            r = A[i, k]
            # Vòng lặp j duyệt tuần tự liên tục trong bộ nhớ
            for j in range(N):
                C[i, j] += r * B[k, j]
    return C


def matmul_tiled(A: np.ndarray, B: np.ndarray, block_size: int = 32) -> np.ndarray:
    """
    Thuật toán phân khối (Tiling / Blocked Matrix Multiplication).
    Chia ma trận lớn thành các khối con (sub-blocks) kích thước block_size x block_size.
    
    Mục đích phần cứng:
      - CPU Intel i7-1165G7 có L1 Data Cache là 48KB mỗi nhân.
      - 3 khối ma trận con 32x32 số float32 chiếm: 3 * 32 * 32 * 4 bytes = 12.288 bytes (~12KB).
      - Toàn bộ 3 khối con nằm trọn vẹn 100% trong L1 Cache, giải phóng tối đa băng thông bộ nhớ RAM.
    """
    assert A.ndim == 2 and B.ndim == 2
    M, K1 = A.shape
    K2, N = B.shape
    assert K1 == K2, f"Không thể nhân ma trận: A={A.shape}, B={B.shape}"

    C = np.zeros((M, N), dtype=A.dtype)

    for i0 in range(0, M, block_size):
        i_max = min(i0 + block_size, M)
        for k0 in range(0, K1, block_size):
            k_max = min(k0 + block_size, K1)
            for j0 in range(0, N, block_size):
                j_max = min(j0 + block_size, N)

                # Thực hiện nhân trên khối con i-k-j nằm trong Cache
                for i in range(i0, i_max):
                    for k in range(k0, k_max):
                        r = A[i, k]
                        for j in range(j0, j_max):
                            C[i, j] += r * B[k, j]
    return C


# ==============================================================================
# 2. Ứng dụng trong Transformer: Scaled Dot-Product Attention Core
# ==============================================================================

def scaled_dot_product_attention_simple(
    Q: np.ndarray, 
    K: np.ndarray, 
    V: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Triển khai cốt lõi của Attention Mechanism:
      Attention(Q, K, V) = softmax((Q K^T) / sqrt(d_k) + Mask) V
      
    Q: (Context_Len, d_k)
    K: (Context_Len, d_k)
    V: (Context_Len, d_v)
    mask: Ma trận Causal Mask (Context_Len, Context_Len), 0 nếu hợp lệ, -inf nếu là token tương lai.
    
    Trả về:
      - Output context matrix: (Context_Len, d_v)
      - Attention weights (probabilities): (Context_Len, Context_Len)
    """
    d_k = Q.shape[-1]
    scale = 1.0 / np.sqrt(d_k)

    # 1. Tính Scores: S = (Q K^T) * scale
    # K.T có shape (d_k, Context_Len) -> S có shape (Context_Len, Context_Len)
    scores = np.matmul(Q, K.T) * scale

    # 2. Áp dụng Causal Mask (nếu có)
    if mask is not None:
        scores = scores + mask

    # 3. Tính Softmax theo hàng để có phân phối xác suất
    # Ổn định số học: trừ đi max của mỗi hàng để tránh tràn số exp
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # 4. Nhân với Values: Output = Attention_weights * V
    output = np.matmul(attention_weights, V)

    return output, attention_weights


def make_causal_mask(context_length: int) -> np.ndarray:
    """
    Tạo ma trận Causal Mask (Tam giác dưới = 0, Tam giác trên = -inf).
    Bảo đảm token ở vị trí t chỉ được chú ý đến các token từ 0 đến t.
    """
    mask = np.triu(np.full((context_length, context_length), -np.inf), k=1)
    return mask
