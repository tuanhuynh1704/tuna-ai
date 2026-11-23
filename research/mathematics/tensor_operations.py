"""
research/mathematics/tensor_operations.py

Thực hành Ngày 3 - Tuần 1:
Nghiên cứu chuyên sâu về Norms, Distance, Quy tắc Broadcasting, 
và các phép biến đổi hình dạng Tensor (Shape Manipulation).
Triển khai LayerNorm, RMSNorm và Gradient Clipping bằng Global Norm.
"""

from typing import Tuple, List, Optional, Union
import numpy as np
import torch


# ==============================================================================
# 1. Các loại Chuẩn (Norms) & Khoảng cách (Distances)
# ==============================================================================

def vector_norm(v: np.ndarray, ord: Union[int, str] = 2) -> float:
    """
    Tính chuẩn vector theo bậc:
      - ord=1: Chuẩn L1 (Manhattan) = sum(|v_i|)
      - ord=2: Chuẩn L2 (Euclidean) = sqrt(sum(v_i^2))
      - ord='inf': Chuẩn cực đại (Chebyshev) = max(|v_i|)
    """
    assert v.ndim == 1, "Đầu vào phải là vector 1 chiều"
    if ord == 1:
        return float(np.sum(np.abs(v)))
    elif ord == 2:
        return float(np.sqrt(np.sum(v ** 2)))
    elif ord == "inf" or ord == float("inf"):
        return float(np.max(np.abs(v)))
    else:
        raise ValueError(f"Không hỗ trợ bậc chuẩn: {ord}")


def frobenius_norm(A: np.ndarray) -> float:
    """
    Chuẩn Frobenius của Ma trận / Tensor: ||A||_F = sqrt(sum(A_{i,j}^2))
    Dùng để đo lường độ lớn của ma trận trọng số (Weight Matrix) trong Regularization.
    """
    return float(np.sqrt(np.sum(A ** 2)))


def root_mean_square(x: np.ndarray, eps: float = 1e-6) -> float:
    """
    Giá trị trung bình bình phương (RMS): RMS(x) = sqrt(1/d * sum(x_i^2) + eps)
    Đây là nền tảng cốt lõi của RMSNorm trong các LLM hiện đại (LLaMA, Gemma, Mistral).
    """
    d = x.shape[-1]
    return float(np.sqrt(np.mean(x ** 2) + eps))


# ==============================================================================
# 2. Chuẩn hóa trong Mạng Nơ-ron (LayerNorm & RMSNorm)
# ==============================================================================

def layer_norm_manual(
    x: np.ndarray, 
    gamma: Optional[np.ndarray] = None, 
    beta: Optional[np.ndarray] = None, 
    eps: float = 1e-5
) -> np.ndarray:
    """
    Layer Normalization (Ba et al., 2016):
      LN(x) = ((x - mu) / sqrt(sigma^2 + eps)) * gamma + beta
    Chuẩn hóa trên chiều đặc trưng cuối cùng (d_model) cho từng token độc lập.
    
    x: (..., d_model)
    gamma: Trọng số co giãn (Scale parameter), shape (d_model,)
    beta: Trọng số dịch chuyển (Shift parameter), shape (d_model,)
    """
    d_model = x.shape[-1]
    if gamma is None:
        gamma = np.ones(d_model, dtype=x.dtype)
    if beta is None:
        beta = np.zeros(d_model, dtype=x.dtype)

    # 1. Tính giá trị trung bình (mean) theo chiều cuối
    mean = np.mean(x, axis=-1, keepdims=True)

    # 2. Tính phương sai (variance) theo chiều cuối
    var = np.var(x, axis=-1, keepdims=True)

    # 3. Chuẩn hóa về mean=0, std=1
    x_hat = (x - mean) / np.sqrt(var + eps)

    # 4. Áp dụng tham số học gamma và beta
    return x_hat * gamma + beta


def rms_norm_manual(
    x: np.ndarray, 
    gamma: Optional[np.ndarray] = None, 
    eps: float = 1e-6
) -> np.ndarray:
    """
    Root Mean Square Normalization (Zhang & Sennrich, 2019):
      RMSNorm(x) = (x / RMS(x)) * gamma
      trong đó RMS(x) = sqrt(mean(x^2) + eps)
    
    Ưu điểm so với LayerNorm:
      - Bỏ qua bước tính mean và trừ mean (Mean-centering invariance).
      - Tiết kiệm 30-50% chi phí tính toán normalization trên CPU/GPU.
      - Ổn định gradient tương đương LayerNorm.
    """
    d_model = x.shape[-1]
    if gamma is None:
        gamma = np.ones(d_model, dtype=x.dtype)

    # Tính RMS theo chiều cuối cùng
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)

    # Chuẩn hóa và scale
    return (x / rms) * gamma


# ==============================================================================
# 3. Cắt tỉa Gradient theo Chuẩn toàn cục (Global Norm Gradient Clipping)
# ==============================================================================

def clip_grad_norm_manual(
    gradients: List[np.ndarray], 
    max_norm: float, 
    eps: float = 1e-6
) -> Tuple[List[np.ndarray], float]:
    """
    Cắt tỉa gradient bằng chuẩn L2 toàn cục:
      total_norm = sqrt(sum(||g_i||_2^2))
      if total_norm > max_norm:
          g_i = g_i * (max_norm / (total_norm + eps))
          
    Đây là kỹ thuật BẮT BUỘC khi huấn luyện Transformer từ đầu (from scratch)
    để ngăn chặn hiện tượng bùng nổ gradient (Exploding Gradients) gây NaN loss.
    """
    # 1. Tính tổng bình phương chuẩn L2 của toàn bộ tham số
    total_norm_sq = 0.0
    for g in gradients:
        if g is not None:
            total_norm_sq += float(np.sum(g ** 2))
    
    total_norm = float(np.sqrt(total_norm_sq))

    # 2. Hệ số tỉ lệ co giãn
    clip_coef = max_norm / (total_norm + eps)
    
    # 3. Nếu tổng chuẩn vượt quá ngưỡng cho phép, co tỉ lệ toàn bộ gradient
    clipped_grads = []
    if clip_coef < 1.0:
        for g in gradients:
            if g is not None:
                clipped_grads.append(g * clip_coef)
            else:
                clipped_grads.append(None)
    else:
        clipped_grads = [g.copy() if g is not None else None for g in gradients]

    return clipped_grads, total_norm


# ==============================================================================
# 4. Biến đổi Hình dạng Tensor (Shape Manipulation trong Multi-Head Attention)
# ==============================================================================

def split_heads(x: np.ndarray, n_heads: int) -> np.ndarray:
    """
    Biến đổi Tensor từ Linear Projection sang cấu trúc Multi-Head Attention:
      Input shape : (Batch_Size, Context_Length, D_Model)
      Output shape: (Batch_Size, N_Heads, Context_Length, D_Head)
      trong đó D_Head = D_Model // N_Heads.
    """
    B, T, D = x.shape
    assert D % n_heads == 0, f"D_Model ({D}) phải chia hết cho n_heads ({n_heads})"
    d_head = D // n_heads

    # 1. Reshape thành (B, T, N_Heads, D_Head)
    x_reshaped = x.reshape(B, T, n_heads, d_head)

    # 2. Hoán vị trục (Transpose / Permute) thành (B, N_Heads, T, D_Head)
    # Trục 0: B, Trục 1: T, Trục 2: n_heads, Trục 3: d_head -> hoán vị (0, 2, 1, 3)
    x_heads = np.transpose(x_reshaped, (0, 2, 1, 3))
    return x_heads


def merge_heads(x: np.ndarray) -> np.ndarray:
    """
    Gộp các Heads sau khi tính xong Attention trở lại hình dạng chuẩn:
      Input shape : (Batch_Size, N_Heads, Context_Length, D_Head)
      Output shape: (Batch_Size, Context_Length, D_Model)
    """
    B, H, T, d_head = x.shape
    D = H * d_head

    # 1. Hoán vị ngược lại: (B, H, T, d_head) -> (B, T, H, d_head)
    x_transposed = np.transpose(x, (0, 2, 1, 3))

    # 2. Gộp 2 chiều cuối: (B, T, D)
    x_merged = x_transposed.reshape(B, T, D)
    return x_merged
