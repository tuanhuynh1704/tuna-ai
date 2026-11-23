"""
research/mathematics/calculus_derivatives.py

Thực hành Ngày 4 - Tuần 1:
Giải tích ứng dụng (Calculus) cho Mô hình Ngôn ngữ Lớn:
- Đạo hàm giải tích (Analytical Derivatives) của các hàm kích hoạt: Sigmoid, Tanh, ReLU, GELU, SiLU.
- Đạo hàm của hàm mất mát Softmax Cross-Entropy (Lõi huấn luyện LLM).
- Kiểm thử Gradient bằng phương pháp sai phân hữu hạn (Finite Differences Gradient Checking).
"""

from typing import Tuple, Callable
import numpy as np
from scipy.special import erf


# ==============================================================================
# 1. Phương pháp Sai phân Hữu hạn (Numerical Gradient Checking)
# ==============================================================================

def numerical_gradient_1d(
    func: Callable[[np.ndarray], np.ndarray], 
    x: np.ndarray, 
    eps: float = 1e-5
) -> np.ndarray:
    """
    Tính đạo hàm số trị bằng công thức sai phân trung tâm (Central Difference):
      f'(x) approx (f(x + eps) - f(x - eps)) / (2 * eps)
    Sai số lý thuyết O(eps^2). Dùng làm 'Ground Truth' để kiểm chứng đạo hàm giải tích.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        orig_val = x[idx]

        # f(x + eps)
        x[idx] = orig_val + eps
        f_plus = func(x).copy()

        # f(x - eps)
        x[idx] = orig_val - eps
        f_minus = func(x).copy()

        # Khôi phục giá trị gốc
        x[idx] = orig_val

        # Nếu func trả về mảng element-wise, lấy giá trị tại idx
        if f_plus.shape == x.shape:
            grad[idx] = (f_plus[idx] - f_minus[idx]) / (2.0 * eps)
        else:
            # Nếu func trả về scalar loss
            grad[idx] = (f_plus - f_minus) / (2.0 * eps)

        it.iternext()
    return grad


# ==============================================================================
# 2. Các hàm kích hoạt (Activation Functions) & Đạo hàm giải tích
# ==============================================================================

class Sigmoid:
    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        # Ổn định số học: tránh tràn số exp với x âm lớn
        return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))

    @staticmethod
    def backward(x: np.ndarray) -> np.ndarray:
        """sigma'(x) = sigma(x) * (1 - sigma(x))"""
        s = Sigmoid.forward(x)
        return s * (1.0 - s)


class Tanh:
    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    @staticmethod
    def backward(x: np.ndarray) -> np.ndarray:
        """tanh'(x) = 1 - tanh^2(x)"""
        t = np.tanh(x)
        return 1.0 - t ** 2


class ReLU:
    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @staticmethod
    def backward(x: np.ndarray) -> np.ndarray:
        """ReLU'(x) = 1 nếu x > 0 else 0 (tại x=0 lấy subgradient = 0)"""
        return np.where(x > 0.0, 1.0, 0.0)


class GELU:
    """
    Gaussian Error Linear Unit (Hendrycks & Gimpel, 2016).
    Được sử dụng trong GPT-2, GPT-3, BERT:
      GELU(x) = x * Phi(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    """
    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))

    @staticmethod
    def backward(x: np.ndarray) -> np.ndarray:
        """
        Đạo hàm chính xác của GELU:
          d/dx [0.5 * x * (1 + erf(x / sqrt(2)))]
          = 0.5 * (1 + erf(x / sqrt(2))) + 0.5 * x * (2 / sqrt(pi)) * exp(-x^2 / 2) * (1 / sqrt(2))
          = 0.5 * (1 + erf(x / sqrt(2))) + (x / sqrt(2 * pi)) * exp(-0.5 * x^2)
        """
        cdf = 0.5 * (1.0 + erf(x / np.sqrt(2.0)))
        pdf = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * (x ** 2))
        return cdf + x * pdf


class SiLU:
    """
    Sigmoid Linear Unit (Swish) (Elfwing et al., 2018):
      SiLU(x) = x * sigma(x)
    Đây là thành phần cốt lõi của SwiGLU trong LLaMA, Mistral, Gemma.
    """
    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        return x * Sigmoid.forward(x)

    @staticmethod
    def backward(x: np.ndarray) -> np.ndarray:
        """
        d/dx [x * sigma(x)] = sigma(x) + x * sigma'(x)
                            = sigma(x) + x * sigma(x) * (1 - sigma(x))
                            = sigma(x) * [1 + x * (1 - sigma(x))]
        """
        s = Sigmoid.forward(x)
        return s + x * s * (1.0 - s)


# ==============================================================================
# 3. Hàm mất mát Softmax Cross-Entropy & Gradient
# ==============================================================================

def softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax ổn định số học (trừ max trước khi exp)."""
    # logits shape: (..., vocab_size)
    shifted_logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(shifted_logits)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


def cross_entropy_loss_and_grad(
    logits: np.ndarray, 
    targets: np.ndarray, 
    eps: float = 1e-15
) -> Tuple[float, np.ndarray]:
    """
    Tính hàm mất mát Cross-Entropy và Gradient đối với Logits (Đầu ra cuối cùng của Transformer).
    
    logits: Ma trận điểm số trước softmax, shape (Batch_Size, Vocab_Size)
    targets: Vector ID của các token mục tiêu đúng, shape (Batch_Size,)
    
    Toán học:
      Loss = - 1/N * sum(log(p_{target}))
      d(Loss) / d(logits_i) = 1/N * (p_i - y_i)
      trong đó y là one-hot vector: y_i = 1 nếu i == target else 0.
    """
    N, V = logits.shape
    probs = softmax(logits)

    # 1. Tính Loss
    # Trích xuất xác suất tại vị trí token đúng
    target_probs = probs[np.arange(N), targets]
    loss = - float(np.mean(np.log(np.clip(target_probs, eps, 1.0))))

    # 2. Tính Gradient cực kỳ thanh lịch: (p - y) / N
    dlogits = probs.copy()
    dlogits[np.arange(N), targets] -= 1.0
    dlogits = dlogits / N

    return loss, dlogits
