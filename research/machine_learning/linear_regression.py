"""
research/machine_learning/linear_regression.py

Mô hình Hồi quy Tuyến tính (Linear Regression) triển khai từ đầu:
- Huấn luyện bằng Gradient Descent & AdamW.
- So sánh kết quả hội tụ với Nghiệm giải tích chính xác (Normal Equation: Closed-form).
"""

from typing import Tuple, List, Optional
import numpy as np
from research.machine_learning.optimizers import Optimizer, SGDMomentum, AdamW


class LinearRegressionModel:
    """
    Mô hình Hồi quy Tuyến tính Đa biến:
      y_pred = X @ W + b
    với X in R^{N x D}, W in R^{D x 1}, b in R
    """
    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        # Khởi tạo trọng số ngẫu nhiên nhỏ
        np.random.seed(42)
        self.W = np.random.randn(input_dim, 1).astype(np.float32) * 0.01
        self.b = np.zeros((1,), dtype=np.float32)

    @property
    def params(self) -> List[np.ndarray]:
        return [self.W, self.b]

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: y_hat = X @ W + b"""
        return np.matmul(X, self.W) + self.b

    def compute_loss_and_grads(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, List[np.ndarray]]:
        """
        Tính Mean Squared Error (MSE) Loss và Gradients giải tích:
          Loss = 1/N * sum((y_hat - y)^2)
          dL/dW = 2/N * X^T @ (y_hat - y)
          dL/db = 2/N * sum(y_hat - y)
        """
        N = X.shape[0]
        y_hat = self.forward(X)
        diff = y_hat - y  # (N, 1)

        # 1. MSE Loss
        loss = float(np.mean(diff ** 2))

        # 2. Gradients
        grad_W = (2.0 / N) * np.matmul(X.T, diff)
        grad_b = (2.0 / N) * np.sum(diff, axis=0)

        return loss, [grad_W, grad_b]

    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        optimizer: Optimizer, 
        epochs: int = 200, 
        batch_size: Optional[int] = None
    ) -> List[float]:
        """Huấn luyện mô hình qua các epochs, trả về lịch sử loss."""
        N = X.shape[0]
        loss_history = []

        for epoch in range(epochs):
            if batch_size is None or batch_size >= N:
                # Full Batch Gradient Descent
                loss, grads = self.compute_loss_and_grads(X, y)
                optimizer.step(grads)
            else:
                # Mini-batch Gradient Descent
                indices = np.random.permutation(N)
                X_shuffled = X[indices]
                y_shuffled = y[indices]

                epoch_losses = []
                for start_idx in range(0, N, batch_size):
                    end_idx = min(start_idx + batch_size, N)
                    X_batch = X_shuffled[start_idx:end_idx]
                    y_batch = y_shuffled[start_idx:end_idx]

                    b_loss, grads = self.compute_loss_and_grads(X_batch, y_batch)
                    optimizer.step(grads)
                    epoch_losses.append(b_loss)
                loss = float(np.mean(epoch_losses))

            loss_history.append(loss)

        return loss_history


def solve_normal_equation(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Nghiệm giải tích chính xác (Closed-form Normal Equation):
      theta* = (X_b^T X_b)^{-1} X_b^T y
      với X_b là ma trận X được bổ sung cột 1 đại diện cho bias.
    Độ phức tạp tính toán: O(D^3) - chỉ khả thi với số chiều D nhỏ.
    """
    N, D = X.shape
    X_b = np.hstack([X, np.ones((N, 1), dtype=X.dtype)])
    
    # theta = (X_b^T @ X_b)^(-1) @ X_b^T @ y
    XtX = np.matmul(X_b.T, X_b)
    XtY = np.matmul(X_b.T, y)
    theta = np.linalg.solve(XtX, XtY)

    W_star = theta[:D]
    b_star = float(theta[D, 0])
    return W_star, b_star
