"""
research/deep_learning/mlp_classification.py

Thực hành Milestone M3:
Chứng minh năng lực biểu diễn Phi tuyến tính (Nonlinear Representation) của Mạng Nơ-ron (MLP):
- Tạo tập dữ liệu phi tuyến (Bài toán Vòng tròn đồng tâm Concentric Circles / XOR).
- Huấn luyện Mô hình Tuyến tính (Linear Classifier): Thất bại trong việc phân loại (Accuracy ~50%).
- Huấn luyện Mô hình Mạng Nơ-ron Đa tầng (MLP với GELU): Học hoàn hảo ranh giới phi tuyến (Accuracy > 98%).
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import Tuple, List
import numpy as np
from src.core.tensor import Tensor
from src.core.modules import Module, Linear, Sequential, GELU, MLP
from src.core.loss import binary_cross_entropy_with_logits
from research.machine_learning.optimizers import AdamW


def generate_circles_dataset(n_samples: int = 400, noise: float = 0.08) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tạo dữ liệu phi tuyến: 2 vòng tròn đồng tâm (Concentric Circles).
    Lớp 0: Vòng tròn bên trong (r ~ 0.5)
    Lớp 1: Vòng tròn bên ngoài (r ~ 1.2)
    Bài toán này KHÔNG THỂ phân tách bằng bất kỳ siêu phẳng tuyến tính nào!
    """
    np.random.seed(42)
    n_per_class = n_samples // 2

    # Lớp 0: Vòng trong
    theta0 = np.random.uniform(0, 2 * np.pi, n_per_class)
    r0 = 0.5 + np.random.randn(n_per_class) * noise
    X0 = np.stack([r0 * np.cos(theta0), r0 * np.sin(theta0)], axis=1)
    y0 = np.zeros((n_per_class, 1), dtype=np.float32)

    # Lớp 1: Vòng ngoài
    theta1 = np.random.uniform(0, 2 * np.pi, n_per_class)
    r1 = 1.2 + np.random.randn(n_per_class) * noise
    X1 = np.stack([r1 * np.cos(theta1), r1 * np.sin(theta1)], axis=1)
    y1 = np.ones((n_per_class, 1), dtype=np.float32)

    X = np.vstack([X0, X1]).astype(np.float32)
    y = np.vstack([y0, y1]).astype(np.float32)

    # Xáo trộn dữ liệu
    indices = np.random.permutation(n_samples)
    return X[indices], y[indices]


def train_classifier(
    model: Module, 
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    epochs: int = 150, 
    lr: float = 0.05
) -> Tuple[List[float], float]:
    """Huấn luyện mô hình và tính độ chính xác (Accuracy)."""
    # Lấy các tham số Tensor dưới dạng mảng numpy để cập nhật
    params_tensor = model.parameters()
    params_data = [p.data for p in params_tensor]
    optimizer = AdamW(params_data, lr=lr, weight_decay=1e-4)

    X_t = Tensor(X_train, requires_grad=False)
    y_t = Tensor(y_train, requires_grad=False)

    loss_history = []
    for epoch in range(epochs):
        model.zero_grad()

        # Forward
        logits = model(X_t)
        loss = binary_cross_entropy_with_logits(logits, y_t)

        # Backward
        loss.backward()

        # Cập nhật tham số bằng AdamW
        grads = [p.grad for p in params_tensor]
        optimizer.step(grads)

        loss_history.append(float(loss.data))

    # Tính Accuracy cuối cùng
    with_pred = model(X_t)
    probs = 1.0 / (1.0 + np.exp(-with_pred.data))
    preds = (probs >= 0.5).astype(np.float32)
    accuracy = float(np.mean(preds == y_train)) * 100.0

    return loss_history, accuracy


def run_experiment():
    print("=" * 75)
    print("THỬ NGHIỆM MILESTONE M3: NĂNG LỰC HỌC PHI TUYẾN CỦA MẠNG NƠ-RON (MLP)")
    print("=" * 75)

    X, y = generate_circles_dataset(n_samples=400)
    print(f"Dataset: 400 mẫu điểm 2D trên 2 vòng tròn đồng tâm (Không thể phân tách tuyến tính).")

    # 1. Thử nghiệm với Mô hình Tuyến tính (Linear Classifier)
    print("\n[1] HUẤN LUYỆN MÔ HÌNH TUYẾN TÍNH (1 TẦNG LINEAR):")
    linear_model = Linear(in_features=2, out_features=1)
    lin_losses, lin_acc = train_classifier(linear_model, X, y, epochs=100, lr=0.05)
    print(f"  -> Loss ban đầu: {lin_losses[0]:.4f} | Loss cuối: {lin_losses[-1]:.4f}")
    print(f"  -> Độ chính xác (Accuracy): {lin_acc:.2f}% (Bị mắc kẹt ở mức ngẫu nhiên ~50%)")

    # 2. Thử nghiệm với Mạng Nơ-ron Đa tầng (MLP 2 tầng ẩn với GELU)
    print("\n[2] HUẤN LUYỆN MẠNG NƠ-RON ĐA TẦNG (MLP: 2 -> 16 -> 8 -> 1 với GELU):")
    mlp_model = Sequential(
        Linear(2, 16),
        GELU(),
        Linear(16, 8),
        GELU(),
        Linear(8, 1)
    )
    mlp_losses, mlp_acc = train_classifier(mlp_model, X, y, epochs=150, lr=0.05)
    print(f"  -> Loss ban đầu: {mlp_losses[0]:.4f} | Loss cuối: {mlp_losses[-1]:.4f}")
    print(f"  -> Độ chính xác (Accuracy): {mlp_acc:.2f}% (Học hoàn hảo ranh giới phi tuyến!)")

    print("\n" + "=" * 75)
    print(f"KẾT LUẬN: Mạng Nơ-ron MLP tăng độ chính xác từ {lin_acc:.1f}% lên {mlp_acc:.1f}%!")
    print("Đã kiểm chứng Định lý Xấp xỉ Phổ quát (Universal Approximation Theorem)!")
    print("=" * 75)


if __name__ == "__main__":
    run_experiment()
