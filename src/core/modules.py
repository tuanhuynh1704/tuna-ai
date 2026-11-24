"""
src/core/modules.py

Thực hành Tuần 2 - Ngày 9 (Nền tảng Cốt lõi của Milestone M3):
Hệ thống Module Mạng Nơ-ron Hướng đối tượng (tương tự torch.nn.Module):
- Lớp cơ sở Module: quản lý tham số (parameters), chế độ train/eval, zero_grad().
- Tầng Linear (Linear Projection Layer): Khởi tạo trọng số Kaiming/He chuẩn hóa.
- Tầng Sequential: Kết nối chuỗi các tầng tính toán.
- Các tầng Activation: ReLU, GELU, Sigmoid, Tanh.
- Khối MLP (Multi-Layer Perceptron): Nền tảng của khối Feed-Forward trong Transformer.
"""

from typing import List, Tuple, Union, Optional
import numpy as np
from src.core.tensor import Tensor


class Module:
    """
    Lớp cơ sở cho toàn bộ các khối mạng nơ-ron trong Tuna-AI.
    Tự động thu thập các Tensor tham số có requires_grad=True.
    """

    def __init__(self):
        self._training: bool = True

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def parameters(self) -> List[Tensor]:
        """Thu thập đệ quy toàn bộ tham số của module và các sub-modules theo thứ tự khai báo."""
        params = []
        for attr in self.__dict__.values():
            if isinstance(attr, Tensor) and attr.requires_grad:
                params.append(attr)
            elif isinstance(attr, Module):
                params.extend(attr.parameters())
            elif isinstance(attr, (list, tuple)):
                for item in attr:
                    if isinstance(item, Tensor) and item.requires_grad:
                        params.append(item)
                    elif isinstance(item, Module):
                        params.extend(item.parameters())
        return params

    def zero_grad(self):
        """Xóa trắng gradient của toàn bộ tham số."""
        for p in self.parameters():
            if p.grad is not None:
                p.grad.fill(0.0)

    def train(self, mode: bool = True):
        """Chuyển đổi chế độ huấn luyện (train) hoặc suy luận (eval)."""
        self._training = mode
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Module):
                attr.train(mode)
            elif isinstance(attr, (list, tuple)):
                for item in attr:
                    if isinstance(item, Module):
                        item.train(mode)

    def eval(self):
        self.train(False)


class Linear(Module):
    """
    Tầng Chiếu Tuyến tính (Fully Connected / Dense Layer):
      Y = X @ W + b
    Trọng số được khởi tạo theo chuẩn Kaiming / He Uniform:
      W ~ U(-sqrt(k), sqrt(k)) với k = 1 / in_features
      b ~ U(-sqrt(k), sqrt(k))
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        # Kaiming uniform initialization (giống hệt PyTorch nn.Linear)
        k = 1.0 / np.sqrt(in_features)
        w_data = np.random.uniform(-k, k, size=(in_features, out_features)).astype(np.float32)
        self.weight = Tensor(w_data, requires_grad=True, label='weight')

        if self.use_bias:
            b_data = np.random.uniform(-k, k, size=(out_features,)).astype(np.float32)
            self.bias = Tensor(b_data, requires_grad=True, label='bias')
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class GELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.gelu()


class Sequential(Module):
    """Container kết nối các module tuần tự."""

    def __init__(self, *layers: Module):
        super().__init__()
        self.layers: List[Module] = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> List[Tensor]:
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params


class MLP(Module):
    """
    Multi-Layer Perceptron (Khối Feed-Forward FFN trong Transformer):
      Input (d_in) -> Linear -> GELU -> Linear -> Output (d_out)
    """

    def __init__(self, in_features: int, hidden_dim: int, out_features: int, activation: str = 'gelu'):
        super().__init__()
        self.fc1 = Linear(in_features, hidden_dim)
        if activation.lower() == 'gelu':
            self.act = GELU()
        elif activation.lower() == 'relu':
            self.act = ReLU()
        else:
            raise ValueError(f"Chưa hỗ trợ activation: {activation}")

        self.fc2 = Linear(hidden_dim, out_features)

    def forward(self, x: Tensor) -> Tensor:
        h = self.act(self.fc1(x))
        out = self.fc2(h)
        return out
