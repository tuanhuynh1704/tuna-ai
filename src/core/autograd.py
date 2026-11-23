"""
src/core/autograd.py

Thực hành Ngày 5 - Tuần 1 (Tiền đề cốt lõi của Milestone M2):
Triển khai Động cơ Tự động Vi phân (Automatic Differentiation Engine - Autograd)
dựa trên Đồ thị Tính toán có hướng không chu trình (DAG) và Sắp xếp Tô-pô (Topological Sort).
Tham khảo kiến trúc của micrograd nhưng được thiết kế độc lập, có type hints và mở rộng.
"""

from typing import Set, List, Union, Tuple, Optional
import math


class Scalar:
    """
    Lớp biểu diễn một nút (Node) trong Đồ thị Tính toán (Computational Graph).
    Lưu trữ giá trị số (data), đạo hàm riêng (grad), các nút cha sinh ra nó (_prev),
    và hàm lan truyền ngược cục bộ (_backward).
    """

    def __init__(
        self, 
        data: float, 
        _children: Tuple['Scalar', ...] = (), 
        _op: str = '', 
        label: str = ''
    ):
        self.data: float = float(data)
        self.grad: float = 0.0
        self._backward = lambda: None
        self._prev: Set['Scalar'] = set(_children)
        self._op: str = _op
        self.label: str = label

    def __repr__(self) -> str:
        label_str = f"'{self.label}', " if self.label else ""
        return f"Scalar({label_str}data={self.data:.6f}, grad={self.grad:.6f})"

    # ==========================================================================
    # Phép Cộng (Addition: z = x + y)
    # dz/dx = 1, dz/dy = 1
    # Chain Rule: dL/dx += dL/dz * 1, dL/dy += dL/dz * 1
    # ==========================================================================
    def __add__(self, other: Union['Scalar', float, int]) -> 'Scalar':
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: Union['Scalar', float, int]) -> 'Scalar':
        return self + other

    # ==========================================================================
    # Phép Nhân (Multiplication: z = x * y)
    # dz/dx = y, dz/dy = x
    # Chain Rule: dL/dx += dL/dz * y, dL/dy += dL/dz * x
    # ==========================================================================
    def __mul__(self, other: Union['Scalar', float, int]) -> 'Scalar':
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: Union['Scalar', float, int]) -> 'Scalar':
        return self * other

    # ==========================================================================
    # Phép Lũy thừa (Power: z = x^n)
    # dz/dx = n * x^(n - 1)
    # Chain Rule: dL/dx += dL/dz * n * x^(n - 1)
    # ==========================================================================
    def __pow__(self, other: Union[int, float]) -> 'Scalar':
        assert isinstance(other, (int, float)), "Chỉ hỗ trợ lũy thừa với số thực/nguyên"
        out = Scalar(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad

        out._backward = _backward
        return out

    # ==========================================================================
    # Các phép toán phái sinh (Sub, Div, Neg)
    # ==========================================================================
    def __neg__(self) -> 'Scalar':
        return self * -1.0

    def __sub__(self, other: Union['Scalar', float, int]) -> 'Scalar':
        return self + (-other)

    def __rsub__(self, other: Union['Scalar', float, int]) -> 'Scalar':
        return other + (-self)

    def __truediv__(self, other: Union['Scalar', float, int]) -> 'Scalar':
        other = other if isinstance(other, Scalar) else Scalar(other)
        return self * (other ** -1.0)

    def __rtruediv__(self, other: Union['Scalar', float, int]) -> 'Scalar':
        other = other if isinstance(other, Scalar) else Scalar(other)
        return other * (self ** -1.0)

    # ==========================================================================
    # Các hàm kích hoạt phi tuyến (Nonlinear Activation Functions)
    # ==========================================================================
    def relu(self) -> 'Scalar':
        out = Scalar(max(0.0, self.data), (self,), 'ReLU')

        def _backward():
            self.grad += (1.0 if self.data > 0.0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> 'Scalar':
        s = 1.0 / (1.0 + math.exp(-self.data)) if self.data >= 0 else math.exp(self.data) / (1.0 + math.exp(self.data))
        out = Scalar(s, (self,), 'Sigmoid')

        def _backward():
            self.grad += (s * (1.0 - s)) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> 'Scalar':
        t = math.tanh(self.data)
        out = Scalar(t, (self,), 'Tanh')

        def _backward():
            self.grad += (1.0 - t ** 2) * out.grad

        out._backward = _backward
        return out

    # ==========================================================================
    # Lan truyền ngược tự động (Automatic Backward Pass qua Topological Sort)
    # ==========================================================================
    def backward(self):
        """
        Thực hiện lan truyền ngược toàn bộ đồ thị tính toán:
        1. Xây dựng thứ tự sắp xếp tô-pô (Topological Order) đảm bảo mọi nút cha
           được tính gradient trước khi truyền tới nút con.
        2. Khởi tạo gradient của nút gốc = 1.0 (dL/dL = 1.0).
        3. Duyệt ngược thứ tự tô-pô và kích hoạt hàm _backward() của từng nút.
        """
        topo: List['Scalar'] = []
        visited: Set['Scalar'] = set()

        def build_topo(v: 'Scalar'):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Gradient của nút output gốc là 1.0
        self.grad = 1.0

        # Duyệt ngược theo chiều từ nút đầu ra về nút đầu vào
        for node in reversed(topo):
            node._backward()
