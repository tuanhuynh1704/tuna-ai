"""
src/core/tensor.py

Thực hành Tuần 2 - Ngày 8:
Động cơ Tự động Vi phân Đa chiều (Tensor Autograd Engine).
Mở rộng từ Scalar Autograd sang Tensor đa chiều trên mảng NumPy:
- Hỗ trợ phép cộng, trừ, nhân element-wise và nhân ma trận (Matmul @).
- Xử lý tự động thu gọn Gradient khi có Broadcasting (Unbroadcasting).
- Tích hợp các hàm kích hoạt phi tuyến (ReLU, Sigmoid, Tanh, GELU).
- Hỗ trợ xây dựng các tầng mạng nơ-ron (Linear Layer) trong Transformer.
"""

from typing import Set, List, Tuple, Union, Optional, Callable
import numpy as np


def unbroadcast(grad: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Thu gọn (Reduce) ma trận gradient về hình dạng gốc trước khi broadcast:
    Khi thực hiện A + B với A.shape=(2, 3, 4) và B.shape=(4,):
      Forward: B được mở rộng ảo thành (2, 3, 4).
      Backward: grad của phép cộng có shape (2, 3, 4). Ta phải cộng dồn (sum)
      grad dọc theo các chiều 0 và 1 để đưa về shape (4,) của B.
    """
    if grad.shape == target_shape:
        return grad

    grad_ndim = grad.ndim
    target_ndim = len(target_shape)

    # 1. Cộng dồn các chiều thừa phía trước (nếu grad có nhiều chiều hơn target)
    if grad_ndim > target_ndim:
        axes_to_sum = tuple(range(grad_ndim - target_ndim))
        grad = np.sum(grad, axis=axes_to_sum)

    # 2. Cộng dồn các chiều có kích thước 1 trong target_shape
    axes_to_keep = []
    for i, (g_dim, t_dim) in enumerate(zip(grad.shape, target_shape)):
        if t_dim == 1 and g_dim > 1:
            axes_to_keep.append(i)

    if axes_to_keep:
        grad = np.sum(grad, axis=tuple(axes_to_keep), keepdims=True)

    return grad


class Tensor:
    """
    Lớp biểu diễn một Tensor đa chiều có khả năng theo dõi đồ thị tính toán
    và tự động tính đạo hàm riêng (Autograd).
    """

    def __init__(
        self, 
        data: Union[float, int, List, np.ndarray], 
        requires_grad: bool = False, 
        _children: Tuple['Tensor', ...] = (), 
        _op: str = '', 
        label: str = ''
    ):
        if isinstance(data, np.ndarray):
            self.data: np.ndarray = data.astype(np.float32)
        else:
            self.data: np.ndarray = np.array(data, dtype=np.float32)

        self.requires_grad: bool = requires_grad
        self.grad: Optional[np.ndarray] = None
        self._backward: Callable[[], None] = lambda: None
        self._prev: Set['Tensor'] = set(_children)
        self._op: str = _op
        self.label: str = label

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def __repr__(self) -> str:
        label_str = f"'{self.label}', " if self.label else ""
        req_str = f", requires_grad={self.requires_grad}" if self.requires_grad else ""
        return f"Tensor({label_str}shape={self.shape}{req_str})"

    # ==========================================================================
    # Phép Cộng (Addition có Broadcasting: Z = X + Y)
    # ==========================================================================
    def __add__(self, other: Union['Tensor', float, int, np.ndarray]) -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other)
        req_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data + other.data, requires_grad=req_grad, _children=(self, other), _op='+')

        def _backward():
            if self.requires_grad:
                grad_self = unbroadcast(out.grad, self.shape)
                self.grad = grad_self if self.grad is None else self.grad + grad_self
            if other.requires_grad:
                grad_other = unbroadcast(out.grad, other.shape)
                other.grad = grad_other if other.grad is None else other.grad + grad_other

        out._backward = _backward
        return out

    def __radd__(self, other: Union['Tensor', float, int, np.ndarray]) -> 'Tensor':
        return self + other

    # ==========================================================================
    # Phép Nhân Element-wise (Hadamard Product: Z = X * Y)
    # ==========================================================================
    def __mul__(self, other: Union['Tensor', float, int, np.ndarray]) -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other)
        req_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data * other.data, requires_grad=req_grad, _children=(self, other), _op='*')

        def _backward():
            if self.requires_grad:
                grad_self = unbroadcast(out.grad * other.data, self.shape)
                self.grad = grad_self if self.grad is None else self.grad + grad_self
            if other.requires_grad:
                grad_other = unbroadcast(out.grad * self.data, other.shape)
                other.grad = grad_other if other.grad is None else other.grad + grad_other

        out._backward = _backward
        return out

    def __rmul__(self, other: Union['Tensor', float, int, np.ndarray]) -> 'Tensor':
        return self * other

    # ==========================================================================
    # Phép Trừ & Đổi dấu
    # ==========================================================================
    def __neg__(self) -> 'Tensor':
        return self * -1.0

    def __sub__(self, other: Union['Tensor', float, int, np.ndarray]) -> 'Tensor':
        return self + (-other)

    def __rsub__(self, other: Union['Tensor', float, int, np.ndarray]) -> 'Tensor':
        return other + (-self)

    # ==========================================================================
    # Phép Nhân Ma trận (Matrix Multiplication / GEMM: C = A @ B)
    # dL/dA = dL/dC @ B^T
    # dL/dB = A^T @ dL/dC
    # ==========================================================================
    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        assert isinstance(other, Tensor), "Toán hạng matmul (@) phải là Tensor"
        req_grad = self.requires_grad or other.requires_grad
        out = Tensor(np.matmul(self.data, other.data), requires_grad=req_grad, _children=(self, other), _op='@')

        def _backward():
            if self.requires_grad:
                # Nếu B là 2D: swapaxes(-1, -2) chính là B^T
                grad_self = np.matmul(out.grad, np.swapaxes(other.data, -1, -2))
                grad_self = unbroadcast(grad_self, self.shape)
                self.grad = grad_self if self.grad is None else self.grad + grad_self

            if other.requires_grad:
                # A^T @ dL/dC
                grad_other = np.matmul(np.swapaxes(self.data, -1, -2), out.grad)
                grad_other = unbroadcast(grad_other, other.shape)
                other.grad = grad_other if other.grad is None else other.grad + grad_other

        out._backward = _backward
        return out

    # ==========================================================================
    # Các hàm kích hoạt phi tuyến trên Tensor
    # ==========================================================================
    def relu(self) -> 'Tensor':
        out = Tensor(np.maximum(0.0, self.data), requires_grad=self.requires_grad, _children=(self,), _op='ReLU')

        def _backward():
            if self.requires_grad:
                grad_self = out.grad * (self.data > 0.0)
                self.grad = grad_self if self.grad is None else self.grad + grad_self

        out._backward = _backward
        return out

    def gelu(self) -> 'Tensor':
        """GELU dùng công thức xấp xỉ nhanh của GPT-2: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))"""
        k = np.sqrt(2.0 / np.pi)
        inner = k * (self.data + 0.044715 * (self.data ** 3))
        tanh_inner = np.tanh(inner)
        out_data = 0.5 * self.data * (1.0 + tanh_inner)

        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,), _op='GELU')

        def _backward():
            if self.requires_grad:
                sech2 = 1.0 - (tanh_inner ** 2)
                d_inner = k * (1.0 + 3.0 * 0.044715 * (self.data ** 2))
                dx = 0.5 * (1.0 + tanh_inner) + 0.5 * self.data * sech2 * d_inner
                grad_self = out.grad * dx
                self.grad = grad_self if self.grad is None else self.grad + grad_self

        out._backward = _backward
        return out

    # ==========================================================================
    # Thu gọn Tensor (Reduction Ops: sum, mean)
    # ==========================================================================
    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'Tensor':
        out_data = np.sum(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,), _op='sum')

        def _backward():
            if self.requires_grad:
                grad = out.grad
                if not keepdims and axis is not None:
                    # Mở rộng lại chiều để broadcast
                    axes = [axis] if isinstance(axis, int) else list(axis)
                    shape = list(self.shape)
                    for ax in axes:
                        shape[ax] = 1
                    grad = grad.reshape(shape)

                grad_self = np.broadcast_to(grad, self.shape)
                self.grad = grad_self if self.grad is None else self.grad + grad_self

        out._backward = _backward
        return out

    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'Tensor':
        n = self.data.size if axis is None else (np.prod([self.shape[ax] for ax in ([axis] if isinstance(axis, int) else axis)]))
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / float(n))

    # ==========================================================================
    # Lan truyền ngược (Full Tensor Backward Pass)
    # ==========================================================================
    def backward(self, grad: Optional[np.ndarray] = None):
        if grad is None:
            assert self.data.size == 1, "Chỉ có thể tự động gọi backward() không tham số cho Scalar Tensor (1 phần tử)"
            grad = np.ones_like(self.data)

        self.grad = grad

        # Sắp xếp tô-pô DFS
        topo: List['Tensor'] = []
        visited: Set['Tensor'] = set()

        def build_topo(v: 'Tensor'):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        for node in reversed(topo):
            node._backward()
