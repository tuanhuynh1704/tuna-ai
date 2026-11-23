import sys
import numpy as np
import torch


def test_python_version():
    """Kiểm tra phiên bản Python tương thích (>= 3.11)."""
    assert sys.version_info >= (3, 11)


def test_numpy_environment():
    """Kiểm tra hoạt động cơ bản của thư viện NumPy."""
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    dot_product = np.dot(a, b)
    assert np.isclose(dot_product, 32.0)


def test_torch_cpu_environment():
    """Kiểm tra hoạt động tính toán tensor thuần CPU của PyTorch."""
    t1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    t2 = torch.tensor([[2.0, 0.0], [1.0, 2.0]])
    res = torch.matmul(t1, t2)
    expected = torch.tensor([[4.0, 4.0], [10.0, 8.0]])
    assert torch.allclose(res, expected)
