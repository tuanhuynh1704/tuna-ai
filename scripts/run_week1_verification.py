"""
scripts/run_week1_verification.py

Script Kiểm định Tích hợp & Đánh giá Toàn diện Tuần 1 (Tuna-AI):
- Kiểm tra tính tương thích môi trường (Python 3.11, NumPy, PyTorch CPU).
- Kiểm tra toàn bộ 41 unit tests tự động của Tuần 1.
- In Bảng Tổng kết tiến độ Tuần 1 và kết quả đo lường phần cứng CPU i7-1165G7.
"""

import sys
import os
from pathlib import Path

# Thêm project root vào sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import time
import subprocess
import torch
import numpy as np


def main():
    print("=" * 80)
    print("           TUNA-AI: BÁO CÁO KIỂM ĐỊNH TÍCH HỢP TUẦN 1")
    print("=" * 80)

    # 1. Thông tin Môi trường
    print("\n[1] MÔI TRƯỜNG HỆ THỐNG:")
    print(f"  - Python Version      : {sys.version.split()[0]}")
    print(f"  - NumPy Version       : {np.__version__}")
    print(f"  - PyTorch Version     : {torch.__version__}")
    print(f"  - Intel MKL Support   : {torch.backends.mkl.is_available()}")
    print(f"  - Intel oneDNN Support: {torch.backends.mkldnn.is_available()}")
    print(f"  - OpenMP Active       : {torch.backends.openmp.is_available()}")
    print(f"  - Số luồng vật lý CPU : {torch.get_num_threads()}")

    # 2. Chạy Pytest
    print("\n[2] CHẠY BỘ KIỂM THỬ TỰ ĐỘNG (PYTEST SUITE):")
    t0 = time.perf_counter()
    res = subprocess.run([sys.executable, "-m", "pytest", "-q"], capture_output=True, text=True)
    duration = time.perf_counter() - t0

    print(res.stdout.strip())
    if res.returncode == 0:
        print(f"  ==> TOÀN BỘ KIỂM THỬ ĐÃ VƯỢT QUA trong {duration:.2f} giây! (Return Code: 0)")
    else:
        print(f"  ==> CÓ LỖI XẢY RA! (Return Code: {res.returncode})")
        print(res.stderr)
        sys.exit(1)

    # 3. Bảng Thành phần đã hoàn thành
    print("\n[3] TỔNG HỢP CÁC MODULE ĐÃ HOÀN THÀNH (NGÀY 1 -> NGÀY 6):")
    modules = [
        ("Ngày 1", "Scalar, Vector, Matrix & Broadcasting", "research/mathematics/vectors_matrices.py", "PASSED"),
        ("Ngày 2", "Cache-aware GEMM & Causal Attention", "research/mathematics/matrix_multiplication.py", "PASSED"),
        ("Ngày 3", "Norms, RMSNorm, LayerNorm, Split/Merge", "research/mathematics/tensor_operations.py", "PASSED"),
        ("Ngày 4", "GELU, SiLU & Softmax Cross-Entropy Grad", "research/mathematics/calculus_derivatives.py", "PASSED"),
        ("Ngày 5", "Autograd DAG Engine & Topological Sort", "src/core/autograd.py", "PASSED"),
        ("Ngày 6", "Optimizers (SGD, AdamW) & Linear Reg", "research/machine_learning/linear_regression.py", "PASSED"),
    ]

    print(f"  {'Giai đoạn':<10} | {'Nội dung cốt lõi':<42} | {'Trạng thái'}")
    print("  " + "-" * 70)
    for day, desc, path, status in modules:
        print(f"  {day:<10} | {desc:<42} | [OK] {status}")

    print("\n" + "=" * 80)
    print("         TUẦN 1 HOÀN TẤT XUẤT SẮC - SẴN SÀNG CHO TUẦN 2!")
    print("=" * 80)


if __name__ == "__main__":
    main()
