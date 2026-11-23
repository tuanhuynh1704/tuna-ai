"""
research/mathematics/benchmark_matmul.py

Script Benchmark đo lường hiệu năng và thông lượng tính toán (GFLOPS)
trên CPU Intel Core i7-1165G7.
So sánh:
1. Python Naive (i-j-k)
2. Python Cache-aware (i-k-j)
3. Python Tiled (Phân khối 32x32)
4. NumPy (BLAS / OpenBLAS vectorization)
5. PyTorch CPU (Intel MKL / oneDNN AVX-512)
"""

import sys
import os
from pathlib import Path

# Thêm root directory vào sys.path để có thể chạy script độc lập
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import time
import numpy as np
import torch
from research.mathematics.matrix_multiplication import (
    matmul_naive_ijk,
    matmul_cache_friendly_ikj,
    matmul_tiled
)


def run_benchmark(sizes: list = [64, 128, 256]):
    print("=" * 70)
    print("BENCHMARK MATRIX MULTIPLICATION (GEMM) TRÊN CPU INTEL i7-1165G7")
    print("=" * 70)
    torch.set_num_threads(4)  # Khóa 4 nhân vật lý

    for N in sizes:
        print(f"\n---> Kích thước ma trận: {N} x {N} (Số phép tính float: {2 * N**3 / 1e6:.2f} MFLOP)")
        A = np.random.randn(N, N).astype(np.float32)
        B = np.random.randn(N, N).astype(np.float32)

        # 1. Naive i-j-k (Chỉ chạy khi N <= 128 để tránh quá lâu)
        if N <= 128:
            t0 = time.perf_counter()
            _ = matmul_naive_ijk(A, B)
            t_ijk = (time.perf_counter() - t0) * 1000.0
            gflops_ijk = (2 * N**3) / (t_ijk / 1000.0) / 1e9
            print(f"  1. Naive (i-j-k)        : {t_ijk:9.2f} ms | {gflops_ijk:.4f} GFLOPS (baseline 1.0x)")
        else:
            t_ijk = None
            print(f"  1. Naive (i-j-k)        :     [Bỏ qua vì N={N} quá chậm]")

        # 2. Cache-friendly i-k-j
        if N <= 256:
            t0 = time.perf_counter()
            _ = matmul_cache_friendly_ikj(A, B)
            t_ikj = (time.perf_counter() - t0) * 1000.0
            gflops_ikj = (2 * N**3) / (t_ikj / 1000.0) / 1e9
            speedup = f"{t_ijk / t_ikj:.1f}x" if t_ijk else "N/A"
            print(f"  2. Cache-aware (i-k-j)  : {t_ikj:9.2f} ms | {gflops_ikj:.4f} GFLOPS (Nhanh hơn: {speedup})")

        # 3. Tiled 32x32
        if N <= 256:
            t0 = time.perf_counter()
            _ = matmul_tiled(A, B, block_size=32)
            t_tile = (time.perf_counter() - t0) * 1000.0
            gflops_tile = (2 * N**3) / (t_tile / 1000.0) / 1e9
            speedup = f"{t_ijk / t_tile:.1f}x" if t_ijk else "N/A"
            print(f"  3. Tiled (32x32 block)  : {t_tile:9.2f} ms | {gflops_tile:.4f} GFLOPS (Nhanh hơn: {speedup})")

        # 4. NumPy BLAS
        loops = 100 if N <= 128 else 20
        t0 = time.perf_counter()
        for _ in range(loops):
            _ = np.matmul(A, B)
        t_np = ((time.perf_counter() - t0) / loops) * 1000.0
        gflops_np = (2 * N**3) / (t_np / 1000.0) / 1e9
        speedup = f"{t_ijk / t_np:.0f}x" if t_ijk else "N/A"
        print(f"  4. NumPy BLAS (AVX)     : {t_np:9.4f} ms | {gflops_np:.4f} GFLOPS (Nhanh hơn: {speedup})")

        # 5. PyTorch CPU (MKL / oneDNN)
        tA = torch.from_numpy(A)
        tB = torch.from_numpy(B)
        t0 = time.perf_counter()
        for _ in range(loops):
            _ = torch.matmul(tA, tB)
        t_pt = ((time.perf_counter() - t0) / loops) * 1000.0
        gflops_pt = (2 * N**3) / (t_pt / 1000.0) / 1e9
        speedup = f"{t_ijk / t_pt:.0f}x" if t_ijk else "N/A"
        print(f"  5. PyTorch MKL (AVX-512): {t_pt:9.4f} ms | {gflops_pt:.4f} GFLOPS (Nhanh hơn: {speedup})")


if __name__ == "__main__":
    run_benchmark(sizes=[64, 128])
