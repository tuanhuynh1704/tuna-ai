# Ghi chép Nghiên cứu Ngày 2 - Tuần 1: Dot Product, Matrix Multiplication (GEMM) & Attention Core

> **Mục tiêu**: Nghiên cứu sâu về Tích vô hướng (Dot Product) và Phép nhân ma trận (General Matrix Multiply - GEMM); phân tích cơ chế phân cấp bộ nhớ đệm (Cache Hierarchy) trên CPU Intel Core i7-1165G7; áp dụng trực tiếp xây dựng lõi Scaled Dot-Product Attention có Causal Mask.

---

## 1. Bản chất Toán học của Phép nhân Ma trận (GEMM)

Cho ma trận $A \in \mathbb{R}^{M \times K}$ và $B \in \mathbb{R}^{K \times N}$, ma trận tích $C = A B \in \mathbb{R}^{M \times N}$ có phần tử tại hàng $i$, cột $j$:
$$C_{i, j} = \sum_{k=1}^K A_{i, k} B_{k, j} = A_{i, :} \cdot B_{:, j}$$

- **Ý nghĩa**: Mỗi phần tử $C_{i, j}$ chính là **tích vô hướng (Dot Product)** giữa hàng thứ $i$ của $A$ và cột thứ $j$ của $B$.
- **Độ phức tạp tính toán**: $O(M \cdot K \cdot N)$. Với hai ma trận vuông kích thước $N \times N$, tổng số phép tính dấu phẩy động (FLOPs) là $2 N^3$ (gồm $N^3$ phép nhân và $N^3$ phép cộng).

---

## 2. Phân tích Phần cứng & Cơ chế Bộ nhớ trên CPU Intel i7-1165G7

### 2.1. Vấn đề của thứ tự vòng lặp truyền thống ($i-j-k$)
Trong bộ nhớ RAM chuẩn C-style (Row-Major), các phần tử trong cùng một hàng được lưu liên tiếp nhau, còn các phần tử cùng một cột nằm cách nhau một khoảng bước nhảy (stride):
- Khi tính $C_{i, j} += A[i, k] * B[k, j]$:
  - $A[i, k]$: Duyệt tuần tự theo hàng $i$ ($\to$ Tốt).
  - $B[k, j]$: Mỗi bước tăng $k$ thì địa chỉ bộ nhớ nhảy một khoảng $N \times 4\text{ bytes}$ ($\to$ Rất xấu).
- **Hậu quả**: Khi ma trận lớn hơn dung lượng L1/L2 Cache, CPU liên tục gặp hiện tượng **Cache Miss**; nhân tính toán phải nhàn rỗi (idle) hàng trăm chu kỳ xung nhịp để chờ RAM nạp dữ liệu.

### 2.2. Tối ưu hóa thứ tự vòng lặp ($i-k-j$)
Đổi vòng lặp $k$ ra ngoài và $j$ vào trong cùng:
- $r = A[i, k]$ được lưu trong thanh ghi (CPU Register).
- Vòng lặp trong cùng duyệt $j$ từ $0 \to N$: Cả $B[k, j]$ và $C[i, j]$ đều truy cập tuần tự theo hàng (stride = 1).
- Tận dụng tối đa mỗi dòng nhớ L1 Cache Line (64 bytes nạp được 16 số float32 cùng lúc).

### 2.3. Sức mạnh của phần cứng AVX-512 & Intel MKL
Trên Intel Core i7-1165G7:
- Nhân phần cứng hỗ trợ thanh ghi vector 512-bit (chứa 16 số `float32` hoặc 64 số `int8` qua VNNI).
- **Intel MKL / oneDNN** sử dụng vi mã Assembly chuyên dụng thực hiện FMA (Fused Multiply-Add: tính $a \times b + c$ trong 1 chu kỳ xung nhịp) trên toàn bộ 4 nhân thực.

---

## 3. Bằng chứng Đo lường Thực tế trên Máy i7-1165G7

Chạy trực tiếp từ script benchmark [benchmark_matmul.py](file:///c:/Users/nkocn/OneDrive/Desktop/Plan/tuna-ai/research/mathematics/benchmark_matmul.py):

| Kích thước | Phương pháp triển khai | Thời gian (ms) | Thông lượng (GFLOPS) | Tốc độ tương đối |
| :---: | :--- | :---: | :---: | :---: |
| **$64 \times 64$** | 1. Python Naive ($i-j-k$) | 105.75 ms | 0.0050 GFLOPS | $1\times$ (baseline) |
| | 2. NumPy BLAS (AVX) | **0.0068 ms** | **77.38 GFLOPS** | **15.608×** |
| | 3. PyTorch MKL (AVX-512) | 0.0320 ms | 16.36 GFLOPS | 3.300× |
| **$128 \times 128$** | 1. Python Naive ($i-j-k$) | 620.01 ms | 0.0068 GFLOPS | $1\times$ (baseline) |
| | 2. NumPy BLAS (AVX) | 0.1766 ms | 23.75 GFLOPS | 3.512× |
| | 3. PyTorch MKL (AVX-512) | **0.0316 ms** | **132.68 GFLOPS** | **19.614×** |

> [!NOTE]
> Ở kích thước $128 \times 128$, **PyTorch MKL đạt 132.68 GFLOPS**, nhanh gấp **19.614 lần** so với Python thông thường. Đây là lý do kiến trúc Transformer bắt buộc phải dựa vào kernel C++/MKL được biên dịch ở tầng thấp.

---

## 4. Ứng dụng vào Cốt lõi Attention Mechanism

Đã triển khai thành công hàm `scaled_dot_product_attention_simple()` và `make_causal_mask()`:
1. **Scaled Dot-Product**:
   $$S = \frac{Q K^T}{\sqrt{d_k}}$$
2. **Causal Masking**: Thêm ma trận mask tam giác trên (các vị trí tương lai gán bằng $-\infty$).
3. **Softmax**:
   $$P = \text{softmax}(S + M)$$
   Khi đi qua Softmax, các giá trị $-\infty$ trở thành xác suất $e^{-\infty} = 0$, ngăn chặn tuyệt đối việc rò rỉ token tương lai.
4. **Context Projection**: Nhân ma trận xác suất với ma trận Values $V$:
   $$\text{Output} = P V$$

---

## 5. Kết quả Kiểm thử (Evidence)
- File mã nguồn: [matrix_multiplication.py](file:///c:/Users/nkocn/OneDrive/Desktop/Plan/tuna-ai/research/mathematics/matrix_multiplication.py)
- File benchmark: [benchmark_matmul.py](file:///c:/Users/nkocn/OneDrive/Desktop/Plan/tuna-ai/research/mathematics/benchmark_matmul.py)
- File unit test: [test_matrix_multiplication.py](file:///c:/Users/nkocn/OneDrive/Desktop/Plan/tuna-ai/tests/test_matrix_multiplication.py)
- Kết quả test: **20/20 passed** (Toàn bộ test Ngày 1 + Ngày 2).
