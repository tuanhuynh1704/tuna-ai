# Báo cáo Tổng kết Tuần 1: Nền tảng Toán học, Giải tích & Động cơ Vi phân

> **Dự án**: Tuna-AI (Bilingual LLM from Scratch)  
> **Thời gian thực hiện**: Tuần 1 (Ngày 1 $\to$ Ngày 7)  
> **Cột mốc liên quan**: **Milestone M1** (Linear Algebra & Calculus ứng dụng) & Tiền đề **Milestone M2** (Autograd)  
> **Nền tảng kiểm định**: Intel Core i7-1165G7, 16GB RAM, Python 3.11, Windows 11  

---

## 1. Trả lời 6 Câu hỏi Tổng kết Tuần (Theo Mục 17 MOTA.md)

### 1.1. Kiến thức nào đã hiểu thấu đáo?
1. **Bản chất của các bậc Tensor trong LLM**: 
   - Scalar ($0$D): Hệ số Attention scale $\frac{1}{\sqrt{d_k}}$, Learning rate $\alpha$, Loss.
   - Vector ($1$D): Token embedding đơn lẻ, Bias vector $b$.
   - Matrix ($2$D): Chuỗi văn bản $(T, D)$, Ma trận trọng số $W_Q, W_K, W_V, W_O$.
   - Tensor $3$D: Hidden states $(B, T, D)$ xuyên suốt các khối Transformer Block.
   - Tensor $4$D: Phân tách Multi-Head Attention $(B, H, T, d_{\text{head}})$.
2. **Cơ chế phân cấp bộ nhớ đệm (Cache Locality)**:
   - Hiểu lý do tại sao vòng lặp $i-j-k$ bị chậm do Cache Misses và cách vòng lặp $i-k-j$ cùng kỹ thuật Tiling $32 \times 32$ tối ưu hóa đường truyền L1 Cache Line (64 bytes).
   - Hiểu cách **Intel MKL / oneDNN** tận dụng tập lệnh **AVX-512 FMA** để tăng tốc tính toán ma trận lên hơn **$19.000$ lần** so với Python thuần trên CPU i7-1165G7.
3. **Cơ chế Chuẩn hóa trong LLM (LayerNorm vs RMSNorm)**:
   - Hiểu tại sao các LLM thế hệ mới (LLaMA, Mistral, Gemma) loại bỏ phép trừ Mean của LayerNorm để chuyển sang **RMSNorm**, giúp tiết kiệm $30 - 50\%$ chi phí tính toán trên CPU.
4. **Quy tắc Dây chuyền & Động cơ Tự động Vi phân (Autograd)**:
   - Bản chất của Đồ thị Tính toán có hướng không chu trình (DAG).
   - Tầm quan trọng sống còn của thuật toán **Sắp xếp Tô-pô (Topological Sort)** và cơ chế **tích lũy gradient (`+=`)** khi một biến rẽ nhánh trong mạng nơ-ron.
5. **Gradient kỳ diệu của Softmax Cross-Entropy**:
   - Chứng minh và hiện thực hóa đạo hàm $\frac{\partial L}{\partial \mathbf{z}} = \frac{\mathbf{p} - \mathbf{y}}{N}$.
6. **Thuật toán Tối ưu hóa AdamW**:
   - Hiểu sự cần thiết của việc tách rời Weight Decay (Decoupled Weight Decay) khỏi các moment thích ứng để bảo toàn khả năng tổng quát hóa của mô hình.

---

### 1.2. Khái niệm nào còn cần làm sâu hơn trong các tuần tới?
- **Rotary Position Embedding (RoPE)**: Cách mã hóa vị trí bằng ma trận xoay phức trong không gian 2D thay vì Positional Embedding học được cổ điển.
- **Tự động vi phân trên Tensor nhiều chiều (Tensor Autograd)**: Mở rộng lớp `Scalar` hiện tại thành cấu trúc `Tensor` hoàn chỉnh hỗ trợ Broadcast backward và Matmul backward ở tầng C/NumPy.

---

### 1.3. Implementation nào đã hoàn thành và vượt qua kiểm thử?

| Ngày | Module | File mã nguồn | Unit Tests | Bằng chứng kiểm định |
| :---: | :--- | :--- | :---: | :--- |
| **Ngày 1** | Phép toán Vector, Matrix, Broadcasting, Embedding Lookup | [vectors_matrices.py](file:///c:/Users/nkocn/OneDrive/Desktop/Plan/tuna-ai/research/mathematics/vectors_matrices.py) | 12 tests | Khớp chuẩn NumPy, cosine sim $\in [-1, 1]$. |
| **Ngày 2** | GEMM Cache-aware, Blocked Tiling, Scaled Dot-Product Attention | [matrix_multiplication.py](file:///c:/Users/nkocn/OneDrive/Desktop/Plan/tuna-ai/research/mathematics/matrix_multiplication.py) | 5 tests | Causal mask chặn rò rỉ token tương lai $100\%$. |
| **Ngày 3** | Norms, LayerNorm, RMSNorm, Gradient Clipping, Split/Merge Heads | [tensor_operations.py](file:///c:/Users/nkocn/OneDrive/Desktop/Plan/tuna-ai/research/mathematics/tensor_operations.py) | 6 tests | Khớp `torch.nn.LayerNorm` sai số $< 10^{-5}$. |
| **Ngày 4** | Đạo hàm giải tích (Sigmoid, Tanh, ReLU, GELU, SiLU), Cross-Entropy | [calculus_derivatives.py](file:///c:/Users/nkocn/OneDrive/Desktop/Plan/tuna-ai/research/mathematics/calculus_derivatives.py) | 7 tests | Khớp Finite Differences & `CrossEntropyLoss`. |
| **Ngày 5** | Động cơ Autograd DAG Engine, Topological Sort, Neuron Perceptron | [autograd.py](file:///c:/Users/nkocn/OneDrive/Desktop/Plan/tuna-ai/src/core/autograd.py) | 5 tests | Khớp `torch.autograd` sai số $< 10^{-6}$. |
| **Ngày 6** | Optimizers (SGDMomentum, AdamW) & Linear Regression | [linear_regression.py](file:///c:/Users/nkocn/OneDrive/Desktop/Plan/tuna-ai/research/machine_learning/linear_regression.py) | 3 tests | $R^2 > 0.99$, khớp Closed-form Normal Equation. |
| **Ngày 7** | Tổng kết Tuần 1 & Script Kiểm định Tích hợp | [run_week1_verification.py](file:///c:/Users/nkocn/OneDrive/Desktop/Plan/tuna-ai/scripts/run_week1_verification.py) | 41 tests | **100% (41/41) tests pass tuyệt đối.** |

---

### 1.4. Những lỗi đã gặp & Cách khắc phục
1. **Lỗi `ModuleNotFoundError` khi chạy script trong thư mục con**:
   - *Nguyên nhân*: Khi chạy `python research/mathematics/benchmark_matmul.py`, thư mục gốc không tự động nằm trong `sys.path`.
   - *Khắc phục*: Bổ sung logic chuẩn hóa `sys.path.insert(0, str(project_root))` vào đầu các script thực thi độc lập.
2. **Lỗi ghi đè Gradient trong Autograd**:
   - *Nguyên nhân*: Trong biểu thức có biến lặp lại như $f = x + x$, nếu dùng phép gán `=` thì gradient chỉ giữ giá trị của nhánh cuối cùng.
   - *Khắc phục*: Luôn sử dụng phép cộng dồn `self.grad += ...` theo đúng Multivariate Chain Rule.
3. **Cảnh báo `asyncio_default_fixture_loop_scope` trong Pytest**:
   - *Khắc phục*: Khai báo tường minh cấu hình `asyncio_default_fixture_loop_scope = "function"` trong [pyproject.toml](file:///c:/Users/nkocn/OneDrive/Desktop/Plan/tuna-ai/pyproject.toml).

---

### 1.5. Kết quả thực nghiệm có tính tái lập (Reproducibility) không?
- **Hoàn toàn tái lập 100%**:
  - Toàn bộ các script tạo dữ liệu ngẫu nhiên đều cố định `np.random.seed(42)` và `torch.manual_seed(42)`.
  - Mọi bài kiểm thử đều chạy tự động và độc lập thông qua lệnh `python -m pytest` hoặc script [run_week1_verification.py](file:///c:/Users/nkocn/OneDrive/Desktop/Plan/tuna-ai/scripts/run_week1_verification.py).

---

### 1.6. Tuần tiếp theo (Tuần 2) cần làm gì?

Theo kế hoạch tổng thể:
- **Chủ đề Tuần 2**: **Phát triển Thư viện Tensor & Mở rộng Động cơ Tự động Vi phân Đa chiều (Autograd Tensor Engine)**.
- **Nội dung trọng tâm**:
  1. Xây dựng lớp `Tensor` đa chiều hỗ trợ lưu trữ mảng dữ liệu contiguous trong bộ nhớ.
  2. Triển khai các phép toán Backward cho Tensor: Matrix Multiplication Backward ($dL/dA = dL/dC \cdot B^T, dL/dB = A^T \cdot dL/dC$).
  3. Xử lý Broadcasting Backward (thu gọn các chiều đã broadcast bằng `np.sum(..., axis=..., keepdims=True)`).
  4. Triển khai Multi-Layer Perceptron (MLP) hoàn chỉnh tự viết và giải quyết bài toán phân loại phi tuyến tính (XOR / Moon dataset) không phụ thuộc vào thư viện ngoài.

---

## 2. Đánh giá Mức độ Hoàn thành Cột mốc (Milestone Review)

- **Milestone M1 (Hiểu Linear Algebra & Calculus ứng dụng)**: **HOÀN THÀNH 100%**.
  - Có đầy đủ mã nguồn giải thuật, bộ kiểm thử Pytest, tài liệu lý thuyết và số liệu đo lường thực tế trên CPU i7-1165G7.
- **Tiền đề Milestone M2 (Automatic Differentiation)**: Đã hoàn thành lõi `Scalar` Autograd Engine và kiểm chứng khớp với PyTorch Autograd.
