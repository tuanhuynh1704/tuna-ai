# Ghi chép Nghiên cứu Ngày 1 - Tuần 1: Scalar, Vector, Matrix & Tensor

> **Mục tiêu**: Hiểu bản chất toán học của các phép toán trên không gian vector và ma trận; đối chiếu giữa thuật toán thủ công và tính toán số (NumPy Vectorization); liên hệ trực tiếp đến kiến trúc Transformer.

---

## 1. Khái niệm cốt lõi & Biểu diễn trong LLM

| Khái niệm | Bậc Tensor | Kích thước mẫu | Ứng dụng cụ thể trong Transformer |
| :--- | :---: | :--- | :--- |
| **Scalar** | 0D | `()` | Hệ số tỉ lệ Attention Scale $\frac{1}{\sqrt{d_k}}$, Learning rate $\alpha$, Loss value. |
| **Vector** | 1D | `(d_model,)` | Vector biểu diễn một token đơn lẻ ($x \in \mathbb{R}^{d}$), Bias vector $b$. |
| **Matrix** | 2D | `(context_len, d_model)` hoặc `(d_in, d_out)` | Chuỗi tokens của một câu; Ma trận trọng số chiếu tuyến tính $W_Q, W_K, W_V, W_O$. |
| **Tensor 3D** | 3D | `(batch_size, context_len, d_model)` | Dữ liệu ẩn (Hidden states) đi qua các khối Transformer Block. |
| **Tensor 4D** | 4D | `(batch_size, n_heads, context_len, d_k)` | Biểu diễn Multi-Head Attention sau khi phân tách đầu. |

---

## 2. Các phép toán nền tảng & Ứng dụng

### 2.1. Tích vô hướng (Dot Product)
- **Công thức**: 
  $$u \cdot v = \sum_{i=1}^d u_i v_i$$
- **Ý nghĩa hình học**: Đo lường góc và độ tương đồng giữa hai vector trong không gian nhiều chiều.
- **Ứng dụng**: Là cơ chế trung tâm của **Attention Mechanism**:
  $$\text{Score}(q, k) = q \cdot k^T$$
  Nếu Query $q$ và Key $k$ hướng về cùng một phía trong không gian vector, tích vô hướng lớn $\to$ điểm Attention cao.

### 2.2. Chuẩn L2 & Cosine Similarity
- **Chuẩn L2**: Độ dài vector $\|v\|_2 = \sqrt{\sum v_i^2}$.
- **Cosine Similarity**: 
  $$\cos(\theta) = \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$
  Dùng để đo sự tương đồng ngữ nghĩa giữa hai token embedding bất kể độ lớn biên độ của vector.

### 2.3. Chuyển vị Ma trận (Transpose) & Tính chất
- **Công thức**: $(A^T)_{i, j} = A_{j, i}$
- **Đẳng thức quan trọng**:
  $$(A B)^T = B^T A^T$$
  Đã được kiểm chứng thực nghiệm tự động trong [test_vectors_matrices.py](file:///c:/Users/nkocn/OneDrive/Desktop/Plan/tuna-ai/tests/test_vectors_matrices.py).

### 2.4. Cơ chế Lan truyền (Broadcasting)
- Khi cộng Tensor 3D `(Batch, Seq_Len, D_Model)` với Bias 1D `(D_Model,)`, thư viện tính toán số tự động mở rộng chiều mà không cấp phát thêm bộ nhớ vật lý:
  $$\text{Output}_{b, t, d} = \text{Input}_{b, t, d} + \text{Bias}_d$$

### 2.5. Token Embedding Lookup
- Về bản chất toán học, tra cứu Embedding là phép nhân ma trận giữa **One-hot vector** $e_i \in \mathbb{R}^{V}$ và Ma trận trọng số $W_E \in \mathbb{R}^{V \times D}$:
  $$x = e_i W_E$$
- Trong thực tế tính toán, để tối ưu tốc độ và bộ nhớ, thao tác này được thực hiện bằng phép lấy chỉ mục hàng (Row Indexing): $W_E[\text{token\_id}]$.

---

## 3. Bằng chứng Thực nghiệm (Evidence)
- File mã nguồn: [vectors_matrices.py](file:///c:/Users/nkocn/OneDrive/Desktop/Plan/tuna-ai/research/mathematics/vectors_matrices.py)
- File unit test: [test_vectors_matrices.py](file:///c:/Users/nkocn/OneDrive/Desktop/Plan/tuna-ai/tests/test_vectors_matrices.py)
- Kết quả kiểm thử: **15/15 tests passed** trong `2.31s`.
