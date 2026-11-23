# Báo cáo Đánh giá Hiện trạng Dự án Tuna-AI (Ngày 23/11/2026)

> **Dự án**: Tuna-AI — Nghiên cứu & Xây dựng Mô hình Ngôn ngữ Lớn Song ngữ (Việt - Anh) từ số 0 (*Training from Scratch*)  
> **Thời điểm đánh giá**: Kết thúc Ngày 8 (Đầu Tuần 2)  
> **Repository**: [https://github.com/tuanhuynh1704/tuna-ai.git](https://github.com/tuanhuynh1704/tuna-ai.git)  

---

## 1. Trả lời Trực diện Câu hỏi Cốt lõi

### ❓ Câu hỏi 1: Ở bước hiện tại, mô hình đã giao tiếp cơ bản được chưa?
> **Trả lời: CHƯA THỂ GIAO TIẾP ĐƯỢC.**  
> Hiện tại, mô hình hoàn toàn **chưa có khả năng sinh văn bản (text generation), nhận câu hỏi hay trò chuyện (chat)**.

### ❓ Câu hỏi 2: Vậy dự án đang ở giai đoạn nào?
> **Trả lời: Dự án đang ở giai đoạn "Phát triển Hệ thống Mạng Nơ-ron & Động cơ Tính toán Nền tảng" (Giai đoạn 1 & 2 / 12 giai đoạn).**  
> Dự án đang xây dựng phần "động cơ cơ khí" bên dưới (Toán học, Tensor, Động cơ tự động vi phân Autograd, Layer Normalization, Matrix GEMM, Lõi Causal Attention, và Khối MLP). 

---

## 2. Tại sao Mô hình Chưa thể Giao tiếp được? (Bản chất Kỹ thuật)

Dự án Tuna-AI được định hướng phát triển theo phương pháp **Training from scratch (Khởi tạo từ trọng số ngẫu nhiên, không dùng mô hình có sẵn của OpenAI, Meta hay Google)**. 

Để một mô hình ngôn ngữ từ các con số ngẫu nhiên có thể trò chuyện được với con người, nó bắt buộc phải đi qua **6 tầng kiến trúc tuần tự**:

```mermaid
flowchart TD
    T1["Tầng 1: Động cơ Toán học & Tensor Autograd (ĐÃ HOÀN THÀNH - M1, M2)"] --> T2["Tầng 2: Mạng Nơ-ron MLP & Hàm kích hoạt (ĐANG TRIỂN KHAI - M3)"]
    T2 --> T3["Tầng 3: Tokenizer Song ngữ BPE (Chuyển chữ viết Việt/Anh thành số - M4)"]
    T3 --> T4["Tầng 4: Lắp ráp Kiến trúc Causal Transformer Decoder hoàn chỉnh (M5)"]
    T4 --> T5["Tầng 5: Pre-training (Học hàng triệu token văn bản từ số 0 - M6, M7, M8)"]
    T5 --> T6["Tầng 6: Instruction Tuning / SFT (Học giao tiếp, trả lời câu hỏi - M10)"]
    
    style T1 fill:#d4edda,stroke:#28a745,stroke-width:2px;
    style T2 fill:#fff3cd,stroke:#ffc107,stroke-width:2px;
    style T3 fill:#f8d7da,stroke:#dc3545,stroke-width:1px;
    style T4 fill:#f8d7da,stroke:#dc3545,stroke-width:1px;
    style T5 fill:#f8d7da,stroke:#dc3545,stroke-width:1px;
    style T6 fill:#f8d7da,stroke:#dc3545,stroke-width:1px;
```

- **Giai đoạn hiện tại (Tầng 1 & 2)**: Giống như chúng ta đang tự đúc từng bánh răng, trục khuỷu và pít-tông của động cơ xe. Xe chưa có bánh, chưa đổ xăng nên chưa thể chạy.
- **Tầng 3 (Milestone M4)**: Xây dựng Tokenizer để mô hình "biết đọc" các ký tự tiếng Việt có dấu và tiếng Anh.
- **Tầng 4 (Milestone M5)**: Lắp ghép các khối Attention, RoPE, RMSNorm thành mô hình Transformer Mini GPT.
- **Tầng 5 (Milestone M6 $\to$ M8)**: Nạp dữ liệu văn bản vào huấn luyện. Khi hoàn tất tầng này, mô hình mới bắt đầu **sinh được câu từ tiếp nối** (ví dụ: gõ *"Hôm nay trời"* $\to$ mô hình đoán tiếp *"rất đẹp"*).
- **Tầng 6 (Milestone M10 - Instruction Tuning)**: **Chỉ sau bước này, mô hình mới chính thức biết giao tiếp dạng Hỏi - Đáp (Chatbot / Assistant) theo ngữ cảnh người dùng.**

---

## 3. Tổng hợp Chi tiết Những gì Dự án ĐÃ ĐẠT ĐƯỢC đến Lúc này

Dù chưa giao tiếp được, hệ thống bên dưới đã đạt được những bước tiến kỹ thuật cực kỳ vững chắc với **45 unit tests tự động vượt qua $100\%$**:

### 3.1. Nghiệm thu Cột mốc Milestone M1 (Đại số Tuyến tính & Giải tích ứng dụng)
1. **Thư viện Toán học Cơ sở**:
   - Tự triển khai các phép toán Vector, Matrix, Tensor 3D/4D.
   - Cơ chế Lan truyền (Broadcasting) và trích xuất vector từ điển (Token Embedding Lookup).
2. **Tối ưu hóa Phép nhân Ma trận (GEMM) trên CPU Intel i7-1165G7**:
   - Triển khai giải thuật Cache-aware ($i-k-j$) và Tiling ($32 \times 32$) giúp dữ liệu nằm trọn trong L1 Cache (48KB/core).
   - Đo lường thực tế: PyTorch Intel MKL + oneDNN đạt **132.68 GFLOPS**, nhanh gấp **$19.614$ lần** so với vòng lặp Python thuần.
3. **Lõi Causal Scaled Dot-Product Attention**:
   - Triển khai công thức $Attention(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$.
   - Tích hợp ma trận Causal Mask tam giác trên ($-\infty$) bảo đảm mô hình không thể "nhìn trộm" token tương lai.
4. **Các lớp Chuẩn hóa Hiện đại (Normalization)**:
   - Triển khai **LayerNorm** (đối chiếu khớp với `torch.nn.LayerNorm`).
   - Triển khai **RMSNorm** (Root Mean Square Normalization) — chuẩn hóa của các LLM hiện đại như LLaMA, Mistral, Gemma giúp **tiết kiệm 30-50% chi phí tính toán trên CPU**.
5. **Đạo hàm Giải tích & Kiểm định Sai phân Hữu hạn (Finite Differences)**:
   - Đạo hàm 5 hàm kích hoạt: Sigmoid, Tanh, ReLU, GELU (GPT-2/3) và SiLU (SwiGLU trong LLaMA).
   - Đạo hàm Softmax Cross-Entropy Loss: $\frac{\partial L}{\partial \mathbf{z}} = \frac{\mathbf{p} - \mathbf{y}}{N}$.
6. **Thuật toán Tối ưu hóa & Huấn luyện Mô hình Đầu tiên**:
   - Triển khai **SGDMomentum** và **AdamW** (Decoupled Weight Decay).
   - Huấn luyện mô hình Linear Regression từ số 0 đạt $R^2 > 0.99$, hội tụ chính xác với nghiệm giải tích Normal Equation.

### 3.2. Nghiệm thu Cột mốc Milestone M2 (Động cơ Tự động Vi phân Đa chiều - Tensor Autograd Engine)
1. **Lớp đại số `Tensor` tại `src/core/tensor.py`**:
   - Hỗ trợ lưu trữ mảng đa chiều NumPy, tự động xây dựng đồ thị tính toán có hướng không chu trình (DAG).
   - Thuật toán **Sắp xếp Tô-pô (Topological Sort qua DFS)** điều phối lan truyền ngược `backward()`.
2. **Kỹ thuật Unbroadcasting Gradient**:
   - Tự động thu gọn (sum-reduce) các chiều được broadcast khi cộng vector bias vào ma trận dữ liệu.
3. **Phép nhân Ma trận Ngược (Matmul Backward)**:
   - $\frac{\partial L}{\partial A} = \frac{\partial L}{\partial C} B^T$ và $\frac{\partial L}{\partial B} = A^T \frac{\partial L}{\partial C}$.
4. **Khối mạng nơ-ron đa tầng (MLP Feed-Forward Network)**:
   - Đã ghép nối thành công khối MLP 2 tầng với hàm kích hoạt GELU.
   - Kết quả Loss và Gradients của toàn bộ trọng số **khớp $100\%$ với PyTorch Autograd (sai số $< 10^{-4}$)**.

---

## 4. Lịch sử Đồng bộ Mã nguồn trên GitHub

Toàn bộ quá trình triển khai đều có bằng chứng rõ ràng, mã nguồn sạch, có unit tests và được commit chi tiết lên GitHub:

| Commit Hash | Thông điệp Commit | Nội dung thực hiện |
| :---: | :--- | :--- |
| `18bea6d` | `feat(day-01)` | Khởi tạo khung dự án, pyproject.toml, cấu hình Model A tiny.json, phép toán vector matrix. |
| `4cb424c` | `feat(day-02)` | Tối ưu GEMM, benchmark bộ nhớ đệm CPU i7-1165G7, lõi Causal Attention. |
| `46d01bd` | `feat(day-03)` | Triển khai Norms, LayerNorm, RMSNorm, Gradient Clipping, Split/Merge Heads. |
| `a7d78d1` | `feat(day-04)` | Đạo hàm các hàm kích hoạt (Sigmoid, ReLU, GELU, SiLU), đạo hàm Softmax Cross-Entropy. |
| `a969d51` | `feat(day-05)` | Động cơ Autograd vi phân tự động (Scalar), đồ thị DAG, Topological Sort. |
| `8a99eb5` | `feat(day-06)` | Triển khai Optimizers (SGD Momentum, AdamW), huấn luyện mô hình Linear Regression. |
| `6be0673` | `feat(day-07)` | Tổng kết Tuần 1, báo cáo nghiệm thu Milestone M1, script kiểm định tích hợp. |
| `ebc91ec` | `feat(day-08)` | Nâng cấp Tensor Autograd Engine đa chiều, Unbroadcasting, nghiệm thu Milestone M2. |

---

## 5. Lộ trình Tiếp theo để Đạt đến Khả năng Giao tiếp

Để đi từ hiện tại đến lúc có thể trò chuyện với mô hình, chúng ta sẽ lần lượt vượt qua các cột mốc:

```text
Hiện tại (Ngày 8 - Tuần 2)
  │
  ├──> [Tuần 2 - 3]: Hoàn thiện Mạng Nơ-ron MLP & Module Linear/Sequential (Milestone M3)
  │
  ├──> [Tuần 4 - 6]: Xây dựng Byte-level BPE Tokenizer Song ngữ (Việt - Anh) (Milestone M4)
  │                  (Giúp mô hình hiểu được từ vựng tiếng Việt có dấu)
  │
  ├──> [Tuần 7 - 8]: Lắp ráp Transformer Decoder hoàn chỉnh (Milestone M5)
  │                  (Embedding + Causal Attention Blocks + RMSNorm + LM Head)
  │
  ├──> [Tuần 9 - 12]: Pre-training Mini GPT trên tập ngữ liệu (Milestone M6 & M8)
  │                   (Mô hình bắt đầu sinh được câu văn có nghĩa)
  │
  └──> [Sau Pre-training]: Instruction Tuning / Chat Fine-Tuning (Milestone M10)
                          (Mô hình CHÍNH THỨC GIAO TIẾP VÀ TRÒ CHUYỆN ĐƯỢC)
```

---

## 6. Kết luận

- Dự án **đang đi hoàn toàn đúng tiến độ khoa học và chuẩn mực kỹ thuật**.
- Việc mô hình chưa giao tiếp được ở Ngày 8 là **hoàn toàn bình thường và tất yếu** đối với phương pháp huấn luyện từ đầu (*from scratch*). 
- Phần nền móng khó nhất (Toán học, Động cơ tính toán Tensor và Tự động vi phân Autograd) đã được xây dựng chuẩn xác $100\%$, tạo bệ phóng vững chắc để bước sang các khối mạng nơ-ron sâu hơn.
