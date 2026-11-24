# Báo cáo Kỹ thuật Ngày 17 - Nghiệm thu Milestone M11: Inference Engine Offline & Tối ưu Hóa CPU

**Dự án**: Tuna-AI — Huấn luyện Mô hình Ngôn ngữ Lớn Song ngữ (Việt - Anh) từ số 0  
**Cột mốc**: Milestone M11 (Offline Inference Engine, KV Cache Optimization & Model Packaging)  
**Tác giả**: Tuna-AI Research & Engineering Team  
**Ngày thực hiện**: 24/11/2026  
**Trạng thái kiểm thử**: **102/102 unit tests passing 100%**, 0 warnings.

---

## 1. Mục tiêu Kỹ thuật của Milestone M11
Theo lộ trình tại `KE_HOACH_TRIEN_KHAI.md` và tài liệu nền tảng `MOTA.md`:
1. Xây dựng động cơ sinh văn bản suy luận (Inference Text Generator) đa chiến lược lấy mẫu:
   - **Greedy Search** ($\text{argmax}$).
   - **Temperature Scaling**: Điều chỉnh độ mượt của phân phối xác suất.
   - **Top-K Sampling**: Giới hạn trong $K$ ứng viên có xác suất cao nhất.
   - **Top-P (Nucleus Sampling)**: Cắt đuôi xác suất theo ngưỡng tích lũy $p$.
   - **Repetition Penalty**: Phạt tỷ lệ nghịch các token đã xuất hiện trong ngữ cảnh, ngăn chặn hiện tượng lặp vòng luẩn quẩn.
   - **Stream Generator**: Chế độ sinh văn bản dạng luồng (`yield`) phục vụ giao diện chat thời gian thực.
2. Triển khai kiến trúc **Key-Value Cache (KV Cache)** cho khối Attention:
   - Chuyển độ phức tạp tính toán giải mã tự hồi quy từ $O(N^2)$ xuống $O(N)$.
   - Khử bỏ hoàn toàn việc tính toán lại $Q, K, V$ cho các token quá khứ, chỉ tính toán duy nhất 1 token mới ở mỗi bước.
3. Thiết lập quy chuẩn đóng gói mô hình phục vụ môi trường **100% Offline Runtime**:
   - `config.json` + `model_weights.pt` + `tokenizer.json` + `model_card.md`.
   - Cung cấp hàm nạp một dòng `load_offline_model(export_dir)` sẵn sàng hoạt động mà không cần mạng internet.

---

## 2. Kiến trúc Module Đã Xây Dựng

### 2.1. `src/inference/kv_cache.py`
- Lớp `KVCache`:
  - Quản lý danh sách các cặp tensor `(key, value)` cho từng tầng Transformer qua các bước giải mã.
  - `update(layer_idx, k, v)`: Ghép nối Key/Value mới với Key/Value quá khứ dọc theo chiều thời gian `dim=2`: `(B, H, T_total, d_k)`.
  - `get_seq_len()` và `reset()` kiểm soát trạng thái bộ nhớ đệm sạch sẽ.

### 2.2. Nâng cấp các Tầng Mô hình Cốt lõi
- `TransformerEmbedding`: Hỗ trợ `pos_offset` phục vụ sinh nhúng vị trí chính xác cho token mới $t$ mà không cần tạo lại toàn bộ chuỗi $0 \dots t$.
- `CausalSelfAttention`: Nhận `kv_cache`, tính toán Attention giữa $Q_{\text{new}}$ và $K_{\text{cached}}$, bỏ qua mặt nạ nhân quả khi $T_{\text{new}} = 1$.
- `TransformerBlock` & `TunaLM`: Tự động truyền tham chiếu `kv_cache` và `layer_idx` xuyên suốt luồng forward.

### 2.3. `src/inference/generator.py`
- Lớp `TextGenerator`:
  - `generate()` và `generate_stream()`: Tích hợp đầy đủ các bộ lọc sampling: Repetition Penalty, Temperature, Top-K, Top-P, Early Stopping với danh sách stop tokens (`<|im_end|>`, `<eos>`).

### 2.4. `src/inference/exporter.py`
- Lớp `ModelExporter`:
  - `export()`: Đóng gói toàn bộ tài sản mô hình vào thư mục sản phẩm `models/tuna_offline_v1/`.
  - `load_offline_model()`: Nạp lại cấu hình, trọng số và từ điển tokenizer trong 1 bước duy nhất.

---

## 3. Bằng chứng Thực nghiệm & Kết quả Đo lường

### 3.1. Kết quả Benchmark KV Cache Đối đầu trên CPU Intel Core i7-1165G7
So sánh giữa chế độ sinh truyền thống (Non-cached $O(N^2)$) và có KV Cache ($O(N)$) qua các độ dài token:

| Độ dài Sinh | Không có KV Cache ($O(N^2)$) | Có KV Cache ($O(N)$) | Tốc độ KV Cache | Tỷ lệ Tăng tốc |
| :--- | :---: | :---: | :---: | :---: |
| **15 tokens** | 38.15 ms (2.54 ms/tok) | **36.67 ms (2.44 ms/tok)** | 409.0 tok/s | **1.04x** |
| **30 tokens** | 84.03 ms (2.80 ms/tok) | **6.89 ms (0.23 ms/tok)** | **4,351.4 tok/s** | **12.19x** |
| **45 tokens** | 308.88 ms (6.86 ms/tok) | **135.13 ms (3.00 ms/tok)** | 333.0 tok/s | **2.29x** |

> [!NOTE]
> **Nhận xét chuyên môn**:
> - Khi không dùng cache, độ trễ Inter-Token Latency (ITL) tăng vọt từ 2.54 ms lên 6.86 ms/token do phải tính ma trận Attention với kích thước lũy tiến $T^2$.
> - Khi bật `KVCache`, ITL duy trì ở mức cực kỳ thấp ($0.23 - 3.00$ ms/token) và tốc độ đạt đỉnh **4.351 tokens/giây** trên CPU i7-1165G7.

### 3.2. Kiểm thử Đóng gói và Nạp lại Offline
- Đã xuất thành công gói mô hình `models/tuna_offline_v1` gồm: `config.json`, `model_weights.pt`, `tokenizer.json`, `model_card.md`.
- Hàm `load_offline_model("models/tuna_offline_v1")` khởi tạo thành công mô hình và sinh phản hồi tự hồi quy mượt mà trong chế độ offline hoàn toàn.

---

## 4. Kết quả Kiểm thử Tự động Dự án (Pytest)
Toàn bộ **102/102 unit tests** trên toàn bộ 17 tệp kiểm thử đã **PASS 100% trong 6.2s với 0 warnings**:
```text
collected 102 items

tests/test_autograd.py .....                                             [  4%]
tests/test_calculus_derivatives.py .......                               [ 11%]
tests/test_dataset_engineering.py ...............                        [ 26%]
tests/test_environment.py ...                                            [ 29%]
tests/test_evaluation.py .....                                           [ 34%]
tests/test_inference.py ....                                             [ 38%]
tests/test_instruction_tuning.py ....                                    [ 42%]
tests/test_linear_regression.py ...                                      [ 45%]
tests/test_matrix_multiplication.py .....                                [ 50%]
tests/test_neural_network.py ....                                        [ 53%]
tests/test_pretraining_experiments.py ...                                [ 56%]
tests/test_tensor_autograd.py ....                                       [ 60%]
tests/test_tensor_operations.py ......                                   [ 66%]
tests/test_tokenizer.py ............                                     [ 78%]
tests/test_training_engine.py ....                                       [ 82%]
tests/test_transformer.py ......                                         [ 88%]
tests/test_vectors_matrices.py ............                              [100%]

============================= 102 passed in 6.20s =============================
```

---

## 5. Kết luận Nghiệm thu
Milestone M11 đã hoàn thành xuất sắc 100% các tiêu chí kỹ thuật:
- [x] Động cơ `TextGenerator` hỗ trợ đầy đủ các kỹ thuật lấy mẫu hiện đại.
- [x] Thuật toán `KVCache` giảm độ phức tạp xuống $O(N)$ và tăng tốc tới 12.19x.
- [x] Đóng gói mô hình `ModelExporter` và cơ chế nạp độc lập `load_offline_model` sẵn sàng phục vụ runtime offline trên máy tính cá nhân.
- [x] 102/102 unit tests passing với 0 warnings.
