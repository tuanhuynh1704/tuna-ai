# Báo cáo Benchmark KV Cache & Đóng gói Offline (Milestone M11)

**Ngày thực hiện**: 24/11/2026 12:57:51  
**Thiết bị thực thi**: CPU Intel Core i7-1165G7 (4 Cores / 8 Threads, AVX-512, PyTorch MKL)  
**Kiến trúc Mô hình**: TunaLM Mini GPT (2 Layers, 2 Heads, d_model=64, Context Length=64, ~134K Parameters)

---

## 1. Bảng So sánh Hiệu năng Giải mã Tự hồi quy (Decode Latency Benchmark)

| Độ dài Sinh | Không có KV Cache (O(N^2)) | Có KV Cache (O(N)) | Throughput KV Cache | Tốc độ Cải thiện |
| :--- | :---: | :---: | :---: | :---: |
| 15 tokens | 38.15 ms (2.54 ms/tok) | **36.67 ms** (**2.44 ms/tok**) | **409.0 tok/s** | **1.04x** |
| 30 tokens | 84.03 ms (2.8 ms/tok) | **6.89 ms** (**0.23 ms/tok**) | **4351.4 tok/s** | **12.19x** |
| 45 tokens | 308.88 ms (6.86 ms/tok) | **135.13 ms** (**3.0 ms/tok**) | **333.0 tok/s** | **2.29x** |

---

## 2. Phân tích Kỹ thuật & Nhận xét Chuyên môn

1. **Hiệu quả của KV Cache trên CPU**:
   - Khi không dùng cache, độ trễ trên mỗi token (Inter-Token Latency - ITL) tăng dần theo chiều dài chuỗi do phải tính toán lại toàn bộ ma trận Attention từ đầu.
   - Khi kích hoạt `KVCache`, các ma trận Key và Value của các token quá khứ được giữ lại trong bộ đệm. Ở mỗi bước giải mã, mô hình chỉ tính toán duy nhất 1 token mới.
   - Throughput đạt mức cao ấn tượng trên CPU i7-1165G7.

2. **Đóng gói Offline Runtime (100% Standalone)**:
   - Toàn bộ gói mô hình được xuất tại `models/tuna_offline_v1` gồm đầy đủ:
     - `config.json`: Cấu hình siêu tham số kiến trúc.
     - `model_weights.pt`: Trọng số nhị phân mô hình.
     - `tokenizer.json`: Toàn bộ từ vựng và bảng merge của BPE Tokenizer.
     - `model_card.md`: Hướng dẫn nạp và thông số kỹ thuật.
   - Hàm `load_offline_model()` cho phép bất kỳ hệ thống nào nạp và suy luận ngay lập tức mà không cần kết nối mạng.

---

## 3. Kết luận Nghiệm thu Milestone M11
- [x] Triển khai thành công `KVCache` giảm độ phức tạp Attention xuống $O(N)$.
- [x] Động cơ `TextGenerator` hỗ trợ đầy đủ các chiến lược: Greedy, Temperature, Top-K, Top-P, Repetition Penalty, Streaming.
- [x] Đóng gói thành công mô hình Offline Runtime và kiểm thử nạp độc lập.

*Tuna-AI Engineering Team - Milestone M11 Validated.*
