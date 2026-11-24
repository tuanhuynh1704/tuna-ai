# Báo cáo Đánh giá Hiệu năng & Năng lực Mô hình (Milestone M9)

**Mô hình**: `TunaLM-Mini-Bilingual-v1.0`  
**Thời điểm đánh giá**: 24/11/2026 12:44:50  
**Số lượng tham số**: 132,672 parameters  
**Phần cứng**: Intel Core i7-1165G7 (4 Cores / 8 Threads, AVX-512, thuần CPU)

---

## 1. Kết quả Đánh giá Ngôn ngữ (Language Modeling Capabilities)

| Tiêu chí Đánh giá | Giá trị Đạt được | Ý nghĩa Kỹ thuật |
| :--- | :---: | :--- |
| **Hold-out Test Loss** | **4.4016** | Hàm mất mát Cross-Entropy trên tập kiểm thử giữ lại |
| **Perplexity (PPL)** | **81.59** | Độ hỗn loạn dự đoán token (càng thấp mô hình càng tự tin) |
| **Next-Token Top-1 Accuracy** | **3.44%** | Tỷ lệ dự đoán chính xác tuyệt đối token kế tiếp |
| **Next-Token Top-5 Accuracy** | **19.38%** | Tỷ lệ token đúng nằm trong Top 5 token xác suất cao nhất |
| **Cloze Test Accuracy** | **25.0%** | Năng lực điền từ ngữ cảnh chuẩn xác theo phân phối log-likelihood |

### Chi tiết Cloze Test theo Danh mục:
- **vi_knowledge**: 25.0%
- **en_knowledge**: 25.0%

---

## 2. Đo lường Hiệu năng Suy luận (Inference Latency & Profiling)

| Chỉ số Hiệu năng | Kết quả Thực tế | Ngưỡng Tiêu chuẩn Đạt được |
| :--- | :---: | :--- |
| **Time To First Token (TTFT)** | **2.48 ms** | Thời gian phản hồi prompt ban đầu cực nhanh (< 20 ms) |
| **Inter-Token Latency (ITL)** | **2.05 ms/token** | Độ trễ giữa 2 token liên tiếp trong quá trình sinh tự hồi quy |
| **Generation Throughput** | **482.99 tok/s** | Tốc độ sinh văn bản trên CPU (~482 tokens/giây) |
| **RAM Footprint** | **290.79 MB** | Tổng mức chiếm dụng RAM an toàn (< 500 MB) |

---

## 3. Kết luận Nghiệm thu
- [x] Harness đánh giá hoạt động độc lập, không phụ thuộc vào vòng lặp training.
- [x] Đo lường chuẩn xác Top-1, Top-5 accuracy và PPL trên hold-out dataset.
- [x] Đạt yêu cầu về độ trễ và tiêu chuẩn vận hành thuần CPU trên laptop cá nhân.

*Tuna-AI Engineering Team - Milestone M9 Validated.*
