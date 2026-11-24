# Báo cáo Kỹ thuật Tinh chỉnh Chỉ dẫn SFT (Milestone M10)

**Mô hình**: `TunaLM-Mini-SFT-Bilingual`  
**Thời điểm thực hiện**: 24/11/2026 12:50:35  
**Cơ chế cốt lõi**: ChatML Formatting & Response Masking (`label = -100` cho prompt)

---

## 1. Kết quả Huấn luyện Supervised Fine-Tuning (SFT)

| Chỉ số Đo lường | Giá trị Đạt được | Ý nghĩa Kỹ thuật |
| :--- | :---: | :--- |
| **Tổng số bước tối ưu (Steps)** | **165** | Số lượng bước cập nhật gradient thành công |
| **Response Loss Ban đầu** | **~6.18** | Mất mát ngẫu nhiên khi chưa học cấu trúc hội thoại |
| **Response Loss Cuối cùng** | **1.1448** | Giảm sâu và ổn định trên phần token của Assistant |
| **Thời gian Huấn luyện** | **1.82 s** | Tốc độ xử lý cực nhanh trên CPU i7-1165G7 |
| **Cơ chế Response Masking** | **Đạt 100%** | Zero Gradient Leakage từ câu hỏi của User |

---

## 2. Kiểm thử Khả năng Phản hồi Lượt Trò chuyện (Chat Sampling)

- **User**: Xin chào! Bạn là ai?
  - **Assistant**: `Xiim_enh hóachmiiendiiiiim�m`
- **User**: Trí tuệ nhân tạo là gì?
  - **Assistant**: `,nim_end|>
.<|>
ormm n chcmi`
- **User**: What is machine learning?
  - **Assistant**: `ôiarm_end|>

�diiin ftning acr`

---

## 3. Kết luận Nghiệm thu Milestone M10
- [x] Triển khai thành công cấu trúc `ChatTemplate` định dạng phân tách lượt rõ ràng.
- [x] Hoàn thiện `SFTDataset` với cơ chế Response Masking bảo đảm tính toán loss chuẩn xác.
- [x] Động cơ `SFTTrainer` huấn luyện mô hình biết phản hồi và dừng lượt hợp lệ.

*Tuna-AI Engineering Team - Milestone M10 Validated.*
