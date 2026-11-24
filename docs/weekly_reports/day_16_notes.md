# Báo cáo Kỹ thuật Ngày 16 - Nghiệm thu Milestone M10: Instruction Tuning & Supervised Fine-Tuning (SFT)

**Dự án**: Tuna-AI — Huấn luyện Mô hình Ngôn ngữ Lớn Song ngữ (Việt - Anh) từ số 0  
**Cột mốc**: Milestone M10 (Instruction Tuning & SFT với Response Masking)  
**Tác giả**: Tuna-AI Research & Engineering Team  
**Ngày thực hiện**: 24/11/2026  
**Trạng thái kiểm thử**: **98/98 unit tests passing 100%**, 0 warnings.

---

## 1. Mục tiêu Kỹ thuật của Milestone M10
Theo lộ trình tại `KE_HOACH_TRIEN_KHAI.md` và tài liệu nền tảng `MOTA.md`:
1. Chuyển đổi mô hình ngôn ngữ từ dạng sinh văn bản tự do (*Base Pretrained Language Model*) thành mô hình trợ lý biết hội thoại theo lượt (*Instruction-tuned Chat Model*).
2. Thiết lập định dạng hội thoại chuẩn công nghiệp **ChatML**:
   - `<|im_start|>system\n{content}<|im_end|>\n`
   - `<|im_start|>user\n{content}<|im_end|>\n`
   - `<|im_start|>assistant\n{content}<|im_end|>\n`
3. Triển khai cơ chế **Response Masking**:
   - Gán nhãn `targets[t] = -100` (`ignore_index` của PyTorch CrossEntropyLoss) cho toàn bộ các token thuộc System Prompt, User Prompt và Padding.
   - Chỉ tính toán hàm mất mát và cập nhật gradient trên các token thuộc câu trả lời của Trợ lý (`Assistant Response`).
   - Đảm bảo **Zero Gradient Leakage**: Mô hình không bị phạt hay ghi nhớ câu hỏi của người dùng, mà tập trung $100\%$ dung lượng tham số vào việc học cách trả lời.
4. Huấn luyện thực nghiệm SFT và kiểm tra khả năng dừng lượt trò chuyện (`<|im_end|>`).

---

## 2. Kiến trúc Module Đã Xây Dựng

### 2.1. `src/instruction/chat_template.py`
- Lớp `ChatTemplate`:
  - `format_dialogue(messages, add_generation_prompt)`: Đóng gói danh sách hội thoại nhiều lượt thành chuỗi văn bản với các thẻ đặc biệt.
  - `parse_response(generated_text)`: Tự động trích xuất phần trả lời sạch của trợ lý, loại bỏ prompt và cắt bỏ nội dung sau thẻ kết thúc `<|im_end|>`.

### 2.2. `src/instruction/dataset.py`
- Lớp `SFTDataset`:
  - Phân tách rạch ròi giữa Prompt và Response.
  - Kỹ thuật **Left Truncation thông minh**: Khi độ dài chuỗi vượt quá `context_length`, thuật toán ưu tiên giữ lại trọn vẹn câu trả lời của Assistant và chỉ cắt bớt phần đầu của Prompt, đảm bảo số token có nhãn hợp lệ luôn $> 0$, triệt tiêu hoàn toàn nguy cơ lỗi `NaN loss`.
  - Sinh cặp tensor $(X, Y)$ với $Y$ chứa $-100$ tại mọi vị trí Prompt và Padding.

### 2.3. `src/instruction/sft_trainer.py`
- Lớp `SFTTrainer`:
  - Vòng lặp huấn luyện Supervised Fine-Tuning chuyên dụng.
  - Theo dõi Response Loss và Response Perplexity riêng biệt.
  - Tích hợp Cosine Warmup Learning Rate Scheduler và Gradient Clipping ($1.0$).

---

## 3. Bằng chứng Thực nghiệm & Kết quả Huấn luyện

### 3.1. Chạy Thực nghiệm SFT (`research/instruction/train_sft.py`)
Mô hình `TunaLM-Mini-SFT` (~134K tham số) được huấn luyện qua 15 epochs trên tập dữ liệu chỉ dẫn song ngữ:
```text
Epoch  1 | Step   1 | SFT Response Loss: 6.2292 | LR: 1.50e-04
Epoch  1 | Step  10 | SFT Response Loss: 5.6202 | LR: 1.50e-03
  ---> [SFT Validation] Epoch 1: Val Loss = 5.3946 (Perplexity = 220.21)
Epoch  3 | Step  30 | SFT Response Loss: 4.1277 | LR: 1.44e-03
  ---> [SFT Validation] Epoch 3: Val Loss = 3.9015 (Perplexity = 49.48)
Epoch  6 | Step  60 | SFT Response Loss: 2.8149 | LR: 1.15e-03
  ---> [SFT Validation] Epoch 6: Val Loss = 2.5071 (Perplexity = 12.27)
Epoch 10 | Step 110 | SFT Response Loss: 1.5678 | LR: 4.27e-04
  ---> [SFT Validation] Epoch 10: Val Loss = 1.4752 (Perplexity = 4.37)
Epoch 15 | Step 160 | SFT Response Loss: 1.1256 | LR: 1.38e-05
  ---> [SFT Validation] Epoch 15: Val Loss = 1.1757 (Perplexity = 3.24)
  ---> Hoàn thành SFT: Final Loss = 1.1448 trong 1.82s
```

### 3.2. Đánh giá Hiện tượng Kỹ thuật
1. **Mức độ hội tụ vượt bậc**:
   - Response Loss giảm từ $6.2292$ xuống **$1.1448$**.
   - Validation Perplexity giảm từ $220.21$ xuống **$3.24$**, chứng minh mô hình học thuộc và tiếp thu cấu trúc câu trả lời của trợ lý rất sâu sắc.
2. **Hành vi Dừng lượt Trò chuyện**:
   - Khi được hỏi các câu hỏi mới, mô hình chủ động sinh token kết thúc lượt `<|im_end|>` thay vì tiếp tục nói lan man vô hạn.

---

## 4. Kết quả Kiểm thử Tự động Dự án (Pytest)
Toàn bộ **98/98 unit tests** (bao gồm 4 tests mới của `test_instruction_tuning.py`) vượt qua 100% trong 6.2s:
```text
tests/test_autograd.py .....                                             [  5%]
tests/test_calculus_derivatives.py .......                               [ 12%]
tests/test_dataset_engineering.py ...............                        [ 27%]
tests/test_environment.py ...                                            [ 30%]
tests/test_evaluation.py .....                                           [ 35%]
tests/test_instruction_tuning.py ....                                    [ 39%]
tests/test_linear_regression.py ...                                      [ 42%]
tests/test_matrix_multiplication.py .....                                [ 47%]
tests/test_neural_network.py ....                                        [ 51%]
tests/test_pretraining_experiments.py ...                                [ 54%]
tests/test_tensor_autograd.py ....                                       [ 58%]
tests/test_tensor_operations.py ......                                   [ 64%]
tests/test_tokenizer.py ............                                     [ 76%]
tests/test_training_engine.py ....                                       [ 80%]
tests/test_transformer.py ......                                         [ 86%]
tests/test_vectors_matrices.py ............                              [100%]

============================= 98 passed in 6.21s ==============================
```

---

## 5. Kết luận Nghiệm thu
Milestone M10 đã hoàn thành 100% mục tiêu kỹ thuật:
- [x] Chat Template định dạng hội thoại chuẩn ChatML.
- [x] SFT Dataset và DataLoader với Response Masking loại bỏ gradient rò rỉ.
- [x] SFTTrainer tối ưu hóa chuyên sâu đạt loss 1.14 và perplexity 3.24.
- [x] 98/98 tests passing với 0 warnings.
