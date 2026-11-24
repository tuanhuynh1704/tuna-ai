# Báo cáo Kỹ thuật Ngày 15 - Nghiệm thu Milestone M9: Evaluation Engine & Benchmark Suite

**Dự án**: Tuna-AI — Huấn luyện Mô hình Ngôn ngữ Lớn Song ngữ (Việt - Anh) từ số 0  
**Cột mốc**: Milestone M9 (Evaluation Engine & Independent Benchmark Suite)  
**Tác giả**: Tuna-AI Research & Engineering Team  
**Ngày thực hiện**: 24/11/2026  
**Trạng thái kiểm thử**: **94/94 unit tests passing 100%**, 0 warnings.

---

## 1. Mục tiêu Kỹ thuật của Milestone M9
Theo đặc tả tại `KE_HOACH_TRIEN_KHAI.md` và tài liệu nền tảng `MOTA.md`:
1. Xây dựng bộ công cụ đánh giá (Evaluation Harness) độc lập, tách rời hoàn toàn khỏi vòng lặp huấn luyện để đảm bảo tính khách quan (*Zero Contamination*).
2. Thiết lập các bài kiểm tra năng lực ngôn ngữ đa chiều:
   - **Hold-out Test Loss & Perplexity (PPL)**: $\text{PPL} = \exp(\text{mean\_loss})$.
   - **Next-Token Prediction Accuracy**: Đo lường Top-1 và Top-5 Accuracy trên ngữ liệu kiểm thử.
   - **Cloze Test Song ngữ**: Đánh giá khả năng hiểu ngữ cảnh và lựa chọn từ đúng thông qua phân phối log-likelihood của các ứng viên: $\arg\max_c \log P(c \mid \text{prefix})$.
3. Phân tích chi tiết hiệu năng suy luận (Inference Profiling) trên CPU Intel Core i7-1165G7:
   - **Time To First Token (TTFT)**.
   - **Inter-Token Latency (ITL)**.
   - **Generation Throughput (tokens/giây)**.
   - **RAM Footprint (Peak RSS memory)**.

---

## 2. Kiến trúc Module Đã Xây Dựng

### 2.1. `src/evaluation/benchmarks.py`
- `ClozeItem`: Cấu trúc dữ liệu đại diện cho 1 câu hỏi điền từ vào chỗ trống, bao gồm tiền tố (`prefix`), đáp án đúng (`target`), danh sách ứng viên (`candidates`) và danh mục (`category`).
- `ClozeBenchmark`: Bộ benchmark câu hỏi song ngữ chuẩn mực:
  - Tiếng Việt: Kiến thức khoa học máy tính, địa lý, ngôn ngữ học (`vi_knowledge`).
  - Tiếng Anh: Deep learning, Transformer, Natural language processing (`en_knowledge`).
  - Tự động cắt lát ngữ cảnh an toàn `prefix_ids[-context_length:]` để ngăn chặn lỗi tràn bộ đệm vị trí (*Position Buffer Overflow*).

### 2.2. `src/evaluation/evaluator.py`
- `ModelEvaluator`: Lớp điều phối đánh giá trung tâm:
  - `evaluate_loss_and_perplexity()`: Tính loss trung bình và Perplexity trên hold-out split của `ShardDataLoader`.
  - `evaluate_next_token_accuracy()`: Tính toán Top-1 và Top-5 accuracy bằng `torch.topk(logits, k=5, dim=-1)`.
  - `benchmark_inference_latency()`: Đo lường chính xác bằng `time.perf_counter()`, phân tách rạch ròi giữa TTFT (prefill stage) và ITL (decode stage).
  - `run_full_evaluation()`: Đóng gói toàn bộ kết quả vào dataclass `EvaluationReport`.

---

## 3. Bằng chứng Thực nghiệm & Kết quả Đo lường

### 3.1. Kết quả Đánh giá Định lượng trên TunaLM-Mini-Bilingual (~132K params)
| Tiêu chí Đánh giá | Kết quả Đo lường | Ý nghĩa & Đánh giá Kỹ thuật |
| :--- | :---: | :--- |
| **Hold-out Test Loss** | **4.4016** | Mức mất mát trên tập kiểm thử giữ lại sau 50 bước |
| **Hold-out Perplexity (PPL)** | **81.59** | Giảm mạnh từ mức ngẫu nhiên ban đầu ($> 500$) |
| **Next-Token Top-1 Accuracy** | **3.44%** | Tỷ lệ đoán chính xác tuyệt đối từ tiếp theo |
| **Next-Token Top-5 Accuracy** | **19.38%** | Gần 20% trường hợp token đúng nằm trong top 5 token có xác suất cao nhất |
| **Cloze Test Accuracy** | **25.00%** | Mô hình bắt đầu thể hiện khả năng thiên kiến lựa chọn token hợp lý |
| **Time To First Token (TTFT)** | **2.48 ms** | Thời gian phản hồi cực nhanh trên CPU i7-1165G7 |
| **Inter-Token Latency (ITL)** | **2.05 ms/token** | Mỗi token chỉ mất ~2ms để giải mã |
| **Generation Throughput** | **482.99 tok/s** | Tốc độ sinh văn bản tự hồi quy thuần CPU |
| **Peak RAM Footprint** | **290.79 MB** | Chiếm dụng cực kỳ khiêm tốn (< 300MB) |

---

## 4. Kết quả Kiểm thử Tự động Dự án (Pytest)
Toàn bộ **94/94 unit tests** trên toàn bộ 15 tệp kiểm thử đã **PASS 100% với 0 warnings**:
```text
collected 94 items

tests/test_autograd.py .....                                             [  5%]
tests/test_calculus_derivatives.py .......                               [ 12%]
tests/test_dataset_engineering.py ...............                        [ 28%]
tests/test_environment.py ...                                            [ 31%]
tests/test_evaluation.py .....                                           [ 37%]
tests/test_linear_regression.py ...                                      [ 40%]
tests/test_matrix_multiplication.py .....                                [ 45%]
tests/test_neural_network.py ....                                        [ 50%]
tests/test_pretraining_experiments.py ...                                [ 53%]
tests/test_tensor_autograd.py ....                                       [ 57%]
tests/test_tensor_operations.py ......                                   [ 63%]
tests/test_tokenizer.py ............                                     [ 76%]
tests/test_training_engine.py ....                                       [ 80%]
tests/test_transformer.py ......                                         [ 87%]
tests/test_vectors_matrices.py ............                              [100%]

============================= 94 passed in 6.45s ==============================
```

---

## 5. Kết luận Nghiệm thu
Milestone M9 đã hoàn thành 100% các tiêu chí nghiệm thu:
- [x] Động cơ đánh giá `ModelEvaluator` hoạt động độc lập và toàn diện.
- [x] Bộ benchmark `ClozeBenchmark` và Top-1/Top-5 accuracy đo lường khách quan.
- [x] Đo lường chi tiết độ trễ suy luận TTFT (2.48ms), ITL (2.05ms) và RAM (< 300MB).
- [x] Xuất báo cáo tự động tại `experiments/reports/m9_evaluation_report.md`.
