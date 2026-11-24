# Báo cáo Kỹ thuật Ngày 13 - Nghiệm thu Milestone M7: Dataset Engineering & Pipeline Tiền Xử Lý Dữ Liệu Song Ngữ

**Dự án**: Tuna-AI — Huấn luyện Mô hình Ngôn ngữ Lớn Song ngữ (Việt - Anh) từ số 0  
**Cột mốc**: Milestone M7 (Dataset Engineering & Preprocessing Pipeline)  
**Tác giả**: Tuna-AI Research & Engineering Team  
**Ngày thực hiện**: 24/11/2026  
**Trạng thái kiểm thử**: **86/86 unit tests passing 100%**, 0 warnings.

---

## 1. Mục tiêu Kỹ thuật của Milestone M7
Theo đặc tả tại **Section 10 (Pipeline Tiền Xử Lý Dữ Liệu 10 Bước)** trong tài liệu nền tảng `MOTA.md`, chất lượng của mô hình ngôn ngữ lớn phụ thuộc quyết định vào chất lượng dữ liệu huấn luyện tiền kỳ (*Garbage In, Garbage Out*). Milestone M7 thiết lập toàn bộ quy trình tiền xử lý dữ liệu chuẩn công nghiệp tự động, nhẹ, hiệu năng cao trên kiến trúc CPU i7-1165G7.

Quy trình 10 bước bao gồm:
1. **Thu thập dữ liệu (Data Acquisition)**: Lựa chọn nguồn ngữ liệu bản quyền mở (CC-BY, MIT, Public Domain) cân bằng song ngữ (50% Việt - 50% Anh).
2. **Lọc định dạng (Format Filtering)**: Loại bỏ các thẻ HTML/XML, boilerplate, mã nguồn rác từ quá trình web-scraping.
3. **Nhận diện ngôn ngữ (Language Identification)**: Phân loại chính xác tiếng Việt (`vi`), tiếng Anh (`en`) hoặc ngôn ngữ lạ (`other`).
4. **Lọc chất lượng dựa trên quy tắc (Rule-based Quality Filtering)**: Loại bỏ câu quá ngắn ($< 5$ từ), quá dài, tỷ lệ ký tự đặc biệt bất thường ($> 35\%$), và spam lặp từ ($> 30\%$).
5. **Khử trùng lặp (Deduplication)**: 
   - *Exact Deduplication*: Băm mã hóa SHA-256 loại bỏ 100% bản sao nguyên văn.
   - *Fuzzy Deduplication (MinHash LSH)*: Sử dụng n-gram shingles kết hợp $K=64$ hàm băm tuyến tính Mersenne $2^{61}-1$ để ước lượng hệ số Jaccard Similarity, loại bỏ bản sao tương tự ($J \ge 0.75$).
6. **Lọc nội dung nhạy cảm & Che giấu PII (Safety & PII Filtering)**: Phát hiện và che giấu tự động Email, Số điện thoại (Việt Nam 10-11 số, quốc tế), Địa chỉ IPv4, Thẻ tín dụng bằng các placeholder an toàn (`[EMAIL]`, `[PHONE]`, `[IP_ADDRESS]`, `[CREDIT_CARD]`).
7. **Chuẩn hóa văn bản (Text Normalization)**: Chuẩn hóa Unicode NFC nghiêm ngặt, xử lý xung đột NFD từ các bộ gõ tiếng Việt Telex/VNI, loại bỏ ký tự điều khiển ẩn và chuẩn hóa khoảng trắng.
8. **Tokenization & Packing**: Sử dụng `BilingualBPETokenizer`, chèn token phân cách `<bos>` và `<eos>`, mã hóa mảng nhị phân `numpy.uint16` tiết kiệm 50% dung lượng RAM so với `int32`.
9. **Siêu dữ liệu Manifest (Metadata Manifest)**: Xuất file `manifest.json` ghi nhận đầy đủ checksum SHA-256, số tài liệu, số token, tỷ lệ ngôn ngữ và giấy phép.
10. **Phân mảnh Sharding (Memory-Safe Streaming)**: Cắt dữ liệu thành các file shard nhỏ (`tuna_shard_00000.bin`, ...) giúp nạp streaming an toàn trong giới hạn 8GB/16GB RAM của CPU i7-1165G7 mà không gây Memory Leak hay Swap SSD.

---

## 2. Kiến trúc Module Đã Xây Dựng

### 2.1. `src/data/cleaner.py`
- `TextCleaner`: Chuẩn hóa Unicode NFC (`unicodedata.normalize('NFC', text)`), bóc tách thẻ HTML/XML, xóa control chars, chuẩn hóa dòng trống và khoảng trắng.
- `RuleQualityFilter`: Kiểm tra độ dài từ (`min_words`, `max_words`), tỷ lệ ký tự chữ/số so với ký tự đặc biệt, kiểm tra tần suất từ xuất hiện cao nhất tránh spam lặp từ.
- `LanguageIdentifier`: Dựa trên tập nguyên âm có dấu tiếng Việt đặc thù và tập stop words song ngữ để phân loại `vi` vs `en` với độ trễ cực thấp.

### 2.2. `src/data/pii_filter.py`
- `PIIFilter`: Hệ thống biểu thức chính quy (Regex) tối ưu hóa phát hiện:
  - Email: `[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}`
  - Phone: Hỗ trợ đầu số Việt Nam (`+84`, `03x`, `05x`, `07x`, `08x`, `09x`, `02x`) và số quốc tế chuẩn E.164.
  - IP: IPv4 4 octets hợp lệ ($0 \to 255$).
  - Credit Card: Chuỗi 16 chữ số phân cách bởi dấu cách hoặc gạch ngang.
- Cung cấp 2 chế độ: `redact(text)` (che giấu) và `scan(text)` (thống kê định lượng).

### 2.3. `src/data/deduplicator.py`
- `ExactDeduplicator`: Lưu trữ hash set SHA-256 của các chuỗi đã chuẩn hóa khoảng trắng. Độ phức tạp $O(1)$ lookup.
- `MinHashDeduplicator`:
  - Shingling đa tầng: Kết hợp word n-grams ($1 \to 3$) và character 4-grams.
  - Ma trận băm tuyến tính: $h_i(x) = (a_i \cdot x + b_i) \pmod{(2^{61}-1)}$ với 64 hoán vị ngẫu nhiên.
  - Ước lượng $J(A, B) \approx \frac{1}{K} \sum_{i=1}^K \mathbb{I}[h_{i,\min}(A) = h_{i,\min}(B)]$.

### 2.4. `src/data/packer.py`
- `DatasetPacker`: Đóng gói văn bản thành token ID kèm `<bos>` và `<eos>`, lưu trữ mảng nhị phân `uint16` theo định dạng `.bin`, tự động ngắt shard khi vượt quá `max_tokens_per_shard`.
- `ManifestGenerator`: Tính toán SHA-256 checksum của từng shard file, tổng hợp số tài liệu, số lượng tokens, kích thước bytes và phân bổ tỷ lệ ngôn ngữ, xuất ra file `data/manifests/manifest.json`.

---

## 3. Bằng chứng Thực nghiệm & Kết quả Kiểm thử

### 3.1. Chạy Thực nghiệm Pipeline 10 bước (`research/dataset/build_bilingual_corpus.py`)
```text
======================================================================
TUNA-AI: PIPELINE TIỀN XỬ LÝ DỮ LIỆU SONG NGỮ 10 BƯỚC (DATASET ENGINEERING)
======================================================================

[1] Tổng số tài liệu thô ban đầu: 15 documents

[2] Số tài liệu vượt qua tất cả các bộ lọc: 11 documents
[3] Số tài liệu bị loại bỏ: 4 documents
    - Doc #11 [Exact Dedup]: Trùng lặp nguyên văn
    - Doc #13 [Quality]: Văn bản quá ngắn (1 từ < 4 từ tối thiểu)
    - Doc #14 [Quality]: Văn bản quá ngắn (1 từ < 4 từ tối thiểu)
    - Doc #15 [Quality]: Tỷ lệ lặp 1 từ quá cao (100.00% > 30.00%)

[4] Thống kê ngôn ngữ tài liệu hợp lệ:
    - Ngôn ngữ 'vi': 7 documents
    - Ngôn ngữ 'en': 4 documents
    - Ngôn ngữ 'other': 0 documents

[5] Nạp thành công BilingualBPETokenizer (Vocab size = 500)

[6] Đã đóng gói thành công 6 binary shards:
    - tuna_shard_00000.bin: 113 tokens, 226 bytes, SHA-256: 703114267eb3c5d7...
    - tuna_shard_00001.bin: 148 tokens, 296 bytes, SHA-256: 062909d31f3aaa23...
    - tuna_shard_00002.bin: 151 tokens, 302 bytes, SHA-256: af21f6c7d44fe39d...
    - tuna_shard_00003.bin: 172 tokens, 344 bytes, SHA-256: 2e4f40c81efb6f8c...
    - tuna_shard_00004.bin: 143 tokens, 286 bytes, SHA-256: 0fe744c36a2f8ccc...
    - tuna_shard_00005.bin: 63 tokens, 126 bytes, SHA-256: 85fcebd955863ee2...

[7] Đã tạo thành công Metadata Manifest: data/manifests/manifest.json
```

### 3.2. Kết quả Unit Tests Toàn diện
Toàn bộ **86/86 unit tests** (bao gồm 15 tests mới của `test_dataset_engineering.py`) vượt qua 100% trong 6.15s:
```text
collected 86 items

tests\test_autograd.py .....                                             [  5%]
tests\test_calculus_derivatives.py .......                               [ 13%]
tests\test_dataset_engineering.py ...............                        [ 31%]
tests\test_environment.py ...                                            [ 34%]
tests\test_linear_regression.py ...                                      [ 38%]
tests\test_matrix_multiplication.py .....                                [ 44%]
tests\test_neural_network.py ....                                        [ 48%]
tests\test_tensor_autograd.py ....                                       [ 53%]
tests\test_tensor_operations.py ......                                   [ 60%]
tests\test_tokenizer.py ............                                     [ 74%]
tests\test_training_engine.py ....                                       [ 79%]
tests\test_transformer.py ......                                         [ 86%]
tests\test_vectors_matrices.py ............                              [100%]

============================= 86 passed in 6.15s ==============================
```

---

## 4. Kết luận Nghiệm thu Milestone M7
Milestone M7 đã chính thức hoàn thành và đáp ứng 100% các tiêu chí kỹ thuật:
- Hệ thống làm sạch và lọc dữ liệu hoạt động chính xác, bảo mật và loại bỏ PII.
- Khử trùng lặp đa tầng (Exact SHA-256 + Fuzzy MinHash LSH) hiệu năng cao.
- Đóng gói định dạng nhị phân uint16 tối ưu băng thông và bộ nhớ cho phần cứng CPU i7-1165G7.
- Xuất manifest JSON truy xuất nguồn gốc và kiểm tra toàn vẹn dữ liệu.
