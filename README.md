# Tuna-AI: Nghiên cứu & Xây dựng Mô hình Ngôn ngữ Lớn Song ngữ (Việt - Anh) từ số 0

[![Python Version](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Unit Tests](https://img.shields.io/badge/Tests-107%2F107%20Passing-brightgreen.svg)](tests/)
[![Warnings](https://img.shields.io/badge/Warnings-0-brightgreen.svg)](pyproject.toml)
[![Hardware](https://img.shields.io/badge/Hardware-Intel%20Core%20i7--1165G7%20(Pure%20CPU)-orange.svg)]()

> **Tuna-AI** là một dự án nghiên cứu và kỹ thuật phần mềm mã nguồn mở nhằm thiết kế, hiện thực hóa và huấn luyện một **Mô hình Ngôn ngữ Lớn (Bilingual Large Language Model)** song ngữ (Việt - Anh) hoàn toàn từ đầu (*Training from Scratch* — ngẫu nhiên hóa trọng số $100\%$, tuyệt đối không dùng pre-trained weights từ bên ngoài). Toàn bộ kiến trúc và thuật toán được tối ưu hóa chuyên sâu để huấn luyện và suy luận hiệu năng cao trên phần cứng máy tính xách tay cá nhân phổ thông (CPU Intel Core i7-1165G7, 16GB RAM, không cần GPU rời).

---

## 📑 Bảng Mục lục
1. [Triết lý Thiết kế & Ràng buộc Phần cứng](#-triết-lý-thiết-kế--ràng-buộc-phần-cứng)
2. [Lộ trình 12 Cột mốc Kỹ thuật (Milestones M1 -> M12)](#-lộ-trình-12-cột-mốc-kỹ-thuật-milestones-m1---m12)
3. [Cấu trúc Thư mục Dự án](#-cấu-trúc-thư-mục-dự-án)
4. [Hướng dẫn Cài đặt & Khởi tạo Môi trường](#-hướng-dẫn-cài-đặt--khởi-tạo-môi-trường)
5. [Hướng dẫn Vận hành Toàn trình](#-hướng-dẫn-vận-hành-toàn-trình)
   - [Chạy Bộ Kiểm thử Đơn vị (107 Tests)](#1-chạy-bộ-kiểm-thử-đơn-vị-unit-tests)
   - [Pipeline Tiền xử lý Dữ liệu 10 bước](#2-pipeline-tiền-xử-lý-dữ-liệu-10-bước)
   - [Huấn luyện Tiền kỳ & Thí nghiệm Kiểm soát](#3-huấn-luyện-tiền-kỳ--chuỗi-thí-nghiệm-kiểm-soát)
   - [Đánh giá Năng lực Ngôn ngữ & Benchmark Suite](#4-đánh-giá-năng-lực-ngôn-ngữ--benchmark-suite)
   - [Tinh chỉnh Chỉ dẫn (Instruction Tuning / SFT)](#5-tinh-chỉnh-chỉ-dẫn-instruction-tuning--sft)
   - [Suy luận Offline với Tăng tốc KV Cache](#6-suy-luận-offline-với-tăng-tốc-kv-cache)
   - [Khởi chạy Model Serving REST API](#7-khởi-chạy-model-serving-rest-api)
6. [Bằng chứng Kiểm định & Số liệu Thực nghiệm](#-bằng-chứng-kiểm-định--số-liệu-thực-nghiệm)

---

## 🎯 Triết lý Thiết kế & Ràng buộc Phần cứng

- **Mục tiêu**: Nắm vững bản chất toán học, giải tích đa biến, lan truyền ngược (Autograd), cơ chế Attention và kỹ thuật tối ưu hóa tài nguyên cấp thấp.
- **Ràng buộc phần cứng mục tiêu**:
  - **CPU**: Intel Core i7-1165G7 (4 nhân vật lý / 8 luồng, tập lệnh mở rộng AVX-512 VNNI, Intel MKL + oneDNN).
  - **RAM**: 16GB (giới hạn ngân sách chạy ứng dụng $\le 8\text{GB}$).
  - **Ổ cứng**: SSD NVMe (cơ chế phân mảnh Sharding & Memory-mapping uint16 chống suy giảm tuổi thọ SSD).
  - **Không phụ thuộc GPU**: Mã nguồn thuần CPU execution, chuẩn hóa đa nền tảng (Windows/Linux/macOS).

---

## 🏆 Lộ trình 12 Cột mốc Kỹ thuật (Milestones M1 -> M12)

Dự án đã hoàn thành xuất sắc **100% tất cả 12 Milestones**:

| Cột mốc | Tên Milestone | Nội dung Kỹ thuật Cốt lõi | Trạng thái |
| :---: | :--- | :--- | :---: |
| **M1** | **Linear Algebra & Calculus** | Tensors 4D, GEMM ($i-k-j$, Cache-aware $32 \times 32$), Softmax CrossEntropy, AdamW | `Hoàn thành` |
| **M2** | **Tensor Autograd Engine** | Động cơ vi phân tự động đa chiều, Unbroadcasting bias, Matmul Backward | `Hoàn thành` |
| **M3** | **Neural Network & Nonlinear MLP** | Khối `Module`, `Linear` Kaiming Uniform, `MLP`, phân loại phi tuyến Circles | `Hoàn thành` |
| **M4** | **Byte-level BPE Tokenizer** | BPE 256 bytes UTF-8 GPT-2 style, chuẩn hóa Unicode NFC tiếng Việt | `Hoàn thành` |
| **M5** | **Causal Transformer Decoder** | Pre-RMSNorm, Causal Multi-Head Attention, FFN GELU, Next-token LM Head | `Hoàn thành` |
| **M6** | **Training Engine & Mini GPT** | Checkpoint xoay vòng, Cosine Warmup, Gradient Clipping, Huấn luyện Mini GPT | `Hoàn thành` |
| **M7** | **Dataset Engineering** | Pipeline 10 bước: TextCleaner, PIIFilter, MinHash LSH Dedup, uint16 Packing | `Hoàn thành` |
| **M8** | **Pre-training Streaming** | `ShardDataLoader` memmap uint16, Thí nghiệm kiểm soát EXP-001, 002, 003 | `Hoàn thành` |
| **M9** | **Evaluation & Benchmark** | Perplexity Hold-out, Top-1/Top-5 Accuracy, Cloze Test song ngữ, TTFT & ITL | `Hoàn thành` |
| **M10** | **Instruction Tuning (SFT)** | ChatML Template, Response Masking (label=-100), Phản hồi hỏi đáp đa lượt | `Hoàn thành` |
| **M11** | **Inference Engine & KV Cache** | `KVCache` giảm $O(N^2) \to O(N)$, Top-P, Repetition Penalty, Streaming, Đóng gói Offline | `Hoàn thành` |
| **M12** | **Model Serving REST API** | FastAPI / Uvicorn Server, endpoints `/health`, `/v1/generate`, `/v1/chat/completions` | `Hoàn thành` |

---

## 📁 Cấu trúc Thư mục Dự án

```text
tuna-ai/
├── configs/                  # Tệp cấu hình JSON siêu tham số
│   ├── model/tiny.json       # Kiến trúc mô hình TunaLM
│   └── tokenizer/            # Từ điển BPE 500 vocab
├── data/                     # Thư mục dữ liệu (được quản lý tự động)
│   ├── manifests/            # Siêu dữ liệu manifest.json
│   ├── processed/            # Binary shards uint16
│   └── experiments/          # Dữ liệu phân mảnh cho thí nghiệm
├── docs/                     # Tài liệu kỹ thuật & nhật ký nghiên cứu
│   └── weekly_reports/       # Báo cáo kỹ thuật chi tiết từng ngày (Day 1 -> Day 18)
├── experiments/reports/      # Báo cáo đo lường thực nghiệm định lượng & metrics
├── models/                   # Gói mô hình đóng gói chạy Offline
│   └── tuna_offline_v1/      # config.json, model_weights.pt, tokenizer.json, model_card.md
├── research/                 # Mã nguồn thí nghiệm, nghiên cứu độc lập
│   ├── mathematics/          # Đại số tuyến tính, đạo hàm giải tích, GEMM benchmark
│   ├── machine_learning/     # Hồi quy tuyến tính, bộ tối ưu hóa AdamW/SGD
│   ├── deep_learning/        # Mạng nơ-ron đa tầng MLP
│   ├── dataset/              # Pipeline tiền xử lý dữ liệu 10 bước
│   ├── experiments/          # Huấn luyện tiền kỳ có kiểm soát (EXP-001 -> EXP-003)
│   ├── evaluation/           # Đánh giá năng lực độc lập
│   ├── instruction/          # Tinh chỉnh chỉ dẫn SFT
│   └── inference/            # Benchmark đối đầu KV Cache
├── scripts/                  # Kịch bản công cụ CLI
│   └── serve_api.py          # Lệnh chạy FastAPI Server
├── src/                      # Package lõi của hệ thống Tuna-AI
│   ├── core/                 # Tensor Autograd, Modules, Loss Functions
│   ├── tokenizer/            # Unicode Normalizer, BPE Trainer & Tokenizer
│   ├── models/               # Causal Transformer Decoder, Attention, Embeddings
│   ├── data/                 # Cleaner, PIIFilter, Deduplicator, ShardDataLoader
│   ├── training/             # Trainer, Scheduler, CheckpointManager, ExperimentRunner
│   ├── evaluation/           # Evaluator, ClozeBenchmark, Metrics
│   ├── instruction/          # ChatTemplate, SFTDataset, SFTTrainer
│   ├── inference/            # KVCache, TextGenerator, ModelExporter
│   └── serving/              # FastAPI REST App, Schemas, CORS Middleware
├── tests/                    # 17 tệp unit test bao phủ toàn bộ 107 bài test
├── pyproject.toml            # Cấu hình dự án & Pytest
├── MOTA.md                   # Tài liệu đặc tả kỹ thuật nền tảng
├── KE_HOACH_TRIEN_KHAI.md    # Kế hoạch & Biên bản nghiệm thu 12 Milestones
└── README.md                 # Tài liệu tổng kết dự án
```

---

## 🚀 Hướng dẫn Cài đặt & Khởi tạo Môi trường

### 1. Yêu cầu Hệ thống
- Hệ điều hành: Windows 10/11, Ubuntu 20.04+, hoặc macOS.
- Python: Phiên bản **3.11** trở lên.
- Trình quản lý gói: `pip`.

### 2. Thiết lập Môi trường Ảo
```bash
# Clone repository
git clone https://github.com/tuanhuynh1704/tuna-ai.git
cd tuna-ai

# Tạo và kích hoạt virtual environment
python -m venv .venv

# Trên Windows:
.venv\Scripts\activate

# Trên Linux/macOS:
source .venv/bin/activate

# Nâng cấp pip và cài đặt phụ thuộc
pip install -r requirements.txt  # hoặc pip install numpy torch pytest fastapi uvicorn httpx psutil
```

---

## 🛠 Hướng dẫn Vận hành Toàn trình

### 1. Chạy Bộ Kiểm thử Đơn vị (Unit Tests)
Dự án được bảo vệ bởi **107 bài test tự động** bao phủ từ nền tảng đại số ma trận đến API máy chủ:
```bash
python -m pytest -q
```
*Kết quả: `107 passed in 6.2s, 0 warnings`.*

### 2. Pipeline Tiền xử lý Dữ liệu 10 bước
Làm sạch văn bản, chuẩn hóa Unicode NFC, khử trùng lặp (Exact SHA-256 + Fuzzy MinHash LSH), che giấu PII và đóng gói shard nhị phân uint16:
```bash
python research/dataset/build_bilingual_corpus.py
```

### 3. Huấn luyện Tiền kỳ & Chuỗi Thí nghiệm Kiểm soát
Chạy chuỗi 3 thí nghiệm có kiểm soát: `EXP-001 (English)`, `EXP-002 (Vietnamese)`, `EXP-003 (Bilingual 50/50)` và đo lường thông lượng:
```bash
python research/experiments/run_pretraining_experiments.py
```

### 4. Đánh giá Năng lực Ngôn ngữ & Benchmark Suite
Đo lường Perplexity trên tập giữ lại, Top-1/Top-5 Next-token accuracy, Cloze Test và độ trễ suy luận TTFT:
```bash
python research/evaluation/run_evaluation.py
```

### 5. Tinh chỉnh Chỉ dẫn (Instruction Tuning / SFT)
Huấn luyện mô hình hội thoại đa lượt với ChatML và cơ chế Response Masking:
```bash
python research/instruction/train_sft.py
```

### 6. Suy luận Offline với Tăng tốc KV Cache
Đo đạc đối đầu hiệu năng giải mã và đóng gói mô hình:
```bash
python research/inference/benchmark_kv_cache.py
```

### 7. Khởi chạy Model Serving REST API
Khởi chạy dịch vụ API cục bộ hỗ trợ chuẩn OpenAI:
```bash
python scripts/serve_api.py --port 8000
```
- Truy cập Swagger UI tại: [http://localhost:8000/docs](http://localhost:8000/docs)
- Kiểm tra sức khỏe: `curl http://localhost:8000/health`
- Gọi API Hội thoại OpenAI chuẩn:
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Xin chào! Bạn là ai?"}],
    "max_tokens": 30,
    "temperature": 0.7
  }'
```

---

## 📊 Bằng chứng Kiểm định & Số liệu Thực nghiệm

| Tiêu chí Thực nghiệm | Kết quả Đo đạc trên CPU i7-1165G7 |
| :--- | :---: |
| **Tốc độ Huấn luyện Pre-training** | **11,800 - 12,100 tokens/giây** |
| **Mức tăng RAM khi đọc Streaming Memmap** | **< 10 MB** (bảo vệ tuyệt đối 16GB RAM) |
| **SFT Response Loss** | Giảm từ **6.2292** xuống **1.1448** (PPL = **3.24**) |
| **Thời gian phản hồi đầu tiên (TTFT)** | **2.48 ms** |
| **Tốc độ giải mã với KV Cache** | **4,351.4 tokens/giây** (nhanh gấp **12.19x**) |
| **RAM Footprint ở chế độ Serving** | **< 290 MB** |
| **Độ bao phủ Kiểm thử (Test Suite)** | **107/107 tests passing (100%), 0 warnings** |

---

## 📜 Giấy phép & Đóng góp
Dự án được phân phối dưới giấy phép **MIT License**. Mọi đóng góp nghiên cứu, cải tiến thuật toán và mở rộng bộ ngữ liệu đều được hoan nghênh qua Pull Requests.
