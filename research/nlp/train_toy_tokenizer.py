"""
research/nlp/train_toy_tokenizer.py

Huấn luyện thử nghiệm Tokenizer Song ngữ Byte-level BPE trên tập dữ liệu mẫu:
- Đánh giá khả năng xử lý dấu thanh tiếng Việt (ngã, hỏi, nặng, huyền, sắc).
- Kiểm tra tính toàn vẹn 100% Round-trip Encode -> Decode.
- Lưu cấu hình ra file configs/tokenizer/bpe_toy_500.json.
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.tokenizer.tokenizer import BilingualBPETokenizer


SAMPLE_CORPUS = [
    # Tiếng Việt
    "Hà Nội mùa này vắng những cơn mưa, cái rét đầu đông đã bắt đầu tràn về trên từng con phố.",
    "Trí tuệ nhân tạo (AI) và mô hình ngôn ngữ lớn (LLM) đang thay đổi cách con người làm việc và sáng tạo.",
    "Dự án Tuna-AI hướng tới việc xây dựng và huấn luyện mô hình ngôn ngữ từ đầu.",
    "Tiếng Việt có hệ thống thanh điệu phong phú: huyền, sắc, hỏi, ngã, nặng.",
    "Khởi tạo trọng số ngẫu nhiên, không sử dụng trọng số được huấn luyện từ trước (pretrained weights).",
    "Phép nhân ma trận và lan truyền ngược là hai thành phần cốt lõi của mạng nơ-ron.",
    "Thử nghiệm các từ có cấu trúc đặc biệt: nghiêng ngả, thoang thoảng, bâng khuâng, ngoằn ngoèo.",
    # Tiếng Anh
    "The quick brown fox jumps over the lazy dog.",
    "Large Language Models (LLMs) are trained on massive text corpora to predict the next token.",
    "Transformers rely on the attention mechanism to capture long-range dependencies in natural language.",
    "Byte-level Byte Pair Encoding ensures zero out-of-vocabulary tokens by falling back to base bytes.",
    "Matrix multiplication efficiency is vital for deep learning inference on edge devices.",
    # Văn bản trộn ngữ (Code, Dấu câu, Thuật ngữ song ngữ)
    "Hàm số def forward(x, weights): return torch.matmul(x, weights) + bias hoạt động rất mượt mà.",
    "Python 3.11 mang lại nhiều cải tiến về tốc độ thực thi so với các phiên bản 3.8 và 3.9.",
    "Đo lường độ trễ: Latency = 15.4ms, Throughput = 65.2 tokens/second trên CPU Intel Core i7-1165G7.",
    "Ký tự đặc biệt: @ # $ % ^ & * ( ) _ + { } [ ] : ; \" ' < > , . ? / ~ ` | \\",
]


def run_experiment():
    print("=" * 80)
    print("THỬ NGHIỆM MILESTONE M4: HUẤN LUYỆN BYTE-LEVEL BPE TOKENIZER SONG NGỮ")
    print("=" * 80)

    # 1. Huấn luyện Tokenizer
    print("\n[1] TIẾN HÀNH HUẤN LUYỆN BPE VỚI TARGET VOCABULARY = 500 TOKENS...")
    tokenizer = BilingualBPETokenizer()
    tokenizer.train(SAMPLE_CORPUS, target_vocab_size=500)
    print(f"  -> Huấn luyện thành công! Kích thước từ vựng thực tế: {len(tokenizer.vocab)} tokens")
    print(f"  -> Số lượng cặp merge pairs: {len(tokenizer.merges)}")

    # 2. Lưu vào thư mục configs/tokenizer/
    out_dir = project_root / "configs" / "tokenizer"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = str(out_dir / "bpe_toy_500.json")
    tokenizer.save(out_file)
    print(f"  -> Đã lưu cấu hình Tokenizer ra: {out_file}")

    # 3. Nạp lại và kiểm tra Round-trip Lossless
    print("\n[2] KIỂM TRA ĐỘ CHÍNH XÁC ROUND-TRIP (DECODE(ENCODE(TEXT)) == TEXT):")
    loaded_tokenizer = BilingualBPETokenizer.load(out_file)

    test_sentences = [
        "Trí tuệ nhân tạo là ngành khoa học máy tính hiện đại.",
        "Deep Learning and Transformer architecture from scratch.",
        "Tiếng Việt dấu thanh: ắ, ằ, ẳ, ẵ, ặ, ế, ề, ể, ễ, ệ, ố, ồ, ổ, ỗ, ộ.",
        "Trộn ngữ: Model Tuna-AI chạy offline 100% trên CPU i7-1165G7!",
        "Ký hiệu toán học: Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V.",
    ]

    all_passed = True
    for s in test_sentences:
        tokens = loaded_tokenizer.encode(s)
        decoded = loaded_tokenizer.decode(tokens)
        bytes_len = len(s.encode("utf-8"))
        tokens_len = len(tokens)
        compression = bytes_len / tokens_len

        match = (s == decoded)
        status = "PASSED" if match else "FAILED"
        if not match:
            all_passed = False

        print(f"\n  Câu gốc   : '{s}'")
        print(f"  Token IDs : {tokens[:10]}... (Tổng: {tokens_len} tokens / {bytes_len} bytes, Tỉ lệ nén: {compression:.2f}x)")
        print(f"  Giải mã   : '{decoded}'")
        print(f"  Kiểm tra  : [{status}]")

    print("\n" + "=" * 80)
    if all_passed:
        print("KẾT LUẬN: 100% VĂN BẢN ĐƯỢC GIẢI MÃ HOÀN HẢO KHÔNG MẤT MÁT DỮ LIỆU!")
        print("CHÍNH THỨC ĐẠT TIÊU CHÍ KỸ THUẬT CỦA BPE TOKENIZER SONG NGỮ (MILESTONE M4)!")
    else:
        print("CÓ LỖI XẢY RA TRONG QUÁ TRÌNH GIẢI MÃ!")
    print("=" * 80)


if __name__ == "__main__":
    run_experiment()
