"""Thực nghiệm xây dựng Dataset Song ngữ (Việt - Anh) theo Quy trình 10 bước của Tuna-AI.

Script này minh họa toàn bộ pipeline tiền xử lý:
1. Thu thập dữ liệu mẫu (Data Acquisition - Song ngữ Việt - Anh)
2. Lọc định dạng rác (HTML, XML)
3. Nhận diện ngôn ngữ (Language Identification)
4. Lọc chất lượng theo quy tắc (Rule Quality Filter)
5. Khử trùng lặp chính xác & mờ (Exact & MinHash Deduplication)
6. Che giấu thông tin cá nhân PII (Email, Phone, IP)
7. Chuẩn hóa văn bản (Unicode NFC)
8. Tokenization & Packing vào uint16
9. Tạo Metadata Manifest (manifest.json)
10. Sharding phân mảnh dữ liệu an toàn cho bộ nhớ
"""

import os
import sys
from pathlib import Path

# Đảm bảo đường dẫn gốc của project có trong sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.cleaner import LanguageIdentifier, RuleQualityFilter, TextCleaner
from src.data.deduplicator import ExactDeduplicator, MinHashDeduplicator
from src.data.packer import DatasetPacker, ManifestGenerator
from src.data.pii_filter import PIIFilter
from src.tokenizer.tokenizer import BilingualBPETokenizer


def get_raw_sample_corpus():
    """Tạo tập hợp văn bản thô đa dạng có chứa cả dữ liệu tốt, dữ liệu nhiễu, PII và trùng lặp."""
    return [
        # Tiếng Việt chất lượng cao
        "<p>Trí tuệ nhân tạo (AI) là một ngành khoa học máy tính nghiên cứu và phát triển các hệ thống thông minh.</p>",
        "Mô hình ngôn ngữ lớn (LLM) dựa trên kiến trúc Transformer đã tạo ra một cuộc cách mạng trong xử lý ngôn ngữ tự nhiên.",
        "Tiếng Việt là ngôn ngữ có thanh điệu phong phú, đòi hỏi việc chuẩn hóa Unicode NFC nghiêm ngặt trước khi huấn luyện mô hình.",
        "Kiến trúc Causal Self-Attention sử dụng mặt nạ nhân quả tam giác trên để ngăn không cho token hiện tại nhìn thấy tương lai.",
        # Tiếng Anh chất lượng cao
        "<div>Artificial Intelligence has transformed modern computing through deep neural networks and self-supervised learning.</div>",
        "The Transformer architecture replaces recurrence and convolutions entirely with self-attention mechanisms.",
        "Gradient clipping is an essential regularization technique that prevents exploding gradients in deep Transformer networks.",
        "Low-Rank Adaptation allows fine-tuning large language models efficiently by updating only small low-rank parameter matrices.",
        # Văn bản chứa PII nhạy cảm
        "Liên hệ hỗ trợ kỹ thuật tại email contact@tuna-ai.vn hoặc gọi số điện thoại 0912345678 để được tư vấn.",
        "Server production đang chạy tại địa chỉ IP 192.168.1.100, vui lòng gửi phản hồi về dev-team@tuna.org ngay.",
        # Bản sao trùng lặp chính xác (Exact duplicates)
        "Mô hình ngôn ngữ lớn (LLM) dựa trên kiến trúc Transformer đã tạo ra một cuộc cách mạng trong xử lý ngôn ngữ tự nhiên.",
        # Bản sao trùng lặp mờ (Fuzzy duplicate - thay đổi 1 vài từ)
        "Trí tuệ nhân tạo AI là một ngành khoa học máy tính nghiên cứu và xây dựng các hệ thống máy học thông minh.",
        # Văn bản rác vi phạm quy tắc chất lượng (quá ngắn hoặc ký tự đặc biệt)
        "Ok.",
        "!@#$%^&*()_+=-~`<>?:{}",
        "spam spam spam spam spam spam spam spam spam spam spam spam spam spam spam",
    ]


def main():
    print("=" * 70)
    print("TUNA-AI: PIPELINE TIỀN XỬ LÝ DỮ LIỆU SONG NGỮ 10 BƯỚC (DATASET ENGINEERING)")
    print("=" * 70)

    cleaner = TextCleaner()
    quality_filter = RuleQualityFilter(min_words=4, max_special_char_ratio=0.35, max_repetition_ratio=0.30)
    lang_id = LanguageIdentifier()
    pii_filter = PIIFilter()
    exact_dedup = ExactDeduplicator()
    minhash_dedup = MinHashDeduplicator(num_perm=64, ngram_size=3, similarity_threshold=0.75)

    raw_docs = get_raw_sample_corpus()
    print(f"\n[1] Tổng số tài liệu thô ban đầu: {len(raw_docs)} documents")

    processed_docs = []
    lang_counts = {"vi": 0, "en": 0, "other": 0}
    filtered_reasons = []

    for idx, doc in enumerate(raw_docs):
        # 1. Làm sạch & Chuẩn hóa NFC
        cleaned = cleaner.clean(doc)

        # 2. Lọc chất lượng
        is_valid, reason = quality_filter.evaluate(cleaned)
        if not is_valid:
            filtered_reasons.append(f"Doc #{idx+1} [Quality]: {reason}")
            continue

        # 3. Khử trùng lặp chính xác
        if exact_dedup.is_duplicate(cleaned):
            filtered_reasons.append(f"Doc #{idx+1} [Exact Dedup]: Trùng lặp nguyên văn")
            continue

        # 4. Khử trùng lặp mờ (MinHash)
        is_fuzzy_dup, sim = minhash_dedup.is_duplicate(cleaned)
        if is_fuzzy_dup:
            filtered_reasons.append(f"Doc #{idx+1} [Fuzzy Dedup]: Trùng lặp mờ (Jaccard = {sim:.2%})")
            continue

        # 5. Che giấu PII
        redacted = pii_filter.redact(cleaned)

        # 6. Nhận diện ngôn ngữ
        lang = lang_id.detect(redacted)
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

        processed_docs.append(redacted)

    print(f"\n[2] Số tài liệu vượt qua tất cả các bộ lọc: {len(processed_docs)} documents")
    print(f"[3] Số tài liệu bị loại bỏ: {len(filtered_reasons)} documents")
    for r in filtered_reasons:
        print(f"    - {r}")

    print(f"\n[4] Thống kê ngôn ngữ tài liệu hợp lệ:")
    for lang, count in lang_counts.items():
        print(f"    - Ngôn ngữ '{lang}': {count} documents")

    # [5] Nạp Tokenizer đã huấn luyện từ Milestone M4
    tokenizer_path = Path("configs/tokenizer/bpe_toy_500.json")
    if not tokenizer_path.exists():
        print(f"Lỗi: Không tìm thấy tokenizer tại {tokenizer_path}")
        return

    tokenizer = BilingualBPETokenizer.from_file(str(tokenizer_path))
    print(f"\n[5] Nạp thành công BilingualBPETokenizer (Vocab size = {tokenizer.vocab_size})")

    # [6] Đóng gói vào Binary Shards (Sharding)
    processed_dir = "data/processed"
    packer = DatasetPacker(
        tokenizer=tokenizer,
        output_dir=processed_dir,
        max_tokens_per_shard=100,  # Ngưỡng nhỏ để kiểm thử sharding đa shard
        shard_prefix="tuna_shard_",
    )
    shard_records = packer.pack(processed_docs)
    print(f"\n[6] Đã đóng gói thành công {len(shard_records)} binary shards:")
    for s in shard_records:
        print(f"    - {s['filename']}: {s['num_tokens']} tokens, {s['file_size_bytes']} bytes, SHA-256: {s['sha256'][:16]}...")

    # [7] Sinh Metadata Manifest
    manifest_gen = ManifestGenerator(manifest_dir="data/manifests")
    manifest_file = manifest_gen.generate(
        dataset_name="Tuna-AI Bilingual Pretraining Corpus",
        shard_records=shard_records,
        language_stats=lang_counts,
        version="1.0.0",
        license_str="MIT / CC-BY-4.0",
        manifest_filename="manifest.json",
    )
    print(f"\n[7] Đã tạo thành công Metadata Manifest: {manifest_file}")
    print("\n" + "=" * 70)
    print("HOÀN THÀNH TOÀN DIỆN PIPELINE TIỀN XỬ LÝ DỮ LIỆU MILESTONE M7")
    print("=" * 70)


if __name__ == "__main__":
    main()
