"""Bộ kiểm thử đơn vị tự động cho Hệ thống Dataset Engineering (Milestone M7).

Kiểm tra:
1. TextCleaner: Chuẩn hóa NFC, loại bỏ thẻ HTML, chuẩn hóa khoảng trắng.
2. RuleQualityFilter: Lọc độ dài từ, ký tự đặc biệt, spam lặp từ.
3. LanguageIdentifier: Phân loại chính xác văn bản tiếng Việt ('vi') và tiếng Anh ('en').
4. PIIFilter: Phát hiện và che giấu email, số điện thoại, IP, credit card.
5. ExactDeduplicator: Phát hiện và loại bỏ bản sao chính xác qua SHA-256.
6. MinHashDeduplicator: Ước lượng Jaccard Similarity và loại bỏ văn bản tương đồng cao.
7. DatasetPacker & ManifestGenerator: Đóng gói ra uint16 .bin shards và tạo manifest.json chuẩn.
"""

import json
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pytest

from src.data.cleaner import LanguageIdentifier, RuleQualityFilter, TextCleaner
from src.data.deduplicator import ExactDeduplicator, MinHashDeduplicator
from src.data.packer import DatasetPacker, ManifestGenerator
from src.data.pii_filter import PIIFilter
from src.tokenizer.tokenizer import BilingualBPETokenizer


class TestTextCleaner:
    def test_html_tag_removal(self):
        cleaner = TextCleaner()
        raw = "<h1>Tiêu đề</h1><p>Đây là <b>nội dung</b> đoạn văn.</p>"
        cleaned = cleaner.clean(raw)
        assert "<" not in cleaned and ">" not in cleaned
        assert "Tiêu đề Đây là nội dung đoạn văn." == cleaned

    def test_unicode_nfc_normalization(self):
        cleaner = TextCleaner()
        # Chuỗi NFD (tổ hợp) vs NFC
        # 'Việt' dạng NFD có e + dấu nặng + dấu mũ tách rời
        nfd_viet = "Vie\u0323\u0302t"
        cleaned = cleaner.clean(nfd_viet)
        assert cleaned == "Việt"

    def test_whitespace_normalization(self):
        cleaner = TextCleaner()
        raw = "   Dòng 1     với    nhiều    khoảng trắng   \n\n\n\n\n   Dòng 2   "
        cleaned = cleaner.clean(raw)
        assert "   " not in cleaned
        assert "\n\n\n" not in cleaned
        assert cleaned == "Dòng 1 với nhiều khoảng trắng\n\nDòng 2"


class TestRuleQualityFilter:
    def test_too_short_text(self):
        qf = RuleQualityFilter(min_words=5)
        is_valid, reason = qf.evaluate("Quá ngắn")
        assert not is_valid
        assert "quá ngắn" in reason.lower()

    def test_special_char_ratio(self):
        qf = RuleQualityFilter(min_words=3, max_special_char_ratio=0.30)
        is_valid, reason = qf.evaluate("Hợp lệ hoàn toàn abc def ghi")
        assert is_valid

        is_valid_junk, reason_junk = qf.evaluate("@@@ ### $$$ %%% ^^^ &&& *** !!! ???")
        assert not is_valid_junk
        assert "đặc biệt" in reason_junk.lower()

    def test_repetition_spam(self):
        qf = RuleQualityFilter(min_words=5, max_repetition_ratio=0.35)
        spam = "spam spam spam spam spam spam spam spam spam spam hợp lệ"
        is_valid, reason = qf.evaluate(spam)
        assert not is_valid
        assert "lặp" in reason.lower()


class TestLanguageIdentifier:
    def test_vietnamese_detection(self):
        lang_id = LanguageIdentifier()
        text_vn = "Trí tuệ nhân tạo và xử lý ngôn ngữ tự nhiên tiếng Việt."
        assert lang_id.detect(text_vn) == "vi"

    def test_english_detection(self):
        lang_id = LanguageIdentifier()
        text_en = "Artificial intelligence and natural language processing with transformer models."
        assert lang_id.detect(text_en) == "en"


class TestPIIFilter:
    def test_redact_email(self):
        pii = PIIFilter()
        text = "Gửi email cho tôi tại support@tuna-ai.com nhé."
        redacted = pii.redact(text)
        assert "support@tuna-ai.com" not in redacted
        assert "[EMAIL]" in redacted

    def test_redact_phone_vn_and_intl(self):
        pii = PIIFilter()
        text = "Số hotline là 0912345678 hoặc +84987654321 để đặt hàng."
        redacted = pii.redact(text)
        assert "0912345678" not in redacted
        assert "[PHONE]" in redacted

    def test_redact_ip_address(self):
        pii = PIIFilter()
        text = "Máy chủ nội bộ đặt tại 192.168.1.250 và cổng 8080."
        redacted = pii.redact(text)
        assert "192.168.1.250" not in redacted
        assert "[IP_ADDRESS]" in redacted

    def test_scan_pii_counts(self):
        pii = PIIFilter()
        text = "Liên hệ dev@tuna.org hoặc hotline 0901234567 tại IP 10.0.0.1."
        counts = pii.scan(text)
        assert counts["email"] == 1
        assert counts["phone"] == 1
        assert counts["ip"] == 1


class TestDeduplicator:
    def test_exact_deduplication(self):
        dedup = ExactDeduplicator()
        doc = "Mô hình ngôn ngữ lớn cần dữ liệu sạch."
        assert not dedup.is_duplicate(doc)
        # Lần 2 với khoảng trắng dư thừa vẫn phải nhận diện là trùng
        assert dedup.is_duplicate("   Mô hình ngôn ngữ lớn    cần dữ liệu sạch.   ")
        # Văn bản khác
        assert not dedup.is_duplicate("Mô hình khác hoàn toàn.")

    def test_minhash_fuzzy_deduplication(self):
        dedup = MinHashDeduplicator(num_perm=64, ngram_size=3, similarity_threshold=0.70)
        doc1 = "Trí tuệ nhân tạo là ngành khoa học máy tính nghiên cứu các mô hình thông minh."
        is_dup1, sim1 = dedup.is_duplicate(doc1)
        assert not is_dup1

        # Doc 2 tương đồng 85% với doc1
        doc2 = "Trí tuệ nhân tạo là một ngành khoa học máy tính nghiên cứu về các mô hình thông minh."
        is_dup2, sim2 = dedup.is_duplicate(doc2)
        assert is_dup2
        assert sim2 >= 0.70

        # Doc 3 hoàn toàn khác biệt
        doc3 = "Hôm nay thời tiết đẹp trời và có gió mùa thu mát mẻ tại thủ đô Hà Nội."
        is_dup3, sim3 = dedup.is_duplicate(doc3)
        assert not is_dup3


class TestDatasetPackerAndManifest:
    @pytest.fixture
    def temp_dirs(self):
        tmp_dir = tempfile.mkdtemp()
        processed_dir = Path(tmp_dir) / "processed"
        manifest_dir = Path(tmp_dir) / "manifests"
        yield processed_dir, manifest_dir
        shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_packer_and_manifest_workflow(self, temp_dirs):
        processed_dir, manifest_dir = temp_dirs

        # Nạp tokenizer toy đã lưu ở Milestone M4
        tok_file = "configs/tokenizer/bpe_toy_500.json"
        assert Path(tok_file).exists()
        tokenizer = BilingualBPETokenizer.from_file(tok_file)

        docs = [
            "Mô hình ngôn ngữ Tuna AI học tiếng Việt và tiếng Anh.",
            "Transformer decoder sử dụng self-attention để dự đoán từ kế tiếp.",
            "Deep learning with PyTorch provides high performance on CPU using AVX-512.",
        ]

        # Đóng gói với max_tokens_per_shard nhỏ để kích hoạt sharding
        packer = DatasetPacker(
            tokenizer=tokenizer,
            output_dir=str(processed_dir),
            max_tokens_per_shard=25,
            shard_prefix="test_shard_",
        )
        shard_records = packer.pack(docs)

        assert len(shard_records) >= 1
        # Kiểm tra tính toàn vẹn của shard nhị phân
        for s in shard_records:
            shard_path = Path(s["path"])
            assert shard_path.exists()
            # Đọc lại mảng uint16
            tokens = np.fromfile(str(shard_path), dtype=np.uint16)
            assert len(tokens) == s["num_tokens"]
            # Kiểm tra token đầu tiên phải là BOS id
            assert tokens[0] == tokenizer.bos_token_id

        # Kiểm tra tạo manifest
        manifest_gen = ManifestGenerator(manifest_dir=str(manifest_dir))
        manifest_path = manifest_gen.generate(
            dataset_name="Test Corpus",
            shard_records=shard_records,
            language_stats={"vi": 2, "en": 1},
        )
        assert Path(manifest_path).exists()

        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["total_documents"] == 3
        assert data["total_tokens"] == sum(s["num_tokens"] for s in shard_records)
        assert data["num_shards"] == len(shard_records)
        assert "vi" in data["language_counts"]
