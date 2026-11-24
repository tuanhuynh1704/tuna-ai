"""
tests/test_tokenizer.py

Bộ kiểm thử Unit Test cho Milestone M4:
Kiểm thử Byte-level BPE Tokenizer Song ngữ (Việt - Anh):
- Chuẩn hóa Unicode NFC (bảo toàn ký tự có dấu tiếng Việt).
- Bảng ánh xạ 256 bytes không trùng lặp.
- Huấn luyện BPE Vocabulary và các quy tắc Merge.
- Kiểm tra tính bảo toàn 100% Round-trip Encode -> Decode (Lossless).
- Kiểm tra Save / Load định dạng JSON.
"""

import tempfile
import unicodedata
from pathlib import Path
import pytest
from src.tokenizer.normalizer import normalize_unicode, normalize_text
from src.tokenizer.bpe import bytes_to_unicode
from src.tokenizer.tokenizer import BilingualBPETokenizer


class TestUnicodeNormalization:
    def test_nfd_to_nfc_conversion(self):
        """
        Kiểm tra chuyển đổi từ tiếng Việt tổ hợp (NFD) sang dựng sẵn (NFC):
        Từ 'Việt':
          - NFC: 4 ký tự mã Unicode (V, i, ệ, t)
          - NFD: 5 ký tự mã Unicode (V, i, e, dấu mũ, dấu nặng)
        """
        text_nfc = "Việt Nam"
        text_nfd = unicodedata.normalize("NFD", text_nfc)

        assert len(text_nfd) > len(text_nfc)
        normalized = normalize_unicode(text_nfd, form="NFC")
        assert len(normalized) == len(text_nfc)
        assert normalized == text_nfc

    def test_normalize_text_cleans_hidden_control_chars(self):
        # Chứa ký tự điều khiển ẩn \x00 và \x0b nhưng giữ lại \n
        dirty_text = "Hà Nội\x00 mùa\x0b đông\nvắng những cơn mưa"
        clean = normalize_text(dirty_text)
        assert "\x00" not in clean
        assert "\x0b" not in clean
        assert "\n" in clean
        assert clean == "Hà Nội mùa đông\nvắng những cơn mưa"


class TestByteLevelBPE:
    def test_bytes_to_unicode_completeness(self):
        """Kiểm tra bảng ánh xạ phủ kín toàn bộ 256 giá trị byte và không có phần tử trùng lặp."""
        mapping = bytes_to_unicode()
        assert len(mapping) == 256
        assert len(set(mapping.values())) == 256
        # Tất cả các ký tự ánh xạ đều là chuỗi độ dài 1
        for char in mapping.values():
            assert len(char) == 1

    def test_tokenizer_training_and_special_tokens(self):
        corpus = [
            "Học máy và mô hình ngôn ngữ lớn.",
            "Machine learning and large language models.",
            "Tuna AI bilingual pretraining from scratch."
        ]
        tokenizer = BilingualBPETokenizer()
        target_vocab = 300
        tokenizer.train(corpus, target_vocab_size=target_vocab)

        assert len(tokenizer.vocab) == target_vocab
        assert tokenizer.special_tokens["<pad>"] == 0
        assert tokenizer.special_tokens["<unk>"] == 1
        assert tokenizer.special_tokens["<bos>"] == 2
        assert tokenizer.special_tokens["<eos>"] == 3

    def test_encode_with_bos_eos(self):
        tokenizer = BilingualBPETokenizer()
        tokenizer.train(["Xin chào thế giới", "Hello world"], target_vocab_size=280)

        tokens = tokenizer.encode("Xin chào", add_bos=True, add_eos=True)
        assert tokens[0] == 2   # <bos>
        assert tokens[-1] == 3  # <eos>

    @pytest.mark.parametrize("sentence", [
        "Hà Nội mùa này vắng những cơn mưa.",
        "Tiếng Việt: ắ, ằ, ẳ, ẵ, ặ, ế, ề, ể, ễ, ệ, ố, ồ, ổ, ỗ, ộ.",
        "The quick brown fox jumps over the lazy dog.",
        "Code test: def forward(x): return x @ W + b",
        "Trộn ngữ: Model Tuna-AI CPU i7-1165G7 đạt 132.68 GFLOPS!",
        "Ký hiệu lạ: ~ ! @ # $ % ^ & * ( ) _ + | \\ / ? > <",
    ])
    def test_round_trip_lossless(self, sentence):
        """Kiểm tra decode(encode(text)) == text tuyệt đối không mất mát ký tự."""
        corpus = [
            "Hà Nội mùa này vắng những cơn mưa.",
            "Tiếng Việt phong phú với các dấu thanh.",
            "The quick brown fox jumps over the lazy dog.",
            "Transformers and attention mechanisms in deep learning."
        ]
        tokenizer = BilingualBPETokenizer()
        tokenizer.train(corpus, target_vocab_size=350)

        token_ids = tokenizer.encode(sentence)
        decoded = tokenizer.decode(token_ids)
        assert decoded == sentence, f"Lỗi không khớp: '{sentence}' != '{decoded}'"

    def test_save_and_load(self):
        """Kiểm tra lưu và nạp cấu hình JSON."""
        tokenizer = BilingualBPETokenizer()
        tokenizer.train(["Kiểm tra lưu trữ cấu hình", "Testing tokenizer serialization"], target_vocab_size=290)

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = str(Path(tmp_dir) / "test_tokenizer.json")
            tokenizer.save(file_path)

            loaded = BilingualBPETokenizer.load(file_path)
            assert len(loaded.vocab) == len(tokenizer.vocab)
            assert len(loaded.merges) == len(tokenizer.merges)

            text = "Kiểm tra sau khi load"
            assert loaded.decode(loaded.encode(text)) == text
