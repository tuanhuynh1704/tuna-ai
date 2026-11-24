"""
src/tokenizer/tokenizer.py

Lớp Tokenizer Song ngữ Byte-level BPE (BilingualBPETokenizer):
- Tự động huấn luyện Vocabulary và các cặp hợp nhất Merges từ kho ngữ liệu.
- Quản lý các Special Tokens: <pad>, <unk>, <bos>, <eos>.
- Bảo đảm tính toàn vẹn 100% khi giải mã: decode(encode(text)) == text cho cả tiếng Việt và tiếng Anh.
- Hỗ trợ lưu trữ và nạp lại cấu hình từ định dạng JSON chuẩn.
"""

from typing import List, Dict, Tuple, Optional, Set
import json
import re
from pathlib import Path

from src.tokenizer.normalizer import normalize_text
from src.tokenizer.bpe import bytes_to_unicode, get_stats, merge_vocab


class BilingualBPETokenizer:
    """
    Byte-level Byte Pair Encoding Tokenizer hỗ trợ song ngữ Việt - Anh.
    Không bao giờ gặp lỗi Out-Of-Vocabulary (OOV) vì luôn có 256 byte cơ sở làm nền tảng.
    """

    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        merges: Optional[List[Tuple[str, str]]] = None,
        special_tokens: Optional[Dict[str, int]] = None
    ):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        self.special_tokens = special_tokens or {
            "<pad>": 0,
            "<unk>": 1,
            "<bos>": 2,
            "<eos>": 3,
        }
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}

        self.vocab: Dict[str, int] = vocab or {}
        self.inverse_vocab: Dict[int, str] = {v: k for k, v in self.vocab.items()}
        self.merges: List[Tuple[str, str]] = merges or []
        self.bpe_ranks: Dict[Tuple[str, str], int] = {pair: i for i, pair in enumerate(self.merges)}

        # Regex phân tách từ: hỗ trợ các ký tự Unicode tiếng Việt có dấu, chữ số và dấu câu
        self.pattern = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\S+|\s+(?!\S)|\s+""",
            re.UNICODE
        )

    def __len__(self) -> int:
        return len(self.vocab)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_token_id(self) -> int:
        return self.special_tokens.get("<pad>", 0)

    @property
    def unk_token_id(self) -> int:
        return self.special_tokens.get("<unk>", 1)

    @property
    def bos_token_id(self) -> int:
        return self.special_tokens.get("<bos>", 2)

    @property
    def eos_token_id(self) -> int:
        return self.special_tokens.get("<eos>", 3)

    def train(self, texts: List[str], target_vocab_size: int = 1000):
        """
        Huấn luyện BPE Tokenizer từ danh sách văn bản:
        1. Khởi tạo vocabulary với Special Tokens và 256 Byte Tokens cơ bản.
        2. Thống kê tần suất và thực hiện merge liên tục cho đến khi đạt target_vocab_size.
        """
        assert target_vocab_size > len(self.special_tokens) + 256, (
            f"target_vocab_size phải lớn hơn {len(self.special_tokens) + 256} để có thể merge thêm token mới"
        )

        # 1. Khởi tạo vocab với Special Tokens
        self.vocab = dict(self.special_tokens)
        current_id = len(self.vocab)

        # 2. Thêm 256 Byte Tokens
        for b_val in range(256):
            char_repr = self.byte_encoder[b_val]
            if char_repr not in self.vocab:
                self.vocab[char_repr] = current_id
                current_id += 1

        # 3. Phân tách corpus và chuyển đổi sang byte-level representation
        word_freqs: Dict[Tuple[str, ...], int] = {}
        for text in texts:
            text = normalize_text(text)
            words = self.pattern.findall(text)
            for w in words:
                # Chuyển chuỗi UTF-8 sang chuỗi ký tự byte-level
                b_chars = tuple(self.byte_encoder[b] for b in w.encode("utf-8"))
                if b_chars:
                    word_freqs[b_chars] = word_freqs.get(b_chars, 0) + 1

        # 4. Vòng lặp Hợp nhất (Merge Loop)
        num_merges = target_vocab_size - len(self.vocab)
        self.merges = []

        for step in range(num_merges):
            stats = get_stats(word_freqs)
            if not stats:
                break

            # Lấy cặp token có tần suất xuất hiện nhiều nhất
            best_pair = max(stats, key=stats.get)
            self.merges.append(best_pair)
            self.bpe_ranks[best_pair] = step

            # Thêm token gộp mới vào vocabulary
            merged_token = best_pair[0] + best_pair[1]
            if merged_token not in self.vocab:
                self.vocab[merged_token] = current_id
                current_id += 1

            # Cập nhật từ điển
            word_freqs = merge_vocab(best_pair, word_freqs)

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def _bpe(self, token_tuple: Tuple[str, ...]) -> List[str]:
        """Áp dụng các quy tắc merge theo thứ tự ưu tiên (ranks) cho 1 từ."""
        word = list(token_tuple)
        if len(word) <= 1:
            return word

        while True:
            # Tìm tất cả các cặp có thể merge trong từ hiện tại
            pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]
            # Tìm cặp có rank thấp nhất (được merge sớm nhất trong training)
            min_rank = float("inf")
            best_pair = None
            for p in pairs:
                rank = self.bpe_ranks.get(p, float("inf"))
                if rank < min_rank:
                    min_rank = rank
                    best_pair = p

            if best_pair is None or min_rank == float("inf"):
                break

            # Thực hiện merge best_pair trong word
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                    new_word.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word

            if len(word) <= 1:
                break

        return word

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Chuyển chuỗi văn bản thành danh sách Token IDs."""
        text = normalize_text(text)
        words = self.pattern.findall(text)

        token_ids: List[int] = []
        if add_bos:
            token_ids.append(self.special_tokens["<bos>"])

        unk_id = self.special_tokens["<unk>"]

        for w in words:
            b_chars = tuple(self.byte_encoder[b] for b in w.encode("utf-8"))
            bpe_tokens = self._bpe(b_chars)
            for tok in bpe_tokens:
                token_ids.append(self.vocab.get(tok, unk_id))

        if add_eos:
            token_ids.append(self.special_tokens["<eos>"])

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        """Chuyển danh sách Token IDs trở lại văn bản gốc chính xác 100%."""
        special_ids = set(self.special_tokens.values())

        tokens = []
        for tid in token_ids:
            if skip_special_tokens and tid in special_ids:
                continue

            if tid in self.inverse_special_tokens:
                tokens.append(self.inverse_special_tokens[tid])
            elif tid in self.inverse_vocab:
                tokens.append(self.inverse_vocab[tid])
            else:
                tokens.append("<unk>")

        # Nối toàn bộ chuỗi ký tự byte-level
        byte_text = "".join(t for t in tokens if t not in self.special_tokens)

        # Giải mã các ký tự byte-level trở lại byte UTF-8 gốc
        byte_array = bytearray([self.byte_decoder[c] for c in byte_text if c in self.byte_decoder])
        return byte_array.decode("utf-8", errors="replace")

    def save(self, file_path: str):
        """Lưu toàn bộ từ điển và merge pairs vào file JSON."""
        data = {
            "vocab": self.vocab,
            "merges": [" ".join(pair) for pair in self.merges],
            "special_tokens": self.special_tokens,
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, file_path: str) -> "BilingualBPETokenizer":
        """Nạp tokenizer từ file JSON đã lưu."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        merges = [tuple(m.split()) for m in data["merges"]]
        return cls(
            vocab=data["vocab"],
            merges=merges,
            special_tokens=data.get("special_tokens")
        )

    @classmethod
    def from_file(cls, file_path: str) -> "BilingualBPETokenizer":
        """Alias cho load(file_path)."""
        return cls.load(file_path)
