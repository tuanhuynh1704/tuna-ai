"""Hệ thống khử trùng lặp dữ liệu chính xác (Exact) và mờ (MinHash LSH).

Đáp ứng bước 5 trong quy trình 10 bước tiền xử lý của Tuna-AI:
- Exact Deduplication: Băm SHA-256 loại bỏ 100% các bản sao nguyên văn.
- Fuzzy Deduplication: MinHash n-gram ước lượng hệ số Jaccard Similarity, loại bỏ bản sao gần giống.
"""

import hashlib
import random
from typing import List, Set, Tuple


class ExactDeduplicator:
    """Khử trùng lặp chính xác bằng hàm băm mật mã học SHA-256."""

    def __init__(self) -> None:
        self.seen_hashes: Set[str] = set()

    def is_duplicate(self, text: str) -> bool:
        """Kiểm tra văn bản đã xuất hiện hay chưa dựa trên SHA-256 hash."""
        text_normalized = " ".join(text.strip().split())
        digest = hashlib.sha256(text_normalized.encode("utf-8")).hexdigest()
        if digest in self.seen_hashes:
            return True
        self.seen_hashes.add(digest)
        return False

    def reset(self) -> None:
        """Xóa trắng bộ nhớ hash."""
        self.seen_hashes.clear()


class MinHashDeduplicator:
    """Khử trùng lặp mờ (Fuzzy Deduplication) sử dụng kỹ thuật MinHash n-grams.
    
    Thuật toán:
    1. Chuyển văn bản thành tập hợp các n-grams (shingles).
    2. Chiếu shingles qua K hàm băm ngẫu nhiên để sinh MinHash signature có độ dài K.
    3. Ước lượng độ tương đồng Jaccard giữa văn bản mới và các văn bản đã lưu:
       J(A, B) ≈ sum(sigA[i] == sigB[i]) / K.
    4. Nếu J(A, B) >= similarity_threshold, văn bản bị loại bỏ do trùng lặp mờ.
    """

    def __init__(
        self,
        num_perm: int = 64,
        ngram_size: int = 3,
        similarity_threshold: float = 0.80,
        seed: int = 42,
    ) -> None:
        self.num_perm = num_perm
        self.ngram_size = ngram_size
        self.similarity_threshold = similarity_threshold
        self.signatures: List[List[int]] = []

        # Khởi tạo K hàm băm tuyến tính: h(x) = (a * x + b) % prime
        # Sử dụng số nguyên tố Mersenne lớn 2^61 - 1
        self.prime = (1 << 61) - 1
        rng = random.Random(seed)
        self.hash_coeffs = [
            (rng.randint(1, self.prime - 1), rng.randint(0, self.prime - 1))
            for _ in range(num_perm)
        ]

    def _get_shingles(self, text: str) -> Set[int]:
        """Tạo tập hợp các n-gram băm từ danh sách từ và ký tự trong văn bản."""
        words = text.lower().split()
        shingles: Set[int] = set()

        # Word n-grams từ 1 đến ngram_size
        for n in range(1, self.ngram_size + 1):
            if len(words) >= n:
                for i in range(len(words) - n + 1):
                    ngram = " ".join(words[i:i + n])
                    h = int.from_bytes(hashlib.md5(ngram.encode("utf-8")).digest()[:8], "little")
                    shingles.add(h)

        # Bổ sung char 4-grams để bắt các biến thể thay thế từ hoặc chính tả
        clean_chars = "".join(words)
        char_n = 4
        if len(clean_chars) >= char_n:
            for i in range(len(clean_chars) - char_n + 1):
                cgram = clean_chars[i:i + char_n]
                h = int.from_bytes(hashlib.md5(cgram.encode("utf-8")).digest()[:8], "little")
                shingles.add(h)

        return shingles

    def compute_signature(self, text: str) -> List[int]:
        """Tính toán MinHash signature gồm num_perm giá trị băm nhỏ nhất."""
        shingles = self._get_shingles(text)
        if not shingles:
            return [0] * self.num_perm

        signature = [self.prime] * self.num_perm
        for s in shingles:
            for i, (a, b) in enumerate(self.hash_coeffs):
                h = (a * s + b) % self.prime
                if h < signature[i]:
                    signature[i] = h
        return signature

    def is_duplicate(self, text: str) -> Tuple[bool, float]:
        """Kiểm tra văn bản có trùng lặp mờ với bất kỳ văn bản nào trước đó không.
        
        Returns:
            (is_dup, max_sim): True nếu vượt ngưỡng tương đồng, kèm độ tương đồng lớn nhất.
        """
        sig = self.compute_signature(text)
        if not self.signatures:
            self.signatures.append(sig)
            return False, 0.0

        max_sim = 0.0
        for stored_sig in self.signatures:
            matches = sum(1 for a, b in zip(sig, stored_sig) if a == b)
            sim = matches / self.num_perm
            if sim > max_sim:
                max_sim = sim
            if max_sim >= self.similarity_threshold:
                return True, max_sim

        # Nếu không trùng lặp, lưu trữ signature văn bản này
        self.signatures.append(sig)
        return False, max_sim

    def reset(self) -> None:
        """Xóa danh sách signatures đã lưu."""
        self.signatures.clear()
