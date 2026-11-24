"""
src/tokenizer/bpe.py

Thuật toán Byte-level BPE (Byte Pair Encoding):
- Xây dựng bảng ánh xạ 256 giá trị Byte sang các ký tự Unicode hiển thị được
  (Kỹ thuật tương tự GPT-2 / Hugging Face tokenizers) giúp tránh hoàn toàn lỗi UnicodeDecodeError.
- Thống kê tần suất các cặp token (Get Stats).
- Hợp nhất các cặp token có tần suất cao nhất (Merge Pairs).
"""

from typing import Dict, Tuple, List, Set


def bytes_to_unicode() -> Dict[int, str]:
    """
    Tạo bảng ánh xạ 256 giá trị byte (0 đến 255) sang các ký tự Unicode có thể in được.
    Thuật toán chuẩn của GPT-2:
    - Các byte tương ứng ký tự in được trong ASCII/Latin-1 giữ nguyên.
    - Các byte điều khiển hoặc khoảng trắng (space, newline) được ánh xạ sang dải ký tự Unicode khác.
    Đảm bảo 100% mọi chuỗi byte UTF-8 bất kỳ đều biểu diễn được dưới dạng chuỗi ký tự hợp lệ!
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_stats(vocab: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, str], int]:
    """Đếm tần suất xuất hiện của tất cả các cặp token liền kề trong từ điển thống kê."""
    pairs: Dict[Tuple[str, str], int] = {}
    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pairs[pair] = pairs.get(pair, 0) + freq
    return pairs


def merge_vocab(pair: Tuple[str, str], vocab: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, ...], int]:
    """Thay thế cặp pair=(first, second) bằng token gộp 'first+second' trong toàn bộ từ điển."""
    v_out: Dict[Tuple[str, ...], int] = {}
    first, second = pair
    merged_token = first + second

    for word, freq in vocab.items():
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                new_word.append(merged_token)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        v_out[tuple(new_word)] = freq
    return v_out
