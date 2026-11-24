"""
src/tokenizer/normalizer.py

Thực hành Milestone M4:
Bộ Chuẩn hóa Văn bản (Text Normalizer) cho Dữ liệu Song ngữ Việt - Anh:
- Chuẩn hóa Unicode sang dạng NFC (Canonical Decomposition followed by Canonical Composition).
  Đặc biệt quan trọng với Tiếng Việt: giải quyết triệt để sự xung đột giữa gõ Tiếng Việt
  dựng sẵn (NFC) và Tiếng Việt tổ hợp (NFD) từ các bộ gõ Unikey, Telex, VNI.
- Chuẩn hóa khoảng trắng liên tiếp và ký tự điều khiển ẩn.
"""

import unicodedata
import re


def normalize_unicode(text: str, form: str = "NFC") -> str:
    """
    Chuẩn hóa văn bản sang định dạng Unicode thống nhất (mặc định là NFC).
    Đảm bảo ký tự có dấu như 'ệ', 'ắ', 'ở' có biểu diễn mã byte duy nhất.
    """
    return unicodedata.normalize(form, text)


def clean_whitespace(text: str) -> str:
    """Loại bỏ các khoảng trắng thừa liên tiếp nhưng giữ lại cấu trúc xuống dòng."""
    # Thay thế các khoảng trắng ngang liên tiếp (spaces, tabs) thành 1 space
    text = re.sub(r'[^\S\r\n]+', ' ', text)
    # Loại bỏ khoảng trắng ở đầu và cuối mỗi dòng
    lines = [line.strip() for line in text.splitlines()]
    return '\n'.join(lines).strip()


def normalize_text(text: str) -> str:
    """Toàn bộ quy trình chuẩn hóa văn bản trước khi đưa vào Tokenizer."""
    # 1. Unicode NFC
    text = normalize_unicode(text, form="NFC")
    # 2. Loại bỏ ký tự điều khiển ẩn ngoại trừ newline \n
    text = "".join(ch for ch in text if ch == '\n' or not unicodedata.category(ch).startswith('C'))
    return text
