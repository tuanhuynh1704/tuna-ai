"""Bộ tiền xử lý văn bản và kiểm tra chất lượng dữ liệu (Cleaner & Quality Filter).

Đáp ứng các bước 2, 3, 4, 7 trong quy trình 10 bước tiền xử lý của Tuna-AI:
- Bước 2: Format Filtering (loại bỏ HTML, định dạng rác)
- Bước 3: Language Identification (phân loại tiếng Việt, tiếng Anh)
- Bước 4: Rule-based Quality Filtering (lọc độ dài, tỷ lệ ký tự rác, ký tự lặp)
- Bước 7: Text Normalization (Unicode NFC, chuẩn hóa khoảng trắng)
"""

import re
import unicodedata
from typing import Dict, Optional, Tuple


class TextCleaner:
    """Bộ làm sạch và chuẩn hóa văn bản cơ bản."""

    def __init__(self) -> None:
        # Regex loại bỏ thẻ HTML / XML
        self.html_tag_pattern = re.compile(r"<[^>]+>")
        # Regex loại bỏ ký tự điều khiển ngoại trừ \n, \t, \r
        self.control_char_pattern = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
        # Regex chuẩn hóa khoảng trắng thừa
        self.multi_space_pattern = re.compile(r"[^\S\r\n]+")
        # Regex chuẩn hóa nhiều dòng trống liên tiếp (tối đa 2 dòng trống)
        self.multi_newline_pattern = re.compile(r"\n{3,}")

    def clean(self, text: str) -> str:
        """Làm sạch và chuẩn hóa toàn diện văn bản.
        
        Args:
            text: Chuỗi văn bản thô đầu vào.
            
        Returns:
            Văn bản đã chuẩn hóa NFC, sạch thẻ HTML và ký tự rác.
        """
        if not text:
            return ""

        # 1. Chuẩn hóa Unicode NFC (tối quan trọng cho tiếng Việt có dấu)
        text = unicodedata.normalize("NFC", text)

        # 2. Loại bỏ các thẻ HTML / XML
        text = self.html_tag_pattern.sub(" ", text)

        # 3. Loại bỏ ký tự điều khiển ẩn
        text = self.control_char_pattern.sub("", text)

        # 4. Chuẩn hóa khoảng trắng ngang (spaces, tabs)
        lines = text.splitlines()
        cleaned_lines = []
        for line in lines:
            line = self.multi_space_pattern.sub(" ", line).strip()
            cleaned_lines.append(line)
        text = "\n".join(cleaned_lines)

        # 5. Rút gọn nhiều dòng trống liên tiếp
        text = self.multi_newline_pattern.sub("\n\n", text)

        return text.strip()


class RuleQualityFilter:
    """Bộ lọc chất lượng văn bản dựa trên quy tắc số liệu thống kê (Heuristics)."""

    def __init__(
        self,
        min_words: int = 5,
        max_words: int = 10000,
        max_special_char_ratio: float = 0.35,
        max_repetition_ratio: float = 0.30,
    ) -> None:
        self.min_words = min_words
        self.max_words = max_words
        self.max_special_char_ratio = max_special_char_ratio
        self.max_repetition_ratio = max_repetition_ratio

    def evaluate(self, text: str) -> Tuple[bool, str]:
        """Đánh giá văn bản xem có đạt tiêu chuẩn chất lượng không.
        
        Returns:
            (is_valid, reason): True nếu đạt, False kèm lý do nếu bị loại.
        """
        words = text.split()
        num_words = len(words)

        if num_words < self.min_words:
            return False, f"Văn bản quá ngắn ({num_words} từ < {self.min_words} từ tối thiểu)"
        if num_words > self.max_words:
            return False, f"Văn bản quá dài ({num_words} từ > {self.max_words} từ tối đa)"

        # Tỷ lệ ký tự đặc biệt / biểu tượng / dấu câu
        total_chars = len(text)
        if total_chars == 0:
            return False, "Văn bản rỗng"

        alphanumeric_count = sum(1 for c in text if c.isalnum() or c.isspace())
        special_count = total_chars - alphanumeric_count
        special_ratio = special_count / total_chars

        if special_ratio > self.max_special_char_ratio:
            return False, f"Tỷ lệ ký tự đặc biệt quá cao ({special_ratio:.2%} > {self.max_special_char_ratio:.2%})"

        # Kiểm tra lặp từ bất thường (Spam repetition)
        if num_words >= 10:
            word_counts: Dict[str, int] = {}
            for w in words:
                w_lower = w.lower()
                word_counts[w_lower] = word_counts.get(w_lower, 0) + 1
            max_word_freq = max(word_counts.values())
            repetition_ratio = max_word_freq / num_words
            if repetition_ratio > self.max_repetition_ratio:
                return False, f"Tỷ lệ lặp 1 từ quá cao ({repetition_ratio:.2%} > {self.max_repetition_ratio:.2%})"

        return True, "Hợp lệ"


class LanguageIdentifier:
    """Nhận diện ngôn ngữ đơn giản và phân loại văn bản Song ngữ (Việt - Anh)."""

    def __init__(self) -> None:
        # Tập các nguyên âm đặc trưng có dấu của tiếng Việt
        self.vietnamese_chars = set(
            "àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"
            "ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ"
        )
        # Một số stop words tiếng Việt phổ biến
        self.vietnamese_stopwords = {
            "và", "là", "của", "cho", "trong", "có", "được", "với", "những", "các",
            "khi", "đã", "một", "này", "về", "người", "không", "thì", "đến", "từ"
        }
        # Một số stop words tiếng Anh phổ biến
        self.english_stopwords = {
            "the", "and", "is", "in", "to", "of", "a", "with", "that", "for",
            "as", "on", "are", "it", "this", "by", "from", "at", "be", "an"
        }

    def detect(self, text: str) -> str:
        """Nhận diện văn bản là 'vi' (tiếng Việt), 'en' (tiếng Anh) hoặc 'other'.
        
        Returns:
            Mã ngôn ngữ: 'vi', 'en', hoặc 'other'.
        """
        if not text:
            return "other"

        # Đếm ký tự dấu tiếng Việt
        vn_char_count = sum(1 for c in text if c in self.vietnamese_chars)
        if vn_char_count >= 2:
            return "vi"

        # Phân tích theo từ vựng stopwords
        words = [w.lower() for w in re.findall(r"\b\w+\b", text)]
        if not words:
            return "other"

        vn_stop_count = sum(1 for w in words if w in self.vietnamese_stopwords)
        en_stop_count = sum(1 for w in words if w in self.english_stopwords)

        if vn_stop_count > en_stop_count:
            return "vi"
        elif en_stop_count > 0:
            return "en"

        # Nếu có dấu tiếng Việt dù chỉ 1 ký tự
        if vn_char_count >= 1:
            return "vi"

        # Mặc định văn bản chữ la-tinh không dấu
        return "en"
