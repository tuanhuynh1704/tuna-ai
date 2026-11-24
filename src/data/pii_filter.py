"""Bộ lọc và che giấu thông tin định danh cá nhân (PII Filter).

Đáp ứng bước 6 trong quy trình tiền xử lý 10 bước của Tuna-AI:
- Loại bỏ/Che giấu Email
- Loại bỏ/Che giấu Số điện thoại (Việt Nam & Quốc tế)
- Loại bỏ/Che giấu Địa chỉ IPv4
- Loại bỏ/Che giấu Số thẻ thanh toán (Credit Card)
"""

import re
from typing import Dict, Tuple


class PIIFilter:
    """Bộ lọc thông tin cá nhân và dữ liệu nhạy cảm."""

    def __init__(
        self,
        redact_email: bool = True,
        redact_phone: bool = True,
        redact_ip: bool = True,
        redact_credit_card: bool = True,
        email_placeholder: str = "[EMAIL]",
        phone_placeholder: str = "[PHONE]",
        ip_placeholder: str = "[IP_ADDRESS]",
        credit_card_placeholder: str = "[CREDIT_CARD]",
    ) -> None:
        self.redact_email = redact_email
        self.redact_phone = redact_phone
        self.redact_ip = redact_ip
        self.redact_credit_card = redact_credit_card

        self.email_placeholder = email_placeholder
        self.phone_placeholder = phone_placeholder
        self.ip_placeholder = ip_placeholder
        self.credit_card_placeholder = credit_card_placeholder

        # Regex phát hiện email
        self.email_pattern = re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
        )

        # Regex phát hiện số điện thoại (+84, 09xx, 08xx, 03xx, 05xx, 07xx, 02xx...)
        self.phone_pattern = re.compile(
            r"(?:\+84|84|0)(?:3|5|7|8|9|2\d)\d{8}\b|"
            r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
        )

        # Regex phát hiện IPv4
        self.ip_pattern = re.compile(
            r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
            r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
        )

        # Regex phát hiện Credit Card (13-16 số phân tách bởi khoảng trắng hoặc dấu gạch ngang)
        self.credit_card_pattern = re.compile(
            r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
        )

    def redact(self, text: str) -> str:
        """Thay thế các thực thể PII bằng placeholder tương ứng.
        
        Args:
            text: Văn bản thô đầu vào.
            
        Returns:
            Văn bản đã được che giấu PII.
        """
        if not text:
            return ""

        # 1. Che giấu Credit Card trước để tránh nhầm với dãy số điện thoại
        if self.redact_credit_card:
            text = self.credit_card_pattern.sub(self.credit_card_placeholder, text)

        # 2. Che giấu Email
        if self.redact_email:
            text = self.email_pattern.sub(self.email_placeholder, text)

        # 3. Che giấu IP Address
        if self.redact_ip:
            text = self.ip_pattern.sub(self.ip_placeholder, text)

        # 4. Che giấu Số điện thoại
        if self.redact_phone:
            text = self.phone_pattern.sub(self.phone_placeholder, text)

        return text

    def scan(self, text: str) -> Dict[str, int]:
        """Thống kê số lượng các loại PII có trong văn bản mà không thay đổi văn bản."""
        if not text:
            return {"email": 0, "phone": 0, "ip": 0, "credit_card": 0}

        return {
            "credit_card": len(self.credit_card_pattern.findall(text)),
            "email": len(self.email_pattern.findall(text)),
            "ip": len(self.ip_pattern.findall(text)),
            "phone": len(self.phone_pattern.findall(text)),
        }
