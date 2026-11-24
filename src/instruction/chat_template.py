"""
src/instruction/chat_template.py

Định dạng Hội thoại (Chat Template) và Quản lý Lượt Trò chuyện cho Tuna-AI:
- Hỗ trợ định dạng ChatML tiêu chuẩn với các thẻ phân cách lượt rõ ràng:
    <|im_start|>system\n{content}<|im_end|>\n
    <|im_start|>user\n{content}<|im_end|>\n
    <|im_start|>assistant\n{content}<|im_end|>\n
- Cho phép chèn `add_generation_prompt=True` khi thực hiện suy luận (Inference).
- Trích xuất phản hồi sạch từ chuỗi sinh của mô hình (parse response).
"""

from typing import Dict, List, Optional


class ChatTemplate:
    """Quản lý cấu trúc và định dạng hội thoại nhiều lượt (Multi-turn Chat Template)."""

    START_TAG = "<|im_start|>"
    END_TAG = "<|im_end|>"

    def __init__(
        self,
        default_system_prompt: str = "Bạn là Tuna-AI, một trợ lý trí tuệ nhân tạo song ngữ (Việt - Anh) thông minh, hữu ích và chính xác."
    ) -> None:
        self.default_system_prompt = default_system_prompt

    def format_dialogue(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = False
    ) -> str:
        """
        Chuyển danh sách thông điệp hội thoại thành chuỗi văn bản hoàn chỉnh.
        
        Args:
            messages: Danh sách các dict với định dạng: [{"role": "user"|"assistant"|"system", "content": str}]
            add_generation_prompt: Nếu True, thêm "<|im_start|>assistant\\n" ở cuối để mồi mô hình trả lời.
            
        Returns:
            Chuỗi văn bản đã đóng khung các lượt hội thoại.
        """
        formatted_parts: List[str] = []

        # Kiểm tra xem có system prompt không; nếu chưa có thì thêm mặc định
        has_system = any(m.get("role") == "system" for m in messages)
        if not has_system and self.default_system_prompt:
            formatted_parts.append(
                f"{self.START_TAG}system\n{self.default_system_prompt}{self.END_TAG}\n"
            )

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "").strip()
            formatted_parts.append(f"{self.START_TAG}{role}\n{content}{self.END_TAG}\n")

        formatted_text = "".join(formatted_parts)

        if add_generation_prompt:
            formatted_text += f"{self.START_TAG}assistant\n"

        return formatted_text

    def parse_response(self, generated_text: str) -> str:
        """
        Trích xuất câu trả lời của trợ lý từ chuỗi sinh tự hồi quy.
        
        Tự động loại bỏ prompt trước đó và dừng tại thẻ <|im_end|> hoặc <eos>.
        """
        # Nếu chuỗi có chứa thẻ mở assistant
        if f"{self.START_TAG}assistant\n" in generated_text:
            generated_text = generated_text.split(f"{self.START_TAG}assistant\n")[-1]

        # Cắt bỏ phần sau thẻ kết thúc lượt
        for stop_tag in [self.END_TAG, "<eos>", "<|end|>"]:
            if stop_tag in generated_text:
                generated_text = generated_text.split(stop_tag)[0]

        return generated_text.strip()
