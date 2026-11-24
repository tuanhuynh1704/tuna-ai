"""
src/instruction/dataset.py

Bộ nạp Dữ liệu Tinh chỉnh Chỉ dẫn (Supervised Fine-Tuning - SFT) với Cơ chế Response Masking:
- Đảm bảo mô hình chỉ học dự đoán câu trả lời của Trợ lý (Assistant).
- Tự động gán nhãn `label = -100` (PyTorch ignore_index) cho toàn bộ các token thuộc System Prompt, User Prompt và Padding.
- Ngăn ngừa tình trạng mô hình bị phạt hoặc ghi nhớ câu hỏi của người dùng thay vì học cách phản hồi.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.instruction.chat_template import ChatTemplate
from src.tokenizer.tokenizer import BilingualBPETokenizer


class SFTDataset(Dataset):
    """Dataset chuyên dụng cho Fine-Tuning theo định dạng Hội thoại."""

    def __init__(
        self,
        samples: List[Dict[str, List[Dict[str, str]]]],
        tokenizer: BilingualBPETokenizer,
        chat_template: Optional[ChatTemplate] = None,
        context_length: int = 64,
        pad_token_id: int = 0,
        ignore_index: int = -100,
    ) -> None:
        """
        Args:
            samples: Danh sách các mẫu dữ liệu [{"messages": [{"role": "...", "content": "..."}]}]
            tokenizer: BilingualBPETokenizer đã huấn luyện
            chat_template: ChatTemplate định dạng hội thoại
            context_length: Độ dài ngữ cảnh tối đa
            pad_token_id: Token ID dùng cho padding
            ignore_index: Giá trị nhãn bỏ qua khi tính loss (-100)
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.chat_template = chat_template or ChatTemplate()
        self.context_length = context_length
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

        self.encoded_samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._prepare_data()

    def _prepare_data(self) -> None:
        """Mã hóa toàn bộ mẫu dữ liệu và xây dựng ma trận nhãn Response Masking."""
        for sample in self.samples:
            messages = sample.get("messages", [])
            if not messages:
                continue

            # Tách riêng phần prompt (system + user) và phần response (assistant cuối cùng)
            prompt_msgs = [m for m in messages if m.get("role") != "assistant"]
            assistant_msgs = [m for m in messages if m.get("role") == "assistant"]

            if not assistant_msgs:
                continue

            # 1. Format prompt có generation prompt ở cuối
            prompt_str = self.chat_template.format_dialogue(prompt_msgs, add_generation_prompt=True)
            # 2. Format response kèm kết thúc lượt
            assistant_content = assistant_msgs[-1].get("content", "").strip()
            response_str = f"{assistant_content}{self.chat_template.END_TAG}\n"

            prompt_ids = self.tokenizer.encode(prompt_str)
            response_ids = self.tokenizer.encode(response_str)

            # Đảm bảo response luôn có ít nhất một phần trong context_length
            if len(response_ids) >= self.context_length:
                # Nếu response quá dài, cắt bớt response để chừa ít nhất 4 tokens cho prompt
                prompt_ids = prompt_ids[-4:] if len(prompt_ids) >= 4 else prompt_ids
                avail_resp_len = self.context_length - len(prompt_ids)
                response_ids = response_ids[:avail_resp_len]
            else:
                # Nếu tổng chiều dài vượt quá context_length, cắt bớt phần đầu của prompt (Left Truncation)
                max_prompt_len = self.context_length - len(response_ids)
                if len(prompt_ids) > max_prompt_len:
                    prompt_ids = prompt_ids[-max_prompt_len:]

            # Ghép chuỗi đầy đủ sau khi đã căn chỉnh an toàn
            full_input_ids = prompt_ids + response_ids
            # Gán -100 cho toàn bộ prompt_ids để PyTorch bỏ qua, chỉ tính loss trên response_ids
            full_targets = [self.ignore_index] * len(prompt_ids) + list(response_ids)

            # Đệm padding nếu còn dư chỗ
            pad_len = self.context_length - len(full_input_ids)
            if pad_len > 0:
                full_input_ids = full_input_ids + [self.pad_token_id] * pad_len
                full_targets = full_targets + [self.ignore_index] * pad_len

            input_tensor = torch.tensor(full_input_ids, dtype=torch.long)
            target_tensor = torch.tensor(full_targets, dtype=torch.long)

            self.encoded_samples.append((input_tensor, target_tensor))

    def __len__(self) -> int:
        return len(self.encoded_samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoded_samples[idx]


def create_sft_dataloader(
    samples: List[Dict[str, List[Dict[str, str]]]],
    tokenizer: BilingualBPETokenizer,
    batch_size: int = 4,
    context_length: int = 64,
    shuffle: bool = True,
) -> DataLoader:
    """Tạo DataLoader cho quá trình Supervised Fine-Tuning."""
    dataset = SFTDataset(
        samples=samples,
        tokenizer=tokenizer,
        context_length=context_length,
        pad_token_id=getattr(tokenizer, "pad_token_id", 0)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
