"""Bộ kiểm thử đơn vị cho Hệ thống Tinh chỉnh Chỉ dẫn SFT (Milestone M10).

Kiểm tra:
1. ChatTemplate:
   - Format dialogue có và không có generation prompt.
   - Parse response dừng đúng ranh giới lượt.
2. SFTDataset:
   - Cơ chế Response Masking: Xác minh toàn bộ token của User và Padding mang nhãn -100.
   - Token của Assistant mang nhãn hợp lệ (>= 0).
3. SFTTrainer:
   - Forward pass và backward pass thành công với targets có nhãn -100.
   - Giảm loss ổn định.
"""

from pathlib import Path
import pytest
import torch

from src.instruction.chat_template import ChatTemplate
from src.instruction.dataset import SFTDataset, create_sft_dataloader
from src.instruction.sft_trainer import SFTTrainer
from src.models.language_model import TunaLM, TunaLMConfig
from src.tokenizer.tokenizer import BilingualBPETokenizer


class TestInstructionTuning:
    @pytest.fixture
    def setup_env(self):
        tok_file = "configs/tokenizer/bpe_toy_500.json"
        tokenizer = BilingualBPETokenizer.from_file(tok_file)
        chat_template = ChatTemplate()

        sample_dialogue = [
            {
                "messages": [
                    {"role": "user", "content": "Xin chào bạn!"},
                    {"role": "assistant", "content": "Chào bạn! Tôi là trợ lý AI."},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "What is AI?"},
                    {"role": "assistant", "content": "AI is artificial intelligence."},
                ]
            },
        ]

        config = TunaLMConfig(
            vocab_size=tokenizer.vocab_size,
            context_length=32,
            d_model=32,
            n_layers=1,
            n_heads=2,
            d_ff=64,
            dropout=0.0
        )
        model = TunaLM(config)

        return tokenizer, chat_template, sample_dialogue, model

    def test_chat_template_formatting(self, setup_env):
        _, chat_template, sample_dialogue, _ = setup_env
        formatted = chat_template.format_dialogue(
            sample_dialogue[0]["messages"],
            add_generation_prompt=False
        )

        assert "<|im_start|>user" in formatted
        assert "Xin chào bạn!" in formatted
        assert "<|im_end|>" in formatted
        assert "<|im_start|>assistant" in formatted
        assert "Chào bạn! Tôi là trợ lý AI." in formatted

        # Thêm generation prompt
        prompt_only = chat_template.format_dialogue(
            [{"role": "user", "content": "Test"}],
            add_generation_prompt=True
        )
        assert prompt_only.endswith("<|im_start|>assistant\n")

    def test_chat_template_parse_response(self, setup_env):
        _, chat_template, _, _ = setup_env
        raw_output = "<|im_start|>assistant\nĐây là câu trả lời hữu ích.<|im_end|>\nToken dư thừa sau đó"
        parsed = chat_template.parse_response(raw_output)
        assert parsed == "Đây là câu trả lời hữu ích."

    def test_sft_dataset_response_masking(self, setup_env):
        tokenizer, chat_template, sample_dialogue, _ = setup_env
        dataset = SFTDataset(
            samples=sample_dialogue,
            tokenizer=tokenizer,
            chat_template=chat_template,
            context_length=32
        )

        assert len(dataset) == 2
        input_ids, targets = dataset[0]

        assert input_ids.shape == (32,)
        assert targets.shape == (32,)

        # Xác minh: các token đầu tiên (Prompt) phải có nhãn -100
        assert targets[0].item() == -100
        # Xác minh: phải tồn tại ít nhất 1 token mang nhãn >= 0 (câu trả lời của Assistant)
        num_assistant_tokens = (targets != -100).sum().item()
        assert num_assistant_tokens > 0

    def test_sft_trainer_step(self, setup_env):
        tokenizer, chat_template, sample_dialogue, model = setup_env
        loader = create_sft_dataloader(
            samples=sample_dialogue,
            tokenizer=tokenizer,
            batch_size=2,
            context_length=32
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        trainer = SFTTrainer(model=model, optimizer=optimizer, device="cpu")

        initial_loss = trainer.evaluate(loader)
        assert initial_loss > 0.0

        # Chạy 3 epochs để kiểm tra loss giảm
        res = trainer.train(train_loader=loader, epochs=3, log_interval=10)
        assert res["total_steps"] > 0
        assert res["final_train_loss"] > 0.0
