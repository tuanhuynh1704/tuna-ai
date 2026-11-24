"""
src/inference/generator.py

Động cơ Sinh Văn bản Đa Chiến lược (Inference Text Generator) cho Tuna-AI:
- Hỗ trợ các chiến lược lấy mẫu tiên tiến:
  1. Greedy Search (argmax)
  2. Temperature Scaling
  3. Top-K Sampling
  4. Top-P (Nucleus Sampling)
  5. Repetition Penalty (chống lặp từ)
- Tích hợp Tăng tốc KV Cache O(N) và chế độ phát trực tuyến Stream Generator.
"""

from typing import Dict, Generator, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F

from src.inference.kv_cache import KVCache
from src.models.language_model import TunaLM
from src.tokenizer.tokenizer import BilingualBPETokenizer


class TextGenerator:
    """Bộ sinh văn bản tự hồi quy chuyên nghiệp cho TunaLM."""

    def __init__(
        self,
        model: TunaLM,
        tokenizer: BilingualBPETokenizer,
        device: str = "cpu"
    ) -> None:
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

    def _apply_sampling_filters(
        self,
        logits: torch.Tensor,
        generated_tokens: List[int],
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """Áp dụng toàn bộ các bộ lọc xác suất lên logits của bước hiện tại."""
        logits = logits.clone()

        # 1. Repetition Penalty
        if repetition_penalty != 1.0 and generated_tokens:
            for token_id in set(generated_tokens):
                if logits[token_id] < 0:
                    logits[token_id] *= repetition_penalty
                else:
                    logits[token_id] /= repetition_penalty

        # 2. Temperature Scaling
        if temperature <= 0.0 or temperature < 1e-4:
            # Greedy: đặt logit lớn nhất thành vô cùng
            top_idx = torch.argmax(logits)
            filtered_logits = torch.full_like(logits, float('-inf'))
            filtered_logits[top_idx] = 0.0
            return filtered_logits

        logits = logits / temperature

        # 3. Top-K Filtering
        if top_k is not None and top_k > 0 and top_k < logits.size(-1):
            top_k_val, _ = torch.topk(logits, k=top_k)
            min_val = top_k_val[-1]
            logits[logits < min_val] = float('-inf')

        # 4. Top-P (Nucleus) Filtering
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Đánh dấu các tokens vượt quá ngưỡng tích lũy top_p
            sorted_indices_to_remove = cumulative_probs > top_p
            # Đảm bảo giữ lại ít nhất 1 token có xác suất cao nhất
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float('-inf')

        return logits

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.1,
        use_cache: bool = True,
        stop_tokens: Optional[List[str]] = None,
    ) -> str:
        """Sinh chuỗi văn bản hoàn chỉnh từ prompt."""
        generated_text = ""
        for token_chunk in self.generate_stream(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            use_cache=use_cache,
            stop_tokens=stop_tokens
        ):
            generated_text += token_chunk
        return generated_text

    @torch.no_grad()
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.1,
        use_cache: bool = True,
        stop_tokens: Optional[List[str]] = None,
    ) -> Generator[str, None, None]:
        """Sinh văn bản dạng luồng (Streaming generator), trả về từng mẩu text ngay khi sinh."""
        self.model.eval()
        prompt_ids = self.tokenizer.encode(prompt)
        if not prompt_ids:
            return

        # Danh sách token dừng
        stop_str_list = stop_tokens or ["<|im_end|>", "<eos>", "<|end|>"]
        stop_id_list = [getattr(self.tokenizer, "eos_token_id", 3)]
        for s in stop_str_list:
            ids = self.tokenizer.encode(s)
            if len(ids) == 1:
                stop_id_list.append(ids[0])

        generated_ids: List[int] = list(prompt_ids)
        new_token_ids: List[int] = []
        prev_decoded_len = 0

        # Khởi tạo KV Cache nếu sử dụng
        kv_cache = None
        if use_cache:
            kv_cache = KVCache(num_layers=self.model.config.n_layers)

        # Prefill prompt: Đảm bảo chừa đủ không gian cho max_new_tokens trong context_length
        max_ctx = self.model.config.context_length
        avail_prompt_len = max(4, max_ctx - max_new_tokens)
        curr_prompt_ids = prompt_ids[-avail_prompt_len:]
        x = torch.tensor([curr_prompt_ids], dtype=torch.long, device=self.device)

        logits, _ = self.model(x, kv_cache=kv_cache)
        next_token_logits = logits[0, -1, :]

        for _ in range(max_new_tokens):
            filtered_logits = self._apply_sampling_filters(
                logits=next_token_logits,
                generated_tokens=generated_ids,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )

            # Lấy mẫu
            probs = F.softmax(filtered_logits, dim=-1)
            next_token_id = int(torch.multinomial(probs, num_samples=1).item())

            # Dừng nếu gặp stop token (sau khi đã sinh ít nhất 2 token nội dung)
            if len(new_token_ids) >= 2 and next_token_id in stop_id_list:
                break

            generated_ids.append(next_token_id)
            new_token_ids.append(next_token_id)

            full_decoded = self.tokenizer.decode(new_token_ids)

            # Kiểm tra thẻ dừng trong chuỗi đã giải mã
            stopped = False
            for st in stop_str_list:
                if st in full_decoded:
                    stopped = True
                    break
            if len(new_token_ids) >= 2 and stopped:
                break

            # Bộ đệm UTF-8 streaming: nếu kết thúc bằng ký tự thay thế \ufffd, chờ token tiếp theo
            if not full_decoded.endswith("\ufffd"):
                new_chunk = full_decoded[prev_decoded_len:]
                if new_chunk:
                    prev_decoded_len = len(full_decoded)
                    yield new_chunk

            # Chuẩn bị cho bước kế tiếp
            if use_cache and kv_cache is not None:
                # Với KV Cache, chỉ cần truyền duy nhất 1 token mới!
                x_next = torch.tensor([[next_token_id]], dtype=torch.long, device=self.device)
                if kv_cache.get_seq_len() >= max_ctx:
                    break
                logits, _ = self.model(x_next, kv_cache=kv_cache)
                next_token_logits = logits[0, -1, :]
            else:
                # Không dùng cache: truyền toàn bộ chuỗi gần nhất
                cond_ids = generated_ids[-max_ctx:]
                x_all = torch.tensor([cond_ids], dtype=torch.long, device=self.device)
                logits, _ = self.model(x_all)
                next_token_logits = logits[0, -1, :]

        # Xả phần đệm ký tự còn lại (nếu có và không chứa thẻ dừng)
        final_decoded = self.tokenizer.decode(new_token_ids)
        if len(final_decoded) > prev_decoded_len:
            remaining = final_decoded[prev_decoded_len:]
            for st in stop_str_list:
                if st in remaining:
                    remaining = remaining.split(st)[0]
            if remaining:
                yield remaining
