"""
src/evaluation/evaluator.py

Hệ thống Đánh giá Hiệu năng và Năng lực Toàn diện cho Mô hình TunaLM:
1. Perplexity & Loss trên Hold-out validation/test set.
2. Next-Token Accuracy (Top-1 và Top-5 Accuracy).
3. Đánh giá Cloze Test song ngữ.
4. Đo lường Độ trễ Suy luận (Inference Latency, TTFT, ITL) và Mức chiếm dụng RAM CPU.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import psutil
import torch
import torch.nn.functional as F

from src.models.language_model import TunaLM
from src.tokenizer.tokenizer import BilingualBPETokenizer
from src.data.loader import ShardDataLoader
from src.evaluation.benchmarks import ClozeBenchmark


@dataclass
class EvaluationReport:
    """Báo cáo tổng hợp số liệu đánh giá mô hình."""
    model_name: str
    timestamp: str
    hold_out_loss: float
    perplexity: float
    top1_accuracy_percent: float
    top5_accuracy_percent: float
    cloze_accuracy_percent: float
    cloze_category_stats: Dict[str, float]
    time_to_first_token_ms: float
    inter_token_latency_ms: float
    generation_throughput_tok_per_sec: float
    ram_usage_mb: float
    parameters_count: int


class ModelEvaluator:
    """Động cơ Đánh giá Toàn diện cho Tuna-AI."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.cloze_suite = ClozeBenchmark()

    @torch.no_grad()
    def evaluate_loss_and_perplexity(
        self,
        model: TunaLM,
        data_loader: ShardDataLoader,
        eval_iters: int = 20
    ) -> Tuple[float, float]:
        """
        Tính toán chính xác Cross-Entropy Loss và Perplexity (PPL) trên hold-out test set.
        
        PPL = exp(mean(loss))
        """
        model.eval()
        total_loss = 0.0

        for _ in range(eval_iters):
            x, y = data_loader.get_batch(split="val")
            _, loss = model(x.to(self.device), targets=y.to(self.device))
            total_loss += loss.item()

        avg_loss = total_loss / eval_iters
        ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
        return round(avg_loss, 4), round(ppl, 2)

    @torch.no_grad()
    def evaluate_next_token_accuracy(
        self,
        model: TunaLM,
        data_loader: ShardDataLoader,
        eval_iters: int = 20,
        top_k: Tuple[int, int] = (1, 5)
    ) -> Tuple[float, float]:
        """
        Đo lường độ chính xác dự đoán token tiếp theo: Top-1 và Top-5 Accuracy.
        
        Top-k Accuracy = tỷ lệ các vị trí mà token thực tế y nằm trong k token có xác suất cao nhất.
        """
        model.eval()
        total_tokens = 0
        correct_top1 = 0
        correct_top5 = 0

        for _ in range(eval_iters):
            x, y = data_loader.get_batch(split="val")
            x, y = x.to(self.device), y.to(self.device)

            logits, _ = model(x)  # shape: (B, T, V)
            B, T, V = logits.shape
            num_tokens_in_batch = B * T
            total_tokens += num_tokens_in_batch

            # Top-1 accuracy
            top1_preds = torch.argmax(logits, dim=-1)  # shape: (B, T)
            correct_top1 += (top1_preds == y).sum().item()

            # Top-5 accuracy
            _, top5_indices = torch.topk(logits, k=min(top_k[1], V), dim=-1)  # shape: (B, T, 5)
            # Mở rộng y thành (B, T, 1) để so sánh broadcasting
            y_expanded = y.unsqueeze(-1)
            correct_top5 += (top5_indices == y_expanded).any(dim=-1).sum().item()

        top1_acc = (correct_top1 / total_tokens) * 100.0 if total_tokens > 0 else 0.0
        top5_acc = (correct_top5 / total_tokens) * 100.0 if total_tokens > 0 else 0.0

        return round(top1_acc, 2), round(top5_acc, 2)

    def benchmark_inference_latency(
        self,
        model: TunaLM,
        tokenizer: BilingualBPETokenizer,
        prompt: str = "Trí tuệ nhân tạo và học sâu",
        max_new_tokens: int = 25,
        warmup_runs: int = 2,
        test_runs: int = 5
    ) -> Dict[str, float]:
        """
        Đo lường độ trễ suy luận chi tiết trên CPU:
        - TTFT (Time To First Token): Thời gian xử lý prompt ban đầu và sinh token đầu tiên.
        - ITL (Inter-Token Latency): Thời gian trung bình để sinh mỗi token tiếp theo.
        - Throughput: Số token sinh được trên mỗi giây (tokens/s).
        - RAM Footprint: Bộ nhớ RSS của tiến trình trong quá trình chạy inference.
        """
        model.eval()
        prompt_ids = tokenizer.encode(prompt)
        x = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        # Warmup
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = model.generate(x, max_new_tokens=5, temperature=1.0)

        ttft_list = []
        itl_list = []
        total_time_list = []

        process = psutil.Process(os.getpid())
        ram_before = process.memory_info().rss / (1024 * 1024)

        for _ in range(test_runs):
            curr_x = x.clone()

            # 1. Đo TTFT (sinh token đầu tiên)
            t0 = time.perf_counter()
            with torch.no_grad():
                logits, _ = model(curr_x)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                curr_x = torch.cat([curr_x, next_token], dim=1)
            t_ttft = (time.perf_counter() - t0) * 1000.0  # ms
            ttft_list.append(t_ttft)

            # 2. Đo các tokens còn lại để tính ITL
            t_start_rest = time.perf_counter()
            with torch.no_grad():
                for _ in range(max_new_tokens - 1):
                    # Giới hạn context length nếu vượt quá
                    cond_x = curr_x if curr_x.size(1) <= model.config.context_length else curr_x[:, -model.config.context_length:]
                    logits, _ = model(cond_x)
                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    curr_x = torch.cat([curr_x, next_token], dim=1)
            t_rest = time.perf_counter() - t_start_rest

            avg_itl = (t_rest / max(1, max_new_tokens - 1)) * 1000.0  # ms
            itl_list.append(avg_itl)
            total_time_list.append(t_ttft / 1000.0 + t_rest)

        ram_after = process.memory_info().rss / (1024 * 1024)
        avg_ttft = float(np.mean(ttft_list))
        avg_itl = float(np.mean(itl_list))
        avg_throughput = max_new_tokens / float(np.mean(total_time_list))

        return {
            "time_to_first_token_ms": round(avg_ttft, 2),
            "inter_token_latency_ms": round(avg_itl, 2),
            "generation_throughput_tok_per_sec": round(avg_throughput, 2),
            "ram_usage_mb": round(ram_after, 2),
        }

    def run_full_evaluation(
        self,
        model: TunaLM,
        tokenizer: BilingualBPETokenizer,
        data_loader: ShardDataLoader,
        model_name: str = "TunaLM-Mini-Bilingual",
        eval_iters: int = 15
    ) -> EvaluationReport:
        """Thực thi đầy đủ bộ kiểm định toàn diện và xuất báo cáo EvaluationReport."""
        # 1. Perplexity & Loss
        loss, ppl = self.evaluate_loss_and_perplexity(model, data_loader, eval_iters=eval_iters)

        # 2. Next-Token Accuracy
        top1, top5 = self.evaluate_next_token_accuracy(model, data_loader, eval_iters=eval_iters)

        # 3. Cloze Test Benchmark
        cloze_results = self.cloze_suite.evaluate(model, tokenizer, device=self.device)

        # 4. Latency & Hardware Metrics
        perf = self.benchmark_inference_latency(model, tokenizer)

        param_count = sum(p.numel() for p in model.parameters())

        report = EvaluationReport(
            model_name=model_name,
            timestamp=datetime.now().isoformat(),
            hold_out_loss=loss,
            perplexity=ppl,
            top1_accuracy_percent=top1,
            top5_accuracy_percent=top5,
            cloze_accuracy_percent=cloze_results["accuracy_percent"],
            cloze_category_stats=cloze_results["category_accuracy_percent"],
            time_to_first_token_ms=perf["time_to_first_token_ms"],
            inter_token_latency_ms=perf["inter_token_latency_ms"],
            generation_throughput_tok_per_sec=perf["generation_throughput_tok_per_sec"],
            ram_usage_mb=perf["ram_usage_mb"],
            parameters_count=param_count
        )
        return report
