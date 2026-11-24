"""
src/evaluation/benchmarks.py

Bộ Benchmark Đánh giá Năng lực Ngôn ngữ Song ngữ (Việt - Anh):
1. Cloze Test (Điền từ vào chỗ trống):
   - Đánh giá khả năng hiểu ngữ cảnh và lựa chọn từ đúng (Multiple-Choice Cloze Task).
   - So sánh xác suất log-likelihood của các từ ứng viên theo tiền tố.
2. Next-Token Accuracy Benchmark:
   - Đo lường Top-1 và Top-5 Accuracy của việc dự đoán token kế tiếp trên tập ngữ liệu kiểm thử.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn.functional as F

from src.models.language_model import TunaLM
from src.tokenizer.tokenizer import BilingualBPETokenizer


@dataclass
class ClozeItem:
    """Một câu hỏi điền từ vào chỗ trống."""
    prefix: str
    target: str
    candidates: List[str]
    category: str  # 'vi_knowledge', 'vi_grammar', 'en_knowledge', 'en_grammar'


class ClozeBenchmark:
    """Bộ câu hỏi Cloze Test Song ngữ tiêu chuẩn cho Tuna-AI."""

    def __init__(self) -> None:
        self.items: List[ClozeItem] = [
            # Tiếng Việt: Kiến thức & Từ vựng
            ClozeItem(
                prefix="Trí tuệ nhân",
                target=" tạo",
                candidates=[" tạo", " loại", " tính", " dân"],
                category="vi_knowledge"
            ),
            ClozeItem(
                prefix="Mô hình ngôn ngữ",
                target=" lớn",
                candidates=[" lớn", " nhỏ", " hẹp", " cao"],
                category="vi_knowledge"
            ),
            ClozeItem(
                prefix="Thủ đô của Việt Nam là",
                target=" Hà",
                candidates=[" Hà", " Sài", " Đà", " Huế"],
                category="vi_knowledge"
            ),
            ClozeItem(
                prefix="Học máy là một nhánh của",
                target=" khoa",
                candidates=[" khoa", " nông", " thương", " du"],
                category="vi_knowledge"
            ),
            # Tiếng Anh: Knowledge & Grammar
            ClozeItem(
                prefix="Artificial intelligence and deep",
                target=" learning",
                candidates=[" learning", " running", " walking", " sleeping"],
                category="en_knowledge"
            ),
            ClozeItem(
                prefix="The transformer model uses self",
                target=" attention",
                candidates=[" attention", " control", " driving", " system"],
                category="en_knowledge"
            ),
            ClozeItem(
                prefix="Natural language",
                target=" processing",
                candidates=[" processing", " cooking", " driving", " eating"],
                category="en_knowledge"
            ),
            ClozeItem(
                prefix="Gradient descent optimizes model",
                target=" parameters",
                candidates=[" parameters", " humans", " buildings", " trees"],
                category="en_knowledge"
            ),
        ]

    def evaluate(
        self,
        model: TunaLM,
        tokenizer: BilingualBPETokenizer,
        device: str = "cpu"
    ) -> Dict[str, Any]:
        """
        Đánh giá mô hình trên toàn bộ danh mục câu hỏi Cloze Test.
        
        Phương pháp tính:
        Với mỗi ứng viên c, tính log-likelihood: log P(c | prefix).
        Ứng viên có log-likelihood cao nhất được mô hình chọn.
        """
        model.eval()
        total_correct = 0
        category_stats: Dict[str, Dict[str, int]] = {}
        details: List[Dict[str, Any]] = []

        with torch.no_grad():
            for item in self.items:
                cat = item.category
                if cat not in category_stats:
                    category_stats[cat] = {"correct": 0, "total": 0}
                category_stats[cat]["total"] += 1

                prefix_ids = tokenizer.encode(item.prefix)
                max_ctx = getattr(model.config, "context_length", 32)
                if len(prefix_ids) > max_ctx:
                    prefix_ids = prefix_ids[-max_ctx:]
                x = torch.tensor([prefix_ids], dtype=torch.long, device=device)

                # Lan truyền thuận lấy logits của token cuối cùng
                logits, _ = model(x)  # shape: (1, seq_len, vocab_size)
                last_token_logits = logits[0, -1, :]  # shape: (vocab_size,)
                log_probs = F.log_softmax(last_token_logits, dim=-1)

                best_cand = None
                best_score = float("-inf")
                cand_scores = {}

                for cand in item.candidates:
                    cand_ids = tokenizer.encode(cand)
                    if cand_ids:
                        first_cand_id = cand_ids[0]
                        score = log_probs[first_cand_id].item()
                    else:
                        score = float("-inf")
                    cand_scores[cand] = round(score, 4)
                    if score > best_score:
                        best_score = score
                        best_cand = cand

                is_correct = (best_cand == item.target)
                if is_correct:
                    total_correct += 1
                    category_stats[cat]["correct"] += 1

                details.append({
                    "prefix": item.prefix,
                    "target": item.target,
                    "prediction": best_cand,
                    "is_correct": is_correct,
                    "scores": cand_scores,
                    "category": item.category,
                })

        accuracy = total_correct / len(self.items) if self.items else 0.0
        cat_accuracies = {
            k: round(v["correct"] / v["total"] * 100, 2)
            for k, v in category_stats.items()
        }

        return {
            "total_items": len(self.items),
            "correct_items": total_correct,
            "accuracy_percent": round(accuracy * 100, 2),
            "category_accuracy_percent": cat_accuracies,
            "details": details,
        }
