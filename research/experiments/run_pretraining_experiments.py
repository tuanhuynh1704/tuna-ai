"""
research/experiments/run_pretraining_experiments.py

Thực thi loạt Thí nghiệm Huấn luyện Tiền kỳ Có Kiểm soát (Milestone M8):
- EXP-001: Huấn luyện trên Ngữ liệu thuần Tiếng Anh.
- EXP-002: Huấn luyện trên Ngữ liệu thuần Tiếng Việt.
- EXP-003: Huấn luyện trên Ngữ liệu Song ngữ (50% Việt - 50% Anh).

Đo lường chi tiết:
- Initial Loss, Final Loss, Validation Loss, Perplexity
- Throughput (tokens/giây)
- Mức tiêu thụ bộ nhớ RAM (MB)
- Khả năng sinh văn bản trước và sau huấn luyện
- Xuất báo cáo so sánh tại experiments/reports/m8_pretraining_experiments.md
"""

import os
import sys
from pathlib import Path
import json

# Thêm root vào sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.packer import DatasetPacker
from src.data.loader import ShardDataLoader
from src.tokenizer.tokenizer import BilingualBPETokenizer
from src.training.experiment import ExperimentConfig, ExperimentRunner


def get_corpora():
    """Tập ngữ liệu chuẩn cho 3 kịch bản thí nghiệm."""
    en_corpus = [
        "Artificial intelligence and deep learning have revolutionized modern software engineering.",
        "The transformer architecture relies entirely on self attention mechanisms without recurrence.",
        "Residual connections and layer normalization stabilize optimization in deep networks.",
        "Cosine learning rate schedule with linear warmup provides optimal convergence characteristics.",
        "Autoregressive language models predict the next token given preceding context tokens.",
        "Gradient clipping limits the global norm of parameter gradients preventing exploding updates.",
        "Byte level byte pair encoding enables lossless subword tokenization across languages.",
        "Pretraining on large scale text builds foundational linguistic representations.",
    ] * 6  # Tăng số lượng câu để tạo shard đầy đủ

    vi_corpus = [
        "Trí tuệ nhân tạo và học sâu đã tạo nên cuộc cách mạng trong ngành kỹ thuật phần mềm hiện đại.",
        "Kiến trúc transformer dựa hoàn toàn vào cơ chế tự chú ý mà không cần mạng hồi quy.",
        "Liên kết phần dư và chuẩn hóa tầng giúp ổn định hóa quá trình tối ưu trong mạng sâu.",
        "Bộ điều độ học tốc độ cosine với khởi động tuyến tính mang lại đặc tính hội tụ tối ưu.",
        "Mô hình ngôn ngữ tự hồi quy dự đoán token tiếp theo dựa trên chuỗi ngữ cảnh phía trước.",
        "Cắt tỉa gradient giới hạn chuẩn toàn cục của gradient tham số ngăn ngừa bùng nổ trọng số.",
        "Mã hóa cặp byte mức byte cho phép phân mảnh từ vựng không mất mát dữ liệu đa ngôn ngữ.",
        "Huấn luyện tiền kỳ trên ngữ liệu văn bản quy mô lớn xây dựng các biểu diễn ngôn ngữ nền tảng.",
    ] * 6

    bilingual_corpus = en_corpus[:24] + vi_corpus[:24]
    return en_corpus, vi_corpus, bilingual_corpus


def prepare_shards(tokenizer: BilingualBPETokenizer, base_dir: Path):
    """Chuẩn bị các file shard uint16 nhị phân cho 3 chế độ ngữ liệu."""
    en_corpus, vi_corpus, bilingual_corpus = get_corpora()

    loaders = {}
    modes = [("en", en_corpus), ("vi", vi_corpus), ("bilingual", bilingual_corpus)]

    for mode, docs in modes:
        mode_dir = base_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        packer = DatasetPacker(
            tokenizer=tokenizer,
            output_dir=str(mode_dir),
            max_tokens_per_shard=250,
            shard_prefix=f"{mode}_shard_",
        )
        records = packer.pack(docs)
        shard_paths = [r["path"] for r in records]
        loaders[mode] = ShardDataLoader(
            shard_paths=shard_paths,
            batch_size=4,
            context_length=32,
            val_split=0.15,
            device="cpu"
        )

    return loaders


def main():
    print("=" * 70)
    print("TUNA-AI: THỰC THI CHUỖI THÍ NGHIỆM TIỀN KỲ CÓ KIỂM SOÁT (MILESTONE M8)")
    print("=" * 70)

    # 1. Nạp tokenizer 500 vocab
    tok_path = Path("configs/tokenizer/bpe_toy_500.json")
    assert tok_path.exists(), f"Không tìm thấy file tokenizer: {tok_path}"
    tokenizer = BilingualBPETokenizer.from_file(str(tok_path))

    # 2. Nạp model config
    config_path = Path("configs/model/tiny.json")
    with open(config_path, "r", encoding="utf-8") as f:
        model_config = json.load(f)

    # Đảm bảo context length và vocab khớp
    model_config["vocab_size"] = tokenizer.vocab_size
    model_config["context_length"] = 32

    # 3. Chuẩn bị Shards cho các thí nghiệm
    exp_shards_dir = Path("data/experiments")
    loaders = prepare_shards(tokenizer, exp_shards_dir)

    # 4. Khởi tạo ExperimentRunner
    runner = ExperimentRunner(
        tokenizer=tokenizer,
        model_config=model_config,
        output_dir="experiments/reports"
    )

    test_prompts = [
        "Trí tuệ nhân tạo",
        "The transformer model",
    ]

    # EXP-001: Pure English Corpus
    exp1_cfg = ExperimentConfig(
        name="EXP-001-English",
        description="Huấn luyện mô hình tiền kỳ trên ngữ liệu thuần tiếng Anh",
        corpus_mode="English Only",
        total_steps=60,
        batch_size=4,
        context_length=32,
        learning_rate=1e-3,
        warmup_steps=10,
        seed=101,
        device="cpu"
    )
    runner.run_experiment(exp1_cfg, loaders["en"], prompts=test_prompts)

    # EXP-002: Pure Vietnamese Corpus
    exp2_cfg = ExperimentConfig(
        name="EXP-002-Vietnamese",
        description="Huấn luyện mô hình tiền kỳ trên ngữ liệu thuần tiếng Việt",
        corpus_mode="Vietnamese Only",
        total_steps=60,
        batch_size=4,
        context_length=32,
        learning_rate=1e-3,
        warmup_steps=10,
        seed=101,
        device="cpu"
    )
    runner.run_experiment(exp2_cfg, loaders["vi"], prompts=test_prompts)

    # EXP-003: Bilingual 50/50 Corpus
    exp3_cfg = ExperimentConfig(
        name="EXP-003-Bilingual",
        description="Huấn luyện mô hình tiền kỳ trên ngữ liệu song ngữ cân bằng (50% Vi - 50% En)",
        corpus_mode="Bilingual 50/50",
        total_steps=60,
        batch_size=4,
        context_length=32,
        learning_rate=1e-3,
        warmup_steps=10,
        seed=101,
        device="cpu"
    )
    runner.run_experiment(exp3_cfg, loaders["bilingual"], prompts=test_prompts)

    # 5. Xuất báo cáo tổng kết
    report_file = runner.generate_report("m8_pretraining_experiments.md")
    print(f"\n{'='*70}")
    print(f"ĐÃ XUẤT BÁO CÁO THỰC NGHIỆM MILESTONE M8 THÀNH CÔNG: {report_file}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
