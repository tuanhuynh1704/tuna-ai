"""
src/inference/exporter.py

Hệ thống Đóng gói và Nạp Mô hình Chạy Offline Độc lập (100% Offline Runtime):
- Đóng gói trọn gói mô hình đã huấn luyện: Trọng số (weights.pt), Siêu tham số (config.json), Từ vựng Tokenizer (tokenizer.json), và Thẻ mô hình (model_card.md).
- Cung cấp hàm nạp một dòng `load_offline_model(export_dir)` sẵn sàng phục vụ suy luận tức thì trên CPU mà không cần kết nối internet hay phụ thuộc vào hạ tầng bên ngoài.
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import torch

from src.models.language_model import TunaLM, TunaLMConfig
from src.tokenizer.tokenizer import BilingualBPETokenizer
from src.inference.generator import TextGenerator


class ModelExporter:
    """Bộ công cụ đóng gói mô hình cho môi trường Offline Production."""

    @staticmethod
    def export(
        model: TunaLM,
        tokenizer: BilingualBPETokenizer,
        output_dir: str,
        model_name: str = "TunaLM-Mini-Bilingual",
        version: str = "1.0.0",
        description: str = "Mô hình ngôn ngữ lớn song ngữ Việt - Anh chạy thuần CPU trên Intel Core i7.",
    ) -> Dict[str, str]:
        """
        Đóng gói toàn bộ mô hình và tài sản liên quan vào thư mục chỉ định.
        
        Returns:
            Dict chứa đường dẫn tới các tệp đã xuất.
        """
        export_path = Path(output_dir)
        export_path.mkdir(parents=True, exist_ok=True)

        # 1. Xuất config.json
        cfg = model.config
        config_dict = {
            "model_name": model_name,
            "version": version,
            "vocab_size": cfg.vocab_size,
            "context_length": cfg.context_length,
            "d_model": cfg.d_model,
            "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads,
            "d_ff": cfg.d_ff,
            "dropout": cfg.dropout,
            "bias": cfg.bias,
            "tie_weights": cfg.tie_weights,
            "exported_at": datetime.now().isoformat(),
        }
        config_file = export_path / "config.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)

        # 2. Xuất weights.pt
        weights_file = export_path / "model_weights.pt"
        torch.save(model.state_dict(), str(weights_file))

        # 3. Xuất tokenizer.json
        tokenizer_file = export_path / "tokenizer.json"
        tokenizer.save(str(tokenizer_file))

        # 4. Xuất model_card.md
        param_count = sum(p.numel() for p in model.parameters())
        model_card_content = f"""# Thẻ Mô hình (Model Card): {model_name}

- **Phiên bản**: {version}
- **Kiến trúc**: Causal Decoder-only Transformer (Pre-RMSNorm, Causal Multi-Head Attention, FFN GELU)
- **Số lượng tham số**: {param_count:,} tham số (~{param_count / 1e6:.2f}M params)
- **Độ dài ngữ cảnh**: {cfg.context_length} tokens
- **Ngôn ngữ**: Song ngữ (Việt - Anh)
- **Tối ưu phần cứng**: Intel Core i7-1165G7 (4 Cores / 8 Threads, AVX-512, Intel MKL)
- **Chế độ vận hành**: Offline 100%, không cần kết nối mạng.
- **Ngày đóng gói**: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}

## Hướng dẫn nạp và suy luận Offline
```python
from src.inference.exporter import load_offline_model

model, tokenizer, generator = load_offline_model("{output_dir}")
response = generator.generate("Trí tuệ nhân tạo là", max_new_tokens=30)
print(response)
```
"""
        model_card_file = export_path / "model_card.md"
        with open(model_card_file, "w", encoding="utf-8") as f:
            f.write(model_card_content)

        return {
            "config": str(config_file).replace("\\", "/"),
            "weights": str(weights_file).replace("\\", "/"),
            "tokenizer": str(tokenizer_file).replace("\\", "/"),
            "model_card": str(model_card_file).replace("\\", "/"),
        }


def load_offline_model(
    export_dir: str,
    device: str = "cpu"
) -> Tuple[TunaLM, BilingualBPETokenizer, TextGenerator]:
    """
    Nạp mô hình đã đóng gói để chạy offline ngay lập tức.
    
    Returns:
        (model, tokenizer, generator)
    """
    path = Path(export_dir)
    config_file = path / "config.json"
    weights_file = path / "model_weights.pt"
    tokenizer_file = path / "tokenizer.json"

    if not config_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file config tại: {config_file}")
    if not weights_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file weights tại: {weights_file}")
    if not tokenizer_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file tokenizer tại: {tokenizer_file}")

    # 1. Nạp config và khởi tạo model
    with open(config_file, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)

    model_config = TunaLMConfig(
        vocab_size=cfg_dict["vocab_size"],
        context_length=cfg_dict["context_length"],
        d_model=cfg_dict["d_model"],
        n_layers=cfg_dict["n_layers"],
        n_heads=cfg_dict["n_heads"],
        d_ff=cfg_dict["d_ff"],
        dropout=cfg_dict.get("dropout", 0.0),
        bias=cfg_dict.get("bias", False),
        tie_weights=cfg_dict.get("tie_weights", True)
    )
    model = TunaLM(model_config)

    # 2. Nạp weights
    state_dict = torch.load(str(weights_file), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 3. Nạp tokenizer
    tokenizer = BilingualBPETokenizer.from_file(str(tokenizer_file))

    # 4. Khởi tạo generator
    generator = TextGenerator(model=model, tokenizer=tokenizer, device=device)

    return model, tokenizer, generator
