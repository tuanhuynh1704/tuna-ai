"""Đóng gói dữ liệu nhị phân (Binary Sharding) và tạo siêu dữ liệu Manifest.

Đáp ứng bước 8, 9, 10 trong quy trình 10 bước tiền xử lý của Tuna-AI:
- Bước 8: Tokenization & Packing (chuyển văn bản thành uint16 kèm thẻ <bos>, <eos>)
- Bước 9: Metadata Manifest (ghi nhận số token, tỷ lệ ngôn ngữ, sha256 checksum)
- Bước 10: Sharding (chia nhỏ dữ liệu thành các file shard.bin tải streaming an toàn cho RAM)
"""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import numpy as np

from src.tokenizer.tokenizer import BilingualBPETokenizer


class DatasetPacker:
    """Đóng gói chuỗi văn bản đã xử lý thành các shard nhị phân numpy uint16."""

    def __init__(
        self,
        tokenizer: BilingualBPETokenizer,
        output_dir: str = "data/processed",
        max_tokens_per_shard: int = 100_000,
        shard_prefix: str = "shard_",
    ) -> None:
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.max_tokens_per_shard = max_tokens_per_shard
        self.shard_prefix = shard_prefix

        self.bos_token_id = getattr(
            self.tokenizer, "bos_token_id", self.tokenizer.special_tokens.get("<bos>", 2)
        )
        self.eos_token_id = getattr(
            self.tokenizer, "eos_token_id", self.tokenizer.special_tokens.get("<eos>", 3)
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def pack(self, documents: Iterable[str]) -> List[Dict[str, Any]]:
        """Mã hóa và ghi các tài liệu thành các file nhị phân shard_xxxxx.bin.
        
        Args:
            documents: Chuỗi các văn bản đã qua làm sạch, lọc PII và khử trùng lặp.
            
        Returns:
            Danh sách metadata của từng shard đã sinh.
        """
        current_shard_tokens: List[int] = []
        shard_records: List[Dict[str, Any]] = []
        shard_idx = 0
        total_docs_in_shard = 0

        for doc in documents:
            if not doc.strip():
                continue

            # Mã hóa văn bản
            doc_token_ids = self.tokenizer.encode(doc)
            # Bao bọc bởi BOS và EOS
            packed_doc = [self.bos_token_id] + doc_token_ids + [self.eos_token_id]
            total_docs_in_shard += 1

            current_shard_tokens.extend(packed_doc)

            # Nếu shard hiện tại đầy, ghi ra file nhị phân
            if len(current_shard_tokens) >= self.max_tokens_per_shard:
                record = self._flush_shard(shard_idx, current_shard_tokens, total_docs_in_shard)
                shard_records.append(record)
                shard_idx += 1
                current_shard_tokens = []
                total_docs_in_shard = 0

        # Ghi nốt phần dư còn lại
        if current_shard_tokens:
            record = self._flush_shard(shard_idx, current_shard_tokens, total_docs_in_shard)
            shard_records.append(record)

        return shard_records

    def _flush_shard(
        self,
        shard_idx: int,
        token_ids: List[int],
        num_docs: int
    ) -> Dict[str, Any]:
        """Ghi mảng uint16 ra tệp nhị phân và tính SHA-256."""
        shard_filename = f"{self.shard_prefix}{shard_idx:05d}.bin"
        shard_path = self.output_dir / shard_filename

        arr = np.array(token_ids, dtype=np.uint16)
        arr.tofile(str(shard_path))

        file_bytes = shard_path.read_bytes()
        sha256_hash = hashlib.sha256(file_bytes).hexdigest()

        return {
            "shard_id": shard_idx,
            "filename": shard_filename,
            "path": str(shard_path).replace("\\", "/"),
            "num_tokens": len(token_ids),
            "num_documents": num_docs,
            "file_size_bytes": len(file_bytes),
            "sha256": sha256_hash,
        }


class ManifestGenerator:
    """Tạo tệp manifest.json ghi nhận siêu dữ liệu huấn luyện."""

    def __init__(self, manifest_dir: str = "data/manifests") -> None:
        self.manifest_dir = Path(manifest_dir)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        dataset_name: str,
        shard_records: List[Dict[str, Any]],
        language_stats: Dict[str, int],
        version: str = "1.0.0",
        license_str: str = "MIT / CC-BY-4.0",
        manifest_filename: str = "manifest.json",
    ) -> str:
        """Xuất file manifest.json chuẩn chỉnh theo đặc tả Section 10."""
        total_tokens = sum(s["num_tokens"] for s in shard_records)
        total_documents = sum(s["num_documents"] for s in shard_records)
        total_bytes = sum(s["file_size_bytes"] for s in shard_records)

        total_lang_docs = sum(language_stats.values()) if language_stats else 1
        lang_distribution = {
            lang: round((count / total_lang_docs) * 100, 2)
            for lang, count in language_stats.items()
        }

        manifest_data = {
            "dataset_name": dataset_name,
            "version": version,
            "created_at": datetime.now().isoformat(),
            "license": license_str,
            "total_documents": total_documents,
            "total_tokens": total_tokens,
            "total_size_bytes": total_bytes,
            "language_counts": language_stats,
            "language_distribution_percent": lang_distribution,
            "num_shards": len(shard_records),
            "shards": shard_records,
        }

        manifest_path = self.manifest_dir / manifest_filename
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, ensure_ascii=False, indent=2)

        return str(manifest_path).replace("\\", "/")
