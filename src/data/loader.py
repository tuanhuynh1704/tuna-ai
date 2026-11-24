"""
src/data/loader.py

Bộ nạp Dữ liệu Huấn luyện & Streaming Shards:
1. get_batch: Trích xuất ngẫu nhiên 1 batch (X, Y) từ mảng token IDs trong RAM.
2. StreamingShardDataset / ShardDataLoader: 
   - Sử dụng np.memmap (Memory-mapped file) để đọc streaming các shard nhị phân uint16 (*.bin).
   - Tránh nạp toàn bộ dữ liệu vào RAM, đảm bảo hoạt động an toàn dưới 8GB/16GB RAM trên CPU i7-1165G7.
   - Hỗ trợ phân chia Train / Validation độc lập và sinh batch (X, Y) dịch chuyển 1 token.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch


def get_batch(
    data: Union[List[int], np.ndarray, torch.Tensor],
    batch_size: int,
    context_length: int,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Trích xuất ngẫu nhiên 1 batch huấn luyện (X, Y):
    data: Mảng token IDs 1 chiều
    batch_size: Số lượng chuỗi trong 1 batch (B)
    context_length: Độ dài ngữ cảnh tối đa (T)
    """
    if isinstance(data, list):
        data_arr = np.array(data, dtype=np.int64)
    elif isinstance(data, torch.Tensor):
        data_arr = data.cpu().numpy()
    else:
        data_arr = data

    num_tokens = len(data_arr)
    assert num_tokens > context_length + 1, (
        f"Số lượng token trong dataset ({num_tokens}) phải lớn hơn context_length + 1 ({context_length + 1})"
    )

    max_idx = num_tokens - context_length - 1
    start_indices = np.random.randint(0, max_idx + 1, size=batch_size)

    x_list = [data_arr[i : i + context_length] for i in start_indices]
    y_list = [data_arr[i + 1 : i + context_length + 1] for i in start_indices]

    x = torch.tensor(np.stack(x_list), dtype=torch.long, device=device)
    y = torch.tensor(np.stack(y_list), dtype=torch.long, device=device)

    return x, y


class ShardDataLoader:
    """
    Bộ nạp dữ liệu từ các Shards nhị phân uint16 bằng np.memmap.
    
    Đặc điểm kỹ thuật:
    - Ánh xạ trực tiếp từ ổ cứng SSD mà không nạp toàn bộ vào RAM (Zero-copy / on-demand paging).
    - Tự động lấy danh sách shard từ manifest.json hoặc danh sách đường dẫn trực tiếp.
    - Hỗ trợ chia train/val theo tỷ lệ không rò rỉ dữ liệu (Split ratio).
    """

    def __init__(
        self,
        shard_paths: List[str],
        batch_size: int = 4,
        context_length: int = 32,
        val_split: float = 0.1,
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        self.shard_paths = [Path(p) for p in shard_paths]
        self.batch_size = batch_size
        self.context_length = context_length
        self.val_split = val_split
        self.device = device
        self.rng = np.random.RandomState(seed)

        # Mở các memmap arrays
        self.shards: List[np.memmap] = []
        self.shard_lengths: List[int] = []
        self.total_tokens = 0

        for path in self.shard_paths:
            if not path.exists():
                raise FileNotFoundError(f"Không tìm thấy file shard tại: {path}")
            # Đọc file nhị phân dưới dạng memmap uint16 (chế độ read-only)
            mmap = np.memmap(str(path), dtype=np.uint16, mode="r")
            self.shards.append(mmap)
            self.shard_lengths.append(len(mmap))
            self.total_tokens += len(mmap)

        assert self.total_tokens > self.context_length + 2, (
            f"Tổng số tokens ({self.total_tokens}) quá nhỏ so với context_length ({self.context_length})"
        )

        # Thiết lập ranh giới train / val cho từng shard
        self.train_ranges: List[Tuple[int, int]] = []
        self.val_ranges: List[Tuple[int, int]] = []

        for length in self.shard_lengths:
            split_idx = int(length * (1.0 - self.val_split))
            # Train range: [0, split_idx)
            self.train_ranges.append((0, split_idx))
            # Val range: [split_idx, length)
            self.val_ranges.append((split_idx, length))

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str,
        batch_size: int = 4,
        context_length: int = 32,
        val_split: float = 0.1,
        device: str = "cpu",
    ) -> "ShardDataLoader":
        """Khởi tạo ShardDataLoader từ file manifest.json."""
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        shard_paths = [s["path"] for s in manifest["shards"]]
        return cls(
            shard_paths=shard_paths,
            batch_size=batch_size,
            context_length=context_length,
            val_split=val_split,
            device=device,
        )

    def get_batch(self, split: str = "train") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Trích xuất 1 batch ngẫu nhiên từ split ('train' hoặc 'val').
        
        Tự động chọn ngẫu nhiên một shard có đủ độ dài, sau đó trích xuất
        cặp token (X, Y) với Y dịch chuyển 1 bước thời gian.
        """
        ranges = self.train_ranges if split == "train" else self.val_ranges

        # Lọc ra các shard có khoảng split đủ lớn (> context_length + 1)
        valid_shard_indices = [
            i for i, (start, end) in enumerate(ranges)
            if (end - start) > self.context_length + 1
        ]

        if not valid_shard_indices:
            # Fallback nếu split quá nhỏ: dùng toàn bộ shard lớn nhất
            valid_shard_indices = [int(np.argmax(self.shard_lengths))]
            shard_idx = valid_shard_indices[0]
            start, end = 0, self.shard_lengths[shard_idx]
        else:
            shard_idx = self.rng.choice(valid_shard_indices)
            start, end = ranges[shard_idx]

        mmap = self.shards[shard_idx]
        max_idx = end - self.context_length - 1
        start_indices = self.rng.randint(start, max_idx + 1, size=self.batch_size)

        x_list = [mmap[i : i + self.context_length].astype(np.int64) for i in start_indices]
        y_list = [mmap[i + 1 : i + self.context_length + 1].astype(np.int64) for i in start_indices]

        x = torch.tensor(np.stack(x_list), dtype=torch.long, device=self.device)
        y = torch.tensor(np.stack(y_list), dtype=torch.long, device=self.device)

        return x, y
