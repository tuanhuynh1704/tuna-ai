"""
src/inference/kv_cache.py

Cơ chế Bộ đệm Key-Value (KV Cache) Tăng tốc Suy luận Tự hồi quy trên CPU:
- Giảm độ phức tạp tính toán Attention cho mỗi bước giải mã (Decode Step) từ O(N^2) xuống O(N).
- Thay vì tính lại toàn bộ chuỗi token quá khứ, chỉ tính toán Query/Key/Value cho token mới và tái sử dụng Key-Value đã được lưu trong bộ đệm.
"""

from typing import Dict, List, Optional, Tuple
import torch


class KVCache:
    """Quản lý bộ nhớ đệm Key và Value cho từng tầng Transformer trong quá trình sinh tự hồi quy."""

    def __init__(self, num_layers: int) -> None:
        self.num_layers = num_layers
        self.cache: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * num_layers

    def update(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cập nhật Key và Value mới vào bộ đệm của tầng layer_idx.
        
        Args:
            layer_idx: Chỉ số tầng Transformer (0 đến num_layers - 1).
            k: Key tensor mới có shape: (B, H, T_new, d_k).
            v: Value tensor mới có shape: (B, H, T_new, d_k).
            
        Returns:
            Tuple (full_k, full_v) sau khi nối các tokens trước đó: (B, H, T_total, d_k).
        """
        if self.cache[layer_idx] is None:
            self.cache[layer_idx] = (k, v)
            return k, v

        prev_k, prev_v = self.cache[layer_idx]
        full_k = torch.cat([prev_k, k], dim=2)
        full_v = torch.cat([prev_v, v], dim=2)
        self.cache[layer_idx] = (full_k, full_v)
        return full_k, full_v

    def get_seq_len(self) -> int:
        """Độ dài chuỗi tokens hiện tại đã lưu trữ trong bộ đệm."""
        if self.cache[0] is None:
            return 0
        return self.cache[0][0].size(2)

    def reset(self) -> None:
        """Xóa sạch bộ đệm sau khi kết thúc chu kỳ sinh văn bản."""
        self.cache = [None] * self.num_layers
