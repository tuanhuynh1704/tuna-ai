"""
src/data/loader.py

Bộ nạp Dữ liệu Huấn luyện (Data Loader & Batch Generator):
- Tạo các mẩu cắt ngẫu nhiên (Chunks) phục vụ huấn luyện Autoregressive LM.
- Sinh cặp tensor (X, Y) với Y được dịch chuyển đúng 1 token về tương lai:
    X: tokens[i     : i + context_length]
    Y: tokens[i + 1 : i + context_length + 1]
"""

from typing import Tuple, List, Union
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

    # Chọn ngẫu nhiên batch_size vị trí bắt đầu
    max_idx = num_tokens - context_length - 1
    start_indices = np.random.randint(0, max_idx + 1, size=batch_size)

    x_list = [data_arr[i : i + context_length] for i in start_indices]
    y_list = [data_arr[i + 1 : i + context_length + 1] for i in start_indices]

    x = torch.tensor(np.stack(x_list), dtype=torch.long, device=device)
    y = torch.tensor(np.stack(y_list), dtype=torch.long, device=device)

    return x, y
