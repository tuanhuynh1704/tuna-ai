# Ghi chép & Xử lý Cảnh báo Hệ thống (Warnings Log)

> **Dự án**: Tuna-AI — LLM Song ngữ Việt - Anh từ số 0  
> **Tài liệu**: Quản lý và theo dõi các cảnh báo môi trường phát triển (Environment & Runtime Warnings)  
> **Cập nhật lần cuối**: 24/11/2026  

---

## 1. Cảnh báo `PytestRemovedIn10Warning` (Đã xử lý)

### 1.1. Chi tiết Thông báo Cảnh báo
Khi thực thi bộ kiểm thử tự động `python -m pytest`, hệ thống hiển thị thông báo:
```text
============================== warnings summary ===============================
..\..\..\..\AppData\Local\Programs\Python\Python311\Lib\site-packages\anyio\pytest_plugin.py:14
  C:\Users\nkocn\AppData\Local\Programs\Python\Python311\Lib\site-packages\anyio\pytest_plugin.py:14: PytestRemovedIn10Warning: 
  _pytest.python.CallSpec2 has been renamed to CallSpec.
  The CallSpec2 alias will be removed in pytest 10.
  Update imports to use CallSpec instead.
  See https://docs.pytest.org/en/stable/deprecations.html#callspec2-renamed
    from _pytest.python import CallSpec2
```

---

### 1.2. Bản chất Kỹ thuật & Nguồn gốc
- **Nguồn gốc phát sinh**: Cảnh báo này xuất phát từ tệp `anyio/pytest_plugin.py` thuộc thư viện bên thứ ba **`anyio`** (một dependency được cài đặt sẵn trong môi trường Python toàn cục của hệ thống phục vụ các tác vụ bất đồng bộ).
- **Không phải lỗi từ mã nguồn dự án**: Tệp này nằm ngoài mã nguồn dự án Tuna-AI (`src/`, `research/`, `tests/`).
- **Nội dung cảnh báo**: Thư viện `anyio` sử dụng tên gọi nội bộ cũ `CallSpec2` của Pytest. Pytest cảnh báo trước rằng ở phiên bản lớn tiếp theo (**Pytest 10**), tên gọi cũ này sẽ chính thức bị loại bỏ để đổi sang `CallSpec`.

---

### 1.3. Mức độ Ảnh hưởng
| Tiêu chí | Mức độ | Đánh giá chi tiết |
| :--- | :---: | :--- |
| **Logic tính toán AI** | **0%** | Không ảnh hưởng đến các phép toán Tensor, Attention, Tokenizer hay Autograd. |
| **Độ chính xác kiểm thử** | **0%** | Toàn bộ 67/67 unit tests đều kiểm định logic chính xác $100\%$. |
| **Tính an toàn hệ thống** | **An toàn tuyệt đối** | Đây là `DeprecationWarning` (cảnh báo tương thích phiên bản tương lai), hoàn toàn **không phải Error hay Exception**. |

---

### 1.4. Giải pháp Xử lý Triệt để
Để màn hình kiểm định luôn sạch sẽ, không bị làm phiền bởi các thông báo lỗi thời từ thư viện bên ngoài, dự án đã bổ sung bộ lọc cảnh báo `filterwarnings` vào tệp cấu hình [pyproject.toml](file:///c:/Users/nkocn/OneDrive/Desktop/Plan/tuna-ai/pyproject.toml):

```toml
[tool.pytest.ini_options]
minversion = "7.0"
addopts = "-ra -q"
asyncio_default_fixture_loop_scope = "function"
testpaths = [
    "tests",
]
pythonpath = [
    ".",
    "src",
]
filterwarnings = [
    "ignore::pytest.PytestRemovedIn10Warning",
    "ignore::DeprecationWarning",
]
```

---

### 1.5. Kết quả Kiểm định sau Xử lý
Thực thi kiểm tra toàn bộ hệ thống kiểm thử:
```powershell
python -m pytest -q
```
**Kết quả thực tế:**
```text
...................................................................      [100%]
67 passed in 3.48s
```
Toàn bộ **67/67 tests passed hoàn hảo với 0 warning**.
