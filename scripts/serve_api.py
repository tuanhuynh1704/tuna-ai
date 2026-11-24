"""
scripts/serve_api.py

Kịch bản Khởi chạy Máy chủ Model Serving API cho Tuna-AI:
- Vận hành trên máy chủ Uvicorn tốc độ cao.
- Mặc định lắng nghe tại: http://localhost:8000
- Hỗ trợ tài liệu tương tác Swagger UI tại: http://localhost:8000/docs
- Hỗ trợ tài liệu Redoc tại: http://localhost:8000/redoc
"""

import argparse
import sys
from pathlib import Path
import uvicorn

# Thêm root vào sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Khởi chạy Tuna-AI Model Serving API")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Địa chỉ IP lắng nghe (mặc định: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Cổng dịch vụ (mặc định: 8000)")
    parser.add_argument("--reload", action="store_true", help="Tự động tải lại mã nguồn khi có thay đổi")
    args = parser.parse_args()

    print("=" * 70)
    print("TUNA-AI: KHỞI CHẠY MODEL SERVING REST API (100% OFFLINE RUNTIME)")
    print("=" * 70)
    print(f"Địa chỉ dịch vụ   : http://{args.host}:{args.port}")
    print(f"Tài liệu tương tác: http://{args.host}:{args.port}/docs")
    print(f"Kiểm tra sức khỏe : http://{args.host}:{args.port}/health")
    print("=" * 70)

    uvicorn.run("src.serving.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
