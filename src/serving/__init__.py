"""Package phục vụ mô hình ngôn ngữ Tuna-AI qua giao diện REST API."""

from src.serving.app import app, create_app

__all__ = ["app", "create_app"]
