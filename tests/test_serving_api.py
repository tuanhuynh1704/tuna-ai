"""Bộ kiểm thử đơn vị tự động cho Máy chủ Phục vụ Mô hình Model Serving API (Milestone M12).

Kiểm tra:
1. GET /health: Trả về status 200, thông tin RAM, CPU threads, model info.
2. GET /v1/models: Trả về danh sách model khả dụng chuẩn OpenAI.
3. POST /v1/generate: Sinh văn bản hợp lệ kèm các thông số thống kê token.
4. POST /v1/chat/completions (Non-stream): Trả về JSON chuẩn OpenAI với choices và usage.
5. POST /v1/chat/completions (Stream): Trả về luồng SSE data chunks và kết thúc bằng [DONE].
"""

import json
import pytest
from fastapi.testclient import TestClient

from src.serving.app import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_check_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "cpu_threads" in data
    assert "ram_usage_mb" in data
    assert data["device"] == "cpu"
    assert data["parameters"] > 0


def test_list_models_endpoint(client):
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) >= 1
    assert data["data"][0]["id"] == "tuna-ai-mini-v1"


def test_generate_endpoint(client):
    payload = {
        "prompt": "Trí tuệ nhân tạo là",
        "max_tokens": 10,
        "temperature": 0.8,
        "use_cache": True
    }
    response = client.post("/v1/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["text"]) > 0
    assert data["prompt_tokens"] > 0
    assert data["completion_tokens"] > 0
    assert data["total_tokens"] == data["prompt_tokens"] + data["completion_tokens"]
    assert data["throughput_tok_per_sec"] > 0


def test_chat_completions_non_stream(client):
    payload = {
        "model": "tuna-ai-mini-v1",
        "messages": [
            {"role": "user", "content": "Xin chào! Bạn là ai?"}
        ],
        "max_tokens": 12,
        "temperature": 0.7,
        "stream": False
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert len(data["choices"][0]["message"]["content"]) >= 0
    assert "usage" in data


def test_chat_completions_streaming(client):
    payload = {
        "model": "tuna-ai-mini-v1",
        "messages": [
            {"role": "user", "content": "What is AI?"}
        ],
        "max_tokens": 8,
        "stream": True
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    lines = response.text.split("\n\n")
    data_lines = [line.strip() for line in lines if line.startswith("data:")]
    assert len(data_lines) > 0
    assert any("[DONE]" in line for line in data_lines)
