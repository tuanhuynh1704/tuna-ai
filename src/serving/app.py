"""
src/serving/app.py

Máy chủ Dịch vụ Mô hình Ngôn ngữ Tuna-AI (Model Serving REST API):
- Phát triển bằng FastAPI & Uvicorn, vận hành 100% Offline trên CPU Intel Core i7-1165G7.
- Cung cấp các giao diện lập trình ứng dụng (API Endpoints):
  1. GET  /health: Trạng thái máy chủ, mức chiếm dụng RAM, luồng CPU và siêu dữ liệu mô hình.
  2. GET  /v1/models: Danh sách các mô hình đang được phục vụ theo chuẩn OpenAI.
  3. POST /v1/generate: Sinh văn bản tự do hỗ trợ đầy đủ các tham số sampling và KV Cache.
  4. POST /v1/chat/completions: Giao thức hội thoại tương thích OpenAI (hỗ trợ cả JSON và Server-Sent Events SSE streaming).
"""

from contextlib import asynccontextmanager
from datetime import datetime
import json
import os
from pathlib import Path
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Union
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import psutil
import torch

from src.inference.exporter import load_offline_model
from src.inference.generator import TextGenerator
from src.instruction.chat_template import ChatTemplate
from src.models.language_model import TunaLM, TunaLMConfig
from src.tokenizer.tokenizer import BilingualBPETokenizer


# --- Pydantic Schemas ---

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Đoạn văn bản đầu vào để mồi mô hình")
    max_tokens: int = Field(default=30, ge=1, le=256, description="Số lượng token tối đa cần sinh")
    temperature: float = Field(default=0.8, ge=0.0, le=2.0, description="Độ mượt của phân phối xác suất (0 = Greedy)")
    top_k: Optional[int] = Field(default=50, ge=0, description="Số lượng token ứng viên hàng đầu")
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0, description="Ngưỡng xác suất tích lũy Nucleus")
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=3.0, description="Hệ số phạt lặp từ")
    use_cache: bool = Field(default=True, description="Kích hoạt bộ đệm KV Cache để tăng tốc giải mã")
    stop: Optional[List[str]] = Field(default=None, description="Danh sách các chuỗi dừng")


class GenerateResponse(BaseModel):
    text: str
    prompt: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    duration_sec: float
    throughput_tok_per_sec: float


class ChatMessage(BaseModel):
    role: str = Field(..., description="Vai trò: system, user, hoặc assistant")
    content: str = Field(..., description="Nội dung thông điệp")


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="tuna-ai-mini-v1", description="Tên mô hình")
    messages: List[ChatMessage] = Field(..., description="Lịch sử hội thoại nhiều lượt")
    max_tokens: int = Field(default=50, ge=1, le=256)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_k: Optional[int] = Field(default=50)
    top_p: Optional[float] = Field(default=0.9)
    stream: bool = Field(default=False, description="Nếu True, trả về luồng Server-Sent Events (SSE)")


# --- Global Model Registry ---

class ModelContext:
    model: Optional[TunaLM] = None
    tokenizer: Optional[BilingualBPETokenizer] = None
    generator: Optional[TextGenerator] = None
    chat_template: Optional[ChatTemplate] = None
    model_name: str = "tuna-ai-mini-v1"
    model_path: str = "models/tuna_offline_v1"


ctx = ModelContext()


def ensure_model_loaded():
    """Khởi tạo và nạp mô hình offline vào bộ nhớ nếu chưa nạp."""
    if ctx.model is not None:
        return

    export_dir = Path(ctx.model_path)
    if export_dir.exists() and (export_dir / "config.json").exists():
        try:
            model, tok, gen = load_offline_model(str(export_dir), device="cpu")
            ctx.model = model
            ctx.tokenizer = tok
            ctx.generator = gen
            ctx.chat_template = ChatTemplate()
            return
        except Exception as e:
            print(f"[Warning] Không thể nạp từ {export_dir}: {e}. Khởi tạo mô hình mặc định...")

    # Fallback: Khởi tạo mô hình tiêu chuẩn với tokenizer 500 vocab
    tok_path = Path("configs/tokenizer/bpe_toy_500.json")
    if not tok_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file tokenizer tại: {tok_path}")

    tokenizer = BilingualBPETokenizer.from_file(str(tok_path))
    config = TunaLMConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=64,
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_ff=256,
        dropout=0.0
    )
    model = TunaLM(config)
    model.eval()

    ctx.model = model
    ctx.tokenizer = tokenizer
    ctx.generator = TextGenerator(model=model, tokenizer=tokenizer, device="cpu")
    ctx.chat_template = ChatTemplate()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle quản lý khởi động và giải phóng tài nguyên của máy chủ."""
    ensure_model_loaded()
    print(f"\n[Serving] Máy chủ Tuna-AI API đã sẵn sàng phục vụ mô hình '{ctx.model_name}' trên CPU!")
    yield
    print("\n[Serving] Giải phóng tài nguyên và dừng máy chủ...")


def create_app() -> FastAPI:
    """Khởi tạo ứng dụng FastAPI với đầy đủ routes và middleware CORS."""
    app = FastAPI(
        title="Tuna-AI Serving API",
        version="1.0.0",
        description="Máy chủ phục vụ suy luận mô hình ngôn ngữ lớn song ngữ Việt - Anh chạy thuần CPU trên Intel Core i7.",
        lifespan=lifespan
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health_check():
        """Kiểm tra tình trạng sức khỏe máy chủ và tài nguyên CPU/RAM."""
        ensure_model_loaded()
        process = psutil.Process(os.getpid())
        ram_mb = process.memory_info().rss / (1024 * 1024)
        num_threads = torch.get_num_threads()
        param_count = sum(p.numel() for p in ctx.model.parameters()) if ctx.model else 0

        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "model_name": ctx.model_name,
            "parameters": param_count,
            "cpu_threads": num_threads,
            "ram_usage_mb": round(ram_mb, 2),
            "device": "cpu",
        }

    @app.get("/v1/models")
    def list_models():
        """Danh sách các mô hình đang khả dụng theo đặc tả chuẩn OpenAI."""
        ensure_model_loaded()
        return {
            "object": "list",
            "data": [
                {
                    "id": ctx.model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "tuna-ai-project",
                    "permission": [],
                    "root": ctx.model_name,
                    "parent": None,
                }
            ]
        }

    @app.post("/v1/generate", response_model=GenerateResponse)
    def generate_text(req: GenerateRequest):
        """Sinh văn bản tự hồi quy từ prompt đầu vào."""
        ensure_model_loaded()
        prompt_ids = ctx.tokenizer.encode(req.prompt)

        t0 = time.perf_counter()
        output_text = ctx.generator.generate(
            prompt=req.prompt,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            use_cache=req.use_cache,
            stop_tokens=req.stop
        )
        duration = time.perf_counter() - t0

        completion_ids = ctx.tokenizer.encode(output_text)
        num_completion_tokens = len(completion_ids)
        total_tokens = len(prompt_ids) + num_completion_tokens
        throughput = num_completion_tokens / max(duration, 1e-4)

        return GenerateResponse(
            text=output_text,
            prompt=req.prompt,
            prompt_tokens=len(prompt_ids),
            completion_tokens=num_completion_tokens,
            total_tokens=total_tokens,
            duration_sec=round(duration, 4),
            throughput_tok_per_sec=round(throughput, 2)
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        """Endpoint tương thích chuẩn OpenAI Chat Completions API."""
        ensure_model_loaded()
        messages_dict = [{"role": m.role, "content": m.content} for m in req.messages]
        prompt_str = ctx.chat_template.format_dialogue(messages_dict, add_generation_prompt=True)
        prompt_ids = ctx.tokenizer.encode(prompt_str)

        # 1. Chế độ Streaming (SSE)
        if req.stream:
            async def event_generator() -> AsyncGenerator[str, None]:
                created_ts = int(time.time())
                for chunk in ctx.generator.generate_stream(
                    prompt=prompt_str,
                    max_new_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_k=req.top_k,
                    top_p=req.top_p,
                    use_cache=True,
                    stop_tokens=["<|im_end|>", "<eos>"]
                ):
                    payload = {
                        "id": f"chatcmpl-{int(time.time()*1000)}",
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": req.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": chunk},
                                "finish_reason": None
                            }
                        ]
                    }
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

                # Gửi tín hiệu hoàn tất
                done_payload = {
                    "id": f"chatcmpl-{int(time.time()*1000)}",
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": req.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }
                    ]
                }
                yield f"data: {json.dumps(done_payload, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        # 2. Chế độ trả lời trọn vẹn (Standard JSON Response)
        raw_output = ctx.generator.generate(
            prompt=prompt_str,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            use_cache=True,
            stop_tokens=["<|im_end|>", "<eos>"]
        )
        reply = ctx.chat_template.parse_response(raw_output)
        completion_ids = ctx.tokenizer.encode(reply)

        return {
            "id": f"chatcmpl-{int(time.time()*1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": reply
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(completion_ids),
                "total_tokens": len(prompt_ids) + len(completion_ids)
            }
        }

    return app


app = create_app()
