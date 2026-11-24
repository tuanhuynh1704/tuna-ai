"""
scripts/chat.py

Giao diện Trò chuyện Tương tác với Tuna-AI trong Terminal (Interactive CLI Chat):
- Nạp trực tiếp mô hình từ `models/tuna_offline_v1` (100% Offline Runtime).
- Hỗ trợ giải mã luồng thời gian thực (Streaming Generation) kèm bộ đệm KV Cache.
- Tự động duy trì ngữ cảnh đàm thoại nhiều lượt (Multi-turn Conversation).
- Gõ 'exit', 'quit' hoặc 'q' để thoát.
"""

import os
import sys
import time
from pathlib import Path

# Thêm root vào sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.exporter import load_offline_model
from src.instruction.chat_template import ChatTemplate


def main():
    print("=" * 70)
    print("🦈 TUNA-AI: GIAO DIỆN HỘI THOẠI TRỰC TIẾP (INTERACTIVE CHAT CLI)")
    print("   Vận hành thuần CPU trên Intel Core i7 (100% Offline)")
    print("=" * 70)

    model_dir = "models/tuna_offline_v1"
    if not Path(model_dir).exists():
        print(f"Lỗi: Không tìm thấy thư mục mô hình tại '{model_dir}'.")
        print("Vui lòng chạy 'python research/inference/benchmark_kv_cache.py' để đóng gói mô hình trước.")
        return

    print(f"\n[1/2] Đang nạp mô hình từ '{model_dir}'...")
    t0 = time.time()
    try:
        model, tokenizer, generator = load_offline_model(model_dir, device="cpu")
    except Exception as e:
        print(f"Lỗi khi nạp mô hình: {e}")
        return

    chat_template = ChatTemplate()
    load_time = time.time() - t0
    param_count = sum(p.numel() for p in model.parameters())

    print(f"[2/2] Nạp thành công! Tham số: {param_count:,} | Thời gian nạp: {load_time:.2f}s")
    print("\n" + "-" * 70)
    print("💡 Mẹo: Nhập câu hỏi và nhấn [Enter]. Gõ 'exit' hoặc 'quit' để thoát.")
    print("        Gõ 'clear' để xóa lịch sử cuộc trò chuyện.")
    print("-" * 70)

    messages = []

    while True:
        try:
            user_input = input("\n🧑 [Bạn]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nĐã thoát giao diện trò chuyện. Tạm biệt!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit", "q"]:
            print("\nCảm ơn bạn đã sử dụng Tuna-AI. Tạm biệt!")
            break

        if user_input.lower() in ["clear", "cls"]:
            messages.clear()
            print("Đã làm mới cuộc trò chuyện!")
            continue

        messages.append({"role": "user", "content": user_input})
        prompt_str = chat_template.format_dialogue(messages, add_generation_prompt=True)

        print("\n🦈 [Tuna-AI]: ", end="", flush=True)

        t_gen_start = time.perf_counter()
        token_count = 0
        full_reply = ""

        # Sinh theo luồng streaming
        for chunk in generator.generate_stream(
            prompt=prompt_str,
            max_new_tokens=40,
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.15,
            use_cache=True,
            stop_tokens=["<|im_end|>", "<eos>"]
        ):
            print(chunk, end="", flush=True)
            full_reply += chunk
            token_count += 1

        t_gen = time.perf_counter() - t_gen_start
        throughput = token_count / max(t_gen, 1e-4)

        clean_reply = chat_template.parse_response(full_reply)
        messages.append({"role": "assistant", "content": clean_reply})

        print(f"\n   ⏱ ({token_count} tokens | {t_gen:.2f}s | {throughput:.1f} tok/s)")


if __name__ == "__main__":
    main()
