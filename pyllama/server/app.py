from __future__ import annotations

import json
import threading
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from pyllama.engine.runtime import Runtime
from pyllama.engine.sampler import Sampler

class ChatRequest(BaseModel):
    messages: list[dict[str, str]]
    max_tokens: int = 128
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.9
    min_p: float = 0.0
    repeat_penalty: float = 1.1
    stream: bool = True
    seed: int | None = None

class Holder:
    def __init__(self):
        self.runtime: Runtime | None = None
        self.lock = threading.Lock()

holder = Holder()

def create_app(model_path: str | Path, *, tokenizer_path=None, dtype="float32", context=None, device="auto") -> FastAPI:
    app = FastAPI(title="PyLlama", version="0.3.0", docs_url=None, redoc_url=None)
    holder.runtime = Runtime(model_path, tokenizer_path=tokenizer_path, dtype=dtype, context=context, device=device)

    @app.get("/api/health")
    def health():
        return {"ok": True}

    @app.get("/api/models")
    def models():
        rt = holder.runtime
        return {"data": [{"id": rt.model_path.stem}]}

    @app.get("/api/info")
    def info():
        return holder.runtime.info()

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest):
        rt = holder.runtime
        if not rt.tokenizer:
            raise RuntimeError("tokenizer.json is required")
        prompt = rt.tokenizer.apply_chat_template(req.messages, add_generation_prompt=True)
        sampler = Sampler(req.temperature, req.top_k, req.top_p, req.repeat_penalty, req.min_p, req.seed)
        if not req.stream:
            with holder.lock:
                answer = []
                for piece, _, _ in rt.stream_generate(prompt, max_tokens=req.max_tokens, sampler=sampler):
                    answer.append(piece)
            content = "".join(answer)
            return {"id": "pyllama", "object": "chat.completion", "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}]}

        def event_stream():
            with holder.lock:
                for piece, _, _ in rt.stream_generate(prompt, max_tokens=req.max_tokens, sampler=sampler):
                    payload = {"id": "pyllama", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app

if __name__ == "__main__":
    import argparse, uvicorn
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer")
    p.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    p.add_argument("--context", type=int)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()
    uvicorn.run(create_app(args.model, tokenizer_path=args.tokenizer, dtype=args.dtype, context=args.context, device=args.device), host=args.host, port=args.port)
