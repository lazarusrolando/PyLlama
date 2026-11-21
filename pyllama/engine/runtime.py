from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from pyllama.formats.gguf import GGUFFile
from pyllama.models.llama_like import LlamaLikeModel
from pyllama.models.registry import get_architecture
from pyllama.tokenizer.bpe import ByteBPE
from pyllama.engine.device import detect_devices
from .sampler import Sampler

@dataclass
class GenerateStats:
    prompt_tokens: int
    generated_tokens: int
    elapsed_seconds: float

    @property
    def tok_per_sec(self) -> float:
        return self.generated_tokens / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0

class Runtime:
    def __init__(self, model_path: str | Path, *, tokenizer_path: str | Path | None = None, dtype: str = "float32", context: int | None = None, device: str = "auto"):
        self.model_path = Path(model_path)
        self.gguf = GGUFFile.open(self.model_path)
        self.architecture = get_architecture(self.gguf.metadata)
        self.model = LlamaLikeModel(self.gguf, weight_dtype=dtype, max_context=context, device=device)
        self.device = self.model.backend.device
        self.tokenizer = None
        tok = Path(tokenizer_path) if tokenizer_path else self._discover_tokenizer()
        if tok and tok.exists():
            self.tokenizer = ByteBPE.from_json(tok)
        else:
            try:
                self.tokenizer = ByteBPE.from_gguf_metadata(self.gguf.metadata)
            except ValueError:
                self.tokenizer = None

    def _discover_tokenizer(self) -> Path | None:
        candidates = [
            self.model_path.parent / "tokenizer.json",
            self.model_path.parent.parent / "tokenizer.json",
        ]
        return next((p for p in candidates if p.exists()), None)

    def close(self) -> None:
        self.gguf.close()

    def info(self) -> dict:
        d = self.gguf.describe()
        d["config"] = self.model.config.__dict__
        d["tokenizer_loaded"] = self.tokenizer is not None
        d["architecture_family"] = self.architecture.family
        d["architecture_supported"] = self.architecture.supported
        d["device"] = self.device
        di = detect_devices()
        d["cpu_available"] = di.cpu_available
        d["gpu_available"] = di.gpu_available
        d["gpu_backend"] = di.gpu_backend
        d["gpu_name"] = di.gpu_name
        d["gpu_count"] = di.gpu_count
        return d

    def encode(self, text: str) -> list[int]:
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not loaded. Pull a model with tokenizer.json or pass --tokenizer.")
        return self.tokenizer.encode(text)

    def decode(self, ids: list[int]) -> str:
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not loaded")
        return self.tokenizer.decode(ids)

    def stream_generate(self, prompt: str, *, max_tokens: int = 128, sampler: Sampler | None = None) -> Iterator[tuple[str, int, int]]:
        ids = self.encode(prompt)
        if not ids:
            raise ValueError("Prompt produced no tokens")
        sampler = sampler or Sampler()
        self.model.reset()
        for pos, tid in enumerate(ids):
            logits = self.model.forward_token(tid, pos)
        recent = ids[-64:].copy()
        generated = 0
        for i in range(max_tokens):
            next_id = sampler.sample(logits, recent)
            recent.append(next_id)
            piece = self.decode([next_id])
            generated += 1
            yield piece, next_id, generated
            logits = self.model.forward_token(next_id, len(ids) + i)
            stop_ids = {self.gguf.metadata.get("tokenizer.ggml.eos_token_id")}
            stop_ids.discard(None)
            if next_id in stop_ids:
                break

    def generate(self, prompt: str, **kwargs) -> tuple[str, GenerateStats]:
        start = time.perf_counter()
        pieces = []
        n = 0
        for piece, _, n in self.stream_generate(prompt, **kwargs):
            pieces.append(piece)
        elapsed = time.perf_counter() - start
        ids = self.encode(prompt)
        return "".join(pieces), GenerateStats(len(ids), n, elapsed)
