from __future__ import annotations
import numpy as np

from .math import softmax

class Sampler:
    def __init__(self, temperature: float = 0.7, top_k: int = 40, top_p: float = 0.9, repeat_penalty: float = 1.1, min_p: float = 0.0, seed: int | None = None):
        if temperature < 0 or top_k < 0 or not 0 < top_p <= 1 or repeat_penalty <= 0 or min_p < 0:
            raise ValueError("Invalid sampling parameters")
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repeat_penalty = repeat_penalty
        self.min_p = min_p
        self.rng = np.random.default_rng(seed)

    def sample(self, logits, recent_tokens: list[int]) -> int:
        try:
            import cupy as cp
            if isinstance(logits, cp.ndarray):
                logits = cp.asnumpy(logits)
        except ImportError:
            pass
        x = np.asarray(logits, dtype=np.float64).copy()
        if self.repeat_penalty != 1.0:
            for tid in set(recent_tokens):
                if 0 <= tid < x.size:
                    x[tid] = x[tid] / self.repeat_penalty if x[tid] > 0 else x[tid] * self.repeat_penalty
        if self.temperature == 0:
            return int(np.argmax(x))
        x /= self.temperature
        if self.top_k and self.top_k < x.size:
            idx = np.argpartition(x, -self.top_k)[-self.top_k:]
            mask = np.ones_like(x, dtype=bool)
            mask[idx] = False
            x[mask] = -np.inf
        probs = softmax(x.astype(np.float32)).astype(np.float64)
        if self.min_p > 0:
            cutoff = float(np.max(probs)) * self.min_p
            probs[probs < cutoff] = 0
            total = probs.sum()
            if total:
                probs /= total
        if self.top_p < 1.0:
            order = np.argsort(probs)[::-1]
            sorted_p = probs[order]
            cumulative = np.cumsum(sorted_p)
            remove = cumulative > self.top_p
            if remove.size:
                remove[0] = False
            probs[order[remove]] = 0
            total = probs.sum()
            if total:
                probs /= total
        return int(self.rng.choice(x.size, p=probs / probs.sum()))
