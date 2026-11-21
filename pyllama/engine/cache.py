from __future__ import annotations


class KVCache:
    def __init__(self, backend, n_layers: int, n_kv_heads: int, head_dim: int, max_context: int, dtype="float32"):
        self.backend = backend
        self.k = [backend.zeros((n_kv_heads, max_context, head_dim), dtype=dtype) for _ in range(n_layers)]
        self.v = [backend.zeros((n_kv_heads, max_context, head_dim), dtype=dtype) for _ in range(n_layers)]
        self.max_context = max_context
        self.position = 0

    def reset(self) -> None:
        self.position = 0

    def write(self, layer: int, k_new, v_new, position: int) -> None:
        if position >= self.max_context:
            raise ValueError(f"Context limit exceeded: {position + 1} > {self.max_context}")
        self.k[layer][:, position, :] = k_new
        self.v[layer][:, position, :] = v_new
        self.position = max(self.position, position + 1)

    def keys(self, layer: int, position: int):
        return self.k[layer][:, : position + 1, :]

    def values(self, layer: int, position: int):
        return self.v[layer][:, : position + 1, :]
