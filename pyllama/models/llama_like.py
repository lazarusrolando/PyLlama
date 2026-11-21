from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from pyllama.engine.cache import KVCache
from pyllama.engine.device import Backend
from pyllama.engine.math import rms_norm, softmax, silu, rope
from pyllama.formats.gguf import GGUFFile
from pyllama.models.registry import get_architecture

@dataclass
class Config:
    architecture: str
    vocab_size: int
    context_length: int
    hidden_size: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    ffn_dim: int
    rope_theta: float
    rope_dim: int
    rms_eps: float
    q_norm: bool
    k_norm: bool

class LlamaLikeModel:
    """NumPy implementation of the decoder-only Llama/Qwen-style execution path."""
    def __init__(self, gguf: GGUFFile, *, weight_dtype: str = "float32", max_context: int | None = None, device: str = "auto"):
        self.gguf = gguf
        self.weight_dtype = weight_dtype
        self.backend = Backend(device)
        self.config = self._read_config()
        if max_context:
            self.config.context_length = min(self.config.context_length, max_context)
        self.cache = KVCache(self.backend, self.config.n_layers, self.config.n_kv_heads, self.config.head_dim, self.config.context_length, dtype="float32")
        self._tensor_cache: dict[str, np.ndarray] = {}

    def _m(self, suffix: str, default=None):
        key = f"{self.config.architecture}.{suffix}"
        return self.gguf.metadata.get(key, default)

    def _read_config(self) -> Config:
        spec = get_architecture(self.gguf.metadata)
        if spec.family not in {"llama_like"}:
            raise NotImplementedError(
                f"Architecture {spec.name!r} is detected, but it does not use the Llama-like graph. "
                "PyLlama will not silently run it through the wrong graph."
            )
        arch = spec.name
        hidden = int(self._raw(arch, "embedding_length", 0))
        layers = int(self._raw(arch, "block_count", 0))
        heads = int(self._raw(arch, "attention.head_count", 0))
        kv_heads = int(self._raw(arch, "attention.head_count_kv", heads))
        head_dim = int(self._raw(arch, "attention.key_length", hidden // heads if heads else 0))
        ffn = int(self._raw(arch, "feed_forward_length", 4 * hidden))
        vocab_meta = self.gguf.metadata.get("tokenizer.ggml.vocab_size")
        vocab_tokens = self.gguf.metadata.get("tokenizer.ggml.tokens")
        vocab = int(vocab_meta or (len(vocab_tokens) if isinstance(vocab_tokens, list) else 0) or self._raw(arch, "vocab_size", 0))
        ctx = int(self._raw(arch, "context_length", 4096))
        theta = float(self._raw(arch, "rope.freq_base", 10000.0))
        rope_dim = int(self._raw(arch, "rope.dimension_count", head_dim))
        eps = float(self._raw(arch, "attention.layer_norm_rms_epsilon", 1e-6))
        q_norm = any(f"blk.0.attn_q_norm.weight" == n for n in self.gguf.tensors)
        k_norm = any(f"blk.0.attn_k_norm.weight" == n for n in self.gguf.tensors)
        missing = []
        if not hidden: missing.append(f"{arch}.embedding_length")
        if not layers: missing.append(f"{arch}.block_count")
        if not heads: missing.append(f"{arch}.attention.head_count")
        if not kv_heads: missing.append(f"{arch}.attention.head_count_kv")
        if not ffn: missing.append(f"{arch}.feed_forward_length")
        if not vocab: missing.append("tokenizer.ggml.tokens/vocab_size")
        if missing:
            raise ValueError("GGUF metadata is missing required decoder fields: " + ", ".join(missing))
        return Config(arch, vocab, ctx, hidden, layers, heads, kv_heads, head_dim, ffn, theta, rope_dim, eps, q_norm, k_norm)

    def _raw(self, arch: str, suffix: str, default=None):
        return self.gguf.metadata.get(f"{arch}.{suffix}", default)

    def _find(self, *names: str) -> str:
        for n in names:
            if n in self.gguf.tensors:
                return n
        raise KeyError(f"None of these tensors were found: {names}")

    def _find_optional(self, *names: str) -> str | None:
        for n in names:
            if n in self.gguf.tensors:
                return n
        return None

    def weight(self, name: str) -> np.ndarray:
        if name not in self._tensor_cache:
            info = self.gguf.tensors[name]
            raw = self.gguf.raw_tensor_bytes(name)
            from pyllama.engine.quant import dequantize
            arr = dequantize(raw, info.type_id, info.n_elements, dtype=self.weight_dtype)
            # Stored GGUF dimensions are reversed relative to normal NumPy / Transformer dimensions.
            arr = arr.reshape(info.shape)
            self._tensor_cache[name] = self.backend.asarray(arr)
        return self._tensor_cache[name]

    def clear_weights(self) -> None:
        self._tensor_cache.clear()

    def reset(self) -> None:
        self.cache.reset()

    def _rms(self, x, name):
        return rms_norm(x, self.weight(name), self.config.rms_eps)

    def forward_token(self, token_id: int, position: int) -> np.ndarray:
        if token_id < 0 or token_id >= self.config.vocab_size:
            raise ValueError(f"Token id {token_id} outside vocabulary")
        x = self.weight(self._find("token_embd.weight", "model.embed_tokens.weight"))[token_id].astype(self.backend.xp.float32)
        c = self.config
        rep = c.n_heads // c.n_kv_heads
        scale = 1.0 / np.sqrt(c.head_dim)

        for layer in range(c.n_layers):
            p = f"blk.{layer}."
            x_in = x
            x_norm = self._rms(x, self._find(p + "attn_norm.weight", p + "input_layernorm.weight"))

            q = x_norm @ self.weight(self._find(p + "attn_q.weight", p + "self_attn.q_proj.weight")).T
            k = x_norm @ self.weight(self._find(p + "attn_k.weight", p + "self_attn.k_proj.weight")).T
            v = x_norm @ self.weight(self._find(p + "attn_v.weight", p + "self_attn.v_proj.weight")).T
            # Qwen2-family checkpoints have learned Q/K/V biases.
            for proj, var in (("q", "q"), ("k", "k"), ("v", "v")):
                bias_name = self._find_optional(
                    p + f"attn_{proj}.bias",
                    p + f"self_attn.{proj}_proj.bias",
                )
                if bias_name is not None:
                    if var == "q": q = q + self.weight(bias_name)
                    elif var == "k": k = k + self.weight(bias_name)
                    else: v = v + self.weight(bias_name)

            q = q.reshape(c.n_heads, c.head_dim)
            k = k.reshape(c.n_kv_heads, c.head_dim)
            v = v.reshape(c.n_kv_heads, c.head_dim)

            qn = p + "attn_q_norm.weight"
            kn = p + "attn_k_norm.weight"
            if qn in self.gguf.tensors:
                q = rms_norm(q, self.weight(qn), c.rms_eps)
            if kn in self.gguf.tensors:
                k = rms_norm(k, self.weight(kn), c.rms_eps)

            q = rope(q, position, c.rope_theta, c.rope_dim)
            k = rope(k, position, c.rope_theta, c.rope_dim)
            self.cache.write(layer, k, v, position)

            keys = self.cache.keys(layer, position)
            vals = self.cache.values(layer, position)
            attn_out = np.empty((c.n_heads, c.head_dim), dtype=np.float32)
            for h in range(c.n_heads):
                kvh = h // rep
                scores = (keys[kvh] @ q[h]) * scale
                probs = softmax(scores)
                attn_out[h] = probs @ vals[kvh]
            attn = attn_out.reshape(c.hidden_size)
            o = self.weight(self._find(p + "attn_output.weight", p + "self_attn.o_proj.weight"))
            x = x_in + attn @ o.T

            x_in = x
            x_norm = self._rms(x, self._find(p + "attn_norm_2.weight", p + "post_attention_layernorm.weight", p + "ffn_norm.weight"))
            gate = x_norm @ self.weight(self._find(p + "ffn_gate.weight", p + "mlp.gate_proj.weight")).T
            up = x_norm @ self.weight(self._find(p + "ffn_up.weight", p + "mlp.up_proj.weight")).T
            hidden = silu(gate) * up
            down = self.weight(self._find(p + "ffn_down.weight", p + "mlp.down_proj.weight"))
            x = x_in + hidden @ down.T

        x = self._rms(x, self._find("output_norm.weight", "model.norm.weight"))
        out_name = "output.weight" if "output.weight" in self.gguf.tensors else "lm_head.weight"
        if out_name in self.gguf.tensors:
            logits = x @ self.weight(out_name).T
        else:
            emb = self.weight(self._find("token_embd.weight", "model.embed_tokens.weight"))
            logits = x @ emb.T
        return logits.astype(self.backend.xp.float32, copy=False)
