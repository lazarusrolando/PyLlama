from __future__ import annotations
import numpy as np


def rms_norm(x, weight, eps: float):
    xp = np if isinstance(x, np.ndarray) else __import__("cupy").get_array_module(x)
    inv = 1.0 / xp.sqrt(xp.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x * inv) * weight


def softmax(x):
    xp = np if isinstance(x, np.ndarray) else __import__("cupy").get_array_module(x)
    x = x - xp.max(x, axis=-1, keepdims=True)
    e = xp.exp(x)
    return e / xp.sum(e, axis=-1, keepdims=True)


def silu(x):
    xp = np if isinstance(x, np.ndarray) else __import__("cupy").get_array_module(x)
    return x / (1.0 + xp.exp(-x))


def rope(x, position: int, theta: float, rotary_dim: int | None = None):
    xp = np if isinstance(x, np.ndarray) else __import__("cupy").get_array_module(x)
    dim = x.shape[-1] if rotary_dim is None else rotary_dim
    if dim <= 0 or dim % 2:
        raise ValueError(f"rotary dimension must be positive and even, got {dim}")
    half = dim // 2
    inv_freq = 1.0 / (theta ** (xp.arange(0, half, dtype=xp.float32) * (2.0 / dim)))
    angles = position * inv_freq
    cos = xp.cos(angles).astype(xp.float32)
    sin = xp.sin(angles).astype(xp.float32)
    y = x.astype(xp.float32, copy=True)
    a = y[..., :half].copy()
    b = y[..., half:dim].copy()
    y[..., :half] = a * cos - b * sin
    y[..., half:dim] = a * sin + b * cos
    return y
