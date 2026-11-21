from __future__ import annotations

import struct
import numpy as np


def _f16(raw: bytes | memoryview, offset: int = 0) -> float:
    return float(np.frombuffer(raw, dtype="<f2", count=1, offset=offset)[0])


def _q4_0(raw: bytes | memoryview, n: int) -> np.ndarray:
    out = np.empty(n, dtype=np.float32)
    bs = 18
    for b, base in enumerate(range(0, len(raw), bs)):
        d = _f16(raw, base)
        q = np.frombuffer(raw, dtype=np.uint8, count=16, offset=base + 2)
        vals = np.empty(32, dtype=np.float32)
        vals[:16] = (q & 0x0F).astype(np.float32) - 8.0
        vals[16:] = (q >> 4).astype(np.float32) - 8.0
        out[b * 32:(b + 1) * 32] = vals * d
    return out


def _q4_1(raw, n):
    out = np.empty(n, dtype=np.float32)
    bs = 20
    for b, base in enumerate(range(0, len(raw), bs)):
        d = _f16(raw, base)
        m = _f16(raw, base + 2)
        q = np.frombuffer(raw, dtype=np.uint8, count=16, offset=base + 4)
        vals = np.empty(32, dtype=np.float32)
        vals[:16] = q & 0x0F
        vals[16:] = q >> 4
        out[b * 32:(b + 1) * 32] = vals * d + m
    return out


def _q5_0(raw, n):
    out = np.empty(n, dtype=np.float32)
    bs = 22
    for b, base in enumerate(range(0, len(raw), bs)):
        d = _f16(raw, base)
        qh = int.from_bytes(bytes(raw[base + 2:base + 6]), "little")
        q = np.frombuffer(raw, dtype=np.uint8, count=16, offset=base + 6)
        vals = np.empty(32, dtype=np.float32)
        lo0, lo1 = q & 0x0F, q >> 4
        hi0 = np.array([(qh >> i) & 1 for i in range(16)], dtype=np.uint8)
        hi1 = np.array([(qh >> (16 + i)) & 1 for i in range(16)], dtype=np.uint8)
        vals[:16] = (lo0 | (hi0 << 4)).astype(np.float32) - 16.0
        vals[16:] = (lo1 | (hi1 << 4)).astype(np.float32) - 16.0
        out[b * 32:(b + 1) * 32] = vals * d
    return out


def _q5_1(raw, n):
    out = np.empty(n, dtype=np.float32)
    bs = 24
    for b, base in enumerate(range(0, len(raw), bs)):
        d = _f16(raw, base)
        m = _f16(raw, base + 2)
        qh = int.from_bytes(bytes(raw[base + 4:base + 8]), "little")
        q = np.frombuffer(raw, dtype=np.uint8, count=16, offset=base + 8)
        vals = np.empty(32, dtype=np.float32)
        hi0 = np.array([(qh >> i) & 1 for i in range(16)], dtype=np.uint8)
        hi1 = np.array([(qh >> (16 + i)) & 1 for i in range(16)], dtype=np.uint8)
        vals[:16] = (q & 0x0F) | (hi0 << 4)
        vals[16:] = (q >> 4) | (hi1 << 4)
        out[b * 32:(b + 1) * 32] = vals.astype(np.float32) * d + m
    return out


def _q8_0(raw, n):
    out = np.empty(n, dtype=np.float32)
    bs = 34
    for b, base in enumerate(range(0, len(raw), bs)):
        d = _f16(raw, base)
        q = np.frombuffer(raw, dtype=np.int8, count=32, offset=base + 2).astype(np.float32)
        out[b * 32:(b + 1) * 32] = q * d
    return out


def _q2_k(raw, n):
    out = np.empty(n, dtype=np.float32)
    for bi, base in enumerate(range(0, len(raw), 84)):
        scales = np.frombuffer(raw, dtype=np.uint8, count=16, offset=base)
        qs = np.frombuffer(raw, dtype=np.uint8, count=64, offset=base + 16)
        d = _f16(raw, base + 80)
        dmin = _f16(raw, base + 82)
        vals = np.empty(256, dtype=np.float32)
        pos = 0
        for g in range(16):
            sc = d * float(scales[g] & 0x0F)
            mn = dmin * float(scales[g] >> 4)
            for i in range(16):
                byte = qs[(g * 16 + i) // 4]
                q = (byte >> (2 * (i % 4))) & 3
                vals[pos] = sc * q - mn
                pos += 1
        out[bi * 256:(bi + 1) * 256] = vals
    return out


def _get_scale_min_k4(j: int, q: np.ndarray) -> tuple[int, int]:
    if j < 4:
        return int(q[j] & 63), int(q[j + 4] & 63)
    return int((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4)), int((q[j + 4] >> 4) | ((q[j] >> 6) << 4))


def _q4_k(raw, n):
    out = np.empty(n, dtype=np.float32)
    block_bytes = 144
    for bi, base in enumerate(range(0, len(raw), block_bytes)):
        d = _f16(raw, base)
        dmin = _f16(raw, base + 2)
        scales = np.frombuffer(raw, dtype=np.uint8, count=12, offset=base + 4)
        qs = np.frombuffer(raw, dtype=np.uint8, count=128, offset=base + 16)
        y = out[bi * 256:(bi + 1) * 256]
        pos = 0
        for j in range(0, 256, 64):
            q = qs[j // 2:j // 2 + 32]
            sc1, mn1 = _get_scale_min_k4((j // 64) * 2, scales)
            sc2, mn2 = _get_scale_min_k4((j // 64) * 2 + 1, scales)
            y[pos:pos + 32] = d * sc1 * (q & 0x0F).astype(np.float32) - dmin * mn1
            y[pos + 32:pos + 64] = d * sc2 * (q >> 4).astype(np.float32) - dmin * mn2
            pos += 64
    return out


def _q5_k(raw, n):
    out = np.empty(n, dtype=np.float32)
    block_bytes = 176
    for bi, base in enumerate(range(0, len(raw), block_bytes)):
        d = _f16(raw, base)
        dmin = _f16(raw, base + 2)
        scales = np.frombuffer(raw, dtype=np.uint8, count=12, offset=base + 4)
        qh = np.frombuffer(raw, dtype=np.uint8, count=32, offset=base + 16)
        ql = np.frombuffer(raw, dtype=np.uint8, count=128, offset=base + 48)
        y = out[bi * 256:(bi + 1) * 256]
        pos = 0
        u1, u2 = 1, 2
        for j in range(0, 256, 64):
            q = ql[j // 2:j // 2 + 32]
            sc1, mn1 = _get_scale_min_k4((j // 64) * 2, scales)
            sc2, mn2 = _get_scale_min_k4((j // 64) * 2 + 1, scales)
            hi1 = np.array([(qh[i] & u1) != 0 for i in range(32)], dtype=np.float32) * 16.0
            hi2 = np.array([(qh[i] & u2) != 0 for i in range(32)], dtype=np.float32) * 16.0
            y[pos:pos + 32] = d * sc1 * ((q & 0x0F).astype(np.float32) + hi1) - dmin * mn1
            y[pos + 32:pos + 64] = d * sc2 * ((q >> 4).astype(np.float32) + hi2) - dmin * mn2
            pos += 64
            u1 <<= 2
            u2 <<= 2
    return out


def _q6_k(raw: bytes | memoryview, n: int) -> np.ndarray:
    """Dequantize GGML Q6_K blocks (256 values / 210 bytes).

    The implementation mirrors the Q6_K storage layout: 128 low-bit bytes,
    64 high-bit bytes, 16 signed int8 scales, and one fp16 super-scale.
    All arithmetic that crosses the signed/unsigned boundary is performed
    after an explicit integer cast to avoid NumPy uint8 underflow/overflow.
    """
    block_bytes = 210
    if n % 256 != 0:
        raise ValueError(f"Q6_K element count must be a multiple of 256, got {n}")
    expected = (n // 256) * block_bytes
    if len(raw) < expected:
        raise ValueError(f"Q6_K buffer is too small: need {expected} bytes, got {len(raw)}")

    blocks = np.frombuffer(raw, dtype=np.uint8, count=expected).reshape(-1, block_bytes)

    ql = blocks[:, :128]
    qh = blocks[:, 128:192]
    scales = blocks[:, 192:208].view(np.int8).astype(np.float32)
    # The fp16 view has a trailing singleton dimension; flatten it so each
    # block's super-scale broadcasts across that block's sixteen group scales.
    d = blocks[:, 208:210].copy().view('<f2').astype(np.float32).reshape(-1)

    # 16 groups x 16 values.  The 6-bit value is encoded as:
    #   q = low_4_bits | (high_2_bits << 4), then interpreted as q - 32.
    # Cast before subtraction so NumPy never performs uint8 underflow.
    ql = ql.reshape(-1, 2, 1, 64) >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)
    ql = (ql & np.uint8(0x0F)).reshape(-1, 8, 32)

    qh = qh.reshape(-1, 2, 1, 32) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape(1, 1, 4, 1)
    qh = (qh & np.uint8(0x03)).reshape(-1, 8, 32)

    q = (ql.astype(np.int16) | (qh.astype(np.int16) << 4)) - 32
    q = q.reshape(-1, 16, 16).astype(np.float32)

    scale = (d[:, None, None] * scales[:, :, None])
    return (scale * q).reshape(-1)[:n].astype(np.float32, copy=False)



def _q3_k(raw: bytes | memoryview, n: int) -> np.ndarray:
    """GGML Q3_K dequantization, 256 values / 110 bytes."""
    if n % 256:
        raise ValueError(f"Q3_K element count must be a multiple of 256, got {n}")
    blocks = np.frombuffer(raw, dtype=np.uint8, count=(n // 256) * 110).reshape(-1, 110)
    hmask, rest = np.hsplit(blocks, [32])
    qs, rest = np.hsplit(rest, [64])
    scales_raw, d_raw = np.hsplit(rest, [12])
    d = d_raw.view('<f2').astype(np.float32)

    # GGML packs sixteen signed 6-bit scales into twelve bytes.
    lsc, hsc = np.hsplit(scales_raw, [8])
    lsc = lsc.reshape((-1, 1, 8)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 2, 1))
    lsc = lsc.reshape((-1, 16))
    hsc = hsc.reshape((-1, 1, 4)) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 4, 1))
    hsc = hsc.reshape((-1, 16))
    scales = (lsc & np.uint8(0x0F)) | ((hsc & np.uint8(0x03)) << np.uint8(4))
    scales = (scales.astype(np.int8) - np.int8(32)).astype(np.float32)

    ql = qs.reshape((qs.shape[0], -1, 1, 32)) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 1, 4, 1))
    ql = (ql.reshape((-1, 16, 16)) & np.uint8(3)).astype(np.int8)
    qh = hmask.reshape((hmask.shape[0], -1, 1, 32)) >> np.array(range(8), dtype=np.uint8).reshape((1, 1, 8, 1))
    qh = (qh.reshape((-1, 16, 16)) & np.uint8(1)) ^ np.uint8(1)
    q = ql - (qh.astype(np.int8) << np.int8(2))
    return (d[:, None, None] * scales[:, :, None] * q.astype(np.float32)).reshape(-1)[:n].astype(np.float32, copy=False)


def _q8_k(raw: bytes | memoryview, n: int) -> np.ndarray:
    """GGML Q8_K dequantization, 256 values / 292 bytes."""
    if n % 256:
        raise ValueError(f"Q8_K element count must be a multiple of 256, got {n}")
    blocks = np.frombuffer(raw, dtype=np.uint8, count=(n // 256) * 292).reshape(-1, 292)
    d = blocks[:, :4].view('<f4').astype(np.float32)
    qs = blocks[:, 4:260].view(np.int8).astype(np.float32)
    return (d * qs).reshape(-1)[:n].astype(np.float32, copy=False)

def dequantize(raw: bytes | memoryview, type_id: int, n_elements: int, *, dtype: str = "float32") -> np.ndarray:
    if type_id == 0:
        arr = np.frombuffer(raw, dtype="<f4", count=n_elements).copy()
    elif type_id == 1:
        arr = np.frombuffer(raw, dtype="<f2", count=n_elements).astype(np.float32)
    elif type_id == 30:
        u = np.frombuffer(raw, dtype="<u2", count=n_elements).astype(np.uint32)
        arr = (u << 16).view(np.float32)
    elif type_id == 2:
        arr = _q4_0(raw, n_elements)
    elif type_id == 3:
        arr = _q4_1(raw, n_elements)
    elif type_id == 6:
        arr = _q5_0(raw, n_elements)
    elif type_id == 7:
        arr = _q5_1(raw, n_elements)
    elif type_id == 8:
        arr = _q8_0(raw, n_elements)
    elif type_id == 10:
        arr = _q2_k(raw, n_elements)
    elif type_id == 11:
        arr = _q3_k(raw, n_elements)
    elif type_id == 12:
        arr = _q4_k(raw, n_elements)
    elif type_id == 13:
        arr = _q5_k(raw, n_elements)
    elif type_id == 14:
        arr = _q6_k(raw, n_elements)
    elif type_id == 15:
        arr = _q8_k(raw, n_elements)
    else:
        raise NotImplementedError(f"GGUF type {type_id} is not supported")
    if dtype == "float16":
        return arr.astype(np.float16)
    return arr.astype(np.float32, copy=False)
