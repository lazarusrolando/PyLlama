import numpy as np
from pyllama.engine.quant import dequantize


def test_q4_0_zero_block():
    raw = bytearray(18)
    raw[0:2] = np.float16(0.5).tobytes()
    raw[2:] = bytes([0x88] * 16)
    assert np.allclose(dequantize(raw, 2, 32), 0)


def test_q8_0_zero_block():
    raw = bytearray(34)
    raw[0:2] = np.float16(0.25).tobytes()
    assert np.allclose(dequantize(raw, 8, 32), 0)


def test_float_types():
    f32 = np.arange(8, dtype=np.float32)
    assert np.allclose(dequantize(f32.tobytes(), 0, 8), f32)
    f16 = f32.astype(np.float16)
    assert np.allclose(dequantize(f16.tobytes(), 1, 8), f32)

def test_q3_k_and_q8_k_smoke():
    import numpy as np
    from pyllama.engine.quant import dequantize
    rng = np.random.default_rng(123)
    q3 = rng.integers(0, 256, size=110, dtype=np.uint8).tobytes()
    q8 = rng.integers(0, 256, size=292, dtype=np.uint8).tobytes()
    a = dequantize(q3, 11, 256)
    b = dequantize(q8, 15, 256)
    assert a.shape == (256,) and np.isfinite(a).all()
    assert b.shape == (256,) and np.isfinite(b).all()
