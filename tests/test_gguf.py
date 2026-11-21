import struct
from pathlib import Path
from pyllama.formats.gguf import GGUFFile


def _s(x: str) -> bytes:
    b = x.encode(); return struct.pack('<Q', len(b)) + b


def make(path: Path):
    # One F32 vector tensor: ggml shape [4], data at 32-byte alignment.
    parts = [b'GGUF', struct.pack('<I', 3), struct.pack('<QQ', 1, 2)]
    parts.append(_s('general.architecture'))
    parts.append(struct.pack('<I', 8) + _s('llama'))
    parts.append(_s('general.alignment'))
    parts.append(struct.pack('<I', 4) + struct.pack('<I', 32))
    parts.append(_s('test'))
    parts.append(struct.pack('<I', 1))
    parts.append(struct.pack('<q', 4))
    parts.append(struct.pack('<I', 0))
    parts.append(struct.pack('<Q', 0))
    blob = b''.join(parts)
    blob += b'\0' * ((32 - len(blob) % 32) % 32)
    blob += struct.pack('<4f', 1, 2, 3, 4)
    path.write_bytes(blob)


def test_parse(tmp_path):
    path = tmp_path / 'x.gguf'; make(path)
    with GGUFFile.open(path) as g:
        assert g.metadata['general.architecture'] == 'llama'
        assert g.tensors['test'].shape == (4,)
        assert bytes(g.raw_tensor_bytes('test')) == struct.pack('<4f', 1, 2, 3, 4)
