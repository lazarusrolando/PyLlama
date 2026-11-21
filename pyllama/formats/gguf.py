from __future__ import annotations

import mmap
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
DEFAULT_ALIGNMENT = 32

# GGUF KV value types.
K_U8 = 0
K_I8 = 1
K_U16 = 2
K_I16 = 3
K_U32 = 4
K_I32 = 5
K_F32 = 6
K_BOOL = 7
K_STRING = 8
K_ARRAY = 9
K_U64 = 10
K_I64 = 11
K_F64 = 12

# ggml type ids used by current GGUF files.
TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
    15: "Q8_K", 20: "IQ4_NL", 23: "IQ4_XS", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0",
    39: "MXFP4", 40: "NVFP4", 41: "Q1_0", 42: "Q2_0",
}

@dataclass(frozen=True)
class TensorInfo:
    name: str
    ggml_shape: tuple[int, ...]
    shape: tuple[int, ...]
    type_id: int
    type_name: str
    offset: int
    shard_index: int = 0

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

@dataclass
class GGUFFile:
    path: Path
    version: int
    tensors: dict[str, TensorInfo]
    metadata: dict[str, Any]
    alignment: int
    tensor_data_start: int
    shards: tuple["GGUFFile", ...] = ()

    def __post_init__(self) -> None:
        self._fh = self.path.open("rb")
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)

    @classmethod
    def open(cls, path: str | Path) -> "GGUFFile":
        path = Path(path)
        first = cls._open_single(path, 0)
        # llama.cpp-compatible multi-shard layout: foo-00001-of-000NN.gguf.
        import re
        m = re.match(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", path.name, re.I)
        if not m or int(m.group(2)) != 1:
            return first
        prefix, _, total_s = m.groups()
        total = int(total_s)
        siblings = []
        for i in range(2, total + 1):
            candidate = path.with_name(f"{prefix}-{i:05d}-of-{total:05d}.gguf")
            if not candidate.exists():
                first.close()
                raise FileNotFoundError(f"Missing GGUF shard: {candidate}")
            siblings.append(cls._open_single(candidate, i - 1))
        # Metadata is normally repeated identically in every shard. Merge tensors.
        for shard in siblings:
            for name, info in shard.tensors.items():
                if name in first.tensors:
                    raise ValueError(f"Duplicate tensor across GGUF shards: {name}")
                first.tensors[name] = info
        first.shards = tuple([first, *siblings])
        return first

    @classmethod
    def _open_single(cls, path: Path, shard_index: int) -> "GGUFFile":
        with path.open("rb") as f:
            magic = f.read(4)
            if magic != GGUF_MAGIC:
                raise ValueError(f"Not a GGUF file: {path}")
            version = struct.unpack("<I", f.read(4))[0]
            if version not in (2, 3):
                raise ValueError(f"Unsupported GGUF version: {version}")
            tensor_count, kv_count = struct.unpack("<QQ", f.read(16))
            metadata: dict[str, Any] = {}
            for _ in range(kv_count):
                key = _read_string(f)
                metadata[key] = _read_value(f)
            tensors: dict[str, TensorInfo] = {}
            for _ in range(tensor_count):
                name = _read_string(f)
                n_dims = struct.unpack("<I", f.read(4))[0]
                ggml_shape = tuple(struct.unpack("<q", f.read(8))[0] for _ in range(n_dims))
                type_id = struct.unpack("<I", f.read(4))[0]
                offset = struct.unpack("<Q", f.read(8))[0]
                shape = tuple(reversed(ggml_shape))
                tensors[name] = TensorInfo(name, ggml_shape, shape, type_id, TYPE_NAMES.get(type_id, f"TYPE_{type_id}"), offset, shard_index)
            alignment = int(metadata.get("general.alignment", DEFAULT_ALIGNMENT))
            tensor_data_start = _align(f.tell(), alignment)
        return cls(path, version, tensors, metadata, alignment, tensor_data_start)

    def close(self) -> None:
        def close_mapping(shard: "GGUFFile") -> None:
            try:
                shard._mm.close()
            except BufferError:
                # An active exception traceback can temporarily retain NumPy
                # views into the mapping. Keep the mapping alive until those
                # views are released, and let the original error propagate.
                pass
            finally:
                shard._fh.close()

        if self.shards:
            # close children first; self is included as shard 0
            for shard in reversed(self.shards):
                close_mapping(shard)
            self.shards = ()
            return
        close_mapping(self)

    def __enter__(self) -> "GGUFFile":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def raw_tensor_bytes(self, name: str) -> memoryview:
        info = self.tensors[name]
        if self.shards:
            source = self.shards[info.shard_index]
        else:
            source = self
        start = source.tensor_data_start + info.offset
        end = start + tensor_nbytes(info.n_elements, info.type_id)
        if end > len(source._mm):
            raise ValueError(f"Tensor {name} exceeds file bounds")
        return memoryview(source._mm)[start:end]

    def describe(self) -> dict[str, Any]:
        arch = self.metadata.get("general.architecture")
        return {
            "path": str(self.path),
            "version": self.version,
            "architecture": arch,
            "alignment": self.alignment,
            "metadata": len(self.metadata),
            "tensors": len(self.tensors),
            "file_size_mb": round(self.path.stat().st_size / 1048576, 2),
            "shards": len(self.shards) if self.shards else 1,
        }


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _read_string(f) -> str:
    n = struct.unpack("<Q", f.read(8))[0]
    raw = f.read(n)
    return raw.decode("utf-8")


def _read_value(f) -> Any:
    typ = struct.unpack("<I", f.read(4))[0]
    if typ == K_U8:
        return struct.unpack("<B", f.read(1))[0]
    if typ == K_I8:
        return struct.unpack("<b", f.read(1))[0]
    if typ == K_U16:
        return struct.unpack("<H", f.read(2))[0]
    if typ == K_I16:
        return struct.unpack("<h", f.read(2))[0]
    if typ == K_U32:
        return struct.unpack("<I", f.read(4))[0]
    if typ == K_I32:
        return struct.unpack("<i", f.read(4))[0]
    if typ == K_F32:
        return struct.unpack("<f", f.read(4))[0]
    if typ == K_BOOL:
        return bool(struct.unpack("<b", f.read(1))[0])
    if typ == K_STRING:
        return _read_string(f)
    if typ == K_ARRAY:
        elem_type = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        return [_read_typed_value(f, elem_type) for _ in range(n)]
    if typ == K_U64:
        return struct.unpack("<Q", f.read(8))[0]
    if typ == K_I64:
        return struct.unpack("<q", f.read(8))[0]
    if typ == K_F64:
        return struct.unpack("<d", f.read(8))[0]
    raise ValueError(f"Unsupported GGUF metadata type {typ}")


def _read_typed_value(f, typ: int) -> Any:
    if typ == K_U8: return struct.unpack("<B", f.read(1))[0]
    if typ == K_I8: return struct.unpack("<b", f.read(1))[0]
    if typ == K_U16: return struct.unpack("<H", f.read(2))[0]
    if typ == K_I16: return struct.unpack("<h", f.read(2))[0]
    if typ == K_U32: return struct.unpack("<I", f.read(4))[0]
    if typ == K_I32: return struct.unpack("<i", f.read(4))[0]
    if typ == K_F32: return struct.unpack("<f", f.read(4))[0]
    if typ == K_BOOL: return bool(struct.unpack("<b", f.read(1))[0])
    if typ == K_STRING: return _read_string(f)
    if typ == K_U64: return struct.unpack("<Q", f.read(8))[0]
    if typ == K_I64: return struct.unpack("<q", f.read(8))[0]
    if typ == K_F64: return struct.unpack("<d", f.read(8))[0]
    raise ValueError(f"Unsupported array element metadata type {typ}")


def tensor_nbytes(n_elements: int, type_id: int) -> int:
    specs = {
        0: (1, 4), 1: (1, 2), 30: (1, 2),
        2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34),
        10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210), 15: (256, 292),
    }
    if type_id not in specs:
        raise NotImplementedError(f"GGUF tensor type {type_id} ({TYPE_NAMES.get(type_id)}) is not supported by PyLlama yet")
    block, size = specs[type_id]
    if n_elements % block:
        raise ValueError(f"Element count {n_elements} is not divisible by block size {block} for type {type_id}")
    return (n_elements // block) * size
