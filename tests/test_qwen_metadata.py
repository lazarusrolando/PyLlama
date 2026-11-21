import struct
from pathlib import Path

from pyllama.formats.gguf import GGUFFile
from pyllama.models.llama_like import LlamaLikeModel
from pyllama.tokenizer.bpe import ByteBPE


def _s(value: str) -> bytes:
    raw = value.encode()
    return struct.pack('<Q', len(raw)) + raw


def _u32(value: int) -> bytes:
    return struct.pack('<I I', 4, value)


def _string(value: str) -> bytes:
    return struct.pack('<I', 8) + _s(value)


def _array_string(values: list[str]) -> bytes:
    return struct.pack('<I I Q', 9, 8, len(values)) + b''.join(_s(v) for v in values)


def _write_minimal_qwen2(path: Path) -> None:
    metadata = [
        ('general.architecture', _string('qwen2')),
        ('qwen2.embedding_length', _u32(16)),
        ('qwen2.block_count', _u32(2)),
        ('qwen2.attention.head_count', _u32(4)),
        ('qwen2.attention.head_count_kv', _u32(2)),
        ('qwen2.feed_forward_length', _u32(32)),
        ('qwen2.context_length', _u32(128)),
        ('tokenizer.ggml.tokens', _array_string(['a', 'b', '<|im_start|>', '<|im_end|>'])),
        ('tokenizer.ggml.merges', _array_string([])),
        ('tokenizer.ggml.eos_token_id', _u32(3)),
    ]
    tensor_info = _s('token_embd.weight') + struct.pack('<I q I Q', 1, 4, 0, 0)
    header = b'GGUF' + struct.pack('<IQQ', 3, 1, len(metadata))
    metadata_bytes = b''.join(_s(k) + v for k, v in metadata)
    base = len(header) + len(metadata_bytes) + len(tensor_info)
    aligned = (base + 31) // 32 * 32
    payload = struct.pack('<4f', 1.0, 2.0, 3.0, 4.0)
    path.write_bytes(header + metadata_bytes + tensor_info + b'\0' * (aligned - base) + payload)


def test_qwen2_vocab_fallback_and_embedded_tokenizer(tmp_path):
    path = tmp_path / 'qwen2.gguf'
    _write_minimal_qwen2(path)
    with GGUFFile.open(path) as gguf:
        model = LlamaLikeModel(gguf)
        assert model.config.vocab_size == 4
        tokenizer = ByteBPE.from_gguf_metadata(gguf.metadata)
        assert tokenizer.encode('<|im_start|>a<|im_end|>') == [2, 0, 3]
