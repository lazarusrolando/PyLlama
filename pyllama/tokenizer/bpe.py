from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

try:
    import regex as regex_mod
except ImportError:  # pragma: no cover
    regex_mod = None

class ByteBPE:
    """Small GPT-2/Qwen-style byte-level BPE tokenizer.

    It can load either tokenizer.json or tokenizer data embedded directly in GGUF.
    """

    QWEN2_PATTERN = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    def __init__(self, vocab: dict[str, int], merges: list[tuple[str, str]], special_tokens: dict[str, int] | None = None, pattern: str | None = None):
        self.vocab = vocab
        self.id_to_token = {int(v): k for k, v in vocab.items()}
        self.merges = {tuple(m): i for i, m in enumerate(merges)}
        self.special_tokens = special_tokens or {}
        self.pattern = pattern
        self.byte_encoder = self._bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

    @classmethod
    def from_gguf_metadata(cls, metadata: dict) -> "ByteBPE":
        tokens = metadata.get("tokenizer.ggml.tokens")
        merges_raw = metadata.get("tokenizer.ggml.merges", [])
        if not isinstance(tokens, list) or not tokens:
            raise ValueError("GGUF does not contain tokenizer.ggml.tokens")

        vocab = {str(token): i for i, token in enumerate(tokens)}
        merges = [tuple(m.split(" ")) if isinstance(m, str) else tuple(m) for m in merges_raw]
        specials: dict[str, int] = {}

        for key in (
            "tokenizer.ggml.bos_token_id",
            "tokenizer.ggml.eos_token_id",
            "tokenizer.ggml.unk_token_id",
            "tokenizer.ggml.pad_token_id",
        ):
            token_id = metadata.get(key)
            if token_id is not None:
                idx = int(token_id)
                if 0 <= idx < len(tokens):
                    specials[str(tokens[idx])] = idx

        # Qwen chat/control tokens are embedded in the GGUF vocabulary.
        for idx, token in enumerate(tokens):
            if isinstance(token, str) and token.startswith("<|") and token.endswith("|>"):
                specials[token] = idx

        return cls(vocab, merges, specials, pattern=cls.QWEN2_PATTERN)

    @classmethod
    def from_json(cls, path: str | Path) -> "ByteBPE":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        model = obj.get("model", {})
        vocab = {str(k): int(v) for k, v in model.get("vocab", {}).items()}
        merges_raw = model.get("merges", [])
        merges = [tuple(m.split(" ")) if isinstance(m, str) else tuple(m) for m in merges_raw]
        specials = {str(x["content"]): int(x["id"]) for x in obj.get("added_tokens", []) if x.get("special")}
        pattern = None
        pretok = obj.get("pre_tokenizer") or {}
        if isinstance(pretok, dict):
            pattern = pretok.get("pattern") or pretok.get("regex")
        return cls(vocab, merges, specials, pattern)

    @staticmethod
    def _bytes_to_unicode() -> dict[int, str]:
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        return dict(zip(bs, map(chr, cs)))

    def _bpe_piece(self, piece: str) -> list[int]:
        # Convert raw UTF-8 bytes through GPT-2/Qwen byte-unicode mapping.
        symbols = [self.byte_encoder[b] for b in piece.encode("utf-8")]
        if not symbols:
            return []
        while len(symbols) > 1:
            ranked = min(
                ((self.merges.get((symbols[i], symbols[i + 1]), 10**9), i) for i in range(len(symbols) - 1)),
                key=lambda x: x[0],
            )
            rank, idx = ranked
            if rank == 10**9:
                break
            symbols[idx:idx + 2] = [symbols[idx] + symbols[idx + 1]]
        ids: list[int] = []
        for s in symbols:
            if s in self.vocab:
                ids.append(self.vocab[s])
            else:
                raise KeyError(f"Tokenizer vocabulary has no token for {s!r}")
        return ids

    def encode(self, text: str, *, add_bos: bool = False) -> list[int]:
        if not text:
            return [self.special_tokens["<|endoftext|>"]] if add_bos and "<|endoftext|>" in self.special_tokens else []
        out: list[int] = []

        def normal_parts(segment: str) -> Iterable[str]:
            if self.pattern and regex_mod is not None:
                return regex_mod.findall(self.pattern, segment)
            return re.findall(r"\s+|[^\s]+", segment)

        # Added/special tokens must be recognized before normal BPE splitting.
        if self.special_tokens:
            import re as _re
            special_pattern = "|".join(_re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True))
            chunks = _re.split(f"({special_pattern})", text) if special_pattern else [text]
            for chunk in chunks:
                if not chunk:
                    continue
                if chunk in self.special_tokens:
                    out.append(self.special_tokens[chunk])
                else:
                    for piece in normal_parts(chunk):
                        out.extend(self._bpe_piece(piece))
        else:
            for piece in normal_parts(text):
                out.extend(self._bpe_piece(piece))
        if add_bos:
            bos = self.special_tokens.get("<|endoftext|>")
            if bos is not None:
                out.insert(0, bos)
        return out

    def decode(self, ids: Iterable[int]) -> str:
        text = "".join(self.id_to_token.get(int(i), "") for i in ids)
        bs = bytearray()
        for ch in text:
            if ch in self.byte_decoder:
                bs.append(self.byte_decoder[ch])
            else:
                bs.extend(ch.encode("utf-8"))
        return bytes(bs).decode("utf-8", errors="replace")

    def apply_chat_template(self, messages: list[dict[str, str]], *, add_generation_prompt: bool = True) -> str:
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)
