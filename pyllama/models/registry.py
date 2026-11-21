from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Type

@dataclass(frozen=True)
class ArchitectureSpec:
    name: str
    family: str
    supported: bool
    description: str

# GGUF's architecture string is authoritative.  Several architectures share
# the same decoder graph, while others require a distinct graph.
ARCHITECTURES = {
    # Llama-family decoder graph
    "llama": ArchitectureSpec("llama", "llama_like", True, "LLaMA/LLaMA2/LLaMA3 style decoder"),
    "mistral": ArchitectureSpec("mistral", "llama_like", True, "Mistral-style decoder"),
    "mixtral": ArchitectureSpec("mixtral", "moe", False, "Mixture-of-Experts decoder"),
    "qwen2": ArchitectureSpec("qwen2", "llama_like", True, "Qwen2/Qwen2.5 decoder"),
    "qwen": ArchitectureSpec("qwen", "llama_like", True, "Qwen1 decoder"),
    "qwen3": ArchitectureSpec("qwen3", "llama_like", True, "Qwen3 decoder"),
    "gemma": ArchitectureSpec("gemma", "gemma", True, "Gemma decoder"),
    "gemma2": ArchitectureSpec("gemma2", "gemma", True, "Gemma 2 decoder"),
    "gemma3": ArchitectureSpec("gemma3", "gemma", True, "Gemma 3 text decoder"),
    "phi2": ArchitectureSpec("phi2", "phi", False, "Phi-2 decoder"),
    "phi3": ArchitectureSpec("phi3", "phi", True, "Phi-3/Phi-3.5 decoder"),
    "stablelm": ArchitectureSpec("stablelm", "llama_like", True, "StableLM decoder"),
    "internlm2": ArchitectureSpec("internlm2", "llama_like", True, "InternLM2 decoder"),
    "minicpm": ArchitectureSpec("minicpm", "llama_like", True, "MiniCPM decoder"),
    "minicpm3": ArchitectureSpec("minicpm3", "llama_like", True, "MiniCPM3 decoder"),
    "starcoder2": ArchitectureSpec("starcoder2", "gpt_neox_like", False, "StarCoder2 decoder"),
    "gpt2": ArchitectureSpec("gpt2", "gpt2", False, "GPT-2 decoder"),
    "gptj": ArchitectureSpec("gptj", "gptj", False, "GPT-J decoder"),
    "gptneox": ArchitectureSpec("gptneox", "gpt_neox_like", False, "GPT-NeoX decoder"),
    "falcon": ArchitectureSpec("falcon", "falcon", False, "Falcon decoder"),
    "mpt": ArchitectureSpec("mpt", "mpt", False, "MPT decoder"),
    "bloom": ArchitectureSpec("bloom", "bloom", False, "BLOOM decoder"),
    "bert": ArchitectureSpec("bert", "encoder", False, "BERT encoder"),
    "modern-bert": ArchitectureSpec("modern-bert", "encoder", False, "Modern BERT encoder"),
    "mamba": ArchitectureSpec("mamba", "mamba", False, "Mamba state-space model"),
}


def get_architecture(metadata: dict) -> ArchitectureSpec:
    arch = str(metadata.get("general.architecture", "unknown"))
    return ARCHITECTURES.get(arch, ArchitectureSpec(arch, "unknown", False, "Unknown GGUF architecture"))


def supported_architectures() -> list[ArchitectureSpec]:
    return [x for x in ARCHITECTURES.values() if x.supported]
