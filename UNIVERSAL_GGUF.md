# PyLlama Universal GGUF Runtime

PyLlama is a local, Python/NumPy inference runtime inspired by the architecture of llama.cpp, but it does not invoke llama.cpp, llama-cpp-python, Hugging Face inference, or a Hugging Face download API.

## Local model workflow

1. Manually download any GGUF model you want.
2. Put the `.gguf` file (or multi-shard GGUF set) under `models/`.
3. Run `run.bat`.
4. PyLlama reads `general.architecture` from GGUF and selects a registered model graph.
5. GGUF tensor types are decoded by the quantization registry.

## Supported GGUF numeric types in this release

- F32, F16, BF16
- Q4_0, Q4_1
- Q5_0, Q5_1
- Q8_0
- Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K

Q4_K_M/Q5_K_M are *file-level quantization mixes*, so individual tensors can use different GGML quantization types. The runtime dispatches each tensor by its own type id.

## Supported decoder graph families in this release

The Llama-like graph is selected for architectures whose GGUF tensor contract matches it, including:

- llama
- qwen
- qwen2 / Qwen2.5
- qwen3
- mistral
- stablelm
- internlm2
- minicpm / minicpm3

Other GGUF architectures are detected but are deliberately rejected until their graph is implemented. This is intentional: GGUF is a container format and does not imply that every model has the same computation graph. Running a different architecture through the wrong graph can produce plausible-looking garbage rather than an actionable error.

## Multi-shard GGUF

Files named like:

`Model-00001-of-00004.gguf`

are automatically discovered and merged into one logical model when all shards are present.

## Important distinction

"All GGUF models" cannot be implemented by a single generic Transformer class. The current llama.cpp architecture registry contains many unrelated graphs (decoder-only Transformers, MoE, recurrent/state-space, encoders, multimodal models, etc.). PyLlama therefore separates:

- GGUF container parsing
- tensor/quantization decoding
- architecture registry
- model graph implementation
- tokenizer/chat formatting
- generation/sampling
- CLI and HTTP API server

New architectures can be added without changing the GGUF loader or CLI.
