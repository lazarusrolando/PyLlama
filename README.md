# PyLlama

PyLlama is a local GGUF inference runtime implemented in Python/NumPy. It follows the **local model workflow** you asked for:

```text
YOU download the model manually
          |
          v
       models\\
          |
          v
      GGUF reader
          |
          v
    quantized tensors
          |
          v
   Python/NumPy runtime
       +------+------+
       |             |
      CLI          HTTP API
```

## Important: no Hugging Face API

PyLlama does **not** use `huggingface_hub`, `HfApi`, `hf_hub_download`, or any Hugging Face inference API.

PyLlama does not download models.

You download a compatible `.gguf` file yourself and copy it into:

```text
models\\
```

The runtime only reads local files from that directory (or an explicit local path).

## Windows

Double-click:

```text
run.bat
```

The launcher creates the virtual environment and installs only the local runtime dependencies. It then provides:

```text
[1] Open models folder
[2] List installed models
[3] Inspect GGUF model
[4] Run CLI chat
[5] Start API server
[6] Generate from prompt
[7] Exit
```

### Manual model installation

Download a GGUF model using your preferred browser/tool and place it in `models\\`.

Example:

```text
PyLlama\\
└── models\\
    └── Qwen3-0.6B-Q4_K_M.gguf
```

Subfolders are also supported:

```text
models\\Qwen3-0.6B\\Qwen3-0.6B-Q4_K_M.gguf
```

### List

```powershell
.venv\\Scripts\\python.exe -m pyllama list
```

### Inspect

```powershell
.venv\\Scripts\\python.exe -m pyllama inspect --model Qwen3-0.6B-Q4_K_M.gguf
```

### CLI chat

```powershell
.venv\\Scripts\\python.exe -m pyllama chat --model Qwen3-0.6B-Q4_K_M.gguf --dtype float16
```

You can also use the `run` alias:

```powershell
.venv\\Scripts\\python.exe -m pyllama run --model Qwen3-0.6B-Q4_K_M.gguf --dtype float16
```

### Generate

```powershell
.venv\\Scripts\\python.exe -m pyllama generate --model Qwen3-0.6B-Q4_K_M.gguf --prompt "Write a Python HTTP server" --max-tokens 128
```

### HTTP API server

```powershell
.venv\\Scripts\\python.exe -m pyllama serve --model Qwen3-0.6B-Q4_K_M.gguf --dtype float16 --port 8080
```

The server exposes `/api/health`, `/api/models`, `/api/info`, and the OpenAI-compatible `/v1/chat/completions` endpoint.


## Architecture

| llama.cpp concept | PyLlama implementation |
|---|---|
| Local model files | `models\\` |
| Model container | GGUF reader |
| Tensor representation | NumPy |
| Quantized tensors | Python dequantization routines |
| Decoder runtime | Python/NumPy Llama/Qwen-style runtime |
| KV cache | `pyllama.engine.cache` |
| Sampling | `pyllama.engine.sampler` |
| CLI | `pyllama chat/generate/run` |
| HTTP server | FastAPI |

## Scope

This is a from-scratch learning/engineering implementation, not a feature-for-feature replacement for llama.cpp. It is CPU/NumPy focused and supports a Llama/Qwen-style decoder path and several GGUF tensor types.

Unsupported architectures or tensor formats are rejected instead of silently producing incorrect results.

## Qwen2.5 GGUF fix

PyLlama reads the vocabulary directly from `tokenizer.ggml.tokens` when a GGUF does not provide a separate `tokenizer.ggml.vocab_size` key. It can also build the byte-level BPE tokenizer from tokenizer metadata embedded in the GGUF, so a standalone `tokenizer.json` is not required for Qwen2-family GGUF files that embed their tokenizer data.


## Universal GGUF runtime

See `UNIVERSAL_GGUF.md` for architecture and quantization support. PyLlama never downloads models; place local GGUF files under `models/`.


## CPU and GPU backend

PyLlama now has an explicit backend layer. CPU uses NumPy. NVIDIA CUDA GPUs use CuPy when installed and when a CUDA device is usable.

Check hardware:

```powershell
python -m pyllama device
```

The command intentionally reports the booleans requested by the runtime:

```text
CPU available == True
GPU available == False
Auto-selected device: cpu
```

On a CUDA machine with CuPy installed:

```text
CPU available == True
GPU available == True
Auto-selected device: cuda
```

Run automatically:

```powershell
python -m pyllama chat --model models\model.gguf --device auto
```

Force CPU:

```powershell
python -m pyllama chat --model models\model.gguf --device cpu
```

Force NVIDIA CUDA:

```powershell
python -m pyllama chat --model models\model.gguf --device cuda
```

For an NVIDIA machine, install the optional GPU backend:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
```

The GPU path accelerates the Transformer tensor operations and KV cache on CUDA, while GGUF quantized weights are decoded by the Python quantizer and then uploaded to the selected backend. This is intentionally a from-scratch Python runtime rather than a wrapper around llama.cpp.
