from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from pyllama import __version__
from pyllama.engine.runtime import Runtime
from pyllama.engine.sampler import Sampler
from pyllama.formats.gguf import GGUFFile
from pyllama.tools.manager import LocalModelManager
from pyllama.engine.device import detect_devices

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"


def _model_path(arg: str) -> Path:
    return LocalModelManager(MODELS).resolve(arg)


def cmd_list(_args):
    mgr = LocalModelManager(MODELS)
    models = mgr.list_models()
    if not models:
        print(f"No GGUF models found in {MODELS}")
        print("Download a model manually and place the .gguf file there.")
        return
    print(f"Installed models in {MODELS}:\n")
    for f in models:
        size = f.stat().st_size / 1048576
        print(f"  {f.stem:40} {size:10.1f} MB")


def cmd_inspect(args):
    path = _model_path(args.model)
    with GGUFFile.open(path) as g:
        d = g.describe()
        print(json.dumps(d, indent=2, ensure_ascii=False))
        print("\nMetadata:")
        for k, v in sorted(g.metadata.items()):
            if isinstance(v, list) and len(v) > 20:
                print(f"  {k}: [{len(v)} items]")
            else:
                print(f"  {k}: {v}")
        print("\nTensors:")
        for name, info in list(g.tensors.items())[:args.max_tensors]:
            print(f"  {name:50} {str(info.shape):20} {info.type_name:8} @ {info.offset}")


def _runtime(args):
    path = _model_path(args.model)
    return Runtime(path, tokenizer_path=args.tokenizer, dtype=args.dtype, context=args.context, device=args.device)


def cmd_generate(args):
    rt = _runtime(args)
    try:
        print(f"PyLlama {__version__} | {rt.model_path.name}")
        print("Generating...\n")
        sampler = Sampler(args.temperature, args.top_k, args.top_p, args.repeat_penalty, args.min_p, args.seed)
        text, stats = rt.generate(args.prompt, max_tokens=args.max_tokens, sampler=sampler)
        print(text)
        print(f"\n\n[generated {stats.generated_tokens} tokens | {stats.tok_per_sec:.2f} tok/s]")
    finally:
        rt.close()


def cmd_chat(args):
    rt = _runtime(args)
    try:
        if not rt.tokenizer:
            raise RuntimeError("Chat requires tokenizer.json")
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        print(f"PyLlama {__version__} | {rt.model_path.name}")
        print("Type /exit to quit.\n")
        while True:
            user = input("You> ").strip()
            if user == "/exit":
                break
            messages.append({"role": "user", "content": user})
            prompt = rt.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            sampler = Sampler(args.temperature, args.top_k, args.top_p, args.repeat_penalty, args.min_p, args.seed)
            print("PyLlama> ", end="", flush=True)
            answer = []
            for piece, _, _ in rt.stream_generate(prompt, max_tokens=args.max_tokens, sampler=sampler):
                print(piece, end="", flush=True)
                answer.append(piece)
            print("\n")
            messages.append({"role": "assistant", "content": "".join(answer)})
    finally:
        rt.close()




def cmd_device(args):
    d = detect_devices()
    print(f"CPU available == {d.cpu_available}")
    print(f"CPU: {d.cpu_name}")
    print(f"CPU threads: {d.cpu_threads}")
    print(f"GPU available == {d.gpu_available}")
    print(f"GPU backend: {d.gpu_backend or 'none'}")
    print(f"GPU name: {d.gpu_name or 'none'}")
    print(f"GPU count: {d.gpu_count}")
    if d.gpu_memory_mb is not None:
        print(f"GPU memory: {d.gpu_memory_mb} MB")
    if d.gpu_error:
        print(f"GPU note: {d.gpu_error}")
    print(f"Auto-selected device: {d.auto_device}")


def cmd_serve(args):
    path = _model_path(args.model)
    import uvicorn
    from pyllama.server.app import create_app
    app = create_app(path, tokenizer_path=args.tokenizer, dtype=args.dtype, context=args.context, device=args.device)
    url = f"http://{args.host}:{args.port}"
    print(f"PyLlama API server: {url}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def build_parser():
    p = argparse.ArgumentParser(prog="pyllama", description="Local GGUF LLM runtime implemented in Python/NumPy")
    sub = p.add_subparsers(dest="cmd", required=True)
    x = sub.add_parser("list", help="List installed models")
    x.set_defaults(func=cmd_list)

    x = sub.add_parser("inspect", help="Inspect GGUF metadata/tensors")
    x.add_argument("--model", required=True)
    x.add_argument("--max-tensors", type=int, default=80)
    x.set_defaults(func=cmd_inspect)

    def add_runtime_args(x):
        x.add_argument("--model", required=True)
        x.add_argument("--tokenizer")
        x.add_argument("--dtype", choices=["float32", "float16"], default="float32")
        x.add_argument("--context", type=int)
        x.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
        x.add_argument("--temperature", type=float, default=0.7)
        x.add_argument("--top-k", type=int, default=40)
        x.add_argument("--top-p", type=float, default=0.9)
        x.add_argument("--min-p", type=float, default=0.0)
        x.add_argument("--repeat-penalty", type=float, default=1.1)
        x.add_argument("--seed", type=int)
        x.add_argument("--max-tokens", type=int, default=128)

    x = sub.add_parser("generate", help="Generate from a prompt")
    add_runtime_args(x)
    x.add_argument("--prompt", required=True)
    x.set_defaults(func=cmd_generate)

    x = sub.add_parser("run", help="Run interactive local chat (alias for chat)")
    add_runtime_args(x)
    x.add_argument("--system", default="You are a helpful assistant.")
    x.set_defaults(func=cmd_chat)

    x = sub.add_parser("chat", help="Interactive chat")
    add_runtime_args(x)
    x.add_argument("--system", default="You are a helpful assistant.")
    x.set_defaults(func=cmd_chat)

    x = sub.add_parser("device", help="Show CPU/GPU availability and automatic selection")
    x.set_defaults(func=cmd_device)

    x = sub.add_parser("serve", help="Start the local HTTP API")
    add_runtime_args(x)
    x.add_argument("--host", default="127.0.0.1")
    x.add_argument("--port", type=int, default=8080)
    x.set_defaults(func=cmd_serve)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as exc:
        print(f"PyLlama error: {exc}", file=sys.stderr)
        raise SystemExit(1)

if __name__ == "__main__":
    main()
