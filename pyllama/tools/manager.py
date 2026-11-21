from __future__ import annotations

from pathlib import Path


class LocalModelManager:
    """Manage models that the user has already downloaded into the local models/ directory.

    This class intentionally has no network code and no model-hosting API dependency.
    """

    def __init__(self, models_dir: str | Path):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def list_models(self) -> list[Path]:
        return sorted(
            (p for p in self.models_dir.rglob("*.gguf") if p.is_file()),
            key=lambda p: str(p).lower(),
        )

    def resolve(self, value: str | Path) -> Path:
        requested = str(value).strip()
        if not requested:
            raise ValueError("Model name/path cannot be empty")

        direct = Path(requested).expanduser()
        if direct.is_file():
            return direct.resolve()
        if direct.is_dir():
            files = sorted(direct.glob("*.gguf"))
            if not files:
                raise FileNotFoundError(f"No .gguf files found in {direct}")
            if len(files) > 1:
                names = "\n".join(f"  - {p.name}" for p in files)
                raise RuntimeError(
                    f"Multiple GGUF files found in {direct}. Specify the GGUF file explicitly:\n{names}"
                )
            return files[0].resolve()

        candidates = self.list_models()
        wanted = requested.lower()

        exact = [
            p for p in candidates
            if p.name.lower() == wanted
            or p.stem.lower() == wanted
            or p.parent.name.lower() == wanted
        ]
        if len(exact) == 1:
            return exact[0].resolve()
        if len(exact) > 1:
            names = "\n".join(f"  - {p}" for p in exact)
            raise RuntimeError(f"Model name is ambiguous:\n{names}")

        partial = [p for p in candidates if wanted in str(p.relative_to(self.models_dir)).lower()]
        if len(partial) == 1:
            return partial[0].resolve()
        if not candidates:
            raise FileNotFoundError(
                f"No GGUF models are installed in {self.models_dir}. "
                "Download a GGUF model manually and place it there."
            )
        available = "\n".join(f"  - {p.relative_to(self.models_dir)}" for p in candidates[:20])
        suffix = "\n  ..." if len(candidates) > 20 else ""
        raise FileNotFoundError(
            f"Model {requested!r} was not found under {self.models_dir}.\n"
            f"Installed models:\n{available}{suffix}"
        )

    def describe(self) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        for p in self.list_models():
            result.append(
                {
                    "name": p.stem,
                    "file": p.name,
                    "path": str(p),
                    "size_mb": round(p.stat().st_size / 1048576, 2),
                    "relative_path": str(p.relative_to(self.models_dir)),
                }
            )
        return result
