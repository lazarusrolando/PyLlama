from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
import platform
import subprocess
from typing import Any


@dataclass(frozen=True)
class DeviceInfo:
    cpu_available: bool
    cpu_threads: int
    cpu_name: str
    gpu_available: bool
    gpu_backend: str | None
    gpu_name: str | None
    gpu_count: int
    gpu_memory_mb: int | None
    gpu_error: str | None

    @property
    def auto_device(self) -> str:
        return "cuda" if self.gpu_available else "cpu"


def _cpu_name() -> str:
    return platform.processor() or platform.machine() or "Unknown CPU"


def _nvidia_smi() -> tuple[bool, list[dict[str, Any]]]:
    """Best-effort hardware probe; it does not make PyLlama depend on nvidia-smi."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return False, []
    if proc.returncode != 0:
        return False, []
    rows: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [x.strip() for x in line.split(",", 1)]
        rows.append({"name": parts[0], "memory_mb": int(float(parts[1])) if len(parts) > 1 and parts[1].strip() else None})
    return bool(rows), rows


def _probe_cuda() -> tuple[bool, str | None, int, int | None, str | None]:
    """Return usable CUDA status. CUDA is considered available only when the GPU backend can execute."""
    if importlib.util.find_spec("cupy") is None:
        hw, rows = _nvidia_smi()
        name = rows[0]["name"] if rows else None
        mem = rows[0]["memory_mb"] if rows else None
        msg = "CuPy is not installed" if hw else "CuPy is not installed and no NVIDIA GPU was detected"
        return False, name, len(rows), mem, msg
    try:
        import cupy as cp
        count = int(cp.cuda.runtime.getDeviceCount())
        if count <= 0:
            return False, None, 0, None, "CUDA reports zero devices"
        names = []
        total_mem = None
        for idx in range(count):
            with cp.cuda.Device(idx):
                raw_name = cp.cuda.runtime.getDeviceProperties(idx)["name"]
                name = raw_name.decode(errors="replace") if isinstance(raw_name, bytes) else str(raw_name)
                names.append(name)
                free_mem, total = cp.cuda.runtime.memGetInfo()
                if idx == 0:
                    total_mem = int(total // (1024 * 1024))
        return True, names[0], count, total_mem, None
    except Exception as exc:  # pragma: no cover - hardware dependent
        return False, None, 0, None, f"CUDA probe failed: {exc}"


def detect_devices() -> DeviceInfo:
    cpu_threads = os.cpu_count() or 1
    cpu_available = cpu_threads > 0
    gpu_available, gpu_name, gpu_count, gpu_mem, gpu_error = _probe_cuda()
    return DeviceInfo(
        cpu_available=cpu_available,
        cpu_threads=cpu_threads,
        cpu_name=_cpu_name(),
        gpu_available=gpu_available,
        gpu_backend="cuda" if gpu_available else None,
        gpu_name=gpu_name,
        gpu_count=gpu_count,
        gpu_memory_mb=gpu_mem,
        gpu_error=gpu_error,
    )


def select_device(requested: str = "auto") -> str:
    requested = requested.lower().strip()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    info = detect_devices()
    if requested == "auto":
        return info.auto_device
    if requested == "cuda" and not info.gpu_available:
        detail = f" ({info.gpu_error})" if info.gpu_error else ""
        raise RuntimeError(f"CUDA GPU backend is not available{detail}")
    if requested == "cpu" and not info.cpu_available:
        raise RuntimeError("CPU backend is not available")
    return requested


class Backend:
    """Small array backend abstraction used by the inference runtime."""

    def __init__(self, device: str = "auto"):
        self.device = select_device(device)
        if self.device == "cuda":
            import cupy as cp
            self.xp = cp
        else:
            import numpy as np
            self.xp = np

    def asarray(self, value: Any, dtype=None):
        return self.xp.asarray(value, dtype=dtype)

    def zeros(self, shape, dtype):
        return self.xp.zeros(shape, dtype=dtype)

    def empty(self, shape, dtype):
        return self.xp.empty(shape, dtype=dtype)

    def to_numpy(self, value):
        if self.device == "cuda":
            import cupy as cp
            return cp.asnumpy(value)
        return value

    def item(self, value):
        try:
            return value.item()
        except AttributeError:
            return value

    def synchronize(self) -> None:
        if self.device == "cuda":
            import cupy as cp
            cp.cuda.Stream.null.synchronize()
