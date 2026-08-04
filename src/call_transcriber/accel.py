"""Choosing where the ONNX models actually run.

Shared by Parakeet and by speaker labelling, because they are the same library
underneath and a machine that can accelerate one can accelerate the other.

The rule throughout is the one the bench PC taught: an accelerator that is
present is not the same as an accelerator that works. A GPU driver without the
libraries behind it will be selected happily and then fail, so anything chosen
automatically has to be able to fall back.
"""

from __future__ import annotations

import logging
import platform

log = logging.getLogger(__name__)


def resolve(chosen: str) -> str:
    """Turn a provider setting into a provider sherpa-onnx will accept."""
    chosen = (chosen or "auto").lower()
    if chosen != "auto":
        return chosen
    return detect()


def detect() -> str:
    """The best accelerator this machine actually has, or the CPU."""
    if _cuda_available():
        return "cuda"
    if platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64"):
        return "coreml"
    return "cpu"


def _cuda_available() -> bool:
    """Is there a usable CUDA execution provider, not merely a GPU?

    onnxruntime lists only the providers it was actually built with and can
    load, which is the question worth asking -- "is there an NVIDIA card" is
    not, since the card is useless here without the runtime beside it.
    """
    try:
        import onnxruntime
    except ImportError:
        return False
    try:
        return "CUDAExecutionProvider" in onnxruntime.get_available_providers()
    except Exception:  # pragma: no cover - defensive
        return False


def describe(chosen: str) -> str:
    """How to say, in the doctor output, where something is going to run."""
    resolved = resolve(chosen)
    names = {"cuda": "NVIDIA GPU", "coreml": "Apple Neural Engine / GPU", "cpu": "CPU"}
    label = names.get(resolved, resolved)
    return f"{label} ({resolved})" if chosen == "auto" else f"{label}, set explicitly"
