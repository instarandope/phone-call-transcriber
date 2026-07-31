"""On-device speech-to-text.

Runs faster-whisper locally. No audio is uploaded anywhere, which is the whole
point when the recordings are client calls.

Handset adapters differ in how they present a call. Some sum both sides into
one signal; some put your voice on the left channel and the caller's on the
right. The second kind is worth detecting, because keeping the sides apart
gives speaker-attributed transcripts for free -- no diarization model needed.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

_model = None
_model_key: tuple | None = None
_model_lock = threading.Lock()


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str | None = None


@dataclass
class Transcript:
    segments: list[Segment]
    language: str
    duration_s: float
    layout: str  # "mono" | "mixed" | "split"

    @property
    def text(self) -> str:
        """Plain text, one line per segment, speaker-prefixed when known."""
        lines = []
        for seg in self.segments:
            body = seg.text.strip()
            if not body:
                continue
            lines.append(f"{seg.speaker}: {body}" if seg.speaker else body)
        return "\n".join(lines)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


def load_model(cfg):
    """Load (and memoize) the whisper model.

    Loading costs seconds and hundreds of megabytes, so it happens once and is
    reused for every call for the life of the process.
    """
    global _model, _model_key
    key = (cfg.model, cfg.device, cfg.compute_type)
    with _model_lock:
        if _model is not None and _model_key == key:
            return _model
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - install-time problem
            raise RuntimeError(
                "faster-whisper is not installed. Re-run install.bat."
            ) from exc

        log.info("loading whisper model %s (%s, %s)", cfg.model, cfg.device, cfg.compute_type)
        try:
            _model = WhisperModel(cfg.model, device=cfg.device, compute_type=cfg.compute_type)
        except Exception as exc:
            # Nearly always either a bad model name or a compute_type the CPU
            # can't do; both are config problems worth naming explicitly.
            raise RuntimeError(
                f"could not load whisper model {cfg.model!r} with compute_type "
                f"{cfg.compute_type!r}: {exc}\n"
                "Check [transcribe] in config.toml -- 'small.en' with 'int8' works on any CPU."
            ) from exc
        _model_key = key
        return _model


def detect_layout(audio: np.ndarray) -> str:
    """Decide whether a 2-channel recording holds one signal or two.

    Correlation is the tell. A duplicated mono feed has near-identical
    channels; genuinely separate sides of a call do not, because only one
    person talks at a time.
    """
    if audio.ndim < 2 or audio.shape[1] < 2:
        return "mono"

    left = audio[:, 0].astype(np.float64)
    right = audio[:, 1].astype(np.float64)

    # A silent channel means the adapter is only wired for one side.
    if np.std(left) < 1.0 or np.std(right) < 1.0:
        return "mono"

    corr = float(np.corrcoef(left, right)[0, 1])
    if not np.isfinite(corr):
        return "mono"
    return "mixed" if abs(corr) > 0.9 else "split"


def transcribe(audio: np.ndarray, sample_rate: int, cfg, stereo_mode: str = "auto") -> Transcript:
    """Transcribe a finished call."""
    model = load_model(cfg)

    layout = detect_layout(audio) if stereo_mode == "auto" else stereo_mode
    if audio.ndim < 2 or audio.shape[1] < 2:
        layout = "mono"
    duration = len(audio) / sample_rate

    if layout == "split":
        segments = _transcribe_split(model, audio, cfg)
    else:
        segments = _run(model, _to_float_mono(audio), cfg)

    return Transcript(
        segments=segments,
        language=cfg.language or "en",
        duration_s=duration,
        layout=layout,
    )


def _transcribe_split(model, audio: np.ndarray, cfg) -> list[Segment]:
    """Transcribe each side separately, then interleave chronologically.

    Which physical channel is the caller varies by how the adapter is wired, so
    the sides are labelled neutrally and the extraction step works out who is
    who from what they actually say.
    """
    merged: list[Segment] = []
    for index, label in ((0, "SIDE A"), (1, "SIDE B")):
        channel = audio[:, index]
        for seg in _run(model, _to_float_mono(channel), cfg):
            seg.speaker = label
            merged.append(seg)
    merged.sort(key=lambda s: s.start)
    return merged


def _run(model, samples: np.ndarray, cfg) -> list[Segment]:
    segments, _info = model.transcribe(
        samples,
        language=cfg.language or None,
        beam_size=cfg.beam_size,
        # Whisper hallucinates fluent-sounding text over silence; its own VAD
        # filter is the cheapest defence, and phone calls have a lot of silence.
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False,
    )
    return [
        Segment(start=s.start, end=s.end, text=s.text.strip())
        for s in segments
        if s.text and s.text.strip()
    ]


def _to_float_mono(audio: np.ndarray) -> np.ndarray:
    """int16 (samples, ch) -> float32 mono in [-1, 1], which is whisper's input."""
    data = audio.astype(np.float32)
    if data.ndim > 1 and data.shape[1] > 1:
        data = data.mean(axis=1)
    data = data.reshape(-1)
    if audio.dtype == np.int16:
        data /= 32768.0
    return np.ascontiguousarray(data)
