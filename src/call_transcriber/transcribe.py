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
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

_model = None
_model_key: tuple | None = None
_model_lock = threading.Lock()

_parakeet = None
_parakeet_key: tuple | None = None


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


class EngineError(RuntimeError):
    """Raised when the configured speech engine cannot be used."""


def load_parakeet(cfg, root: Path):
    """Load (and memoize) the Parakeet recognizer.

    Parakeet is a transducer rather than an autoregressive decoder, scores
    better than whisper on English, and does not invent sentences over silence.

    It is not, however, faster than a small whisper. Measured on a seven minute
    call on an i5-4570: base.en 26s, parakeet 85s. Parakeet is 600M parameters
    against base.en's 74M, and the "several times faster" figure that gets
    quoted is against whisper small or large. It is chosen for accuracy.
    """
    global _parakeet, _parakeet_key

    folder = Path(cfg.parakeet_dir) if cfg.parakeet_dir else root / "models/parakeet"
    if not folder.is_absolute():
        folder = root / folder
    key = (str(folder), cfg.num_threads, cfg.beam_size)

    with _model_lock:
        if _parakeet is not None and _parakeet_key == key:
            return _parakeet

        try:
            import sherpa_onnx
        except ImportError as exc:
            raise EngineError(
                "sherpa-onnx is not installed, so the parakeet engine cannot run. "
                "Re-run install.bat."
            ) from exc

        needed = {
            "encoder": folder / "encoder.int8.onnx",
            "decoder": folder / "decoder.int8.onnx",
            "joiner": folder / "joiner.int8.onnx",
            "tokens": folder / "tokens.txt",
        }
        missing = [str(p) for p in needed.values() if not p.exists()]
        if missing:
            raise EngineError(
                "the parakeet model is not downloaded yet -- missing "
                f"{missing[0]}.\n  Run:  run.bat models --parakeet"
            )

        log.info("loading parakeet from %s", folder)
        _parakeet = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(needed["encoder"]),
            decoder=str(needed["decoder"]),
            joiner=str(needed["joiner"]),
            tokens=str(needed["tokens"]),
            num_threads=cfg.num_threads,
            sample_rate=16000,
            feature_dim=80,
            # NeMo transducers use a different blank/joiner convention to the
            # icefall ones; naming the type is what selects it.
            model_type="nemo_transducer",
            decoding_method="greedy_search" if cfg.beam_size <= 1 else "modified_beam_search",
        )
        _parakeet_key = key
        return _parakeet


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


def transcribe(
    audio: np.ndarray,
    sample_rate: int,
    cfg,
    stereo_mode: str = "auto",
    turns=None,
    root: Path | None = None,
) -> Transcript:
    """Transcribe a finished call.

    `turns` is optional diarization output. When present the recording is
    transcribed turn by turn and each line carries a speaker, which is the
    whole point -- a mixed handset tap otherwise gives no way to tell the
    person answering from the person calling.
    """
    root = root or Path.cwd()
    layout = detect_layout(audio) if stereo_mode == "auto" else stereo_mode
    if audio.ndim < 2 or audio.shape[1] < 2:
        layout = "mono"
    duration = len(audio) / sample_rate

    if layout == "split":
        segments = _transcribe_split(audio, cfg, root)
    elif turns:
        segments = _transcribe_turns(to_float_mono(audio), sample_rate, cfg, turns, root)
        layout = "diarized"
    else:
        segments = _run(to_float_mono(audio), cfg, root)

    return Transcript(
        segments=segments,
        language=cfg.language or "en",
        duration_s=duration,
        layout=layout,
    )


def _transcribe_turns(samples, sample_rate, cfg, turns, root) -> list[Segment]:
    """One pass per speaker turn, so every line knows who said it.

    The turns are widened to cover the silence between them first. Only what is
    inside a turn ever reaches the speech engine, so every moment the diarizer
    left unassigned is a moment nobody transcribes -- and the moments it misses
    are the short quiet ones, which on a service call are the customer
    confirming an address or agreeing to a time.
    """
    from . import diarize as diarize_mod

    merged = diarize_mod.close_gaps(
        diarize_mod.merge_adjacent(turns), len(samples) / sample_rate
    )
    names = diarize_mod.label(merged)

    out: list[Segment] = []
    for turn in merged:
        first = max(0, int(turn.start * sample_rate))
        last = min(len(samples), int(turn.end * sample_rate))
        # Only a guard against decoding a sliver of nothing. It used to sit at
        # 250 ms, which is about the length of the word "yes".
        if last - first < sample_rate // 20:  # 50 ms
            continue
        for segment in _run(samples[first:last], cfg, root):
            out.append(
                Segment(
                    start=turn.start + segment.start,
                    end=turn.start + segment.end,
                    text=segment.text,
                    speaker=names.get(turn.speaker, f"SPEAKER {turn.speaker}"),
                )
            )
    out.sort(key=lambda s: s.start)
    return out


def _transcribe_split(audio: np.ndarray, cfg, root) -> list[Segment]:
    """Transcribe each side separately, then interleave chronologically.

    Which physical channel is the caller varies by how the adapter is wired, so
    the sides are labelled neutrally and the extraction step works out who is
    who from what they actually say.
    """
    merged: list[Segment] = []
    for index, label in ((0, "SIDE A"), (1, "SIDE B")):
        channel = audio[:, index]
        for seg in _run(to_float_mono(channel), cfg, root):
            seg.speaker = label
            merged.append(seg)
    merged.sort(key=lambda s: s.start)
    return merged


def _initial_prompt(cfg) -> str | None:
    """Prime whisper with the vocabulary of this trade.

    Whisper picks between similar-sounding words partly on how likely each is
    in context. Seeing the trade's terms first shifts that: "cables" stops
    losing to "key balls" because cables are now expected and key balls are
    not. Cheaper and more targeted than a larger model, which would still have
    no idea what business this is.
    """
    vocabulary = (getattr(cfg, "vocabulary", "") or "").strip()
    if not vocabulary:
        return None
    return f"The following conversation may mention: {vocabulary}."


def _run(samples: np.ndarray, cfg, root: Path) -> list[Segment]:
    """Transcribe one stretch of mono audio with whichever engine is chosen."""
    engine = (getattr(cfg, "engine", "whisper") or "whisper").lower()
    if engine == "parakeet":
        return _run_parakeet(samples, cfg, root)
    return _run_whisper(samples, cfg)


def _run_parakeet(samples: np.ndarray, cfg, root: Path) -> list[Segment]:
    """Transcribe with Parakeet, a few windows at a time.

    The batch size is the whole point of this loop. sherpa-onnx runs the
    encoder over every stream handed to `decode_streams` at once, padded to the
    longest, and holds the activations for all of them -- measured at roughly
    175 MB per window with this model. Decoding a whole call in one call is
    therefore not a batch, it is a memory leak with a duration on it:

        10 minutes, 28 windows  ->  7.6 GB
        30 minutes, 83 windows  ->  killed by the OOM reaper at 16 GB

    A long call would take the program out on the machine it was written for.
    In groups of four it is bounded at well under a gigabyte however long the
    call runs, and the padding waste is small because the windows are of
    similar length by construction.
    """
    recognizer = load_parakeet(cfg, root)

    windows = _speech_windows(samples, cfg)
    if not windows:
        return []

    size = max(1, int(getattr(cfg, "batch_size", 4) or 1))
    out: list[Segment] = []
    for first in range(0, len(windows), size):
        group = windows[first : first + size]
        streams = []
        for start, end in group:
            stream = recognizer.create_stream()
            stream.accept_waveform(16000, np.ascontiguousarray(
                samples[int(start * 16000) : int(end * 16000)], dtype=np.float32
            ))
            streams.append(stream)

        recognizer.decode_streams(streams)
        for (start, end), stream in zip(group, streams):
            text = stream.result.text.strip() if stream.result.text else ""
            if text:
                out.append(Segment(start=start, end=end, text=text))
        # Let the encoder activations go before building the next group.
        streams.clear()

    return out


def _speech_windows(samples: np.ndarray, cfg) -> list[tuple[float, float]]:
    from .vad import speech_windows

    # A turn already bounded by diarization is short enough to send whole.
    duration = len(samples) / 16000
    if duration <= 30:
        return [(0.0, duration)]
    return speech_windows(samples, 16000)


def _run_whisper(samples: np.ndarray, cfg) -> list[Segment]:
    model = load_model(cfg)
    segments, _info = model.transcribe(
        samples,
        language=cfg.language or None,
        beam_size=cfg.beam_size,
        initial_prompt=_initial_prompt(cfg),
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


def to_float_mono(audio: np.ndarray) -> np.ndarray:
    """int16 (samples, ch) -> float32 mono in [-1, 1], which is whisper's input."""
    data = audio.astype(np.float32)
    if data.ndim > 1 and data.shape[1] > 1:
        data = data.mean(axis=1)
    data = data.reshape(-1)
    if audio.dtype == np.int16:
        data /= 32768.0
    return np.ascontiguousarray(data)
