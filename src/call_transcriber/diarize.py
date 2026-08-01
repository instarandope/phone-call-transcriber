"""Working out who is speaking.

A handset tap that mixes both sides gives one signal and no idea who said
what, which is why "my name is Johan" from the person answering kept being
recorded as the customer. Every fix for that so far has been the model
inferring it from context. This settles it from the audio instead.

The useful shortcut here is that a phone call has exactly two people on it.
General diarization spends most of its difficulty estimating how many speakers
there are; we know, so we tell it, and what is left is the easy part.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


class DiarizationError(RuntimeError):
    """Raised when diarization is asked for but cannot run."""


@dataclass(frozen=True)
class Turn:
    start: float
    end: float
    speaker: int

    @property
    def duration(self) -> float:
        return self.end - self.start


def available() -> bool:
    try:
        import sherpa_onnx  # noqa: F401
    except ImportError:
        return False
    return True


def _resolve(root: Path, configured: str, fallback: str) -> Path:
    path = Path(configured) if configured else root / fallback
    return path if path.is_absolute() else root / path


def load(cfg, root: Path):
    """Build the diarizer. Raises DiarizationError with something actionable."""
    try:
        import sherpa_onnx
    except ImportError as exc:
        raise DiarizationError(
            "sherpa-onnx is not installed, so speaker labelling cannot run. "
            "Re-run install.bat."
        ) from exc

    segmentation = _resolve(root, cfg.segmentation_model, "models/diarize-segmentation/model.onnx")
    embedding = _resolve(
        root, cfg.embedding_model, "models/diarize-embedding/nemo_en_titanet_small.onnx"
    )

    for label, path in (("segmentation", segmentation), ("embedding", embedding)):
        if not path.exists():
            raise DiarizationError(
                f"the {label} model is missing from {path}. "
                f"Run:  run.bat models --diarize"
            )

    settings = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(segmentation)
            ),
            num_threads=cfg.num_threads,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(embedding), num_threads=cfg.num_threads
        ),
        # A phone call is two people. Saying so removes the part of the problem
        # that diarizers are worst at.
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=cfg.speakers),
        min_duration_on=cfg.min_speech_s,
        min_duration_off=cfg.min_silence_s,
    )
    if not settings.validate():
        raise DiarizationError(
            "the diarization models were found but rejected as invalid. "
            "Delete the models/diarize-* folders and run:  run.bat models --diarize"
        )
    return sherpa_onnx.OfflineSpeakerDiarization(settings)


def diarize(samples: np.ndarray, sample_rate: int, cfg, root: Path) -> list[Turn]:
    """Return who spoke when. `samples` is float32 mono in [-1, 1]."""
    diarizer = load(cfg, root)

    if sample_rate != diarizer.sample_rate:
        raise DiarizationError(
            f"the diarization models expect {diarizer.sample_rate} Hz but the "
            f"recording is {sample_rate} Hz"
        )

    result = diarizer.process(np.ascontiguousarray(samples, dtype=np.float32))
    turns = [
        Turn(start=float(r.start), end=float(r.end), speaker=int(r.speaker))
        for r in result.sort_by_start_time()
    ]
    log.info(
        "diarization found %d turns across %d speaker(s)",
        len(turns), len({t.speaker for t in turns}),
    )
    return turns


def merge_adjacent(turns: list[Turn], gap_s: float = 0.4) -> list[Turn]:
    """Join consecutive turns by the same speaker.

    Diarizers emit a new turn across every short pause, which would otherwise
    chop one person's sentence into several transcript lines and force the
    speech model to start cold on each fragment.
    """
    merged: list[Turn] = []
    for turn in turns:
        if merged and merged[-1].speaker == turn.speaker and turn.start - merged[-1].end <= gap_s:
            previous = merged.pop()
            merged.append(Turn(previous.start, turn.end, turn.speaker))
        else:
            merged.append(turn)
    return merged


def label(turns: list[Turn]) -> dict[int, str]:
    """Name the speakers neutrally, in the order they first speak.

    Deliberately not "customer" and "agent": which is which is a judgement
    about content, and it belongs in the extraction step where the words are.
    Guessing it here would just move the same mistake earlier.
    """
    names = ("SIDE A", "SIDE B", "SIDE C", "SIDE D")
    order: list[int] = []
    for turn in turns:
        if turn.speaker not in order:
            order.append(turn.speaker)
    return {
        speaker: names[index] if index < len(names) else f"SPEAKER {speaker}"
        for index, speaker in enumerate(order)
    }
