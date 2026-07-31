"""Where results go, and making sure the audio doesn't stay.

Default behaviour is that the recording exists only long enough to be
transcribed. It is written to a temp file, read once by whisper, then
overwritten and unlinked.

On the overwrite: it makes the data unrecoverable by ordinary means, which is
the realistic threat here. It is not a guarantee against forensic recovery --
SSD wear levelling means the original blocks may survive somewhere the
filesystem can no longer address. Full-disk encryption is the real answer if
that matters to you.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def call_dir(root: Path, started_at: float, name_slug: str) -> Path:
    """output/2026-07-31/1422-jane-doe/ -- sorted by day, then by call."""
    day = time.strftime("%Y-%m-%d", time.localtime(started_at))
    stamp = time.strftime("%H%M%S", time.localtime(started_at))
    path = root / day / f"{stamp}-{name_slug}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> Path:
    """Write int16 PCM. Uses soundfile when present, else the stdlib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = audio if audio.ndim > 1 else audio[:, None]
    if data.dtype != np.int16:
        data = (np.clip(data, -1.0, 1.0) * 32767.0).astype(np.int16)

    try:
        import soundfile as sf

        sf.write(str(path), data, sample_rate, subtype="PCM_16")
        return path
    except ImportError:
        pass

    import wave

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(data.shape[1])
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(data.tobytes())
    return path


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read an audio file to (samples, channels) int16. Used by `test`."""
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "soundfile is needed to read audio files. Re-run install.bat."
        ) from exc

    data, rate = sf.read(str(path), dtype="int16", always_2d=True)
    return data, int(rate)


def shred(path: Path) -> None:
    """Overwrite a file with zeros, then delete it."""
    try:
        if not path.exists():
            return
        size = path.stat().st_size
        with open(path, "r+b", buffering=0) as handle:
            chunk = b"\x00" * 1024 * 1024
            written = 0
            while written < size:
                handle.write(chunk[: min(len(chunk), size - written)])
                written += len(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        path.unlink()
        log.debug("shredded %s (%d bytes)", path.name, size)
    except OSError as exc:
        # Never let cleanup failure lose the work order that just succeeded.
        log.warning("could not shred %s: %s", path, exc)
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def save_results(
    folder: Path,
    work_order: str,
    transcript_text: str,
    extracted: dict,
    *,
    save_transcript: bool = True,
    meta: dict | None = None,
) -> dict[str, Path]:
    """Write the outputs and return what was written."""
    written: dict[str, Path] = {}

    order_path = folder / "work_order.txt"
    order_path.write_text(work_order, encoding="utf-8")
    written["work_order"] = order_path

    payload = dict(extracted)
    if meta:
        payload["_call"] = meta
    json_path = folder / "extracted.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    written["extracted"] = json_path

    if save_transcript:
        transcript_path = folder / "transcript.txt"
        transcript_path.write_text(transcript_text, encoding="utf-8")
        written["transcript"] = transcript_path

    return written


AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


def purge_audio(root: Path, everything: bool = False) -> tuple[int, int]:
    """Shred kept recordings under `root`. Returns (files removed, bytes).

    Only relevant if output.keep_audio was ever switched on -- normal operation
    never writes audio at all. This is the way back from a testing session.
    """
    if not root.exists():
        return 0, 0

    files = 0
    freed = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if not everything and path.suffix.lower() not in AUDIO_SUFFIXES:
            continue
        try:
            freed += path.stat().st_size
        except OSError:
            pass
        shred(path)
        if not path.exists():
            files += 1

    if everything:
        # Remove the directories left behind, deepest first.
        for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    pass
    return files, freed


def latest_work_order(root: Path) -> Path | None:
    """The most recently written work order, for reopening a closed window."""
    if not root.exists():
        return None
    written = list(root.rglob("work_order.txt"))
    if not written:
        return None
    return max(written, key=lambda p: p.stat().st_mtime)


def read_call(folder: Path) -> dict:
    """Load a finished call back off disk.

    The popup is not the record -- these files are. Reading them back is how a
    window closed by accident stops being a lost call.
    """
    out: dict = {"folder": folder, "work_order": "", "transcript": "", "extracted": {}}

    order = folder / "work_order.txt"
    if order.exists():
        out["work_order"] = order.read_text(encoding="utf-8")

    transcript = folder / "transcript.txt"
    if transcript.exists():
        out["transcript"] = transcript.read_text(encoding="utf-8")

    extracted = folder / "extracted.json"
    if extracted.exists():
        try:
            out["extracted"] = json.loads(extracted.read_text(encoding="utf-8"))
        except ValueError:
            pass
    return out


def open_folder(path: Path) -> None:
    """Reveal a folder in the OS file manager."""
    try:
        if os.name == "nt":
            os.startfile(str(path))  # noqa: S606 - Windows-only API
        else:
            import subprocess
            import sys

            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.Popen([opener, str(path)])
    except Exception as exc:
        log.warning("could not open %s: %s", path, exc)
