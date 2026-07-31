"""Capture from the telephone recording adapter.

The LRX-40USB is not a recorder -- it is a USB sound card that taps the
handset cord. So there is no folder to watch: we open it like a microphone and
listen continuously. Whatever the adapter's native rate is, this module hands
downstream code a steady stream of 16 kHz int16 frames, which is what both
webrtcvad and whisper want.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Iterator

import numpy as np

# Imported lazily in open_stream() so that `devices`/`doctor` can give a clean
# error message instead of an ImportError traceback on a half-finished install.
sd = None
soxr = None


class AudioError(RuntimeError):
    """Raised for anything the user can fix by plugging something in."""


@dataclass(frozen=True)
class InputDevice:
    index: int
    name: str
    channels: int
    samplerate: float

    def __str__(self) -> str:
        return f"[{self.index}] {self.name} ({self.channels}ch @ {self.samplerate:.0f} Hz)"


def _load_sounddevice():
    global sd
    if sd is None:
        try:
            import sounddevice as _sd
        except (ImportError, OSError) as exc:
            # OSError covers a missing PortAudio DLL, which is a different fix
            # from a missing package, so say both.
            raise AudioError(
                f"could not load the audio backend ({exc}). "
                "Re-run install.bat, which installs sounddevice and its PortAudio DLL."
            ) from exc
        sd = _sd
    return sd


def list_input_devices() -> list[InputDevice]:
    _load_sounddevice()
    devices = []
    for index, info in enumerate(sd.query_devices()):
        if info["max_input_channels"] > 0:
            devices.append(
                InputDevice(
                    index=index,
                    name=info["name"],
                    channels=int(info["max_input_channels"]),
                    samplerate=float(info["default_samplerate"]),
                )
            )
    return devices


def find_device(match: str) -> InputDevice:
    """Find the adapter by name substring.

    Matching on the name rather than the index matters: Windows renumbers audio
    devices when anything else is plugged in or removed, so a hardcoded index
    silently starts recording the webcam mic instead.
    """
    devices = list_input_devices()
    if not devices:
        raise AudioError("no audio input devices found at all -- is the adapter plugged in?")

    needle = match.strip().lower()
    if not needle:
        raise AudioError("audio.device_match is empty; set it to part of your adapter's name")

    hits = [d for d in devices if needle in d.name.lower()]
    if not hits:
        listing = "\n".join(f"    {d}" for d in devices)
        raise AudioError(
            f"no input device matching {match!r}.\n"
            f"  Inputs available right now:\n{listing}\n"
            f"  Set audio.device_match in config.toml to part of the right name."
        )
    if len(hits) > 1:
        listing = "\n".join(f"    {d}" for d in hits)
        raise AudioError(
            f"{match!r} matches {len(hits)} devices, so I can't tell which is the "
            f"adapter:\n{listing}\n  Make audio.device_match more specific."
        )
    return hits[0]


class Capture:
    """Continuous capture, resampled to `target_rate`, as 20 ms frames.

    Used as a context manager; iterate `frames()` to consume. Frames come back
    shaped (samples, channels) so a stereo adapter that puts each side of the
    call on its own channel keeps that separation all the way to transcription.
    """

    def __init__(
        self,
        device: InputDevice,
        target_rate: int = 16000,
        frame_ms: int = 20,
        queue_seconds: float = 30.0,
    ):
        self.device = device
        self.target_rate = target_rate
        self.frame_ms = frame_ms
        self.frame_samples = int(target_rate * frame_ms / 1000)
        # Bound the queue: if transcription somehow blocks the consumer, drop
        # audio rather than growing the heap until the process dies.
        self.maxsize = max(4, int(queue_seconds * 1000 / frame_ms))

        self.channels = min(device.channels, 2)
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=self.maxsize)
        self._stream = None
        self._resampler = None
        self._native_rate = target_rate
        self._residual = np.zeros((0, self.channels), dtype=np.float32)
        self._overflows = 0
        self._drops = 0
        self._lock = threading.Lock()
        self._closed = threading.Event()

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "Capture":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def start(self) -> None:
        _load_sounddevice()
        self._native_rate = self._negotiate_rate()

        if self._native_rate != self.target_rate:
            self._resampler = _make_resampler(
                self._native_rate, self.target_rate, self.channels
            )

        blocksize = int(self._native_rate * self.frame_ms / 1000)
        try:
            self._stream = sd.InputStream(
                device=self.device.index,
                channels=self.channels,
                samplerate=self._native_rate,
                blocksize=blocksize,
                dtype="float32",
                callback=self._on_audio,
            )
            self._stream.start()
        except Exception as exc:  # sounddevice raises several unrelated types
            raise AudioError(
                f"could not open {self.device.name!r} ({exc}). "
                "Another program may already have it open -- close any dialer or "
                "recording software and try again."
            ) from exc

    def stop(self) -> None:
        self._closed.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        # Unblock any consumer parked on get().
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

    def _negotiate_rate(self) -> int:
        """Prefer the target rate; fall back to the device's native rate.

        Opening at 16 kHz directly lets WASAPI do the resampling, which is both
        cheaper and better than doing it ourselves -- but not every adapter
        will accept it, so we check rather than assume.
        """
        for rate in (self.target_rate, int(self.device.samplerate), 48000, 44100):
            try:
                sd.check_input_settings(
                    device=self.device.index,
                    channels=self.channels,
                    samplerate=rate,
                    dtype="float32",
                )
                return rate
            except Exception:
                continue
        raise AudioError(
            f"{self.device.name!r} rejected every sample rate tried "
            f"({self.target_rate}, {int(self.device.samplerate)}, 48000, 44100 Hz)."
        )

    # -- capture callback --------------------------------------------------

    def _on_audio(self, indata, frames, time_info, status) -> None:
        if status:
            with self._lock:
                self._overflows += 1
        # Copy: PortAudio reuses this buffer after the callback returns.
        block = np.asarray(indata, dtype=np.float32).copy()
        if block.ndim == 1:
            block = block[:, None]

        if self._resampler is not None:
            block = self._resampler(block)
            if block.size == 0:
                return

        self._emit(block)

    def _emit(self, block: np.ndarray) -> None:
        """Slice into exact-length frames; webrtcvad rejects anything else."""
        buf = np.concatenate([self._residual, block]) if self._residual.size else block
        n = self.frame_samples
        count = len(buf) // n
        for i in range(count):
            frame = buf[i * n : (i + 1) * n]
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                with self._lock:
                    self._drops += 1
        self._residual = buf[count * n :]

    # -- consumption -------------------------------------------------------

    def frames(self, timeout: float = 1.0) -> Iterator[np.ndarray]:
        """Yield (frame_samples, channels) float32 frames until stopped."""
        while not self._closed.is_set():
            try:
                frame = self._queue.get(timeout=timeout)
            except queue.Empty:
                continue
            if frame is None:
                return
            yield frame

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "native_rate": self._native_rate,
                "channels": self.channels,
                "overflows": self._overflows,
                "dropped_frames": self._drops,
            }


def _make_resampler(in_rate: int, out_rate: int, channels: int):
    """Streaming resampler, with a linear fallback if soxr is unavailable."""
    global soxr
    if soxr is None:
        try:
            import soxr as _soxr

            soxr = _soxr
        except ImportError:
            soxr = False

    if soxr:
        stream = soxr.ResampleStream(
            in_rate, out_rate, channels, dtype="float32", quality="QQ"
        )
        return lambda block: stream.resample_chunk(block)

    # Linear interpolation is audibly worse but keeps the app running on a box
    # where the soxr wheel didn't install. Narrowband phone audio survives it.
    ratio = out_rate / in_rate

    def linear(block: np.ndarray) -> np.ndarray:
        n_out = int(round(len(block) * ratio))
        if n_out == 0:
            return np.zeros((0, block.shape[1]), dtype=np.float32)
        src = np.arange(len(block), dtype=np.float32)
        dst = np.linspace(0, len(block) - 1, n_out, dtype=np.float32)
        return np.stack(
            [np.interp(dst, src, block[:, c]) for c in range(block.shape[1])],
            axis=1,
        ).astype(np.float32)

    return linear


def to_mono_int16(frame: np.ndarray) -> bytes:
    """Downmix and convert to the little-endian int16 webrtcvad expects."""
    mono = frame.mean(axis=1) if frame.ndim > 1 and frame.shape[1] > 1 else frame.reshape(-1)
    clipped = np.clip(mono, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def dbfs(frame: np.ndarray) -> float:
    """RMS level in dBFS. -inf for digital silence."""
    if frame.size == 0:
        return -np.inf
    rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))
    if rms <= 1e-9:
        return -np.inf
    return 20.0 * np.log10(rms)
