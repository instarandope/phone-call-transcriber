"""Deciding when a call starts and stops.

Nobody presses a button. The adapter is live whenever the handset is off the
cradle, so this module watches the signal and infers call boundaries: sustained
speech opens a recording, a long enough silence closes it.

Two guards keep it honest. A noise floor stops line hiss from being heard as
speech, and a pre-roll buffer means the recording starts slightly *before* the
speech that triggered it -- otherwise every transcript would open mid-word.
"""

from __future__ import annotations

import collections
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np

from .audio import dbfs, to_mono_int16


class State(str, Enum):
    IDLE = "idle"
    IN_CALL = "in_call"


@dataclass
class Call:
    """A finished recording, still in memory."""

    audio: np.ndarray  # (samples, channels) int16
    sample_rate: int
    started_at: float  # unix time
    duration_s: float
    ended_reason: str  # "silence" | "max_length" | "shutdown"

    @property
    def channels(self) -> int:
        return 1 if self.audio.ndim == 1 else self.audio.shape[1]


@dataclass
class _Window:
    """Sliding vote over the last N frames."""

    size: int
    votes: collections.deque = field(init=False)

    def __post_init__(self):
        self.votes = collections.deque(maxlen=max(1, self.size))

    def push(self, voiced: bool) -> None:
        self.votes.append(voiced)

    @property
    def full(self) -> bool:
        return len(self.votes) == self.votes.maxlen

    @property
    def ratio(self) -> float:
        return (sum(self.votes) / len(self.votes)) if self.votes else 0.0

    def clear(self) -> None:
        self.votes.clear()


class CallDetector:
    """Turns a stream of fixed-size frames into completed Call objects."""

    # Fraction of frames in the trigger window that must be speech before we
    # commit to a recording. High enough that a cough or a door doesn't start
    # one, low enough that a normal "Hello?" does.
    TRIGGER_RATIO = 0.7

    # Seconds of audio kept before the trigger so the first word survives.
    # Only needs to cover the trigger window plus the syllable that opened it;
    # anything more just prepends silence to every recording.
    PREROLL_S = 0.6

    def __init__(
        self,
        cfg,
        sample_rate: int = 16000,
        frame_ms: int = 20,
        on_state_change: Callable[[State], None] | None = None,
    ):
        self.cfg = cfg
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.on_state_change = on_state_change

        self._vad = _make_vad(cfg.vad_aggressiveness)
        self.state = State.IDLE

        frames_per_s = 1000 / frame_ms
        self._trigger = _Window(max(1, round(cfg.speech_trigger_ms / frame_ms)))
        self._preroll: collections.deque[np.ndarray] = collections.deque(
            maxlen=max(1, round(self.PREROLL_S * frames_per_s))
        )
        self._hangup_frames = max(1, round(cfg.hangup_silence_s * frames_per_s))
        self._dead_frames = max(1, round(cfg.line_dead_s * frames_per_s))
        self._max_frames = max(1, round(cfg.max_call_s * frames_per_s))

        self._recorded: list[np.ndarray] = []
        self._silence_run = 0
        self._dead_run = 0
        self._started_at = 0.0
        self._last_end_at = 0.0
        self.discarded = 0  # calls dropped for being too short

    # -- main entry point --------------------------------------------------

    def push(self, frame: np.ndarray) -> Call | None:
        """Feed one frame. Returns a Call on the frame where one completes."""
        level = dbfs(frame)
        voiced = self._is_speech(frame, level)

        if self.state is State.IDLE:
            self._preroll.append(frame)
            self._trigger.push(voiced)
            if self._trigger.full and self._trigger.ratio >= self.TRIGGER_RATIO:
                self._begin()
            return None

        self._recorded.append(_as_int16(frame))
        self._silence_run = 0 if voiced else self._silence_run + 1
        self._dead_run = self._dead_run + 1 if level < self.cfg.line_dead_dbfs else 0

        # An open phone line is never digitally silent -- there is always line
        # noise and room tone coming through the handset. So "nobody is
        # talking" and "the handset went back on the cradle" are different
        # signals, and only the second one means the call is over.
        #
        # Ending on the dead line is what makes a long pause safe: someone
        # walking off to read a model number off the water heater leaves the
        # line open, so the recording keeps running.
        if self._dead_run >= self._dead_frames:
            return self._finish("hangup", trim=self._dead_run)
        # Fallback for lines that stay noisy after the other end hangs up, or
        # a handset left off the cradle in a quiet room. Deliberately long.
        if self._silence_run >= self._hangup_frames:
            return self._finish("silence", trim=self._silence_run)
        if len(self._recorded) >= self._max_frames:
            return self._finish("max_length")
        return None

    def flush(self) -> Call | None:
        """Close out an in-progress call, e.g. on Ctrl-C or a tray Quit."""
        if self.state is State.IN_CALL:
            return self._finish("shutdown")
        return None

    # -- internals ---------------------------------------------------------

    def _is_speech(self, frame: np.ndarray, level: float) -> bool:
        # The energy gate comes first and is the important one. webrtcvad will
        # happily label steady line hiss as speech; a level check will not.
        if level < self.cfg.noise_floor_dbfs:
            return False
        try:
            return self._vad.is_speech(to_mono_int16(frame), self.sample_rate)
        except Exception:
            # A malformed frame should not take the process down; treat it as
            # silence and let the next one decide.
            return False

    def _begin(self) -> None:
        self._recorded = [_as_int16(f) for f in self._preroll]
        self._preroll.clear()
        self._trigger.clear()
        self._silence_run = 0
        self._dead_run = 0
        # The pre-roll means the recording begins before the speech that
        # triggered it, so wind the timestamp back by the same amount -- but
        # never past the end of the previous call, or two calls would appear to
        # overlap and their output folders would sort out of order.
        rewound = time.time() - len(self._recorded) * self.frame_ms / 1000
        self._started_at = max(rewound, self._last_end_at)
        self._set_state(State.IN_CALL)

    def _finish(self, reason: str, trim: int = 0) -> Call | None:
        frames = self._recorded
        self._recorded = []
        self._trigger.clear()
        self._preroll.clear()
        self._set_state(State.IDLE)

        if not frames:
            return None

        # Drop the trailing quiet that ended the call, but leave half a second
        # so the last word doesn't get clipped.
        keep_tail = max(0, trim - round(0.5 * 1000 / self.frame_ms))
        if keep_tail:
            frames = frames[:-keep_tail] or frames[:1]

        audio = np.concatenate(frames, axis=0)
        duration = len(audio) / self.sample_rate
        self._silence_run = 0
        self._dead_run = 0
        self._last_end_at = self._started_at + duration

        if duration < self.cfg.min_call_s:
            self.discarded += 1
            return None

        return Call(
            audio=audio,
            sample_rate=self.sample_rate,
            started_at=self._started_at,
            duration_s=duration,
            ended_reason=reason,
        )

    def _set_state(self, state: State) -> None:
        if state is not self.state:
            self.state = state
            if self.on_state_change:
                self.on_state_change(state)


class ManualDetector:
    """Records between an explicit start and stop, ignoring the line entirely.

    Automatic detection assumes the loudest thing near the phone is the phone.
    In a busy workshop or a shared office that is false, and the room triggers
    recordings that were never calls. This is the answer for those rooms: a
    hotkey decides, and nothing else does.

    Same interface as CallDetector so the runner does not care which it has.
    """

    # Below this a recording is treated as a mis-hit rather than a decision --
    # a double-tap on the hotkey, or a stop that landed early.
    MIN_S = 2.0

    def __init__(
        self,
        cfg,
        sample_rate: int = 16000,
        frame_ms: int = 20,
        on_state_change: Callable[[State], None] | None = None,
    ):
        self.cfg = cfg
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.on_state_change = on_state_change
        self.state = State.IDLE
        self.discarded = 0

        self._max_frames = max(1, round(cfg.max_call_s * 1000 / frame_ms))
        self._recorded: list[np.ndarray] = []
        self._started_at = 0.0
        # Set from the hotkey thread, read from the capture thread. An Event is
        # the whole synchronisation story -- there is nothing else shared.
        self._want = threading.Event()

    @property
    def recording(self) -> bool:
        return self._want.is_set()

    def toggle(self) -> bool:
        """Flip recording on or off. Returns the new state. Thread-safe."""
        if self._want.is_set():
            self._want.clear()
        else:
            self._want.set()
        return self._want.is_set()

    def push(self, frame: np.ndarray) -> Call | None:
        want = self._want.is_set()

        if self.state is State.IDLE:
            if not want:
                return None
            self._begin()

        self._recorded.append(_as_int16(frame))

        if not want:
            return self._finish("manual")
        if len(self._recorded) >= self._max_frames:
            # Stop asking for more, or the next frame would start a new one.
            self._want.clear()
            return self._finish("max_length")
        return None

    def flush(self) -> Call | None:
        if self.state is State.IN_CALL:
            self._want.clear()
            return self._finish("shutdown")
        return None

    def _begin(self) -> None:
        self._recorded = []
        self._started_at = time.time()
        self.state = State.IN_CALL
        if self.on_state_change:
            self.on_state_change(State.IN_CALL)

    def _finish(self, reason: str) -> Call | None:
        frames = self._recorded
        self._recorded = []
        self.state = State.IDLE
        if self.on_state_change:
            self.on_state_change(State.IDLE)

        if not frames:
            return None

        audio = np.concatenate(frames, axis=0)
        duration = len(audio) / self.sample_rate

        # The floor only guards against a mis-hit on the hotkey. Hitting the
        # length cap or shutting down is not a mis-hit, and discarding those
        # would throw away a recording nobody asked to discard.
        if reason == "manual" and duration < self.MIN_S:
            self.discarded += 1
            return None

        return Call(
            audio=audio,
            sample_rate=self.sample_rate,
            started_at=self._started_at,
            duration_s=duration,
            ended_reason=reason,
        )


def _as_int16(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.int16:
        return frame
    block = frame if frame.ndim > 1 else frame[:, None]
    return (np.clip(block, -1.0, 1.0) * 32767.0).astype(np.int16)


def _make_vad(aggressiveness: int):
    try:
        import webrtcvad
    except ImportError as exc:  # pragma: no cover - install-time problem
        raise RuntimeError(
            "webrtcvad is not installed, so calls cannot be detected. "
            "Re-run install.bat."
        ) from exc
    return webrtcvad.Vad(aggressiveness)
