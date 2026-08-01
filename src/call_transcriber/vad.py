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

        # Presses are queued rather than collapsed into a single flag. Frames
        # arrive every 20 ms, and someone ending one call and starting the next
        # can easily press stop and start inside one frame. A flag would show
        # only the final value -- still "recording" -- and the two calls would
        # silently merge into one recording. The queue guarantees every press
        # produces its transition, even if it lands a frame or two later.
        self._lock = threading.Lock()
        self._pending: collections.deque[bool] = collections.deque()
        self._recording = False

    @property
    def recording(self) -> bool:
        with self._lock:
            return self._recording

    def toggle(self) -> bool:
        """Flip recording on or off. Returns the new state. Thread-safe."""
        with self._lock:
            self._recording = not self._recording
            self._pending.append(self._recording)
            return self._recording

    def _next_command(self) -> bool | None:
        """Take one queued press, so a rapid pair is two recordings not one."""
        with self._lock:
            return self._pending.popleft() if self._pending else None

    def push(self, frame: np.ndarray) -> Call | None:
        command = self._next_command()

        if self.state is State.IDLE:
            if command is not True:
                return None
            self._begin()
            self._recorded.append(_as_int16(frame))
            return None

        self._recorded.append(_as_int16(frame))

        if command is False:
            return self._finish("manual")
        if len(self._recorded) >= self._max_frames:
            # Drop any queued presses too: they were aimed at a recording that
            # no longer exists.
            with self._lock:
                self._recording = False
                self._pending.clear()
            return self._finish("max_length")
        return None

    def flush(self) -> Call | None:
        with self._lock:
            self._recording = False
            self._pending.clear()
        if self.state is State.IN_CALL:
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


def speech_windows(
    samples: np.ndarray,
    sample_rate: int = 16000,
    # 20 ms, matching CallDetector. webrtcvad's verdict genuinely differs by
    # frame length, and having two parts of one app disagree about what counts
    # as speech is a bug waiting to be blamed on something else.
    frame_ms: int = 20,
    aggressiveness: int = 2,
    max_window_s: float = 25.0,
    min_silence_s: float = 0.4,
) -> list[tuple[float, float]]:
    """Cut a finished recording into pieces Parakeet can take in one go.

    Whisper slices long audio up internally; Parakeet does not -- it recognises
    one utterance, so a long call has to be cut somewhere.

    **The windows returned are contiguous and cover the whole recording.** That
    is the part that matters. An earlier version returned only the stretches
    webrtcvad called speech and quietly dropped everything between them, which
    on a real call lost the customer's short replies -- "it's a house", "that's
    perfect", "nope", "yes, it is" -- because a brief, quiet answer from the far
    end of a phone line is precisely what a voice detector is least certain
    about. Those replies are the ones that confirm the fields on the work order,
    so losing them is far worse than handing the engine some silence.

    The detector therefore decides only *where* to cut, never *what* to keep.

    `samples` is float32 mono in [-1, 1]. Returns (start, end) in seconds.
    """
    duration = len(samples) / sample_rate
    step = int(sample_rate * frame_ms / 1000)
    if duration <= max_window_s or step <= 0 or len(samples) < step:
        return [(0.0, duration)]

    voiced = _voiced_frames(samples, sample_rate, frame_ms, aggressiveness, step)

    # An all-or-nothing guard, not a filter. A recording with no speech in it
    # anywhere is worth skipping; a quiet moment *within* one is not.
    if voiced and not any(voiced):
        return []

    cuts = _pauses(voiced, frame_ms, min_silence_s)
    return _windows_between(cuts, duration, max_window_s, samples, sample_rate)


def _voiced_frames(
    samples: np.ndarray, sample_rate: int, frame_ms: int, aggressiveness: int, step: int
) -> list[bool]:
    """Which 20 ms frames webrtcvad thinks contain speech. [] if it is absent."""
    try:
        import webrtcvad
    except ImportError:  # pragma: no cover - install-time problem
        return []

    detector = webrtcvad.Vad(aggressiveness)
    voiced: list[bool] = []
    for start in range(0, len(samples) - step + 1, step):
        frame = samples[start : start + step]
        pcm = (np.clip(frame, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        try:
            voiced.append(detector.is_speech(pcm, sample_rate))
        except Exception:
            voiced.append(False)
    return voiced


def _pauses(voiced: list[bool], frame_ms: int, min_silence_s: float) -> list[float]:
    """The midpoint of every silence long enough to be worth cutting at."""
    needed = max(1, round(min_silence_s * 1000 / frame_ms))
    out: list[float] = []
    run = 0
    for index, is_speech in enumerate(voiced):
        if is_speech:
            if run >= needed:
                out.append((index - run / 2) * frame_ms / 1000)
            run = 0
        else:
            run += 1
    if run >= needed:
        out.append((len(voiced) - run / 2) * frame_ms / 1000)
    return out


def _windows_between(
    cuts: list[float],
    duration: float,
    limit: float,
    samples: np.ndarray,
    sample_rate: int,
) -> list[tuple[float, float]]:
    """Contiguous windows, each no longer than the limit, split at the pauses."""
    bounds = [0.0]
    index = 0
    while duration - bounds[-1] > limit:
        ceiling = bounds[-1] + limit
        latest = None
        while index < len(cuts) and cuts[index] <= ceiling:
            if cuts[index] > bounds[-1]:
                latest = cuts[index]
            index += 1

        # Nobody drew breath for a whole window. Cut at the quietest instant
        # near the limit rather than exactly on it -- landing mid-vowel is what
        # turned "basically" into "...a trolley based" / "Basically, closer to
        # the ceiling" on a real call.
        if latest is None:
            latest = _quietest(samples, sample_rate, bounds[-1], ceiling)
        bounds.append(latest)

    bounds.append(duration)
    return list(zip(bounds, bounds[1:]))


def _quietest(
    samples: np.ndarray,
    sample_rate: int,
    start: float,
    end: float,
    tail: float = 0.25,
    block_ms: int = 20,
) -> float:
    """The quietest moment in the last stretch before `end`.

    Searching only the tail keeps windows close to the limit; searching at all
    is what stops the cut landing in the middle of a word.
    """
    from_time = end - (end - start) * tail
    first = int(from_time * sample_rate)
    last = min(len(samples), int(end * sample_rate))
    block = max(1, int(sample_rate * block_ms / 1000))
    if last - first < block * 2:
        return end

    usable = (last - first) // block * block
    blocks = samples[first : first + usable].reshape(-1, block)
    quietest = int(np.argmin(np.abs(blocks).mean(axis=1)))
    return (first + (quietest + 0.5) * block) / sample_rate


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
