"""Wiring: audio in, work order out.

Three threads and no shared mutable state between them beyond two queues.

  PortAudio callback  -> frames queue
  detector thread     -> drains frames, emits finished calls
  worker thread       -> transcribe, extract, save, notify

The split matters: transcription of a ten-minute call takes tens of seconds,
and the detector must not be blocked during it or the next incoming call is
missed entirely.
"""

from __future__ import annotations

import concurrent.futures
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import audio as audio_mod
from . import extract, hotkey, notify, storage, transcribe, workorder
from .vad import Call, CallDetector, ManualDetector, State

log = logging.getLogger(__name__)


@dataclass
class Result:
    folder: Path | None
    work_order: str
    transcript: str
    extracted: dict
    warnings: list[str] = field(default_factory=list)


CLIP_THRESHOLD = 32700  # int16 full scale is 32767
CLIP_FRACTION = 0.005  # half a percent of samples pinned is audible damage
QUIET_PEAK_DBFS = -20.0


def level_warnings(audio: np.ndarray) -> list[str]:
    """Flag recording levels that will quietly ruin the transcript.

    Both failure modes trace back to one physical knob on the adapter, so the
    messages name it. Clipping is the worse of the two: whisper copes with a
    quiet recording far better than with a distorted one, because clipping
    destroys the waveform rather than just shrinking it.
    """
    if audio.size == 0:
        return []

    peak = int(np.abs(audio).max())
    clipped = float(np.count_nonzero(np.abs(audio) >= CLIP_THRESHOLD)) / audio.size

    if clipped > CLIP_FRACTION:
        return [
            f"{clipped:.1%} of this recording is clipped -- the level is too high "
            f"and the distortion will cost you accuracy. Turn the record level "
            f"dial on the adapter down."
        ]

    peak_dbfs = 20 * np.log10(peak / 32768) if peak else -np.inf
    if peak_dbfs < QUIET_PEAK_DBFS:
        return [
            f"This recording peaked at only {peak_dbfs:.0f} dBFS, which is very "
            f"quiet. Turn the record level dial on the adapter up, aiming for "
            f"peaks around -12 dBFS."
        ]
    return []


def process_call(call: Call, cfg, *, ui=None) -> Result:
    """Transcribe, extract, save and surface one finished call."""
    warnings: list[str] = level_warnings(call.audio)
    started = time.monotonic()

    # The recording is transcribed straight out of memory and, unless
    # keep_audio is on, is never written anywhere. There is no temp file to
    # leak, no window during which a copy exists on disk, and nothing to clean
    # up if this function raises.
    log.info("transcribing %.0fs of audio", call.duration_s)
    result = transcribe.transcribe(
        call.audio, call.sample_rate, cfg.transcribe, cfg.audio.stereo_mode
    )
    transcript_text = result.text

    if result.is_empty:
        warnings.append(
            "Nothing intelligible was transcribed -- check the adapter is "
            "tapping the handset and that [detect] noise_floor_dbfs isn't too high."
        )
        extracted = extract.empty_result()
    else:
        log.info(
            "transcribed %d segments (%s layout); extracting fields",
            len(result.segments), result.layout,
        )
        try:
            extracted = extract.extract(transcript_text, cfg.extract, cfg.business)
        except extract.ExtractionError as exc:
            # The transcript is the expensive part and it survived, so save it
            # with a warning rather than throwing the call away.
            warnings.append(f"Field extraction failed: {exc}")
            extracted = extract.empty_result()

    order = workorder.render(
        extracted,
        started_at=call.started_at,
        duration_s=call.duration_s,
        business=cfg.business,
    )

    folder = storage.call_dir(cfg.output_dir, call.started_at, workorder.slug(extracted))
    storage.save_results(
        folder,
        order,
        transcript_text,
        extracted,
        save_transcript=cfg.output.save_transcript,
        meta={
            "started_at": call.started_at,
            "duration_s": round(call.duration_s, 2),
            "ended_reason": call.ended_reason,
            "audio_layout": result.layout,
            "whisper_model": cfg.transcribe.model,
            "extract_model": cfg.extract.model,
            "processing_s": round(time.monotonic() - started, 1),
        },
    )

    if cfg.output.keep_audio:
        kept = storage.write_wav(folder / "call.wav", call.audio, call.sample_rate)
        warnings.append(f"Audio kept at {kept} (output.keep_audio is on)")

    copied = notify.copy(order) if cfg.output.copy_to_clipboard else False

    log.info("work order ready in %s", folder)
    if ui is not None:
        ui.request(
            notify.Popup(
                title=workorder.headline(extracted),
                work_order=order,
                transcript=transcript_text,
                folder=folder,
                warnings=warnings,
                copied=copied,
            )
        )

    return Result(folder, order, transcript_text, extracted, warnings)


def process_file(path: Path, cfg, *, ui=None) -> Result:
    """Run the pipeline over an existing audio file, for testing."""
    data, rate = storage.read_wav(path)
    call = Call(
        audio=data,
        sample_rate=rate,
        started_at=path.stat().st_mtime,
        duration_s=len(data) / rate,
        ended_reason="file",
    )
    if rate != cfg.audio.sample_rate:
        log.info("file is %d Hz; whisper resamples internally", rate)
    return process_call(call, cfg, ui=ui)


class Runner:
    """Owns the live capture loop."""

    # A phone line is quiet almost all day. An input that reads as an active
    # call nearly continuously is not a phone line -- it is a microphone
    # hearing the room, which means the adapter is on the wrong jack or in the
    # wrong mode. Worth saying out loud, because the symptom otherwise looks
    # like "it started recording the moment I ran it" and reads as a bug.
    HOT_INPUT_AFTER_FRAMES = 15_000  # five minutes of 20 ms frames
    HOT_INPUT_RATIO = 0.9

    def __init__(self, cfg, ui=None):
        self.cfg = cfg
        self.ui = ui
        self.stop_event = threading.Event()
        self.paused = threading.Event()
        self.state = State.IDLE
        self.calls_handled = 0
        self._frames_seen = 0
        self._frames_in_call = 0
        self._warned_hot = False
        self._thread: threading.Thread | None = None
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="transcribe"
        )
        self._error: Exception | None = None
        self._detector = None
        self._hotkey_listener = None

    @property
    def manual(self) -> bool:
        return self.cfg.control.mode == "manual"

    def toggle_recording(self) -> bool | None:
        """Start or stop recording by hand. Returns the new state.

        Called from the hotkey listener thread and from the tray menu. Only
        does anything in manual mode.
        """
        detector = self._detector
        if detector is None or not hasattr(detector, "toggle"):
            return None
        started = detector.toggle()
        log.info("recording started by hotkey" if started else "recording stopped by hotkey")
        if self.cfg.control.beep:
            notify.beep(started)
        return started

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="detect", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        hotkey.stop(self._hotkey_listener)
        self._hotkey_listener = None
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._pool.shutdown(wait=True)
        if self.ui is not None:
            self.ui.shutdown()

    @property
    def error(self) -> Exception | None:
        return self._error

    # -- capture loop ------------------------------------------------------

    def _loop(self) -> None:
        try:
            device = audio_mod.find_device(
                self.cfg.audio.device_match, self.cfg.audio.device_index
            )
            log.info("listening on %s", device)

            detector_class = ManualDetector if self.manual else CallDetector
            detector = detector_class(
                self.cfg.detect,
                sample_rate=self.cfg.audio.sample_rate,
                on_state_change=self._on_state,
            )
            self._detector = detector

            if self.manual:
                # Fatal in manual mode: without the hotkey there is no way to
                # record at all, and failing quietly would look like the audio
                # is broken.
                self._hotkey_listener = hotkey.start(
                    self.cfg.control.hotkey, self.toggle_recording
                )

            with audio_mod.Capture(device, target_rate=self.cfg.audio.sample_rate) as capture:
                if self.manual:
                    log.info(
                        "ready -- press %s to start recording, and again to stop",
                        self.cfg.control.hotkey.upper(),
                    )
                else:
                    log.info(
                        "ready -- calls are detected automatically (the line going "
                        "quiet for %.0fs ends one)",
                        self.cfg.detect.line_dead_s,
                    )
                for frame in capture.frames():
                    if self.stop_event.is_set():
                        break
                    if self.paused.is_set():
                        continue
                    call = detector.push(frame)
                    if call is not None:
                        self._submit(call)
                    self._note_frame(detector.state)

                final = detector.flush()
                if final is not None:
                    self._submit(final)

                stats = capture.stats
                if stats["dropped_frames"]:
                    log.warning(
                        "dropped %d audio frames -- the machine could not keep up",
                        stats["dropped_frames"],
                    )
        except Exception as exc:
            self._error = exc
            log.error("capture stopped: %s", exc)
        finally:
            self.stop_event.set()
            if self.ui is not None:
                self.ui.shutdown()

    def _note_frame(self, state: State) -> None:
        """Watch for an input that is never idle, and say so once."""
        self._frames_seen += 1
        if state is State.IN_CALL:
            self._frames_in_call += 1

        if self.manual or self._warned_hot or self._frames_seen < self.HOT_INPUT_AFTER_FRAMES:
            return
        ratio = self._frames_in_call / self._frames_seen
        if ratio < self.HOT_INPUT_RATIO:
            return

        self._warned_hot = True
        log.warning(
            "This input has counted as an active call for %.0f%% of the last %d "
            "minutes. A phone line is silent when nobody is on it, so the adapter "
            "is almost certainly hearing the room rather than the handset.",
            ratio * 100,
            self.HOT_INPUT_AFTER_FRAMES // 3000,
        )
        log.warning(
            "Stop with Ctrl-C and run `run.bat levels`. Put the handset down and "
            "watch the meter: if it stays high, the problem is which jack the "
            "adapter is plugged into or which position its switch is in -- not "
            "anything in config.toml."
        )

    def _submit(self, call: Call) -> None:
        self.calls_handled += 1
        log.info(
            "call #%d ended after %.0fs (%s) -- processing",
            self.calls_handled, call.duration_s, call.ended_reason,
        )
        self._pool.submit(self._safe_process, call)

    def _safe_process(self, call: Call) -> None:
        try:
            process_call(call, self.cfg, ui=self.ui)
        except Exception as exc:
            log.exception("failed to process call: %s", exc)
        finally:
            # Drop the reference promptly; an hour of stereo audio is ~460 MB.
            call.audio = np.zeros((0, 1), dtype=np.int16)

    def _on_state(self, state: State) -> None:
        self.state = state
        if not self.manual:
            log.info("recording started" if state is State.IN_CALL else "call ended")
