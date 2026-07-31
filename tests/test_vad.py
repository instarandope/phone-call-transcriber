import numpy as np
import pytest

from call_transcriber import audio, vad
from call_transcriber.config import DetectConfig

webrtcvad = pytest.importorskip("webrtcvad", reason="webrtcvad-wheels not installed")

RATE = 16000
FRAME_MS = 20
FRAME = RATE * FRAME_MS // 1000


def speech_frame(amplitude=0.3):
    """A voiced-sounding frame: a low harmonic stack, like a voice on a phone."""
    t = np.arange(FRAME) / RATE
    wave = sum(np.sin(2 * np.pi * f * t) / (i + 1) for i, f in enumerate((150, 300, 450, 900)))
    return (amplitude * wave / 2).astype(np.float32).reshape(-1, 1)


def silent_frame(amplitude=0.0):
    return np.full((FRAME, 1), amplitude, dtype=np.float32)


def detector(**overrides):
    cfg = DetectConfig(**{"min_call_s": 0.0, "hangup_silence_s": 0.2, **overrides})
    return vad.CallDetector(cfg, sample_rate=RATE, frame_ms=FRAME_MS)


def feed(det, frame, count):
    result = None
    for _ in range(count):
        call = det.push(frame)
        if call is not None:
            result = call
    return result


# -- level metering --------------------------------------------------------


def test_digital_silence_is_negative_infinity():
    assert audio.dbfs(np.zeros((FRAME, 1), dtype=np.float32)) == -np.inf


def test_full_scale_is_about_zero_dbfs():
    assert abs(audio.dbfs(np.ones((FRAME, 1), dtype=np.float32))) < 0.1


def test_mono_conversion_produces_the_byte_count_webrtcvad_expects():
    assert len(audio.to_mono_int16(speech_frame())) == FRAME * 2


# -- state machine ---------------------------------------------------------


def test_silence_never_starts_a_recording():
    det = detector()
    feed(det, silent_frame(), 100)
    assert det.state is vad.State.IDLE


def test_line_hiss_below_the_noise_floor_never_starts_a_recording():
    # Quiet enough to be under the floor, but structured enough that the VAD
    # alone might well call it speech. The energy gate is what stops it.
    det = detector(noise_floor_dbfs=-40.0)
    feed(det, speech_frame(amplitude=0.001), 200)
    assert det.state is vad.State.IDLE


def test_sustained_speech_starts_a_recording():
    det = detector()
    feed(det, speech_frame(), 40)
    assert det.state is vad.State.IN_CALL


def test_silence_after_speech_ends_the_call():
    det = detector(hangup_silence_s=0.2)
    feed(det, speech_frame(), 40)
    call = feed(det, silent_frame(), 40)

    assert call is not None
    assert call.ended_reason == "silence"
    assert det.state is vad.State.IDLE


def test_the_recording_includes_audio_from_before_the_trigger():
    """Otherwise every transcript opens mid-word."""
    det = detector()
    feed(det, speech_frame(), 40)
    call = feed(det, silent_frame(), 40)

    trigger_s = DetectConfig().speech_trigger_ms / 1000
    assert call.duration_s > 40 * FRAME_MS / 1000 - trigger_s


def test_a_pause_mid_call_does_not_split_it_in_two():
    det = detector(hangup_silence_s=1.0)
    feed(det, speech_frame(), 40)
    assert feed(det, silent_frame(), 20) is None  # 0.4s pause
    assert det.state is vad.State.IN_CALL
    assert feed(det, speech_frame(), 20) is None


def test_calls_shorter_than_the_minimum_are_discarded():
    det = detector(min_call_s=30.0, hangup_silence_s=0.2)
    feed(det, speech_frame(), 40)
    call = feed(det, silent_frame(), 40)

    assert call is None
    assert det.discarded == 1
    assert det.state is vad.State.IDLE


def test_a_handset_left_off_the_cradle_cannot_record_forever():
    det = detector(max_call_s=1.0, hangup_silence_s=60.0)
    call = feed(det, speech_frame(), 200)

    assert call is not None
    assert call.ended_reason == "max_length"
    assert call.duration_s <= 1.1


def test_flush_closes_an_in_progress_call_on_shutdown():
    det = detector()
    feed(det, speech_frame(), 40)
    call = det.flush()

    assert call is not None
    assert call.ended_reason == "shutdown"
    assert det.flush() is None


def test_state_changes_are_reported_once_each():
    seen = []
    cfg = DetectConfig(min_call_s=0.0, hangup_silence_s=0.2)
    det = vad.CallDetector(cfg, RATE, FRAME_MS, on_state_change=seen.append)

    feed(det, speech_frame(), 40)
    feed(det, silent_frame(), 40)

    assert seen == [vad.State.IN_CALL, vad.State.IDLE]


def test_captured_audio_is_int16_and_keeps_its_channels():
    det = detector()
    feed(det, speech_frame(), 40)
    call = feed(det, silent_frame(), 40)

    assert call.audio.dtype == np.int16
    assert call.audio.ndim == 2
    assert call.channels == 1


def test_back_to_back_calls_are_kept_separate():
    det = detector(hangup_silence_s=0.2)
    feed(det, speech_frame(), 40)
    first = feed(det, silent_frame(), 40)

    feed(det, speech_frame(), 40)
    second = feed(det, silent_frame(), 40)

    assert first is not None and second is not None
    assert second.started_at >= first.started_at
