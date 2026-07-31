"""Manual recording: the hotkey decides, nothing else does."""

import numpy as np
import pytest

from call_transcriber import hotkey
from call_transcriber.config import DetectConfig
from call_transcriber.vad import ManualDetector, State

RATE = 16000
FRAME_MS = 20
FRAME = RATE * FRAME_MS // 1000


def loud_frame():
    return np.full((FRAME, 1), 0.4, dtype=np.float32)


def detector(**overrides):
    return ManualDetector(DetectConfig(**overrides), RATE, FRAME_MS)


def feed(det, count, frame=None):
    result = None
    for _ in range(count):
        call = det.push(loud_frame() if frame is None else frame)
        if call is not None:
            result = call
    return result


# -- the point of the mode -------------------------------------------------


def test_a_loud_room_records_nothing_until_told():
    """The reason this mode exists."""
    det = detector()
    feed(det, 2000)  # 40 seconds of noise

    assert det.state is State.IDLE
    assert det.discarded == 0


def test_the_hotkey_starts_and_stops_it():
    det = detector()
    assert det.toggle() is True

    assert feed(det, 200) is None
    assert det.state is State.IN_CALL

    assert det.toggle() is False
    call = feed(det, 1)

    assert call is not None
    assert call.ended_reason == "manual"
    assert det.state is State.IDLE


def test_the_recording_covers_the_time_between_presses():
    det = detector()
    det.toggle()
    feed(det, 500)  # 10 seconds
    det.toggle()
    call = feed(det, 1)

    assert 9.9 < call.duration_s < 10.2


def test_silence_during_a_manual_recording_does_not_end_it():
    """No pause is ever a hangup here -- only the hotkey ends a recording."""
    det = detector()
    det.toggle()
    feed(det, 3000, frame=np.zeros((FRAME, 1), dtype=np.float32))

    assert det.state is State.IN_CALL


def test_a_double_tap_is_discarded_rather_than_transcribed():
    det = detector()
    det.toggle()
    feed(det, 5)
    det.toggle()

    assert feed(det, 1) is None
    assert det.discarded == 1


def test_a_forgotten_stop_cannot_record_forever():
    det = detector(max_call_s=1.0)
    det.toggle()
    call = feed(det, 200)

    assert call is not None
    assert call.ended_reason == "max_length"
    assert det.recording is False


def test_it_does_not_immediately_start_again_after_the_length_cap():
    det = detector(max_call_s=1.0)
    det.toggle()
    feed(det, 200)

    assert feed(det, 100) is None
    assert det.state is State.IDLE


def test_back_to_back_recordings_stay_separate():
    det = detector()
    det.toggle(); feed(det, 300); det.toggle()
    first = feed(det, 1)

    det.toggle(); feed(det, 600); det.toggle()
    second = feed(det, 1)

    assert first.duration_s < second.duration_s
    assert second.duration_s < first.duration_s * 3


def test_flush_saves_a_recording_left_running_at_shutdown():
    det = detector()
    det.toggle()
    feed(det, 300)
    call = det.flush()

    assert call is not None
    assert call.ended_reason == "shutdown"
    assert det.flush() is None


def test_state_changes_are_reported_for_the_tray():
    seen = []
    det = ManualDetector(DetectConfig(), RATE, FRAME_MS, on_state_change=seen.append)

    det.toggle(); feed(det, 300); det.toggle(); feed(det, 1)

    assert seen == [State.IN_CALL, State.IDLE]


# -- hotkey parsing --------------------------------------------------------


@pytest.mark.parametrize("written,expected", [
    ("ctrl+alt+r", "<ctrl>+<alt>+r"),
    ("CTRL+ALT+R", "<ctrl>+<alt>+r"),
    ("ctrl + shift + p", "<ctrl>+<shift>+p"),
    ("f9", "<f9>"),
    ("ctrl+space", "<ctrl>+<space>"),
    ("win+r", "<cmd>+r"),
    ("control+option+r", "<ctrl>+<alt>+r"),
])
def test_hotkeys_are_written_the_friendly_way(written, expected):
    assert hotkey.to_pynput(written) == expected


def test_an_empty_hotkey_is_rejected():
    with pytest.raises(hotkey.HotkeyError):
        hotkey.to_pynput("  ")


def test_a_short_recording_survives_if_it_was_not_a_double_tap():
    """Only a manual stop can be a mis-hit; a length cap or shutdown cannot."""
    det = detector(max_call_s=0.5)
    det.toggle()
    call = feed(det, 100)

    assert call is not None
    assert call.duration_s < ManualDetector.MIN_S
    assert det.discarded == 0


# -- switching from one caller straight to the next ------------------------


def test_stop_then_start_between_frames_still_makes_two_recordings():
    """Ending one call and answering the next waiting on hold.

    Both presses land inside a single 20 ms frame, so a plain flag would show
    only "recording" and the two callers would end up in one work order.
    """
    det = detector()
    det.toggle()
    feed(det, 300)

    det.toggle()  # stop caller one
    det.toggle()  # start caller two -- same frame, no push in between

    first = feed(det, 1)
    assert first is not None
    assert first.ended_reason == "manual"

    feed(det, 600)
    det.toggle()
    second = feed(det, 1)

    assert second is not None
    assert second.duration_s > first.duration_s
    # Caller two's recording must not contain caller one.
    assert second.duration_s < first.duration_s * 3


def test_the_second_recording_starts_clean():
    det = detector()
    det.toggle(); feed(det, 500); det.toggle(); feed(det, 1)

    det.toggle()
    feed(det, 250)
    det.toggle()
    second = feed(det, 1)

    assert 4.9 < second.duration_s < 5.2


def test_a_burst_of_presses_does_not_lose_the_final_state():
    det = detector()
    for _ in range(6):  # even count -- ends up back at "not recording"
        det.toggle()

    assert det.recording is False
    feed(det, 400)
    assert det.state is State.IDLE


def test_an_odd_burst_of_presses_leaves_it_recording():
    det = detector()
    for _ in range(5):
        det.toggle()

    assert det.recording is True
    feed(det, 400)
    assert det.state is State.IN_CALL


def test_presses_queued_while_idle_are_honoured_in_order():
    det = detector()
    det.toggle()   # start
    det.toggle()   # stop, before a single frame has been processed

    assert feed(det, 200) is None      # too short to keep, but it did run
    assert det.discarded == 1
    assert det.state is State.IDLE
