"""Turning speaker turns into a labelled transcript.

The diarizer itself needs sherpa-onnx and a model download, so what is tested
here is everything around it: the turn tidying, the labelling, and that a
missing model degrades to an unlabelled transcript rather than losing the call.
"""

import numpy as np
import pytest

from call_transcriber import config, diarize, pipeline, transcribe
from call_transcriber.diarize import Turn


# -- tidying the turns -----------------------------------------------------


def test_a_speaker_carrying_on_across_a_breath_is_one_turn():
    """Diarizers cut on every pause; a sentence is not several turns."""
    merged = diarize.merge_adjacent([
        Turn(0.0, 2.0, 0),
        Turn(2.2, 4.0, 0),
        Turn(4.1, 5.0, 0),
    ])

    assert len(merged) == 1
    assert merged[0].start == 0.0
    assert merged[0].end == 5.0


def test_the_other_person_starting_ends_the_turn():
    merged = diarize.merge_adjacent([
        Turn(0.0, 2.0, 0),
        Turn(2.1, 4.0, 1),
        Turn(4.1, 6.0, 0),
    ])

    assert [t.speaker for t in merged] == [0, 1, 0]


def test_a_long_gap_is_left_as_two_turns():
    merged = diarize.merge_adjacent([Turn(0.0, 2.0, 0), Turn(9.0, 11.0, 0)])
    assert len(merged) == 2


def test_nothing_to_merge_is_not_an_error():
    assert diarize.merge_adjacent([]) == []


# -- naming them -----------------------------------------------------------


def test_speakers_are_named_in_the_order_they_first_speak():
    names = diarize.label([Turn(0.0, 1.0, 3), Turn(1.0, 2.0, 1), Turn(2.0, 3.0, 3)])
    assert names == {3: "SIDE A", 1: "SIDE B"}


def test_the_labels_say_nothing_about_who_is_the_customer():
    """That is a judgement about content, and belongs where the words are."""
    names = diarize.label([Turn(0.0, 1.0, 0), Turn(1.0, 2.0, 1)])
    for name in names.values():
        assert "customer" not in name.lower()
        assert "agent" not in name.lower()


def test_more_speakers_than_expected_still_get_names():
    turns = [Turn(float(i), float(i + 1), i) for i in range(6)]
    names = diarize.label(turns)
    assert len(names) == 6
    assert len(set(names.values())) == 6


# -- building the transcript -----------------------------------------------


@pytest.fixture
def whisper_stub(monkeypatch):
    """Return one segment per slice, so timing offsets are checkable."""
    def fake_run(samples, cfg, root):
        seconds = len(samples) / 16000
        return [transcribe.Segment(0.0, seconds, f"[{seconds:.1f}s of speech]")]

    monkeypatch.setattr(transcribe, "_run", fake_run)


def test_each_line_carries_the_speaker_who_said_it(whisper_stub):
    audio = (np.random.randn(16000 * 6, 1) * 3000).astype(np.int16)
    cfg = config.load(__import__("pathlib").Path("/nope.toml")).transcribe

    result = transcribe.transcribe(
        audio, 16000, cfg, "mono",
        turns=[Turn(0.0, 2.0, 0), Turn(2.0, 4.0, 1), Turn(4.0, 6.0, 0)],
    )

    assert result.layout == "diarized"
    assert [s.speaker for s in result.segments] == ["SIDE A", "SIDE B", "SIDE A"]
    assert result.text.startswith("SIDE A:")


def test_timestamps_are_shifted_into_the_whole_call(whisper_stub):
    audio = (np.random.randn(16000 * 6, 1) * 3000).astype(np.int16)
    cfg = config.load(__import__("pathlib").Path("/nope.toml")).transcribe

    result = transcribe.transcribe(
        audio, 16000, cfg, "mono", turns=[Turn(0.0, 2.0, 0), Turn(4.0, 6.0, 1)],
    )

    # The second turn's text starts four seconds in, not at zero.
    assert result.segments[1].start == pytest.approx(4.0, abs=0.1)


def test_a_flicker_of_a_turn_is_skipped(whisper_stub):
    """A tenth of a second is a cough, not a sentence."""
    audio = (np.random.randn(16000 * 6, 1) * 3000).astype(np.int16)
    cfg = config.load(__import__("pathlib").Path("/nope.toml")).transcribe

    result = transcribe.transcribe(
        audio, 16000, cfg, "mono", turns=[Turn(0.0, 0.1, 0), Turn(1.0, 3.0, 1)],
    )

    assert len(result.segments) == 1


# -- failing softly --------------------------------------------------------


def test_missing_models_lose_the_labels_not_the_call(tmp_path, monkeypatch):
    cfg = config.load(tmp_path / "none.toml", root=tmp_path)
    cfg.diarize.enabled = True

    monkeypatch.setattr(
        pipeline.transcribe, "transcribe",
        lambda *a, **k: transcribe.Transcript(
            [transcribe.Segment(0, 1, "hello")], "en", 1.0, "mono"
        ),
    )
    monkeypatch.setattr(pipeline.extract, "extract", lambda *a, **k: {})
    monkeypatch.setattr(pipeline.notify, "copy", lambda text: True)

    call = pipeline.Call(
        audio=(np.random.randn(16000, 1) * 3000).astype(np.int16),
        sample_rate=16000, started_at=1785500000.0, duration_s=1.0,
        ended_reason="manual",
    )
    result = pipeline.process_call(call, cfg)

    assert "hello" in result.transcript
    assert any("Speaker labelling was skipped" in w for w in result.warnings)
    assert result.problems == []


def test_diarization_reports_whether_it_can_run():
    assert diarize.available() in (True, False)
