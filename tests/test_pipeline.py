"""End-to-end wiring, with the two model calls stubbed.

These exercise the part most likely to break silently: what happens to the
recording and the outputs when a stage fails.
"""

import json

import numpy as np
import pytest

from call_transcriber import config, extract, pipeline, transcribe
from call_transcriber.vad import Call


@pytest.fixture
def cfg(tmp_path):
    return config.load(tmp_path / "no-such-config.toml", root=tmp_path)


@pytest.fixture
def call():
    audio = (np.random.randn(16000 * 3, 1) * 3000).astype(np.int16)
    return Call(
        audio=audio,
        sample_rate=16000,
        started_at=1785500000.0,
        duration_s=3.0,
        ended_reason="silence",
    )


@pytest.fixture
def stub_models(monkeypatch):
    """Stand in for whisper and Ollama, and record that they were called."""
    seen = {}

    def fake_transcribe(audio, rate, tcfg, stereo_mode="auto", turns=None, root=None):
        seen["transcribed"] = True
        return transcribe.Transcript(
            segments=[
                transcribe.Segment(0.0, 1.5, "Hi, this is Jane Doe at 12 Oak Street."),
                transcribe.Segment(1.5, 3.0, "My furnace is making a banging noise."),
            ],
            language="en",
            duration_s=3.0,
            layout="mono",
        )

    def fake_extract(text, ecfg, business=None):
        seen["extract_input"] = text
        data = extract.empty_result()
        data.update({
            "caller_name": "Jane Doe",
            "service_address": "12 Oak Street",
            "issue_summary": "Furnace making a banging noise",
            "urgency": "routine",
            "missing_info": ["PHONE"],
        })
        return data

    def fake_copy(text):
        seen["copied"] = text
        return True  # the real notify.copy returns whether it succeeded

    monkeypatch.setattr(pipeline.transcribe, "transcribe", fake_transcribe)
    monkeypatch.setattr(pipeline.extract, "extract", fake_extract)
    monkeypatch.setattr(pipeline.notify, "copy", fake_copy)
    return seen


# -- the happy path --------------------------------------------------------


def test_a_call_produces_a_work_order_on_disk(cfg, call, stub_models):
    result = pipeline.process_call(call, cfg)

    assert result.folder.is_dir()
    assert "Jane Doe" in result.work_order
    assert "12 Oak Street" in result.work_order
    assert (result.folder / "work_order.txt").exists()
    assert (result.folder / "transcript.txt").exists()
    assert (result.folder / "extracted.json").exists()


def test_the_transcript_is_what_gets_sent_to_the_extractor(cfg, call, stub_models):
    pipeline.process_call(call, cfg)
    assert "banging noise" in stub_models["extract_input"]


def test_the_work_order_lands_on_the_clipboard(cfg, call, stub_models):
    result = pipeline.process_call(call, cfg)
    assert stub_models["copied"] == result.work_order


def test_metadata_is_recorded_alongside_the_fields(cfg, call, stub_models):
    result = pipeline.process_call(call, cfg)
    payload = json.loads((result.folder / "extracted.json").read_text(encoding="utf-8"))

    assert payload["caller_name"] == "Jane Doe"
    assert payload["_call"]["duration_s"] == 3.0
    assert payload["_call"]["ended_reason"] == "silence"
    assert payload["_call"]["audio_layout"] == "mono"


# -- the recording does not stick around -----------------------------------


def test_no_audio_is_left_behind(cfg, call, stub_models):
    pipeline.process_call(call, cfg)
    assert list(cfg.output_dir.rglob("*.wav")) == []


def test_audio_never_reaches_the_disk_at_all(cfg, call, monkeypatch):
    """Not 'written then deleted' -- never written. There is no window."""
    written = []
    monkeypatch.setattr(
        pipeline.storage, "write_wav",
        lambda path, *a, **k: written.append(path) or path,
    )
    monkeypatch.setattr(
        pipeline.transcribe, "transcribe",
        lambda *a, **k: transcribe.Transcript([], "en", 3.0, "mono"),
    )
    monkeypatch.setattr(pipeline.notify, "copy", lambda text: None)

    pipeline.process_call(call, cfg)
    assert written == []


def test_a_crash_mid_transcription_leaves_no_audio(cfg, call, monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("model crashed")

    monkeypatch.setattr(pipeline.transcribe, "transcribe", explode)

    with pytest.raises(RuntimeError, match="model crashed"):
        pipeline.process_call(call, cfg)

    assert list(cfg.output_dir.rglob("*.wav")) == []


def test_audio_is_kept_only_when_explicitly_asked_for(cfg, call, stub_models):
    cfg.output.keep_audio = True
    result = pipeline.process_call(call, cfg)

    kept = result.folder / "call.wav"
    assert kept.exists()
    assert any("keep_audio" in w for w in result.warnings)


# -- degrading instead of failing ------------------------------------------


def test_a_dead_extractor_still_saves_the_transcript(cfg, call, monkeypatch, stub_models):
    def refuse(*args, **kwargs):
        raise extract.ExtractionError("Ollama is not running")

    monkeypatch.setattr(pipeline.extract, "extract", refuse)

    result = pipeline.process_call(call, cfg)

    assert "banging noise" in (result.folder / "transcript.txt").read_text(encoding="utf-8")
    assert "Nothing could be extracted" in result.work_order

    # A failure, not a note -- and it says how to pick the call back up.
    assert any("Ollama is not running" in p for p in result.problems)
    assert any("run.bat compare" in p for p in result.problems)
    assert not result.warnings


def test_silence_produces_a_warning_rather_than_a_confident_blank_form(
    cfg, call, monkeypatch, stub_models
):
    monkeypatch.setattr(
        pipeline.transcribe,
        "transcribe",
        lambda *a, **k: transcribe.Transcript([], "en", 3.0, "mono"),
    )

    result = pipeline.process_call(call, cfg)
    assert any("Nothing intelligible" in p for p in result.problems)
    assert "Nothing could be extracted" in result.work_order


def test_the_popup_gets_the_work_order_and_the_transcript(cfg, call, stub_models):
    posted = []

    class Ui:
        def request(self, popup):
            posted.append(popup)

    pipeline.process_call(call, cfg, ui=Ui())

    assert len(posted) == 1
    assert posted[0].title == "Jane Doe - Furnace making a banging noise"
    assert "banging noise" in posted[0].transcript
    assert posted[0].folder.is_dir()
    assert posted[0].copied is True


def test_the_popup_does_not_claim_a_copy_that_did_not_happen(cfg, call, stub_models):
    cfg.output.copy_to_clipboard = False
    posted = []

    class Ui:
        def request(self, popup):
            posted.append(popup)

    pipeline.process_call(call, cfg, ui=Ui())
    assert posted[0].copied is False


# -- an input that is never idle -------------------------------------------


def _run_frames(runner, state, count):
    for _ in range(count):
        runner._note_frame(state)


@pytest.fixture
def auto_cfg(cfg):
    """The never-idle warning is about automatic detection, so ask for it."""
    cfg.control.mode = "auto"
    return cfg


def test_a_constantly_hot_input_is_called_out(auto_cfg, caplog):
    """The 'it started recording the moment I ran it' symptom."""
    runner = pipeline.Runner(auto_cfg)
    with caplog.at_level("WARNING"):
        _run_frames(runner, pipeline.State.IN_CALL, runner.HOT_INPUT_AFTER_FRAMES)

    assert "hearing the room" in caplog.text
    assert "run.bat levels" in caplog.text


def test_the_warning_is_said_once_not_every_frame(auto_cfg, caplog):
    runner = pipeline.Runner(auto_cfg)
    with caplog.at_level("WARNING"):
        _run_frames(runner, pipeline.State.IN_CALL, runner.HOT_INPUT_AFTER_FRAMES * 3)

    assert caplog.text.count("hearing the room") == 1


def test_a_normal_phone_line_is_never_warned_about(auto_cfg, caplog):
    """Mostly idle with real calls in it -- the expected shape."""
    runner = pipeline.Runner(auto_cfg)
    with caplog.at_level("WARNING"):
        for _ in range(10):
            _run_frames(runner, pipeline.State.IN_CALL, 1_500)
            _run_frames(runner, pipeline.State.IDLE, 15_000)

    assert "hearing the room" not in caplog.text


def test_it_waits_for_enough_evidence_before_complaining(auto_cfg, caplog):
    """One long call early on is not proof of anything."""
    runner = pipeline.Runner(auto_cfg)
    with caplog.at_level("WARNING"):
        _run_frames(runner, pipeline.State.IN_CALL, runner.HOT_INPUT_AFTER_FRAMES - 1)

    assert "hearing the room" not in caplog.text


# -- recording level -------------------------------------------------------


def _tone(peak, n=16000):
    t = np.linspace(0, 1, n, dtype=np.float64)
    return (np.sin(2 * np.pi * 300 * t) * peak).astype(np.int16).reshape(-1, 1)


def test_a_healthy_level_says_nothing():
    assert pipeline.level_warnings(_tone(12000)) == []


def test_clipping_names_the_dial_to_turn_down():
    clipped = np.full((16000, 1), 32767, dtype=np.int16)
    warnings = pipeline.level_warnings(clipped)

    assert len(warnings) == 1
    assert "clipped" in warnings[0]
    assert "down" in warnings[0]


def test_a_few_clipped_samples_are_tolerated():
    """A brief peak is not distortion worth complaining about."""
    audio = _tone(12000)
    audio[:20] = 32767
    assert pipeline.level_warnings(audio) == []


def test_a_very_quiet_recording_names_the_dial_to_turn_up():
    warnings = pipeline.level_warnings(_tone(1000))

    assert len(warnings) == 1
    assert "quiet" in warnings[0]
    assert "up" in warnings[0]


def test_clipping_is_reported_rather_than_quietness_when_both_could_apply():
    """Clipping is the more damaging fault, so it wins."""
    audio = np.zeros((16000, 1), dtype=np.int16)
    audio[:1000] = 32767
    warnings = pipeline.level_warnings(audio)

    assert len(warnings) == 1
    assert "clipped" in warnings[0]


def test_digital_silence_does_not_divide_by_zero():
    warnings = pipeline.level_warnings(np.zeros((16000, 1), dtype=np.int16))
    assert "quiet" in warnings[0]


def test_an_empty_recording_is_not_complained_about():
    assert pipeline.level_warnings(np.zeros((0, 1), dtype=np.int16)) == []


def test_the_level_warning_reaches_the_popup(cfg, stub_models):
    quiet = Call(
        audio=_tone(500, 16000 * 3),
        sample_rate=16000,
        started_at=1785500000.0,
        duration_s=3.0,
        ended_reason="hangup",
    )
    posted = []

    class Ui:
        def request(self, popup):
            posted.append(popup)

    pipeline.process_call(quiet, cfg, ui=Ui())
    assert any("record level dial" in w for w in posted[0].warnings)


def test_manual_mode_never_warns_about_a_busy_input(cfg, caplog):
    """A long recording in manual mode is a decision, not a symptom."""
    runner = pipeline.Runner(cfg)
    assert runner.manual

    with caplog.at_level("WARNING"):
        _run_frames(runner, pipeline.State.IN_CALL, runner.HOT_INPUT_AFTER_FRAMES * 2)

    assert "hearing the room" not in caplog.text


def test_a_level_note_is_not_dressed_up_as_a_failure(cfg, stub_models):
    """A quiet recording that transcribed fine is not a failure."""
    quiet = Call(
        audio=_tone(500, 16000 * 3),
        sample_rate=16000,
        started_at=1785500000.0,
        duration_s=3.0,
        ended_reason="hangup",
    )
    result = pipeline.process_call(quiet, cfg)

    assert any("record level dial" in w for w in result.warnings)
    assert result.problems == []


# -- working through a folder of recordings ---------------------------------
#
# For testing against real calls without waiting for the phone to ring: drop
# the recordings in a folder and walk away.


def _stub_batch(monkeypatch, outcomes):
    """Replace processing with a scripted result per filename."""
    from call_transcriber import __main__ as cli, pipeline as pl

    seen = []

    def fake(path, cfg, ui=None):
        seen.append(path.name)
        outcome = outcomes[path.name]
        if isinstance(outcome, Exception):
            raise outcome
        return pl.Result(
            folder=None, work_order=outcome, transcript="t", extracted={},
        )

    monkeypatch.setattr(cli.pipeline, "process_file", fake)
    return seen


def test_every_recording_in_the_folder_is_processed(tmp_path, monkeypatch, capsys):
    from call_transcriber import __main__ as cli, config

    for name in ("b.wav", "a.wav", "c.mp3"):
        (tmp_path / name).write_bytes(b"")
    seen = _stub_batch(monkeypatch, {n: f"ORDER {n}" for n in ("a.wav", "b.wav", "c.mp3")})

    assert cli._cmd_test(config.load(root=tmp_path), tmp_path) == 0
    assert seen == ["a.wav", "b.wav", "c.mp3"], "processed out of order"
    assert "3 of 3 processed" in capsys.readouterr().out


def test_one_bad_recording_does_not_stop_the_rest(tmp_path, monkeypatch, capsys):
    """Twenty calls in, failing the batch on the fifth would waste the lot."""
    from call_transcriber import __main__ as cli, config

    for name in ("a.wav", "b.wav", "c.wav"):
        (tmp_path / name).write_bytes(b"")
    _stub_batch(monkeypatch, {
        "a.wav": "ORDER a",
        "b.wav": RuntimeError("corrupt"),
        "c.wav": "ORDER c",
    })

    assert cli._cmd_test(config.load(root=tmp_path), tmp_path) == 1
    out = capsys.readouterr().out
    assert "ORDER a" in out and "ORDER c" in out
    assert "1 failed" in out


def test_files_that_are_not_recordings_are_ignored(tmp_path, monkeypatch):
    from call_transcriber import __main__ as cli, config

    (tmp_path / "call.wav").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("not audio")
    (tmp_path / "config.toml").write_text("")
    seen = _stub_batch(monkeypatch, {"call.wav": "ORDER"})

    cli._cmd_test(config.load(root=tmp_path), tmp_path)

    assert seen == ["call.wav"]


def test_an_empty_folder_says_what_it_was_looking_for(tmp_path, capsys):
    from call_transcriber import __main__ as cli, config

    assert cli._cmd_test(config.load(root=tmp_path), tmp_path) == 1
    assert ".wav" in capsys.readouterr().out
