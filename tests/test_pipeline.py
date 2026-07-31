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

    def fake_transcribe(audio, rate, tcfg, stereo_mode="auto"):
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
    assert any("Ollama is not running" in w for w in result.warnings)
    assert "Nothing could be extracted" in result.work_order


def test_silence_produces_a_warning_rather_than_a_confident_blank_form(
    cfg, call, monkeypatch, stub_models
):
    monkeypatch.setattr(
        pipeline.transcribe,
        "transcribe",
        lambda *a, **k: transcribe.Transcript([], "en", 3.0, "mono"),
    )

    result = pipeline.process_call(call, cfg)
    assert any("Nothing intelligible" in w for w in result.warnings)
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
