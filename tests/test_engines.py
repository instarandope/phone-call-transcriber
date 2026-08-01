"""Choosing a speech engine, and cutting long audio for the one that needs it."""

import numpy as np
import pytest

from call_transcriber import config, transcribe, vad


def tcfg(**overrides):
    settings = config.load(__import__("pathlib").Path("/nope.toml")).transcribe
    for key, value in overrides.items():
        setattr(settings, key, value)
    return settings


# -- picking an engine -----------------------------------------------------


def test_whisper_is_the_default(tmp_path):
    assert config.load(tmp_path / "none.toml", root=tmp_path).transcribe.engine == "whisper"


def test_parakeet_can_be_asked_for(tmp_path):
    (tmp_path / "config.toml").write_text(
        '[transcribe]\nengine = "parakeet"\n', encoding="utf-8"
    )
    cfg = config.load(root=tmp_path)
    assert cfg.transcribe.engine == "parakeet"
    assert cfg.warnings == []


def test_a_misspelled_engine_says_so_rather_than_failing_later(tmp_path):
    (tmp_path / "config.toml").write_text(
        '[transcribe]\nengine = "parrakeet"\n', encoding="utf-8"
    )
    cfg = config.load(root=tmp_path)
    assert cfg.transcribe.engine == "whisper"
    assert any("parrakeet" in w for w in cfg.warnings)


def test_the_engine_choice_reaches_the_transcriber(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        transcribe, "_run_parakeet",
        lambda samples, cfg, root: seen.setdefault("engine", "parakeet") or [],
    )
    monkeypatch.setattr(
        transcribe, "_run_whisper",
        lambda samples, cfg: seen.setdefault("engine", "whisper") or [],
    )

    audio = np.zeros((16000, 1), dtype=np.int16)
    transcribe.transcribe(audio, 16000, tcfg(engine="parakeet"), "mono")
    assert seen["engine"] == "parakeet"


def test_a_missing_parakeet_download_names_the_command_to_run(tmp_path):
    pytest.importorskip("sherpa_onnx", reason="sherpa-onnx not installed")

    with pytest.raises(transcribe.EngineError, match="run.bat models --parakeet"):
        transcribe.load_parakeet(tcfg(parakeet_dir=str(tmp_path / "absent")), tmp_path)


# -- cutting long audio ----------------------------------------------------


def speech(seconds, rate=16000):
    """Something webrtcvad will accept: harmonics, an envelope, and breath.

    A steady tone is not speech and webrtcvad is right to say so, so a bare
    sine stack tests nothing except the VAD's ability to reject it.
    """
    rng = np.random.default_rng(0)
    t = np.arange(int(seconds * rate)) / rate
    harmonics = sum(
        np.sin(2 * np.pi * f * t) / (i + 1) for i, f in enumerate((140, 280, 560, 1120))
    )
    syllables = 0.6 + 0.4 * np.sin(2 * np.pi * 4.5 * t)  # ~4.5 syllables a second
    breath = rng.normal(0, 0.02, len(t))
    return (0.35 * (harmonics / 2 * syllables + breath)).astype(np.float32)


def silence(seconds, rate=16000):
    return np.zeros(int(seconds * rate), dtype=np.float32)


def covers_everything(windows, samples, rate=16000):
    """Contiguous, in order, and accounting for every sample."""
    duration = len(samples) / rate
    assert windows[0][0] == 0.0
    assert windows[-1][1] == pytest.approx(duration)
    for (_, end), (next_start, _) in zip(windows, windows[1:]):
        assert next_start == pytest.approx(end), "a gap here is audio nobody transcribes"
    return True


def test_nothing_is_ever_dropped_between_windows():
    """The bug this replaced: only the stretches webrtcvad called speech were
    kept, so the customer's short quiet answers -- the ones that confirm the
    address and the appointment -- were thrown away before the engine saw them."""
    pytest.importorskip("webrtcvad")
    samples = np.concatenate(
        [speech(20), silence(2), speech(0.4) * 0.15, silence(2), speech(20)]
    )

    windows = vad.speech_windows(samples)

    assert covers_everything(windows, samples)


def test_a_pause_between_sentences_is_where_the_cut_lands():
    pytest.importorskip("webrtcvad")
    samples = np.concatenate([speech(20), silence(3), speech(20)])

    windows = vad.speech_windows(samples)

    assert len(windows) > 1
    assert covers_everything(windows, samples)
    # The seam belongs in the silence, not in either sentence.
    assert 20.0 <= windows[0][1] <= 23.0


def test_a_breath_mid_sentence_is_not_treated_as_a_pause():
    pytest.importorskip("webrtcvad")
    samples = np.concatenate([speech(2), silence(0.1), speech(2)])

    assert vad.speech_windows(samples) == [(0.0, pytest.approx(len(samples) / 16000))]


def test_short_audio_is_sent_whole():
    assert vad.speech_windows(speech(8)) == [(0.0, pytest.approx(8.0))]


def test_someone_who_never_pauses_is_still_cut_up():
    """Parakeet recognises an utterance, so something has to bound it."""
    pytest.importorskip("webrtcvad")
    samples = speech(70)
    windows = vad.speech_windows(samples, max_window_s=25.0)

    assert len(windows) >= 3
    assert all(end - start <= 25.1 for start, end in windows)
    assert covers_everything(windows, samples)


def test_an_unavoidable_cut_goes_to_the_quietest_moment_available():
    """Cutting on the clock landed mid-vowel and split a word in half."""
    pytest.importorskip("webrtcvad")
    rate = 16000
    # Continuous speech with one dip at 22s -- too short for the pause detector
    # to offer it, but the best place available before the 25s limit.
    samples = speech(70).copy()
    samples[int(21.9 * rate) : int(22.1 * rate)] *= 0.01

    first_cut = vad.speech_windows(samples, max_window_s=25.0)[0][1]

    assert 21.9 <= first_cut <= 22.1, f"cut at {first_cut}, not at the quiet dip"


def test_silence_all_the_way_through_yields_nothing():
    """All-or-nothing. A recording with no speech anywhere is worth skipping;
    a quiet moment inside a real call is not."""
    pytest.importorskip("webrtcvad")
    assert vad.speech_windows(silence(40)) == []


def test_windows_never_run_past_the_recording():
    pytest.importorskip("webrtcvad")
    samples = np.concatenate([silence(0.2), speech(40), silence(0.2)])
    duration = len(samples) / 16000

    for start, end in vad.speech_windows(samples):
        assert start >= 0.0
        assert end <= duration + 1e-6


def test_audio_shorter_than_one_frame_is_handled():
    assert vad.speech_windows(np.zeros(10, dtype=np.float32)) == [(0.0, 10 / 16000)]


def test_a_short_turn_is_sent_whole_rather_than_re_split():
    """Diarization already bounded it; cutting again would only lose words."""
    assert transcribe._speech_windows(speech(8), tcfg()) == [(0.0, 8.0)]


# -- `compare` should not transcribe more times than it has to ---------------


def _fake_result(text, duration=60.0):
    from call_transcriber import transcribe as t

    return t.Transcript(
        segments=[t.Segment(start=0.0, end=duration, text=text)],
        language="en",
        duration_s=duration,
        layout="mono",
    )


def _compare_counting(monkeypatch, tmp_path, **kwargs):
    """Run `compare` with transcription stubbed, and count the passes."""
    from call_transcriber import __main__ as cli, config, storage, transcribe

    calls = []
    monkeypatch.setattr(storage, "read_wav", lambda p: (np.zeros(16000), 16000))

    def counted(audio, rate, settings, stereo_mode, root=None):
        calls.append(cli._engine_label(settings))
        return _fake_result("the door will not close")

    monkeypatch.setattr(transcribe, "transcribe", counted)

    wav = tmp_path / "call.wav"
    wav.write_bytes(b"")
    cfg = config.load(root=tmp_path)
    cli._cmd_compare(cfg, wav, kwargs.get("models"), kwargs.get("engines"))
    return calls


def test_comparing_engines_transcribes_once_per_engine_and_no_more(monkeypatch, tmp_path):
    """It used to transcribe a third time first, with the configured engine,
    and then throw that away -- 88 wasted seconds on a seven minute call."""
    calls = _compare_counting(
        monkeypatch, tmp_path, engines="whisper:base.en,parakeet"
    )

    assert calls == ["whisper:base.en", "parakeet"]


def test_comparing_extraction_models_transcribes_exactly_once(monkeypatch, tmp_path):
    from call_transcriber import extract

    monkeypatch.setattr(
        extract, "extract", lambda text, settings, business: {"issue_summary": "x"}
    )
    calls = _compare_counting(monkeypatch, tmp_path, models="gemma4:e4b,gemma3:4b")

    assert len(calls) == 1


def test_the_engine_named_in_the_output_is_the_one_that_ran(monkeypatch, tmp_path):
    """It announced base.en while loading parakeet, because it printed
    transcribe.model regardless of transcribe.engine."""
    from call_transcriber import config

    settings = config.load(root=tmp_path).transcribe
    settings.engine = "parakeet"
    from call_transcriber import __main__ as cli

    assert cli._engine_label(settings) == "parakeet"
    settings.engine = "whisper"
    assert cli._engine_label(settings) == "whisper:base.en"
