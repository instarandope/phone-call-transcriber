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


def test_a_pause_between_sentences_becomes_a_cut():
    pytest.importorskip("webrtcvad")
    samples = np.concatenate([speech(2), silence(2), speech(2)])

    windows = vad.speech_windows(samples)
    assert len(windows) == 2
    assert windows[0][0] < 0.5
    assert windows[1][0] > 3.0


def test_a_breath_mid_sentence_does_not():
    pytest.importorskip("webrtcvad")
    samples = np.concatenate([speech(2), silence(0.3), speech(2)])

    assert len(vad.speech_windows(samples)) == 1


def test_someone_who_never_pauses_is_still_cut_up():
    """Parakeet recognises an utterance, so something has to bound it."""
    pytest.importorskip("webrtcvad")
    windows = vad.speech_windows(speech(70), max_window_s=25.0)

    assert len(windows) >= 3
    assert all(end - start <= 25.1 for start, end in windows)


def test_silence_yields_nothing_to_transcribe():
    pytest.importorskip("webrtcvad")
    assert vad.speech_windows(silence(5)) == []


def test_windows_never_run_past_the_recording():
    pytest.importorskip("webrtcvad")
    samples = np.concatenate([silence(0.2), speech(1.0), silence(0.2)])
    duration = len(samples) / 16000

    for start, end in vad.speech_windows(samples):
        assert start >= 0.0
        assert end <= duration + 1e-6


def test_audio_shorter_than_one_frame_is_handled():
    assert vad.speech_windows(np.zeros(10, dtype=np.float32)) == [(0.0, 10 / 16000)]


def test_a_short_turn_is_sent_whole_rather_than_re_split():
    """Diarization already bounded it; cutting again would only lose words."""
    assert transcribe._speech_windows(speech(8), tcfg()) == [(0.0, 8.0)]
