import numpy as np

from call_transcriber import transcribe


def test_mono_input_is_reported_as_mono():
    audio = (np.random.randn(16000, 1) * 3000).astype(np.int16)
    assert transcribe.detect_layout(audio) == "mono"


def test_duplicated_channels_are_recognised_as_one_mixed_signal():
    signal = (np.random.randn(16000) * 3000).astype(np.int16)
    audio = np.stack([signal, signal], axis=1)
    assert transcribe.detect_layout(audio) == "mixed"


def test_slightly_different_copies_of_one_signal_are_still_mixed():
    signal = (np.random.randn(16000) * 3000).astype(np.int16)
    noise = (np.random.randn(16000) * 100).astype(np.int16)
    audio = np.stack([signal, signal + noise], axis=1)
    assert transcribe.detect_layout(audio) == "mixed"


def test_independent_channels_are_recognised_as_two_sides_of_a_call():
    # Only one person talks at a time, so the two sides barely correlate.
    left = np.zeros(16000, dtype=np.int16)
    right = np.zeros(16000, dtype=np.int16)
    left[:8000] = (np.random.randn(8000) * 3000).astype(np.int16)
    right[8000:] = (np.random.randn(8000) * 3000).astype(np.int16)
    assert transcribe.detect_layout(np.stack([left, right], axis=1)) == "split"


def test_a_silent_second_channel_is_treated_as_mono():
    signal = (np.random.randn(16000) * 3000).astype(np.int16)
    audio = np.stack([signal, np.zeros(16000, dtype=np.int16)], axis=1)
    assert transcribe.detect_layout(audio) == "mono"


def test_int16_is_converted_to_the_float_range_whisper_expects():
    audio = np.full((100, 1), 16384, dtype=np.int16)
    out = transcribe.to_float_mono(audio)
    assert out.dtype == np.float32
    assert out.ndim == 1
    assert abs(float(out[0]) - 0.5) < 0.01


def test_stereo_is_downmixed_to_one_channel():
    audio = np.stack(
        [np.full(100, 16384, dtype=np.int16), np.zeros(100, dtype=np.int16)], axis=1
    )
    out = transcribe.to_float_mono(audio)
    assert out.shape == (100,)
    assert abs(float(out[0]) - 0.25) < 0.01


def test_transcript_text_prefixes_speakers_only_when_known():
    plain = transcribe.Transcript(
        segments=[transcribe.Segment(0, 1, "hello")],
        language="en", duration_s=1, layout="mono",
    )
    assert plain.text == "hello"

    labelled = transcribe.Transcript(
        segments=[transcribe.Segment(0, 1, "hello", speaker="SIDE A")],
        language="en", duration_s=1, layout="split",
    )
    assert labelled.text == "SIDE A: hello"


def test_a_transcript_of_only_blank_segments_counts_as_empty():
    result = transcribe.Transcript(
        segments=[transcribe.Segment(0, 1, "   ")],
        language="en", duration_s=1, layout="mono",
    )
    assert result.is_empty


import sys

import pytest

from call_transcriber import config


def tcfg(**overrides):
    settings = config.load(__import__("pathlib").Path("/nope.toml")).transcribe
    for key, value in overrides.items():
        setattr(settings, key, value)
    return settings


# -- a half-installed GPU must not take whisper down -------------------------
#
# device = "auto" on a PC with an NVIDIA driver selects CUDA, and then dies
# loading cublas64_12.dll -- the runtime DLLs are not part of the driver, so a
# gaming PC that never installed the CUDA toolkit fails exactly this way. The
# fix is the CPU, not an error message.


def _gpu_fallback_setup(monkeypatch, error_message, cpu_ok=True):
    attempts = []

    class FakeWhisperModel:
        def __init__(self, model, device="auto", compute_type="int8"):
            attempts.append(device)
            if device != "cpu":
                raise RuntimeError(error_message)
            if not cpu_ok:
                raise RuntimeError("cpu is broken too")

    import types

    fake = types.ModuleType("faster_whisper")
    fake.WhisperModel = FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)
    monkeypatch.setattr(transcribe, "_model", None)
    monkeypatch.setattr(transcribe, "_model_key", None)
    return attempts


def test_a_missing_cuda_runtime_falls_back_to_the_cpu(monkeypatch):
    attempts = _gpu_fallback_setup(
        monkeypatch, "Library cublas64_12.dll is not found or cannot be loaded"
    )

    transcribe.load_model(tcfg())

    assert attempts == ["auto", "cpu"]


def test_a_missing_cudnn_falls_back_too(monkeypatch):
    attempts = _gpu_fallback_setup(monkeypatch, "Unable to load libcudnn_ops.so.9")

    transcribe.load_model(tcfg())

    assert attempts == ["auto", "cpu"]


def test_an_ordinary_load_failure_is_still_an_error(monkeypatch):
    """A bad model name must not be retried on the CPU and then blamed on it."""
    attempts = _gpu_fallback_setup(monkeypatch, "Invalid model size 'basee.en'")

    with pytest.raises(RuntimeError, match="could not load whisper model"):
        transcribe.load_model(tcfg(model="basee.en"))

    assert attempts == ["auto"]


def test_asking_for_cpu_outright_never_retries(monkeypatch):
    attempts = _gpu_fallback_setup(monkeypatch, "irrelevant", cpu_ok=False)

    with pytest.raises(RuntimeError):
        transcribe.load_model(tcfg(device="cpu"))

    assert attempts == ["cpu"]


def test_a_broken_cpu_after_a_broken_gpu_reports_the_cpu_error(monkeypatch):
    _gpu_fallback_setup(monkeypatch, "cublas64_12.dll not found", cpu_ok=False)

    with pytest.raises(RuntimeError, match="cpu is broken too"):
        transcribe.load_model(tcfg())
