import json
import wave

import numpy as np

from call_transcriber import extract, storage


def test_wav_roundtrips_through_the_stdlib_fallback(tmp_path, monkeypatch):
    # Force the no-soundfile path so the fallback is actually exercised.
    monkeypatch.setitem(__import__("sys").modules, "soundfile", None)

    audio = (np.random.randn(1600, 2) * 3000).astype(np.int16)
    path = storage.write_wav(tmp_path / "call.wav", audio, 16000)

    with wave.open(str(path), "rb") as handle:
        assert handle.getnchannels() == 2
        assert handle.getframerate() == 16000
        assert handle.getsampwidth() == 2
        assert handle.getnframes() == 1600


def test_shred_removes_the_file():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "secret.wav"
        path.write_bytes(b"sensitive audio" * 1000)
        storage.shred(path)
        assert not path.exists()


def test_shredding_a_missing_file_is_not_an_error(tmp_path):
    storage.shred(tmp_path / "never-existed.wav")


def test_call_dir_is_sorted_by_day_then_time(tmp_path):
    import time

    when = time.mktime((2026, 7, 31, 14, 22, 5, 0, 0, -1))
    folder = storage.call_dir(tmp_path, when, "jane-doe")
    assert folder.parent.name == "2026-07-31"
    assert folder.name == "142205-jane-doe"
    assert folder.is_dir()


def test_results_are_written_and_metadata_is_attached(tmp_path):
    data = extract.empty_result()
    data["caller_name"] = "Jane Doe"

    written = storage.save_results(
        tmp_path, "WORK ORDER\nCUSTOMER  Jane Doe", "SIDE A: hello", data,
        meta={"duration_s": 12.5},
    )

    assert written["work_order"].read_text(encoding="utf-8").startswith("WORK ORDER")
    assert written["transcript"].read_text(encoding="utf-8") == "SIDE A: hello"

    payload = json.loads(written["extracted"].read_text(encoding="utf-8"))
    assert payload["caller_name"] == "Jane Doe"
    assert payload["_call"]["duration_s"] == 12.5


def test_purge_removes_audio_but_keeps_the_paperwork(tmp_path):
    day = tmp_path / "2026-07-31" / "142205-jane-doe"
    day.mkdir(parents=True)
    (day / "call.wav").write_bytes(b"\x00" * 2048)
    (day / "work_order.txt").write_text("WORK ORDER", encoding="utf-8")
    (day / "transcript.txt").write_text("hello", encoding="utf-8")

    files, freed = storage.purge_audio(tmp_path)

    assert files == 1
    assert freed == 2048
    assert not (day / "call.wav").exists()
    assert (day / "work_order.txt").exists()
    assert (day / "transcript.txt").exists()


def test_purge_all_clears_the_whole_tree(tmp_path):
    day = tmp_path / "2026-07-31" / "142205-jane-doe"
    day.mkdir(parents=True)
    (day / "call.wav").write_bytes(b"\x00" * 100)
    (day / "work_order.txt").write_text("WORK ORDER", encoding="utf-8")

    files, _ = storage.purge_audio(tmp_path, everything=True)

    assert files == 2
    assert list(tmp_path.rglob("*")) == []


def test_purging_a_missing_folder_is_harmless(tmp_path):
    assert storage.purge_audio(tmp_path / "nope") == (0, 0)


def test_transcript_can_be_withheld(tmp_path):
    written = storage.save_results(
        tmp_path, "order", "transcript", extract.empty_result(), save_transcript=False
    )
    assert "transcript" not in written
    assert not (tmp_path / "transcript.txt").exists()
