"""Updating in place.

The thing that must never happen here is an update replacing something that
belongs to the installation rather than the project -- a config.toml, a
recording, a downloaded model. None of those are in the published archive, so
the risk is theoretical; these tests are what keep it that way.
"""

import io
import zipfile

import pytest

from call_transcriber import update


def archive(files, root="phone-call-transcriber-main"):
    """Build a ZIP shaped like GitHub's, with one folder at the top."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        for name, content in files.items():
            bundle.writestr(f"{root}/{name}", content)
    return buffer.getvalue()


def test_a_changed_file_is_replaced(tmp_path):
    (tmp_path / "run.bat").write_text("old")

    result = update.apply(archive({"run.bat": "new"}), tmp_path)

    assert (tmp_path / "run.bat").read_text() == "new"
    assert result.changed == ["run.bat"]


def test_a_new_file_is_added(tmp_path):
    result = update.apply(archive({"config.bat": "hello"}), tmp_path)

    assert (tmp_path / "config.bat").read_text() == "hello"
    assert result.added == ["config.bat"]


def test_an_identical_file_is_left_alone(tmp_path):
    (tmp_path / "run.bat").write_text("same")
    before = (tmp_path / "run.bat").stat().st_mtime_ns

    result = update.apply(archive({"run.bat": "same"}), tmp_path)

    assert result.changed == [] and result.added == []
    assert (tmp_path / "run.bat").stat().st_mtime_ns == before


def test_nested_files_arrive_in_the_right_place(tmp_path):
    update.apply(archive({"src/call_transcriber/vad.py": "code"}), tmp_path)

    assert (tmp_path / "src" / "call_transcriber" / "vad.py").read_text() == "code"


def test_your_settings_are_never_replaced(tmp_path):
    """config.toml is not in the archive today. If it ever were -- committed
    upstream by accident -- it would silently replace the settings on every
    machine, which is a failure this project has already been bitten by."""
    (tmp_path / "config.toml").write_text('engine = "parakeet"')

    result = update.apply(archive({"config.toml": 'engine = "whisper"'}), tmp_path)

    assert (tmp_path / "config.toml").read_text() == 'engine = "parakeet"'
    assert "config.toml" in result.protected


def test_recordings_and_models_are_never_replaced(tmp_path):
    for folder, name in (("output", "call.wav"), ("models", "encoder.onnx")):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / name).write_text("mine")

    result = update.apply(
        archive({"output/call.wav": "theirs", "models/encoder.onnx": "theirs"}),
        tmp_path,
    )

    assert (tmp_path / "output" / "call.wav").read_text() == "mine"
    assert (tmp_path / "models" / "encoder.onnx").read_text() == "mine"
    assert len(result.protected) == 2


def test_a_changed_dependency_list_asks_for_install_bat(tmp_path):
    (tmp_path / "requirements.txt").write_text("numpy")

    result = update.apply(archive({"requirements.txt": "numpy\nsherpa-onnx"}), tmp_path)

    assert result.needs_install


def test_an_ordinary_update_does_not(tmp_path):
    (tmp_path / "run.bat").write_text("old")

    assert not update.apply(archive({"run.bat": "new"}), tmp_path).needs_install


def test_a_path_escaping_the_folder_is_refused(tmp_path):
    """A zip may name ../../anywhere. Extraction is the classic place to
    forget that."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr("../escaped.txt", "no")

    with pytest.raises(update.UpdateError, match="escapes"):
        update.apply(buffer.getvalue(), tmp_path)


def test_something_that_is_not_an_archive_says_so(tmp_path):
    with pytest.raises(update.UpdateError, match="not a valid archive"):
        update.apply(b"<!doctype html><h1>404</h1>", tmp_path)


def test_an_archive_of_the_wrong_shape_is_refused(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr("one/a.txt", "a")
        bundle.writestr("two/b.txt", "b")

    with pytest.raises(update.UpdateError, match="does not look like this project"):
        update.apply(buffer.getvalue(), tmp_path)


def test_a_download_failure_explains_itself(monkeypatch):
    import requests

    def boom(*args, **kwargs):
        raise requests.ConnectionError("no route to host")

    monkeypatch.setattr(requests, "get", boom)

    with pytest.raises(update.UpdateError, match="could not download"):
        update.fetch()
