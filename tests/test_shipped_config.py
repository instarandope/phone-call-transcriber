"""The example file and the built-in defaults must not drift apart.

Both are shipped, and either can be the one in effect: config.toml is copied
from the example on install, but a missing or deleted config.toml falls through
to the dataclass defaults. If they disagree, deleting config.toml silently
changes behaviour -- which is the sort of thing nobody thinks to check.
"""

import pytest
import tomllib
from pathlib import Path

from call_transcriber import config

EXAMPLE = Path(__file__).resolve().parents[1] / "config.example.toml"


def shipped() -> dict:
    return tomllib.loads(EXAMPLE.read_text(encoding="utf-8"))


def test_the_example_file_is_valid_toml():
    assert shipped()


def test_every_shipped_value_matches_the_built_in_default():
    defaults = config.load(Path("/does-not-exist.toml"))

    for section, table in shipped().items():
        for key, value in table.items():
            built_in = getattr(getattr(defaults, section), key)
            assert built_in == value, (
                f"{section}.{key} is {value!r} in config.example.toml but "
                f"{built_in!r} in config.py -- deleting config.toml would "
                f"change behaviour"
            )


def test_the_example_names_no_setting_the_code_does_not_have():
    """A typo'd key in the example would be copied into every install."""
    defaults = config.load(Path("/does-not-exist.toml"))

    for section, table in shipped().items():
        assert hasattr(defaults, section), f"unknown section [{section}]"
        for key in table:
            assert hasattr(getattr(defaults, section), key), f"unknown [{section}] {key}"


def test_the_example_loads_without_complaint():
    """Whatever ships must not itself produce config warnings."""
    assert config.load(EXAMPLE).warnings == []


# -- fetching the optional models ------------------------------------------


def test_a_failed_download_says_how_to_do_it_by_hand():
    """A company network that inspects HTTPS breaks this, and that is fixable."""
    from call_transcriber import models

    bundle = models.BY_KEY["parakeet"]
    message = models._manual_instructions(bundle, "certificate verify failed")

    assert bundle.url in message
    assert "--file" in message
    assert "browser" in message
    assert "not the model being unavailable" in message


def test_every_bundle_can_explain_itself():
    from call_transcriber import models

    for bundle in models.BUNDLES:
        message = models._manual_instructions(bundle, "boom")
        assert f"--{bundle.key}" in message
        assert bundle.url in message


def test_installing_from_a_missing_file_says_so(tmp_path):
    from call_transcriber import models

    with pytest.raises(RuntimeError, match="does not exist"):
        models.install(
            models.BY_KEY["parakeet"], tmp_path, archive=tmp_path / "nothing.tar.bz2"
        )


def test_an_archive_that_escapes_its_folder_is_refused(tmp_path):
    """A tar member named ../../evil must not be able to write there."""
    import tarfile

    from call_transcriber import models

    payload = tmp_path / "payload.txt"
    payload.write_text("x", encoding="utf-8")
    archive = tmp_path / "bad.tar"
    with tarfile.open(archive, "w") as tar:
        tar.add(payload, arcname="../escaped.txt")

    with pytest.raises(RuntimeError, match="escapes the target folder"):
        models._unpack(archive, tmp_path / "target")
