import pytest

from call_transcriber import config


def test_defaults_when_no_file(tmp_path):
    """The shipped defaults are the working setup, so config.toml is optional."""
    cfg = config.load(tmp_path / "missing.toml", root=tmp_path)

    assert cfg.transcribe.model == "base.en"
    assert cfg.extract.model == "gemma4:e4b"
    assert cfg.control.mode == "manual"
    assert cfg.control.hotkey == "f9"
    assert cfg.audio.device_match == "USB PnP"
    assert cfg.business.name
    assert cfg.output.keep_audio is False
    assert cfg.warnings == []


def test_the_shipped_thresholds_are_internally_consistent(tmp_path):
    """A dead-line threshold above the speech floor would chop every call."""
    cfg = config.load(tmp_path / "missing.toml", root=tmp_path)
    assert cfg.detect.line_dead_dbfs < cfg.detect.noise_floor_dbfs


def test_the_shipped_vocabulary_covers_the_words_that_were_misheard(tmp_path):
    vocabulary = config.load(tmp_path / "missing.toml", root=tmp_path).transcribe.vocabulary
    for word in ("cables", "Manaras", "Grunthal", "torsion spring"):
        assert word in vocabulary


def test_values_are_read_and_coerced(tmp_path):
    (tmp_path / "config.toml").write_text(
        """
        [audio]
        device_match = "USB Audio"

        [detect]
        hangup_silence_s = 4

        [output]
        keep_audio = true
        """,
        encoding="utf-8",
    )
    cfg = config.load(root=tmp_path)
    assert cfg.audio.device_match == "USB Audio"
    assert cfg.detect.hangup_silence_s == 4.0
    assert isinstance(cfg.detect.hangup_silence_s, float)
    assert cfg.output.keep_audio is True


def test_unknown_keys_are_reported_not_silently_dropped(tmp_path):
    (tmp_path / "config.toml").write_text(
        "[output]\nkeep_audio = true\nkeep_audios = true\n", encoding="utf-8"
    )
    cfg = config.load(root=tmp_path)
    assert any("keep_audios" in w for w in cfg.warnings)
    assert cfg.output.keep_audio is True


def test_unknown_section_is_reported(tmp_path):
    (tmp_path / "config.toml").write_text("[nonsense]\nx = 1\n", encoding="utf-8")
    cfg = config.load(root=tmp_path)
    assert any("[nonsense]" in w for w in cfg.warnings)


def test_broken_toml_falls_back_to_defaults(tmp_path):
    (tmp_path / "config.toml").write_text("[audio\nbroken", encoding="utf-8")
    cfg = config.load(root=tmp_path)
    assert cfg.audio.device_match == "USB PnP"
    assert any("not valid TOML" in w for w in cfg.warnings)


def test_out_of_range_values_are_clamped(tmp_path):
    (tmp_path / "config.toml").write_text(
        "[detect]\nvad_aggressiveness = 9\n\n[audio]\nsample_rate = 12345\n",
        encoding="utf-8",
    )
    cfg = config.load(root=tmp_path)
    assert cfg.detect.vad_aggressiveness == 3
    assert cfg.audio.sample_rate == 16000
    assert len(cfg.warnings) == 2


def test_a_dead_line_threshold_above_the_speech_floor_is_corrected(tmp_path):
    """Otherwise an ordinary pause reads as a hangup and chops every call up."""
    (tmp_path / "config.toml").write_text(
        "[detect]\nnoise_floor_dbfs = -50.0\nline_dead_dbfs = -30.0\n", encoding="utf-8"
    )
    cfg = config.load(root=tmp_path)
    assert cfg.detect.line_dead_dbfs == -60.0
    assert any("line_dead_dbfs" in w for w in cfg.warnings)


def test_sensible_thresholds_are_left_alone(tmp_path):
    (tmp_path / "config.toml").write_text(
        "[detect]\nnoise_floor_dbfs = -45.0\nline_dead_dbfs = -58.0\n", encoding="utf-8"
    )
    cfg = config.load(root=tmp_path)
    assert cfg.detect.line_dead_dbfs == -58.0
    assert cfg.warnings == []


def test_the_hangup_fallback_defaults_long_not_short(tmp_path):
    """A short value here is what splits one call into two."""
    assert config.load(root=tmp_path).detect.hangup_silence_s >= 30.0


def test_min_longer_than_max_would_discard_every_call(tmp_path):
    (tmp_path / "config.toml").write_text(
        "[detect]\nmin_call_s = 100\nmax_call_s = 60\n", encoding="utf-8"
    )
    cfg = config.load(root=tmp_path)
    assert cfg.detect.min_call_s == 0.0
    assert any("discarded" in w for w in cfg.warnings)


def test_output_dir_resolves_against_root(tmp_path):
    cfg = config.load(root=tmp_path)
    assert cfg.output_dir == tmp_path / "output"


# -- nothing leaves this machine -------------------------------------------


@pytest.mark.parametrize("url", [
    "http://127.0.0.1:11434",
    "http://localhost:11434",
    "http://[::1]:11434",
    "http://127.0.0.2:11434",
])
def test_addresses_on_this_machine_are_recognised(url):
    assert config.is_local(url) is True


@pytest.mark.parametrize("url", [
    "http://192.168.1.50:11434",
    "https://api.example.com",
    "http://ollama.mycompany.internal:11434",
    "http://10.0.0.5:11434",
    "",
])
def test_anything_else_is_not(url):
    assert config.is_local(url) is False


def test_pointing_extraction_off_the_machine_is_called_out(tmp_path):
    """Transcripts carry names, addresses and phone numbers."""
    (tmp_path / "config.toml").write_text(
        '[extract]\nbase_url = "https://api.example.com"\n', encoding="utf-8"
    )
    cfg = config.load(root=tmp_path)

    assert any("NOT this machine" in w for w in cfg.warnings)


def test_the_shipped_default_stays_on_this_machine(tmp_path):
    assert config.is_local(config.load(tmp_path / "none.toml", root=tmp_path).extract.base_url)
