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


def _with_extract(tmp_path, body):
    (tmp_path / "config.toml").write_text(f"[extract]\n{body}\n", encoding="utf-8")
    return config.load(root=tmp_path)


def test_pointing_extraction_at_the_internet_is_called_out(tmp_path):
    """Transcripts carry names, addresses and phone numbers."""
    cfg = _with_extract(tmp_path, 'base_url = "https://api.example.com"')

    assert any("public internet" in w for w in cfg.warnings)


def test_the_internet_is_refused_even_when_the_lan_is_allowed(tmp_path):
    """allow_lan permits the machine next door. It is not a blanket override,
    or it would be a way to turn the guard off by accident."""
    cfg = _with_extract(
        tmp_path, 'base_url = "https://api.example.com"\nallow_lan = true'
    )

    assert any("public internet" in w for w in cfg.warnings)


def test_a_machine_on_your_own_network_asks_to_be_meant(tmp_path):
    cfg = _with_extract(tmp_path, 'base_url = "http://192.168.1.50:11434"')

    assert any("allow_lan" in w for w in cfg.warnings)


def test_and_is_accepted_once_it_is(tmp_path):
    cfg = _with_extract(
        tmp_path, 'base_url = "http://192.168.1.50:11434"\nallow_lan = true'
    )

    assert cfg.warnings == []


def test_which_addresses_count_as_your_own_network():
    for private in ("http://192.168.1.50", "http://10.0.0.4", "http://172.16.5.1"):
        assert config.is_private_network(private), private
    for not_private in ("https://api.example.com", "http://8.8.8.8", "http://127.0.0.1"):
        assert not config.is_private_network(not_private), not_private


def test_a_hostname_is_not_assumed_to_be_next_door(tmp_path):
    """It may well resolve to the machine in the corner, but nothing here can
    tell -- so it is treated as the internet rather than guessed at."""
    cfg = _with_extract(tmp_path, 'base_url = "http://macmini.local:11434"\nallow_lan = true')

    assert any("public internet" in w for w in cfg.warnings)


def test_the_shipped_default_stays_on_this_machine(tmp_path):
    assert config.is_local(config.load(tmp_path / "none.toml", root=tmp_path).extract.base_url)


# ---------------------------------------------------------------------------
# Knowing which file the settings came from.
#
# A download of this project never contains config.toml, so updating by
# unzipping it again produces a folder with no settings. The defaults then take
# over, every check still passes, and the transcriber runs a setup nobody
# chose. That happened, so these are the guards against it happening quietly.
# ---------------------------------------------------------------------------


def test_no_config_file_is_visible_as_such(tmp_path):
    assert config.load(root=tmp_path).path is None


def test_the_file_actually_read_is_recorded(tmp_path):
    (tmp_path / "config.toml").write_text("[transcribe]\nengine = \"parakeet\"\n")
    cfg = config.load(root=tmp_path)

    assert cfg.path == tmp_path / "config.toml"
    assert cfg.transcribe.engine == "parakeet"


def test_an_unreadable_file_does_not_claim_to_have_been_read(tmp_path):
    """Falling back to defaults after a parse error is fine. Saying the file
    was used when it was not is what makes the failure invisible."""
    (tmp_path / "config.toml").write_text("[transcribe\nengine =")
    cfg = config.load(root=tmp_path)

    assert cfg.path is None
    assert cfg.warnings


def test_differences_lists_only_what_was_changed(tmp_path):
    (tmp_path / "config.toml").write_text(
        "[transcribe]\nengine = \"parakeet\"\n\n[diarize]\nenabled = true\n"
    )
    changed = dict(config.differences(config.load(root=tmp_path)))

    assert changed == {"transcribe.engine": "parakeet", "diarize.enabled": True}


def test_a_default_config_differs_in_nothing(tmp_path):
    assert config.differences(config.load(root=tmp_path)) == []


def _install(folder):
    (folder / "src" / "call_transcriber").mkdir(parents=True)
    (folder / "src" / "call_transcriber" / "config.py").write_text("")
    (folder / "config.example.toml").write_text("[transcribe]\nengine = \"whisper\"\n")
    return folder


def test_settings_are_carried_over_from_the_install_being_replaced(tmp_path):
    old = _install(tmp_path / "phone-call-transcriber-main")
    (old / "config.toml").write_text("[transcribe]\nengine = \"parakeet\"\n")
    new = _install(tmp_path / "phone-call-transcriber-main (1)")

    what, source = config.adopt_or_create(new)

    assert what == "adopted"
    assert source == old / "config.toml"
    assert config.load(root=new).transcribe.engine == "parakeet"


def test_the_most_recently_touched_install_is_the_one_copied(tmp_path):
    stale = _install(tmp_path / "old")
    (stale / "config.toml").write_text("[transcribe]\nmodel = \"tiny.en\"\n")
    recent = _install(tmp_path / "newer")
    (recent / "config.toml").write_text("[transcribe]\nmodel = \"small.en\"\n")
    import os

    os.utime(stale / "config.toml", (1, 1))
    os.utime(recent / "config.toml", (2, 2))

    _, source = config.adopt_or_create(_install(tmp_path / "target"))

    assert source == recent / "config.toml"


def test_an_unrelated_neighbour_is_not_mistaken_for_a_previous_install(tmp_path):
    stranger = tmp_path / "some other project"
    stranger.mkdir()
    (stranger / "config.toml").write_text("[transcribe]\nmodel = \"tiny.en\"\n")

    what, _ = config.adopt_or_create(_install(tmp_path / "target"))

    assert what == "created"


def test_an_existing_config_is_never_overwritten(tmp_path):
    old = _install(tmp_path / "previous")
    (old / "config.toml").write_text("[transcribe]\nmodel = \"tiny.en\"\n")
    current = _install(tmp_path / "current")
    (current / "config.toml").write_text("[transcribe]\nmodel = \"small.en\"\n")

    what, _ = config.adopt_or_create(current)

    assert what == "kept"
    assert config.load(root=current).transcribe.model == "small.en"


# ---------------------------------------------------------------------------
# `config --value`, which install.bat reads settings through.
# ---------------------------------------------------------------------------


def test_one_setting_prints_alone(tmp_path, capsys):
    from call_transcriber import __main__ as cli

    (tmp_path / "config.toml").write_text('[extract]\nmodel = "gemma4:e4b"\n')
    assert cli._cmd_config(config.load(root=tmp_path), False, "extract.model") == 0
    assert capsys.readouterr().out == "gemma4:e4b\n"


def test_booleans_print_as_the_config_file_spells_them(tmp_path, capsys):
    """install.bat compares against "true"; Python would say "True"."""
    from call_transcriber import __main__ as cli

    (tmp_path / "config.toml").write_text("[diarize]\nenabled = true\n")
    cli._cmd_config(config.load(root=tmp_path), False, "diarize.enabled")
    assert capsys.readouterr().out == "true\n"


def test_asking_for_a_setting_that_does_not_exist_fails_loudly(tmp_path, capsys):
    from call_transcriber import __main__ as cli

    cfg = config.load(root=tmp_path)
    for asked in ("nonsense", "extract.nonsense", "nonsense.model", "root.parent"):
        assert cli._cmd_config(cfg, False, asked) == 1, asked
        captured = capsys.readouterr()
        assert captured.out == "", asked
        assert "no such setting" in captured.err
