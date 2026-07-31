from call_transcriber import config


def test_defaults_when_no_file(tmp_path):
    cfg = config.load(tmp_path / "missing.toml", root=tmp_path)
    assert cfg.transcribe.model == "small.en"
    assert cfg.output.keep_audio is False
    assert cfg.warnings == []


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
    assert cfg.audio.device_match == "LRX"
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
