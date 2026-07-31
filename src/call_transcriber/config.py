"""Configuration loading.

Every field has a default that works on a stock Windows box, so a missing or
partial config.toml is never fatal -- unknown keys are reported rather than
silently ignored, because a typo'd setting that quietly does nothing is worse
than one that complains.
"""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AudioConfig:
    device_match: str = "LRX"
    # -1 means "match by name". Set to a number from `run.bat devices` only
    # when several genuinely different devices share a name.
    device_index: int = -1
    sample_rate: int = 16000
    stereo_mode: str = "auto"  # auto | mixed | split


@dataclass
class DetectConfig:
    vad_aggressiveness: int = 2
    speech_trigger_ms: int = 300
    # Long on purpose. This is the fallback for ending a call, not the primary
    # one -- line_dead_* below is what normally detects a hangup. A short value
    # here splits calls in half whenever someone goes quiet for a moment.
    hangup_silence_s: float = 45.0
    min_call_s: float = 10.0
    max_call_s: float = 3600.0
    noise_floor_dbfs: float = -50.0
    # The line going properly silent means the handset is back on the cradle.
    line_dead_dbfs: float = -60.0
    line_dead_s: float = 3.0


@dataclass
class TranscribeConfig:
    model: str = "small.en"
    # Trade vocabulary, fed to whisper so it expects these words. Far more
    # effective on domain terms than moving to a larger model.
    vocabulary: str = ""
    device: str = "auto"
    compute_type: str = "int8"
    beam_size: int = 1
    language: str = "en"


@dataclass
class ExtractConfig:
    base_url: str = "http://127.0.0.1:11434"
    model: str = "gemma3:4b"
    temperature: float = 0.0
    num_ctx: int = 8192
    chunk_chars: int = 12000
    timeout_s: int = 180


@dataclass
class ControlConfig:
    # manual -- nothing is recorded until the hotkey is pressed. The default,
    #           because recording without being asked is the worse mistake.
    # auto   -- the app decides when a call starts and stops
    mode: str = "manual"
    hotkey: str = "ctrl+alt+r"
    beep: bool = True


@dataclass
class OutputConfig:
    dir: str = "output"
    keep_audio: bool = False
    show_popup: bool = True
    copy_to_clipboard: bool = True
    save_transcript: bool = True


@dataclass
class BusinessConfig:
    name: str = ""
    # Comma-separated names of whoever answers the phone. Without this, an
    # unlabelled transcript gives the model no way to tell the person saying
    # "my name is X" from the customer, and it will take whichever name it saw.
    staff: str = ""
    default_service_area: str = ""


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    transcribe: TranscribeConfig = field(default_factory=TranscribeConfig)
    extract: ExtractConfig = field(default_factory=ExtractConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    business: BusinessConfig = field(default_factory=BusinessConfig)

    root: Path = field(default_factory=Path.cwd)
    warnings: list[str] = field(default_factory=list)

    @property
    def output_dir(self) -> Path:
        d = Path(self.output.dir)
        return d if d.is_absolute() else self.root / d


def _build(cls, table: dict, prefix: str, warnings: list[str]):
    """Instantiate a config dataclass from a TOML table, coercing scalars."""
    known = {f.name: f for f in dataclasses.fields(cls)}
    kwargs = {}
    for key, value in table.items():
        spec = known.get(key)
        if spec is None:
            warnings.append(f"unknown setting [{prefix}] {key!r} -- ignored")
            continue
        # `from __future__ import annotations` means field types arrive as
        # strings, so match on the name rather than the type object.
        declared = spec.type if isinstance(spec.type, str) else spec.type.__name__
        try:
            # bool before int: bool is a subclass of int and would coerce to 1/0.
            if declared == "bool":
                kwargs[key] = bool(value)
            elif declared == "int":
                kwargs[key] = int(value)
            elif declared == "float":
                kwargs[key] = float(value)
            else:
                kwargs[key] = str(value)
        except (TypeError, ValueError):
            warnings.append(
                f"[{prefix}] {key} = {value!r} is not a valid {declared} -- using default"
            )
    return cls(**kwargs)


def load(path: Path | None = None, root: Path | None = None) -> Config:
    """Load config.toml, falling back to defaults for anything absent."""
    root = root or Path.cwd()
    path = path or root / "config.toml"

    warnings: list[str] = []
    data: dict = {}
    if path.exists():
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            warnings.append(f"{path.name} is not valid TOML ({exc}) -- using defaults")
        except OSError as exc:
            warnings.append(f"could not read {path} ({exc}) -- using defaults")

    sections = {
        "audio": AudioConfig,
        "detect": DetectConfig,
        "control": ControlConfig,
        "transcribe": TranscribeConfig,
        "extract": ExtractConfig,
        "output": OutputConfig,
        "business": BusinessConfig,
    }
    for name in data:
        if name not in sections:
            warnings.append(f"unknown config section [{name}] -- ignored")

    cfg = Config(
        **{
            name: _build(cls, data.get(name, {}), name, warnings)
            for name, cls in sections.items()
        },
        root=root,
        warnings=warnings,
    )
    return _validate(cfg)


def _validate(cfg: Config) -> Config:
    """Clamp values that would crash a downstream library if passed through."""
    if not 0 <= cfg.detect.vad_aggressiveness <= 3:
        cfg.warnings.append(
            f"detect.vad_aggressiveness must be 0-3, got "
            f"{cfg.detect.vad_aggressiveness} -- clamped"
        )
        cfg.detect.vad_aggressiveness = min(3, max(0, cfg.detect.vad_aggressiveness))

    # webrtcvad only accepts these rates, and 16 kHz is what whisper wants too.
    if cfg.audio.sample_rate not in (8000, 16000, 32000, 48000):
        cfg.warnings.append(
            f"audio.sample_rate {cfg.audio.sample_rate} is not supported -- using 16000"
        )
        cfg.audio.sample_rate = 16000

    if cfg.audio.stereo_mode not in ("auto", "mixed", "split"):
        cfg.warnings.append(
            f"audio.stereo_mode {cfg.audio.stereo_mode!r} is not valid -- using 'auto'"
        )
        cfg.audio.stereo_mode = "auto"

    # The dead-line threshold has to sit below the speech threshold. If it were
    # above, an ordinary pause in conversation would register as a hangup and
    # every call would be chopped into pieces.
    if cfg.detect.line_dead_dbfs >= cfg.detect.noise_floor_dbfs:
        corrected = cfg.detect.noise_floor_dbfs - 10.0
        cfg.warnings.append(
            f"detect.line_dead_dbfs ({cfg.detect.line_dead_dbfs}) must be below "
            f"noise_floor_dbfs ({cfg.detect.noise_floor_dbfs}), or pauses would end "
            f"calls -- using {corrected}"
        )
        cfg.detect.line_dead_dbfs = corrected

    if cfg.control.mode not in ("auto", "manual"):
        cfg.warnings.append(
            f"control.mode {cfg.control.mode!r} is not valid -- using 'auto'. "
            f"Choose 'auto' to detect calls automatically or 'manual' to record "
            f"only when the hotkey is pressed."
        )
        cfg.control.mode = "auto"

    if cfg.control.mode == "manual" and not cfg.control.hotkey.strip():
        cfg.warnings.append(
            "control.mode is 'manual' but control.hotkey is empty, which would "
            "leave no way to record -- using ctrl+alt+r"
        )
        cfg.control.hotkey = "ctrl+alt+r"

    if cfg.detect.min_call_s > cfg.detect.max_call_s:
        cfg.warnings.append(
            "detect.min_call_s is greater than max_call_s -- every call would be "
            "discarded, so min_call_s has been reset to 0"
        )
        cfg.detect.min_call_s = 0.0

    return cfg
