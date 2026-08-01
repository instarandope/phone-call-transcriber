"""Configuration loading.

Every field has a default that works on a stock Windows box, so a missing or
partial config.toml is never fatal -- unknown keys are reported rather than
silently ignored, because a typo'd setting that quietly does nothing is worse
than one that complains.
"""

from __future__ import annotations

import dataclasses
import ipaddress
import shutil
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse


@dataclass
class AudioConfig:
    device_match: str = "USB PnP"
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
    # Real calls top out around 25 minutes here, so 30 leaves headroom while
    # still catching a hotkey that never got pressed a second time.
    max_call_s: float = 1800.0
    noise_floor_dbfs: float = -48.0
    # The line going properly silent means the handset is back on the cradle.
    line_dead_dbfs: float = -59.0
    line_dead_s: float = 3.0


@dataclass
class TranscribeConfig:
    # whisper  -- faster-whisper. Fetches its own weights, and biases well to
    #             a vocabulary hint. base.en is quick even on an old CPU.
    # parakeet -- NVIDIA Parakeet TDT via sherpa-onnx. Scores better on
    #             English and does not hallucinate over silence, but is around
    #             three times slower than base.en -- it is eight times the
    #             model. Needs `run.bat models --parakeet` first.
    engine: str = "whisper"
    parakeet_dir: str = ""
    num_threads: int = 4
    model: str = "base.en"
    # Trade vocabulary, fed to whisper so it expects these words. Far more
    # effective on domain terms than moving to a larger model.
    vocabulary: str = (
        "torsion spring, cables, rollers, tracks, panels, hoist, pulleys, "
        "opener, operator, keypad, drum, bracket, weather seal, chain, rail, "
        "trolley, jamb, header, Chamberlain, LiftMaster, Manaras, "
        "Hanover Door Systems, Steinbach, Mitchell, Grunthal, Manitoba"
    )
    device: str = "auto"
    compute_type: str = "int8"
    beam_size: int = 1
    language: str = "en"


@dataclass
class DiarizeConfig:
    # Off by default: it needs models downloading and adds a minute or so per
    # call. Worth turning on because a mixed handset tap otherwise gives no way
    # to tell who said what, which is where most extraction mistakes start.
    enabled: bool = False
    segmentation_model: str = ""
    embedding_model: str = ""
    # A phone call is two people. Saying so outright removes the part of
    # diarization that goes wrong most.
    speakers: int = 2
    num_threads: int = 4
    min_speech_s: float = 0.3
    min_silence_s: float = 0.5


@dataclass
class ExtractConfig:
    base_url: str = "http://127.0.0.1:11434"
    model: str = "gemma4:e4b"
    # Reasoning models spend tokens thinking before answering. For filling in a
    # fixed schema from a transcript there is nothing to reason about, and on a
    # slow CPU it is minutes of pure cost.
    think: bool = False
    temperature: float = 0.0
    num_ctx: int = 8192
    chunk_chars: int = 12000
    # An eleven-minute call takes a capable local model several minutes on an
    # older CPU. Processing is in the background, so a generous ceiling costs
    # nothing when things work and only matters when something is stuck.
    timeout_s: int = 900


@dataclass
class ControlConfig:
    # manual -- nothing is recorded until the hotkey is pressed. The default,
    #           because recording without being asked is the worse mistake.
    # auto   -- the app decides when a call starts and stops
    mode: str = "manual"
    hotkey: str = "f9"
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
    name: str = "Hanover Doors Systems"
    # Comma-separated names of whoever answers the phone. Without this, an
    # unlabelled transcript gives the model no way to tell the person saying
    # "my name is X" from the customer, and it will take whichever name it saw.
    staff: str = "Johan, Derek"
    default_service_area: str = "Steinbach or Mitchell"


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    transcribe: TranscribeConfig = field(default_factory=TranscribeConfig)
    diarize: DiarizeConfig = field(default_factory=DiarizeConfig)
    extract: ExtractConfig = field(default_factory=ExtractConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    business: BusinessConfig = field(default_factory=BusinessConfig)

    root: Path = field(default_factory=Path.cwd)
    warnings: list[str] = field(default_factory=list)
    # The file the settings actually came from, or None when there was none to
    # read and every value below is a built-in default. Worth recording: a
    # missing config file and a config file that happens to agree with the
    # defaults produce identical settings, and only one of them means the edit
    # you just made is being ignored.
    path: Path | None = None

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
    read_from: Path | None = None
    if path.exists():
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
            read_from = path
        except tomllib.TOMLDecodeError as exc:
            warnings.append(f"{path.name} is not valid TOML ({exc}) -- using defaults")
        except OSError as exc:
            warnings.append(f"could not read {path} ({exc}) -- using defaults")

    sections = {
        "audio": AudioConfig,
        "detect": DetectConfig,
        "control": ControlConfig,
        "transcribe": TranscribeConfig,
        "diarize": DiarizeConfig,
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
        path=read_from,
    )
    return _validate(cfg)


CONFIG_NAME = "config.toml"
EXAMPLE_NAME = "config.example.toml"


def differences(cfg: Config) -> list[tuple[str, object]]:
    """Settings that are not the built-in default, as dotted key/value pairs.

    This is the answer to "I changed something and nothing happened". Either
    the change is in this list or it never reached the program.
    """
    stock = Config()
    out: list[tuple[str, object]] = []
    for section in dataclasses.fields(Config):
        mine = getattr(cfg, section.name)
        # root/warnings/path are bookkeeping, not settings. `field.type` is a
        # string here because of `from __future__ import annotations`, so ask
        # the value rather than the annotation.
        if not dataclasses.is_dataclass(mine):
            continue
        theirs = getattr(stock, section.name)
        for entry in dataclasses.fields(mine):
            value = getattr(mine, entry.name)
            if value != getattr(theirs, entry.name):
                out.append((f"{section.name}.{entry.name}", value))
    return out


def _looks_like_install(folder: Path) -> bool:
    return (folder / "src" / "call_transcriber" / "config.py").exists()


def adopt_or_create(root: Path) -> tuple[str, Path | None]:
    """Make sure root/config.toml exists, carrying settings over if it can.

    config.toml is deliberately not in the repository -- it holds business
    details and machine paths, and those are nobody else's business. The
    side effect is that updating the program by downloading it again lands in
    a folder with no settings at all, whereupon the defaults quietly take over
    and the next call runs with a different engine than the one that was
    chosen. Nothing about that is visible until you go looking.

    So when this folder has no settings, look next door for the install this
    one replaces and bring its settings across.

    Returns (what happened, where they came from).
    """
    destination = root / CONFIG_NAME
    if destination.exists():
        return "kept", destination

    previous = _previous_config(root)
    if previous is not None:
        shutil.copyfile(previous, destination)
        return "adopted", previous

    example = root / EXAMPLE_NAME
    if not example.exists():
        raise RuntimeError(f"{example} is missing -- the download is incomplete")
    shutil.copyfile(example, destination)
    return "created", example


def _previous_config(root: Path) -> Path | None:
    """The config.toml of the most recently used other install, if there is one."""
    try:
        siblings = sorted(root.parent.iterdir())
    except OSError:
        return None

    candidates = []
    for folder in siblings:
        if folder == root or not folder.is_dir() or not _looks_like_install(folder):
            continue
        config_file = folder / CONFIG_NAME
        if config_file.exists():
            candidates.append(config_file)

    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


LOOPBACK_NAMES = {"localhost", "ip6-localhost", "ip6-loopback"}


def is_local(url: str) -> bool:
    """Is this address on this machine?

    The whole point of the project is that call audio and customer details
    never leave the building. Everything is local by construction except one
    thing -- the address extraction posts transcripts to -- and that is a
    setting, so it is the one place a mistake or an edit could quietly start
    sending client conversations somewhere else.
    """
    host = (urlparse(url).hostname or "").lower()
    if not host:
        return False
    if host in LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


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

    if not is_local(cfg.extract.base_url):
        cfg.warnings.append(
            f"extract.base_url points at {cfg.extract.base_url}, which is NOT this "
            f"machine. Every call transcript -- names, addresses, phone numbers -- "
            f"would be sent there. Set it back to http://127.0.0.1:11434 unless you "
            f"genuinely intend that."
        )

    if cfg.transcribe.engine not in ("whisper", "parakeet"):
        cfg.warnings.append(
            f"transcribe.engine {cfg.transcribe.engine!r} is not valid -- using "
            f"'whisper'. Choose 'whisper' or 'parakeet'."
        )
        cfg.transcribe.engine = "whisper"

    if cfg.diarize.speakers < 1:
        cfg.warnings.append(
            f"diarize.speakers must be at least 1, got {cfg.diarize.speakers} -- using 2"
        )
        cfg.diarize.speakers = 2

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
