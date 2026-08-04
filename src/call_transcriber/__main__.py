"""Command line entry point.

    run       listen for calls (default)
    devices   list audio inputs, so you can find the adapter's name
    doctor    check everything the app needs before you rely on it
    test      run the full pipeline over an existing audio file
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
from pathlib import Path

from . import __version__, config, notify, pipeline, tray


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)

    root = _root()
    _setup_logging(args.verbose, root)
    cfg = config.load(Path(args.config) if args.config else None, root=root)

    for warning in cfg.warnings:
        logging.warning("config: %s", warning)

    if args.command == "devices":
        return _cmd_devices()
    if args.command == "doctor":
        return _cmd_doctor(cfg)
    if args.command == "config":
        return _cmd_config(cfg, args.create, args.value)
    if args.command == "prompt":
        return _cmd_prompt(cfg)
    if args.command == "last":
        return _cmd_last(cfg)
    if args.command == "models":
        return _cmd_models(cfg, args)
    if args.command == "levels":
        return _cmd_levels(cfg, args.seconds)
    if args.command == "test":
        return _cmd_test(cfg, Path(args.file))
    if args.command == "compare":
        return _cmd_compare(
            cfg, Path(args.file) if args.file else None, args.models, args.engines
        )
    if args.command == "purge":
        return _cmd_purge(cfg, args.purge_all)
    return _cmd_run(cfg, use_tray=args.tray)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="call-transcriber",
        description="Transcribe phone calls locally and produce service work orders.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-c", "--config", help="path to config.toml")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")

    sub = parser.add_subparsers(dest="command")
    run = sub.add_parser("run", help="listen for calls (default)")
    run.add_argument(
        "--tray",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="system tray icon (on by default; --no-tray to suppress)",
    )
    sub.add_parser("devices", help="list audio input devices")
    sub.add_parser("doctor", help="check the setup")
    settings = sub.add_parser(
        "config", help="show which config.toml is in use and what it changed"
    )
    settings.add_argument(
        "--create",
        action="store_true",
        help="create config.toml if absent, reusing a previous install's settings",
    )
    settings.add_argument(
        "--value",
        metavar="SECTION.KEY",
        help="print one setting and nothing else, e.g. extract.model",
    )
    sub.add_parser("prompt", help="show exactly what the model is told to extract")
    sub.add_parser("last", help="reprint the most recent work order")
    fetch = sub.add_parser("models", help="download the optional speech models")
    fetch.add_argument("--parakeet", action="store_true", help="the Parakeet speech engine")
    fetch.add_argument("--diarize", action="store_true", help="speaker labelling models")
    fetch.add_argument("--all", action="store_true", dest="all_models", help="both")
    fetch.add_argument("--force", action="store_true", help="re-download even if present")
    fetch.add_argument(
        "--file",
        dest="archive",
        help="install from a file you downloaded yourself, instead of fetching it",
    )
    levels = sub.add_parser("levels", help="measure your line and suggest thresholds")
    levels.add_argument(
        "--seconds", type=int, default=45, help="how long to listen (default 45)"
    )
    test = sub.add_parser("test", help="process an existing audio file")
    test.add_argument("file", help="path to a .wav/.mp3/.m4a recording")
    compare = sub.add_parser(
        "compare", help="run one recording through several extraction models"
    )
    compare.add_argument(
        "file",
        nargs="?",
        help="recording to use; defaults to the most recent kept call",
    )
    compare.add_argument(
        "--models",
        help="comma-separated Ollama models to compare, e.g. gemma3:4b,gemma4:e4b",
    )
    compare.add_argument(
        "--engines",
        help="comma-separated speech engines to compare, e.g. whisper:base.en,"
        "whisper:small.en,parakeet -- needs an audio file, not a transcript",
    )
    purge = sub.add_parser("purge", help="securely delete kept recordings")
    purge.add_argument(
        "--all",
        action="store_true",
        dest="purge_all",
        help="delete work orders and transcripts too, not just audio",
    )

    parser.set_defaults(command="run", tray=True)
    return parser


def _root() -> Path:
    """Project root: where config.toml and output/ live, regardless of cwd."""
    return Path(__file__).resolve().parents[2]


LOG_FILENAME = "call-transcriber.log"


def _setup_logging(verbose: bool, root: Path) -> None:
    """Log to the console when there is one, and always to a file.

    The file matters more than it looks. Started from the tray shortcut there
    is no console at all, so without it a failure to find the adapter would be
    completely silent -- no window, no icon, no explanation.
    """
    handlers: list[logging.Handler] = []

    # pythonw.exe leaves sys.stderr as None, and a StreamHandler over None
    # raises on the first record rather than at construction.
    if sys.stderr is not None:
        handlers.append(logging.StreamHandler())

    try:
        from logging.handlers import RotatingFileHandler

        handlers.append(
            RotatingFileHandler(
                root / LOG_FILENAME, maxBytes=1_000_000, backupCount=2, encoding="utf-8"
            )
        )
    except OSError:
        pass  # read-only folder; the console handler is still there

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
    # These are chatty at INFO and say nothing useful here.
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    _catch_hard_crashes(root)


CRASH_FILENAME = "call-transcriber-crash.log"


def _catch_hard_crashes(root: Path) -> None:
    """Leave a trace when the process dies below Python.

    Most of this program's dependencies are C libraries -- ctranslate2,
    onnxruntime, libsndfile, PortAudio -- and when one of those falls over it
    takes the interpreter with it. There is no exception to catch and no
    traceback: the process simply stops, mid-sentence, and the console returns
    to a prompt as if nothing had been asked of it.

    faulthandler is the only thing that says anything at all in that case. It
    writes the C-level stack to a file at the moment of the fault, which turns
    "it printed nothing" into something that can be read.
    """
    try:
        handle = open(root / CRASH_FILENAME, "a", buffering=1, encoding="utf-8")
    except OSError:
        return  # read-only folder; not worth failing the run over

    try:
        import faulthandler

        # Kept open deliberately for the life of the process: faulthandler
        # writes to this descriptor from a signal handler, so it has to stay
        # valid right up to the crash.
        faulthandler.enable(file=handle, all_threads=True)
    except Exception:
        handle.close()


def _fatal(message: str) -> int:
    """Report a startup failure, including when there is no console to see it."""
    logging.error("%s", message)
    if sys.stderr is None:
        # Started from the tray shortcut: a dialog is the only way to say so.
        try:
            import tkinter as tk
            from tkinter import messagebox

            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Call Transcriber could not start", message)
            root.destroy()
        except Exception:
            pass
    return 1


# -- commands --------------------------------------------------------------


def _cmd_devices() -> int:
    from . import audio

    try:
        devices = audio.list_input_devices()
    except audio.AudioError as exc:
        print(f"error: {exc}")
        return 1

    if not devices:
        print("No audio input devices found. Is the adapter plugged in?")
        return 1

    # Windows lists every device once per driver stack, so a single adapter
    # shows up three or four times under near-identical names. Presenting the
    # raw list makes it look like four devices when there is one.
    groups: dict[str, list] = {}
    for device in devices:
        groups.setdefault(device.identity, []).append(device)

    print(f"{len(groups)} input device(s), listed by Windows as {len(devices)} entries.")
    print("Each device appears once per driver stack; those are grouped here.\n")

    for number, entries in enumerate(groups.values(), start=1):
        best = audio.best_hostapi(entries)
        print(f"  {number}. {best.name}")
        print(
            f"       would use index {best.index} via {best.hostapi or 'default'}, "
            f"{best.channels}ch @ {best.samplerate:.0f} Hz"
        )
        others = [e for e in entries if e.index != best.index]
        if others:
            print(
                "       also listed as "
                + ", ".join(f"[{e.index}] {e.hostapi or 'unknown'}" for e in others)
            )
        print()

    print("Set audio.device_match in config.toml to part of the right name -- matching")
    print("is case-insensitive, and choosing the driver stack happens automatically.")
    print()
    print("Not sure which is the phone adapter? Pick the likely one, then run")
    print("`run.bat levels` and talk into the handset. If the meter moves, it is the")
    print("right one. If two different devices share a name, set audio.device_index")
    print("to one of the numbers above instead.")
    return 0


def _cmd_models(cfg, args) -> int:
    """Fetch the optional model bundles. Neither ships with the code."""
    from . import models

    wanted: list[str] = []
    if args.all_models or args.parakeet:
        wanted.append("parakeet")
    if args.all_models or args.diarize:
        wanted.extend(models.DIARIZATION_KEYS)

    if not wanted:
        print("Optional models:\n")
        for bundle, installed in models.status(cfg.root):
            mark = "installed" if installed else "not installed"
            print(f"  [{mark:>13}]  {bundle.what}  (~{bundle.size_mb} MB)")
        print()
        print("  run.bat models --parakeet    more accurate speech engine, slower")
        print("  run.bat models --diarize     label who is speaking on each line")
        print("  run.bat models --all         both")
        return 0

    archive = Path(args.archive) if getattr(args, "archive", None) else None
    if archive is not None and len(wanted) != 1:
        print(
            "error: --file installs one bundle, so name exactly one.\n"
            "  e.g.  run.bat models --parakeet --file C:\\path\\to\\model.tar.bz2"
        )
        return 1

    for key in wanted:
        bundle = models.BY_KEY[key]
        try:
            models.install(bundle, cfg.root, force=args.force, archive=archive)
        except Exception as exc:
            print(f"\nerror: could not install {bundle.what}\n  {exc}")
            return 1

    print("\nDone. To use them, in config.toml:")
    if "parakeet" in wanted:
        print('    [transcribe]\n    engine = "parakeet"')
    if "segmentation" in wanted:
        print("    [diarize]\n    enabled = true")
    return 0


def _cmd_last(cfg) -> int:
    """Reprint the most recent work order and put it back on the clipboard."""
    from . import notify, storage

    path = storage.latest_work_order(cfg.output_dir)
    if path is None:
        print(f"No work orders yet in {cfg.output_dir}.")
        return 1

    call = storage.read_call(path.parent)
    print(call["work_order"])
    print(f"  from {path.parent}")
    if cfg.output.copy_to_clipboard and notify.copy(call["work_order"]):
        print("  copied to the clipboard - just paste")
    return 0


def _cmd_prompt(cfg) -> int:
    """Print the instructions the model actually receives.

    Editing fields.py changes the prompt indirectly, which makes it easy to
    write something you think is clear and never see how it reads in context.
    """
    from . import extract, fields

    print("=" * 72)
    print("STANDING RULES  (edit in src/call_transcriber/extract.py)")
    print("=" * 72)
    print(extract.SYSTEM_PROMPT)

    if cfg.business.name or cfg.business.default_service_area:
        print("=" * 72)
        print("YOUR BUSINESS  (edit under [business] in config.toml)")
        print("=" * 72)
        if cfg.business.name:
            print(f"  {cfg.business.name}")
        if cfg.business.default_service_area:
            print(f"  serving {cfg.business.default_service_area}")
        print()

    print("=" * 72)
    print(f"FIELDS TO EXTRACT  ({len(fields.FIELDS)} of them)")
    print("edit in src/call_transcriber/fields.py")
    print("=" * 72)
    print(fields.instructions())
    print()
    print(f"Model: {cfg.extract.model}")
    return 0


def _cmd_config(cfg, create: bool, value: str | None = None) -> int:
    """Where the settings came from, and which of them are not the default.

    The question this answers is "I edited config.toml and nothing changed".
    There are only two possible reasons -- the file being read is not the file
    that was edited, or the edit did not parse -- and both are visible here.
    """
    if value:
        # install.bat needs a setting or two before it can carry on. It used to
        # pick them out of the file with findstr, which cannot tell one
        # section's `model` from another's and quietly returned whichever came
        # last. Asking the loader is the only way to get the same answer the
        # program will act on.
        section, _, key = value.partition(".")
        table = getattr(cfg, section, None)
        if not hasattr(table, "__dataclass_fields__") or not key or not hasattr(table, key):
            print(f"error: no such setting: {value}", file=sys.stderr)
            return 1
        found = getattr(table, key)
        # Booleans print the way you would write them in the file, not the way
        # Python spells them, so a batch script comparing against "true" and a
        # person reading config.toml agree.
        print(str(found).lower() if isinstance(found, bool) else found)
        return 0

    if create:
        try:
            what, source = config.adopt_or_create(cfg.root)
        except (OSError, RuntimeError) as exc:
            print(f"  [XX] could not create config.toml: {exc}")
            return 1

        if what == "kept":
            print("  [ok] config.toml already exists (left alone)")
        elif what == "adopted":
            print(f"  [ok] Brought your settings across from {source}")
            print("       Delete config.toml and re-run install.bat to start fresh.")
        else:
            print("  [ok] Created config.toml from the example")
        cfg = config.load(root=cfg.root)

    if cfg.path is None:
        print(f"No {config.CONFIG_NAME} in {cfg.root}")
        print("Every setting below is a built-in default.\n")
    else:
        print(f"Reading {cfg.path}\n")

    changed = config.differences(cfg)
    if not changed:
        print("Nothing differs from the built-in defaults.")
    else:
        print("Changed from the defaults:")
        for setting, current in changed:
            rendered = str(current)
            if len(rendered) > 60:
                rendered = rendered[:57] + "..."
            print(f"  {setting} = {rendered}")

    for warning in cfg.warnings:
        print(f"\n  [!!] {warning}")
    return 0


def _cmd_doctor(cfg) -> int:
    from . import audio, extract

    ok = True

    def check(label: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        print(f"  [{'ok' if passed else 'XX'}] {label}" + (f"\n       {detail}" if detail else ""))
        ok = ok and passed

    print(f"call-transcriber {__version__}\n")

    # First, because it frames everything below it. Downloading the project
    # again gives you a folder with no config.toml, the defaults silently take
    # over, and every check still passes -- while running a setup nobody chose.
    print("Settings")
    if cfg.path is not None:
        check(f"reading {cfg.path}", True)
    else:
        print(f"  [--] no {config.CONFIG_NAME} here -- built-in defaults in use")
        print(f"       Expected it at {cfg.root / config.CONFIG_NAME}")
        print("       If you edited a config.toml, that was not this one.")
        print("       Run install.bat to create it here.")

    print("\nPython")
    check(
        f"version {sys.version_info.major}.{sys.version_info.minor}",
        sys.version_info >= (3, 11),
        "" if sys.version_info >= (3, 11) else "3.11 or newer is required",
    )

    # Updating the code can introduce a dependency the existing virtual
    # environment does not have, and the failure then shows up much later as a
    # feature that will not start. Checking here is what makes re-running
    # install.bat the obvious fix.
    print("\nPackages")
    required = [
        ("numpy", "numpy"),
        ("sounddevice", "sounddevice"),
        ("soundfile", "soundfile"),
        ("webrtcvad", "webrtcvad-wheels"),
        ("faster_whisper", "faster-whisper"),
        ("requests", "requests"),
        ("pyperclip", "pyperclip"),
    ]
    if cfg.control.mode == "manual":
        required.append(("pynput", "pynput"))
    # Only when something actually uses it. Reporting it unconditionally would
    # put a red mark next to a setup that works perfectly well without it.
    if cfg.transcribe.engine == "parakeet" or cfg.diarize.enabled:
        required.append(("sherpa_onnx", "sherpa-onnx"))

    for module, package in required:
        try:
            __import__(module)
            check(package, True)
        except Exception as exc:
            check(package, False, f"{exc} -- re-run install.bat")

    for module, package in (("pystray", "pystray"), ("PIL", "Pillow")):
        try:
            __import__(module)
        except Exception:
            print(f"  [--] {package} missing -- no tray icon, everything else works")

    print("\nAudio")
    try:
        device = audio.find_device(cfg.audio.device_match, cfg.audio.device_index)
        check(f"adapter found: {device}", True)
    except audio.AudioError as exc:
        check(f"adapter matching {cfg.audio.device_match!r}", False, str(exc))

    print("\nControl")
    if cfg.control.mode == "manual":
        try:
            from . import hotkey

            check(f"hotkey {cfg.control.hotkey.upper()} -> {hotkey.to_pynput(cfg.control.hotkey)}", True)
        except Exception as exc:
            check(f"hotkey {cfg.control.hotkey!r}", False, str(exc))
    else:
        check("automatic call detection (nothing to press)", True)

    print("\nLocal model")
    reachable, detail = extract.check_server(cfg.extract)
    check(f"Ollama / {cfg.extract.model}", reachable, "" if reachable else detail)

    # Loading each model for real is the point. Checking that a file exists
    # would pass right up until the first call, which is the worst place to
    # discover a half-finished install.
    print("\nSpeech engine")
    from . import transcribe

    if cfg.transcribe.engine == "parakeet":
        try:
            transcribe.load_parakeet(cfg.transcribe, cfg.root)
            check("parakeet loads", True)
        except Exception as exc:
            check("parakeet", False, str(exc))
    else:
        try:
            transcribe.load_model(cfg.transcribe)
            check(f"whisper {cfg.transcribe.model} loads", True)
        except Exception as exc:
            check(f"whisper {cfg.transcribe.model}", False, str(exc))

    # A setting that quietly does nothing is the thing this project keeps
    # getting caught by, and this is one: the trade vocabulary reaches whisper
    # as an initial prompt, and parakeet has nowhere to put it. sherpa-onnx
    # does take a hotwords file, but only with modified_beam_search, and that
    # decoder is not implemented for NeMo transducers -- asking for it
    # segfaults rather than refusing.
    if cfg.transcribe.engine == "parakeet" and cfg.transcribe.vocabulary.strip():
        print("  [--] transcribe.vocabulary is set but parakeet cannot use it")
        print("       Only whisper takes the word list. Parakeet has no hotword")
        print("       support in sherpa-onnx yet, so those terms are not biased.")

    print("\nSpeaker labelling")
    if not cfg.diarize.enabled:
        print("  [--] off -- transcripts will not say who is speaking")
        print("       Set diarize.enabled = true in config.toml to turn it on.")
    else:
        from . import diarize

        try:
            diarize.load(cfg.diarize, cfg.root)
            check(f"diarization loads, expecting {cfg.diarize.speakers} speakers", True)
        except Exception as exc:
            check("diarization", False, str(exc))

    print("\nWhere the data goes")
    local = config.is_local(cfg.extract.base_url)
    check(
        f"transcripts go to {cfg.extract.base_url}"
        + (" (this machine)" if local else " -- OFF THIS MACHINE"),
        local,
        "" if local else
        "Every transcript would be sent to that address. Set extract.base_url "
        "back to http://127.0.0.1:11434 unless you genuinely intend this.",
    )
    print("  [ok] call audio is never sent anywhere -- it is transcribed in memory")
    print("  [--] the internet is used once, to download models, and never again")

    print("\nOutput")
    try:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        probe = cfg.output_dir / ".write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        check(f"{cfg.output_dir} is writable", True)
    except OSError as exc:
        check(f"{cfg.output_dir}", False, str(exc))

    check(
        "audio is deleted after transcribing",
        not cfg.output.keep_audio,
        "" if not cfg.output.keep_audio
        else "output.keep_audio is ON -- recordings are being kept. Turn it off "
             "in config.toml when you finish testing.",
    )

    print("\n" + ("All good. Run `run.bat` and take a call." if ok
                  else "Fix the items marked XX above, then run doctor again."))
    return 0 if ok else 1


def _cmd_test(cfg, path: Path) -> int:
    if not path.exists():
        print(f"error: {path} does not exist")
        return 1

    ui = notify.make_ui(cfg.output.show_popup)
    stop = threading.Event()
    result_box: list = []

    def work():
        try:
            result_box.append(pipeline.process_file(path, cfg, ui=ui))
        except Exception as exc:
            logging.exception("failed: %s", exc)
            result_box.append(exc)
        finally:
            stop.set()

    worker = threading.Thread(target=work, name="test", daemon=True)
    worker.start()
    ui.run(stop)
    worker.join(timeout=10)

    if not result_box or isinstance(result_box[0], Exception):
        return 1

    result = result_box[0]
    print("\n" + result.work_order)
    if result.folder:
        print(f"saved to {result.folder}")
    return 0


def _cmd_levels(cfg, seconds: int) -> int:
    """Measure the line so the two thresholds come from real numbers.

    The whole call-detection scheme rests on one assumption: an open line is
    measurably louder than a closed one, even when nobody is speaking. This
    checks that on your actual phone and tells you where to put the thresholds.
    """
    import time

    import numpy as np

    from . import audio

    try:
        device = audio.find_device(cfg.audio.device_match, cfg.audio.device_index)
    except audio.AudioError as exc:
        print(f"error: {exc}")
        return 1

    print(f"Listening on {device} for {seconds} seconds.\n")
    print("  Spend roughly a third of the time on each of these:")
    print("    1. handset ON the cradle, don't touch it")
    print("    2. handset lifted, say nothing")
    print("    3. handset lifted, talk normally\n")
    print("  Starting now.\n")

    frame_ms = 20
    per_window = max(1, int(500 / frame_ms))
    levels: list[float] = []
    window: list[float] = []
    deadline = time.monotonic() + seconds

    try:
        with audio.Capture(device, target_rate=cfg.audio.sample_rate) as capture:
            for frame in capture.frames():
                value = audio.dbfs(frame)
                window.append(value if np.isfinite(value) else -100.0)
                if len(window) >= per_window:
                    level = float(np.mean(window))
                    levels.append(level)
                    print(f"  {_bar(level)} {level:7.1f} dBFS")
                    window = []
                if time.monotonic() >= deadline:
                    break
    except audio.AudioError as exc:
        print(f"error: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\n  stopped early")

    if len(levels) < 10:
        print("\nNot enough audio was captured to suggest anything.")
        return 1

    quiet = float(np.percentile(levels, 10))
    loud = float(np.percentile(levels, 90))
    spread = loud - quiet

    print(f"\n  Quietest (10th pct): {quiet:7.1f} dBFS")
    print(f"  Loudest  (90th pct): {loud:7.1f} dBFS")
    print(f"  Spread:              {spread:7.1f} dB\n")

    if spread < 10:
        print(
            "The quiet and loud levels are too close together for the adapter to\n"
            "tell an open line from a closed one. Check that the handset cord runs\n"
            "through the adapter, and that Windows input volume for it isn't at\n"
            "either extreme. Re-run this once the spread is at least 15 dB."
        )
        return 1

    dead = round(quiet + 4)
    floor = round(max(quiet + 10, (quiet + loud) / 2))

    print("Put these in config.toml under [detect]:\n")
    print(f"    noise_floor_dbfs = {floor}.0")
    print(f"    line_dead_dbfs = {dead}.0\n")
    print(
        "  noise_floor_dbfs is the level speech has to beat to start a recording.\n"
        "  line_dead_dbfs is the level below which the line counts as hung up.\n"
        "  If calls end while someone is just thinking, lower line_dead_dbfs.\n"
        "  If recordings start when nobody called, raise noise_floor_dbfs."
    )
    return 0


def _bar(level: float, width: int = 40) -> str:
    """A meter from -80 dBFS to 0."""
    filled = int(max(0.0, min(1.0, (level + 80) / 80)) * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


TEXT_SUFFIXES = {".txt", ".md"}


def _latest_input(cfg) -> Path | None:
    """Newest transcript, or failing that newest kept recording.

    Transcripts are preferred and are what this defaults to: comparing
    extraction models only needs the text, and transcripts are saved for every
    call while audio is deleted unless keep_audio is on. Re-transcribing to
    compare a step that never sees the audio is wasted minutes.
    """
    for pattern in ("transcript.txt", "call.wav"):
        found = list(cfg.output_dir.rglob(pattern))
        if found:
            return max(found, key=lambda p: p.stat().st_mtime)
    return None


def _parse_engine(spec: str, cfg):
    """'parakeet' or 'whisper:small.en' -> a transcribe config to run with."""
    import copy

    settings = copy.copy(cfg.transcribe)
    engine, _, model = spec.partition(":")
    settings.engine = engine.strip().lower() or "whisper"
    if model.strip():
        settings.model = model.strip()
    return settings


def _cmd_compare(cfg, path: Path | None, models: str | None, engines: str | None) -> int:
    """Run one recording through several extraction models.

    Which model to use is an empirical question about a specific machine and a
    specific trade's vocabulary, and no amount of reading benchmarks settles
    it. Transcription happens once and is shared, so what is being compared is
    the extraction step alone.
    """
    import copy
    import time as _time

    from . import extract, storage, transcribe, workorder

    if path is None:
        path = _latest_input(cfg)
        if path is None:
            print(
                "Nothing to compare against yet.\n\n"
                "  This needs one call's transcript. Take a call, then run it again --\n"
                "  or point it at any transcript.txt, or at a saved recording."
            )
            return 1
        print(f"Using {path}\n")

    if not path.exists():
        print(f"error: {path} does not exist")
        return 1

    wanted = [m.strip() for m in (models or "").split(",") if m.strip()]
    engine_specs = [e.strip() for e in (engines or "").split(",") if e.strip()]
    if not wanted and not engine_specs:
        print("error: give --models, --engines, or both")
        return 1

    is_text = path.suffix.lower() in TEXT_SUFFIXES
    if engine_specs and is_text:
        print(
            "error: --engines compares speech recognition, so it needs a "
            "recording rather than a transcript.\n"
            "  Set keep_audio = true under [output], take a call, and point "
            "this at the call.wav."
        )
        return 1

    duration_s = None
    transcript_text = ""
    if is_text:
        transcript_text = path.read_text(encoding="utf-8").strip()
        print(f"Read {len(transcript_text)} characters of transcript.\n")
    else:
        audio, rate = storage.read_wav(path)
        # An engine comparison run on a clipped recording measures the
        # engines' tolerance for distortion, not their accuracy -- worth
        # knowing before reading anything into the results.
        for note in pipeline.level_warnings(audio):
            print(f"  [!!] {note}\n")

    if engine_specs:
        # The engines each transcribe the whole recording below, so
        # transcribing it once more up here would be the same work a third
        # time -- it used to, on a seven minute call, for nothing.
        transcript_text, duration_s = _compare_engines(cfg, audio, rate, engine_specs)
    elif not is_text:
        print(f"Transcribing {path.name} once with {_engine_label(cfg.transcribe)} ...")
        started = _time.monotonic()
        result = transcribe.transcribe(audio, rate, cfg.transcribe, cfg.audio.stereo_mode)
        transcript_text = result.text
        duration_s = result.duration_s
        print(f"  {_time.monotonic() - started:.0f}s, {len(result.segments)} segments\n")

    if not transcript_text:
        print("That transcript is empty, so there is nothing to compare.")
        return 1

    if not wanted:
        return 0

    scored = []
    for model in wanted:
        print("=" * 72)
        print(f"  {model}")
        print("=" * 72)

        settings = copy.copy(cfg.extract)
        settings.model = model
        started = _time.monotonic()
        try:
            data = extract.extract(transcript_text, settings, cfg.business)
        except extract.ExtractionError as exc:
            print(f"  failed: {exc}\n")
            continue
        elapsed = _time.monotonic() - started

        print(workorder.render(data, duration_s=duration_s, business=cfg.business))
        filled = sum(
            1 for f in fields_with_values(data)
        )
        print(f"  {elapsed:.0f}s, {filled} fields filled\n")
        scored.append((model, elapsed, filled))

    if len(scored) > 1:
        print("=" * 72)
        print(f"  {'model':<24} {'seconds':>9} {'fields':>8}")
        print("-" * 72)
        for model, elapsed, filled in scored:
            print(f"  {model:<24} {elapsed:>9.0f} {filled:>8}")
        print()
        print("  More fields filled is not automatically better -- a model that")
        print("  invents an address scores well here and sends a tech to the wrong")
        print("  house. Read the work orders above against the transcript.")
    return 0


def _engine_label(settings) -> str:
    """How an engine is named in output: parakeet, or whisper plus its size."""
    engine = (getattr(settings, "engine", "whisper") or "whisper").lower()
    return f"{engine}:{settings.model}" if engine == "whisper" else engine


def _compare_engines(cfg, audio, rate: int, specs: list[str]) -> tuple[str, float | None]:
    """Transcribe the same recording with each engine and print both.

    Speed is only half the question. Read the transcripts: the one that gets
    the address, the part name and the phone number right is the one that
    matters, and that does not always follow the clock.

    Returns the first engine's transcript, so that pairing --engines with
    --models compares extraction against one fixed transcript rather than a
    different one per model.
    """
    import time as _time

    from . import transcribe

    first = ""
    duration_s = None
    timings = []
    for spec in specs:
        settings = _parse_engine(spec, cfg)
        label = _engine_label(settings)
        print("=" * 72)
        print(f"  {label}")
        print("=" * 72)

        started = _time.monotonic()
        try:
            result = transcribe.transcribe(
                audio, rate, settings, cfg.audio.stereo_mode, root=cfg.root
            )
        except Exception as exc:
            print(f"  failed: {exc}\n")
            continue
        elapsed = _time.monotonic() - started

        print(result.text or "  (nothing transcribed)")
        speed = result.duration_s / elapsed if elapsed else 0
        print(f"\n  {elapsed:.0f}s for {result.duration_s:.0f}s of audio "
              f"({speed:.1f}x real time), {len(result.segments)} segments\n")
        timings.append((label, elapsed, speed, len(result.text)))
        if not first:
            first, duration_s = result.text, result.duration_s

    if len(timings) > 1:
        print("=" * 72)
        print(f"  {'engine':<22} {'seconds':>9} {'x real time':>12} {'chars':>8}")
        print("-" * 72)
        for label, elapsed, speed, chars in timings:
            print(f"  {label:<22} {elapsed:>9.0f} {speed:>12.1f} {chars:>8}")
        print()
        print("  Read them, do not just compare the numbers. More characters is")
        print("  not better if the extra ones are wrong, and the engine that gets")
        print("  the address right wins however long it took.\n")
    return first, duration_s


def fields_with_values(data: dict) -> list:
    from . import fields

    out = []
    for f in fields.FIELDS:
        value = data.get(f.name)
        if f.kind == "list":
            if value:
                out.append(f.name)
        elif value not in (None, "", "unknown"):
            out.append(f.name)
    return out


def _cmd_purge(cfg, everything: bool) -> int:
    from . import storage

    target = cfg.output_dir
    if not target.exists():
        print(f"Nothing to purge -- {target} does not exist.")
        return 0

    what = "EVERYTHING in" if everything else "all audio recordings under"
    print(f"About to permanently delete {what} {target}")
    answer = input("Type 'yes' to confirm: ").strip().lower()
    if answer != "yes":
        print("Cancelled. Nothing was deleted.")
        return 1

    files, freed = storage.purge_audio(target, everything=everything)
    print(f"Shredded {files} file(s), {freed / 1_048_576:.1f} MB.")
    if not everything:
        print("Work orders and transcripts were left in place.")
    return 0


def _cmd_run(cfg, use_tray: bool) -> int:
    ui = notify.make_ui(cfg.output.show_popup)
    runner = pipeline.Runner(cfg, ui=ui)

    if cfg.output.keep_audio:
        logging.warning(
            "output.keep_audio is ON -- call recordings will be kept on disk. "
            "Turn it off in config.toml when you are done testing."
        )

    # Loading whisper takes seconds; do it now so the first real call isn't the
    # one that pays for it.
    try:
        from . import transcribe

        transcribe.load_model(cfg.transcribe)
    except Exception as exc:
        return _fatal(str(exc))

    from . import extract

    reachable, detail = extract.check_server(cfg.extract)
    if not reachable:
        # Not fatal: transcripts are still worth having, and Ollama may come up
        # on its own a moment later.
        logging.warning("%s", detail)

    runner.start()
    if use_tray:
        tray.start(runner, cfg)

    logging.info("press Ctrl-C to stop")
    try:
        ui.run(runner.stop_event)
    except KeyboardInterrupt:
        logging.info("stopping")
    finally:
        runner.stop()

    if runner.error is not None:
        return _fatal(str(runner.error))
    return 0


if __name__ == "__main__":
    sys.exit(main())
