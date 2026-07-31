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
    if args.command == "prompt":
        return _cmd_prompt(cfg)
    if args.command == "levels":
        return _cmd_levels(cfg, args.seconds)
    if args.command == "test":
        return _cmd_test(cfg, Path(args.file))
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
    sub.add_parser("prompt", help="show exactly what the model is told to extract")
    levels = sub.add_parser("levels", help="measure your line and suggest thresholds")
    levels.add_argument(
        "--seconds", type=int, default=45, help="how long to listen (default 45)"
    )
    test = sub.add_parser("test", help="process an existing audio file")
    test.add_argument("file", help="path to a .wav/.mp3/.m4a recording")
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


def _cmd_doctor(cfg) -> int:
    from . import audio, extract

    ok = True

    def check(label: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        print(f"  [{'ok' if passed else 'XX'}] {label}" + (f"\n       {detail}" if detail else ""))
        ok = ok and passed

    print(f"call-transcriber {__version__}\n")
    print("Python")
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

    print("\nSpeech model")
    try:
        from . import transcribe

        transcribe.load_model(cfg.transcribe)
        check(f"whisper {cfg.transcribe.model} loads", True)
    except Exception as exc:
        check(f"whisper {cfg.transcribe.model}", False, str(exc))

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
