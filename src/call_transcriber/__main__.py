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

    _setup_logging(args.verbose)
    cfg = config.load(Path(args.config) if args.config else None, root=_root())

    for warning in cfg.warnings:
        logging.warning("config: %s", warning)

    if args.command == "devices":
        return _cmd_devices()
    if args.command == "doctor":
        return _cmd_doctor(cfg)
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
    run.add_argument("--tray", action="store_true", help="show a system tray icon")
    sub.add_parser("devices", help="list audio input devices")
    sub.add_parser("doctor", help="check the setup")
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

    parser.set_defaults(command="run", tray=False)
    return parser


def _root() -> Path:
    """Project root: where config.toml and output/ live, regardless of cwd."""
    return Path(__file__).resolve().parents[2]


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    # These are chatty at INFO and say nothing useful here.
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


# -- commands --------------------------------------------------------------


def _cmd_devices() -> int:
    from . import audio

    try:
        devices = audio.list_input_devices()
    except audio.AudioError as exc:
        print(f"error: {exc}")
        return 1

    if not devices:
        print("No audio input devices found.")
        return 1

    print("Audio inputs:\n")
    for device in devices:
        print(f"  {device}")
    print(
        "\nPut a distinctive piece of your adapter's name into audio.device_match\n"
        "in config.toml (matching is case-insensitive)."
    )
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

    print("\nPackages")
    for module, package in (
        ("numpy", "numpy"),
        ("sounddevice", "sounddevice"),
        ("soundfile", "soundfile"),
        ("webrtcvad", "webrtcvad-wheels"),
        ("faster_whisper", "faster-whisper"),
        ("requests", "requests"),
    ):
        try:
            __import__(module)
            check(package, True)
        except Exception as exc:
            check(package, False, f"{exc} -- re-run install.bat")

    print("\nAudio")
    try:
        device = audio.find_device(cfg.audio.device_match)
        check(f"adapter found: {device}", True)
    except audio.AudioError as exc:
        check(f"adapter matching {cfg.audio.device_match!r}", False, str(exc))

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
        device = audio.find_device(cfg.audio.device_match)
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
        logging.error("%s", exc)
        return 1

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
        logging.error("%s", runner.error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
