"""Optional system tray icon.

Entirely cosmetic -- the app records with or without it. It exists so the thing
running invisibly in the background can be paused, checked on, and quit without
hunting for a console window.

Every failure here is swallowed. A tray icon that won't load is not a reason to
stop transcribing calls.
"""

from __future__ import annotations

import logging
import threading

log = logging.getLogger(__name__)

# The tray renders these at 16x16, so the two states have to differ in shape
# as well as colour -- a small disc changing hue is not readable at that size.
# Idle is a hollow ring, recording is a solid red disc.
IDLE_COLOR = (110, 120, 135, 255)
RECORDING_COLOR = (220, 38, 38, 255)


def start(runner, cfg) -> threading.Thread | None:
    """Start the tray in a background thread. Returns None if unavailable."""
    try:
        import pystray  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError:
        log.info("pystray/Pillow not installed -- running without a tray icon")
        return None

    thread = threading.Thread(target=_run, args=(runner, cfg), name="tray", daemon=True)
    thread.start()
    return thread


def _run(runner, cfg) -> None:
    try:
        import pystray
        from PIL import Image, ImageDraw

        from . import storage
        from .vad import State

        def icon_image(recording):
            # Transparent, so it sits correctly on a light or dark taskbar.
            image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)
            if recording:
                draw.ellipse((4, 4, 60, 60), fill=RECORDING_COLOR)
            else:
                draw.ellipse((6, 6, 58, 58), outline=IDLE_COLOR, width=9)
            return image

        def status_text(_item):
            if runner.state is State.IN_CALL:
                return "RECORDING"
            if runner.manual:
                return f"Ready ({runner.calls_handled} recorded today)"
            if runner.paused.is_set():
                return "Paused"
            return f"Listening ({runner.calls_handled} calls today)"

        def toggle_pause(icon, _item):
            if runner.paused.is_set():
                runner.paused.clear()
            else:
                runner.paused.set()
            icon.update_menu()

        def open_output(_icon, _item):
            cfg.output_dir.mkdir(parents=True, exist_ok=True)
            storage.open_folder(cfg.output_dir)

        def quit_app(icon, _item):
            runner.stop_event.set()
            icon.stop()

        icon = pystray.Icon(
            "call_transcriber",
            icon_image(False),
            "Call Transcriber",
            menu=pystray.Menu(
                pystray.MenuItem(status_text, None, enabled=False),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem(
                    lambda _i: (
                        "Stop recording" if runner.state is State.IN_CALL
                        else f"Start recording  ({cfg.control.hotkey.upper()})"
                    ),
                    lambda icon, _item: (runner.toggle_recording(), icon.update_menu()),
                    visible=runner.manual,
                ),
                pystray.MenuItem(
                    lambda _i: "Resume" if runner.paused.is_set() else "Pause",
                    toggle_pause,
                    visible=not runner.manual,
                ),
                pystray.MenuItem("Open work orders", open_output),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("Quit", quit_app),
            ),
        )

        def watch():
            """Redraw the icon so a glance tells you whether it is recording."""
            last = None
            while not runner.stop_event.wait(0.3):
                current = runner.state is State.IN_CALL
                if current == last:
                    continue
                try:
                    icon.icon = icon_image(current)
                    icon.title = (
                        "Call Transcriber - RECORDING" if current else "Call Transcriber"
                    )
                    icon.update_menu()
                except Exception as exc:
                    # The icon may not be on screen yet. Leave `last` alone so
                    # the next tick tries again rather than giving up silently.
                    log.debug("could not update the tray icon: %s", exc)
                    continue
                last = current
            try:
                icon.stop()
            except Exception:
                pass

        threading.Thread(target=watch, name="tray-watch", daemon=True).start()
        icon.run()
    except Exception as exc:
        log.info("tray icon unavailable (%s) -- continuing without it", exc)
