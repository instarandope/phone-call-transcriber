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

IDLE_COLOR = (90, 100, 115)
RECORDING_COLOR = (200, 40, 40)


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

        def icon_image(color):
            image = Image.new("RGB", (64, 64), (245, 245, 245))
            draw = ImageDraw.Draw(image)
            draw.ellipse((10, 10, 54, 54), fill=color)
            return image

        def status_text(_item):
            if runner.paused.is_set():
                return "Paused"
            if runner.state is State.IN_CALL:
                return "Recording a call..."
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
            icon_image(IDLE_COLOR),
            "Call Transcriber",
            menu=pystray.Menu(
                pystray.MenuItem(status_text, None, enabled=False),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem(
                    lambda _i: "Resume" if runner.paused.is_set() else "Pause",
                    toggle_pause,
                ),
                pystray.MenuItem("Open work orders", open_output),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("Quit", quit_app),
            ),
        )

        def watch():
            """Recolour the icon so a glance tells you if it's recording."""
            last = None
            while not runner.stop_event.wait(0.5):
                current = runner.state is State.IN_CALL and not runner.paused.is_set()
                if current != last:
                    try:
                        icon.icon = icon_image(RECORDING_COLOR if current else IDLE_COLOR)
                    except Exception:
                        pass
                    last = current
            try:
                icon.stop()
            except Exception:
                pass

        threading.Thread(target=watch, name="tray-watch", daemon=True).start()
        icon.run()
    except Exception as exc:
        log.info("tray icon unavailable (%s) -- continuing without it", exc)
