"""Global hotkey for manual recording.

Global means it works while another window has focus -- you press it without
leaving whatever you are typing in.

The listener runs on its own thread and touches nothing but a threading.Event
in the detector, so there is no lock to get wrong.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# pynput writes modifiers and named keys inside angle brackets; single
# characters go in bare. Users should not have to know that.
MODIFIER_ALIASES = {
    "win": "cmd",
    "windows": "cmd",
    "super": "cmd",
    "control": "ctrl",
    "option": "alt",
}


class HotkeyError(RuntimeError):
    """Raised when a hotkey cannot be registered."""


def to_pynput(hotkey: str) -> str:
    """'ctrl+alt+r' -> '<ctrl>+<alt>+r'."""
    parts = [p.strip().lower() for p in hotkey.split("+") if p.strip()]
    if not parts:
        raise HotkeyError("the hotkey is empty")

    out = []
    for part in parts:
        part = MODIFIER_ALIASES.get(part, part)
        out.append(part if len(part) == 1 else f"<{part}>")
    return "+".join(out)


def start(hotkey: str, on_press):
    """Listen for `hotkey` and call `on_press` each time. Returns the listener.

    Raises HotkeyError if it cannot be set up, because in manual mode a hotkey
    that silently does nothing leaves no way at all to record.
    """
    try:
        from pynput import keyboard
    except ImportError as exc:
        raise HotkeyError(
            "pynput is not installed, so the recording hotkey cannot work. "
            "Re-run install.bat."
        ) from exc

    combo = to_pynput(hotkey)
    try:
        listener = keyboard.GlobalHotKeys({combo: on_press})
        listener.daemon = True
        listener.start()
    except Exception as exc:
        raise HotkeyError(
            f"could not register the hotkey {hotkey!r} ({exc}). "
            f"Another program may already have claimed it -- try a different "
            f"combination in config.toml under [control]."
        ) from exc

    log.info("press %s to start and stop recording", hotkey.upper())
    return listener


def stop(listener) -> None:
    if listener is None:
        return
    try:
        listener.stop()
    except Exception:
        pass
