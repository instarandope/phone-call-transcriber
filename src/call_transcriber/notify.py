"""Putting the finished work order in front of you.

Two things happen when a call finishes: the text lands on the clipboard, and a
window appears showing it. The clipboard copy is the part that matters -- by
the time you look at the popup, Ctrl-V already works.

Tk owns the main thread. Everything else in the app runs on background threads
and posts here through a queue, because Tk is not thread-safe and calling into
it from the transcription worker is a reliable way to hang the process.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


OK_GREEN = "#15803d"
ERROR_RED = "#b91c1c"
WARN_AMBER = "#b45309"


@dataclass
class Popup:
    title: str
    work_order: str
    transcript: str = ""
    folder: Path | None = None
    # Kept apart on purpose. A note about recording level and the reason
    # extraction failed are not the same kind of thing, and listing them
    # together sends people to investigate the wrong one.
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # Whether the pipeline already put this on the clipboard, so the window can
    # say so truthfully instead of claiming it either way.
    copied: bool = False


def beep(started: bool) -> None:
    """Confirm a hotkey press without needing to look at anything.

    Rising for start, falling for stop, so the two are distinguishable while
    you are on the phone and looking somewhere else entirely.
    """
    try:
        import winsound
    except ImportError:
        return  # not Windows; the tray icon and log still report state

    tones = ((880, 90), (1320, 90)) if started else ((1320, 90), (660, 120))
    try:
        for frequency, duration in tones:
            winsound.Beep(frequency, duration)
    except Exception as exc:
        log.debug("could not beep: %s", exc)


def copy(text: str) -> bool:
    """Put text on the system clipboard. False if no backend is available."""
    try:
        import pyperclip

        pyperclip.copy(text)
        return True
    except Exception as exc:
        log.warning("clipboard copy failed: %s", exc)
        return False


class NullUi:
    """Used when popups are switched off or Tk is missing."""

    def request(self, popup: Popup) -> None:
        log.info("work order ready: %s", popup.title)

    def run(self, stop_event: threading.Event) -> None:
        # Nothing to pump; just idle until the worker is finished.
        while not stop_event.wait(0.5):
            pass

    def shutdown(self) -> None:
        pass


class UiHost:
    """A hidden Tk root that spawns a window per finished call."""

    POLL_MS = 150

    def __init__(self):
        self._queue: queue.Queue[Popup] = queue.Queue()
        self._root = None
        self._stop: threading.Event | None = None

    def request(self, popup: Popup) -> None:
        self._queue.put(popup)

    def run(self, stop_event: threading.Event) -> None:
        import tkinter as tk

        self._stop = stop_event
        self._root = tk.Tk()
        self._root.withdraw()
        self._root.after(self.POLL_MS, self._poll)
        try:
            self._root.mainloop()
        finally:
            self._root = None

    def shutdown(self) -> None:
        root = self._root
        if root is not None:
            try:
                root.after(0, root.quit)
            except Exception:
                pass

    def _poll(self) -> None:
        while True:
            try:
                popup = self._queue.get_nowait()
            except queue.Empty:
                break
            try:
                self._show(popup)
            except Exception as exc:
                log.error("could not show work order window: %s", exc)

        if self._stop is not None and self._stop.is_set():
            if self._root is not None:
                self._root.quit()
            return
        if self._root is not None:
            self._root.after(self.POLL_MS, self._poll)

    def _show(self, popup: Popup) -> None:
        import tkinter as tk
        from tkinter import ttk

        win = tk.Toplevel(self._root)
        win.title(popup.title)
        win.geometry("720x620")
        win.minsize(520, 360)
        win.attributes("-topmost", True)

        frame = ttk.Frame(win, padding=10)
        frame.pack(fill="both", expand=True)

        ttk.Label(
            frame, text=popup.title, font=("Segoe UI", 11, "bold"), anchor="w"
        ).pack(fill="x", pady=(0, 6))

        for message in popup.problems:
            ttk.Label(
                frame, text=f"FAILED: {message}", foreground=ERROR_RED,
                font=("Segoe UI", 9, "bold"),
                wraplength=680, anchor="w", justify="left",
            ).pack(fill="x", pady=(0, 4))

        for message in popup.warnings:
            ttk.Label(
                frame, text=f"Note: {message}", foreground=WARN_AMBER,
                wraplength=680, anchor="w", justify="left",
            ).pack(fill="x", pady=(0, 4))

        text = tk.Text(frame, wrap="none", font=("Consolas", 10), height=20)
        scroll_y = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scroll_y.set)
        text.insert("1.0", popup.work_order)
        # Read-only, but still selectable so partial copies work.
        text.configure(state="disabled")
        scroll_y.pack(side="right", fill="y")
        text.pack(side="top", fill="both", expand=True)

        buttons = ttk.Frame(win, padding=(10, 0, 10, 10))
        buttons.pack(fill="x")

        status = ttk.Label(
            buttons,
            text="Copied to clipboard - just paste" if popup.copied else "",
            foreground=OK_GREEN,
        )
        status.pack(side="left")

        def copy_to(text: str, label: str):
            def handler():
                ok = copy(text)
                status.configure(
                    text=f"{label} copied" if ok else "Copy failed",
                    foreground=OK_GREEN if ok else ERROR_RED,
                )

            return handler

        ttk.Button(buttons, text="Close", command=win.destroy).pack(side="right")
        if popup.folder is not None:
            from . import storage

            ttk.Button(
                buttons,
                text="Open folder",
                command=lambda: storage.open_folder(popup.folder),
            ).pack(side="right", padx=6)
        if popup.transcript:
            ttk.Button(
                buttons,
                text="Copy transcript",
                command=copy_to(popup.transcript, "Transcript"),
            ).pack(side="right", padx=6)
        ttk.Button(
            buttons,
            text="Copy work order",
            command=copy_to(popup.work_order, "Work order"),
        ).pack(side="right", padx=6)

        win.bind("<Escape>", lambda _e: win.destroy())
        win.lift()
        win.focus_force()
        # Drop the always-on-top flag once it has surfaced, so it doesn't sit
        # over everything else for the rest of the day.
        win.after(1200, lambda: win.attributes("-topmost", False))


def make_ui(enabled: bool):
    if not enabled:
        return NullUi()
    try:
        import tkinter  # noqa: F401
    except ImportError:
        log.warning("tkinter is not available; work orders will only be written to disk")
        return NullUi()
    return UiHost()
