"""Rendering extracted fields into something you can paste straight into a job.

Fields the caller never mentioned are left out entirely rather than printed as
"N/A" -- a work order with eleven blank lines is harder to read than one with
four filled ones. The exception is STILL NEEDED, which is precisely the list of
what is missing and so is always worth printing.
"""

from __future__ import annotations

import time

from . import fields

RULE = "-" * 60


def render(data: dict, *, started_at: float | None = None,
           duration_s: float | None = None, business=None) -> str:
    """Build the copy-ready work order text."""
    lines: list[str] = []

    header = "WORK ORDER"
    if business is not None and business.name:
        header = f"WORK ORDER - {business.name}"
    lines.append(header)

    stamp = time.strftime("%a %d %b %Y, %I:%M %p", time.localtime(started_at or time.time()))
    meta = f"Call received {stamp}"
    if duration_s:
        meta += f"  ({_duration(duration_s)})"
    lines.append(meta)
    lines.append(RULE)

    width = max(len(f.label) for f in fields.FIELDS) + 2
    body: list[str] = []

    # Decided up front, because STILL NEEDED prints even when empty and would
    # otherwise make a completely failed extraction look like a filled-in form
    # that simply had nothing outstanding.
    if not has_content(data):
        lines.append("Nothing could be extracted from this call.")
        lines.append("Check transcript.txt in the call folder.")
        lines.append("")
        lines.append(RULE)
        return "\n".join(lines).rstrip() + "\n"

    for f in fields.SCALAR_FIELDS:
        value = data.get(f.name)
        if not value or (f.kind == "enum" and value == "unknown"):
            continue
        text = str(value).strip()
        if f.kind == "enum":
            text = text.upper()
        if f.block or len(text) > 66:
            body.append(f.label)
            body.extend(f"  {line}" for line in _wrap(text, 66))
            body.append("")
        else:
            body.append(f"{f.label.ljust(width)}{text}")

    for f in fields.LIST_FIELDS:
        items = data.get(f.name) or []
        # STILL NEEDED earns its space even when empty -- "nothing missing" is
        # information a dispatcher wants.
        if not items and f.name != "missing_info":
            continue
        body.append("")
        body.append(f.label)
        if items:
            for item in items:
                wrapped = _wrap(str(item).strip(), 64)
                body.append(f"  - {wrapped[0]}")
                body.extend(f"    {line}" for line in wrapped[1:])
        else:
            body.append("  - nothing outstanding")

    lines.extend(body)
    lines.append("")
    lines.append(RULE)
    return "\n".join(lines).rstrip() + "\n"


def has_content(data: dict) -> bool:
    """Did the extraction actually find anything?

    An enum sitting at its "unknown" default is not a finding, and neither is
    an empty list -- so a result made entirely of those means the call yielded
    nothing and the work order should say so rather than print an empty form.
    """
    for f in fields.FIELDS:
        value = data.get(f.name)
        if f.kind == "list":
            if value:
                return True
        elif f.kind == "enum":
            if value not in (None, "", "unknown"):
                return True
        elif value not in (None, ""):
            return True
    return False


def headline(data: dict) -> str:
    """One line for a notification title or a log entry."""
    who = data.get("caller_name") or "Unknown caller"
    what = data.get("issue_summary") or "no issue captured"
    return f"{who} - {what}"


def slug(data: dict) -> str:
    """Filesystem-safe fragment for the call folder name."""
    raw = (data.get("caller_name") or "").strip().lower()
    kept = [c if c.isalnum() else "-" for c in raw]
    out = "".join(kept).strip("-")
    while "--" in out:
        out = out.replace("--", "-")
    return out[:40] or "unknown"


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines, current = [], words[0]
    for word in words[1:]:
        if len(current) + 1 + len(word) <= width:
            current = f"{current} {word}"
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m {secs:02d}s" if minutes else f"{secs}s"
