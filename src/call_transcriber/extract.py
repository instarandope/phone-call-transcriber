"""Turning a raw transcript into work-order fields with a local LLM.

Talks to Ollama on localhost, so the transcript never leaves the machine
either. Ollama's structured-output mode constrains decoding to the schema in
fields.py, which means the reply is always parseable JSON with every key
present -- no regex salvage, no retry loop for malformed output.

The prompt is deliberately hostile to invention. A work order with a blank
address is annoying; one with a confidently wrong address sends a tech to the
wrong house.
"""

from __future__ import annotations

import json
import logging
import time

import requests

from . import fields

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You extract service work-order details from a transcript of a phone call to a \
service business.

Rules, in order of importance:
1. Use ONLY what is actually said in the transcript. Never infer, complete, or \
invent a value. If a detail was not stated, the value is null (or an empty \
array for list fields).
2. Do not repeat the caller's phrasing when it carries no information. Strip \
greetings, apologies, hold music chatter, thanks, small talk about weather, \
and anything the transcript repeats. A dispatcher should be able to read the \
result in five seconds.
3. Transcripts are imperfect. If a name, street or number is garbled to the \
point of being a guess, treat it as not stated and add it to missing_info \
rather than writing down something that might be wrong.
4. Keep numbers, addresses, model numbers and prices exactly as spoken. Do not \
normalise, reformat, or "fix" them.
5. The transcript usually does not say who is speaking. The person who answers \
by naming the business, and anyone introduced as working there, are STAFF -- \
never the customer. The customer is the other party: the one describing a \
problem, giving an address to visit, and being asked for their details. If the \
staff member says "my name is X", X is not the customer's name.
6. Reply with JSON matching the schema. No commentary.
"""

SPEAKER_NOTE = """\
This transcript is labelled SIDE A and SIDE B because the recorder captured \
each side of the line separately, but it does not know which is which. Work \
out from the content which side is the caller (the customer) and which is the \
business answering. Extract details about the CALLER only -- if the person \
answering states their own name or the business address, that is not the \
customer's name or the service address.
"""


class ExtractionError(RuntimeError):
    """Raised when the local model cannot be reached or fails outright."""


# Whether a given model accepts the `think` parameter. Ollama rejects it for
# models with no reasoning mode, so the answer is learned from the first
# attempt and remembered rather than guessed from the model's name.
_THINK_SUPPORT: dict[str, bool] = {}


def _strip_reasoning(content: str) -> str:
    """Remove any reasoning the model emitted before its answer.

    Structured output should prevent this, and `think: false` should prevent it
    again, but a reasoning model that ignores both would otherwise fail at
    json.loads with a wall of prose. Falling back to the outermost braces
    handles whatever tag convention a future model invents.
    """
    text = content.strip()
    for opener, closer in (("<think>", "</think>"), ("<|think|>", "<|/think|>")):
        while opener in text and closer in text:
            start = text.index(opener)
            end = text.index(closer, start) + len(closer)
            text = (text[:start] + text[end:]).strip()

    if text.startswith("{"):
        return text
    first, last = text.find("{"), text.rfind("}")
    return text[first : last + 1] if 0 <= first < last else text


def empty_result() -> dict:
    """A blank result, so a failed extraction still produces a usable file."""
    return fields.empty()


def check_server(cfg) -> tuple[bool, str]:
    """Is Ollama up and does it have the configured model? Used by `doctor`."""
    try:
        resp = requests.get(f"{cfg.base_url}/api/tags", timeout=5)
        resp.raise_for_status()
    except requests.RequestException as exc:
        return False, (
            f"cannot reach Ollama at {cfg.base_url} ({exc}). "
            "Install it from https://ollama.com/download and make sure it is running."
        )

    try:
        installed = [m.get("name", "") for m in resp.json().get("models", [])]
    except ValueError:
        return False, f"Ollama at {cfg.base_url} returned a response that wasn't JSON"

    # Ollama reports "gemma3:4b"; a config of "gemma3" should still match.
    wanted = cfg.model
    if any(name == wanted or name.split(":")[0] == wanted.split(":")[0] for name in installed):
        return True, f"Ollama is running with {wanted}"
    return False, (
        f"Ollama is running but {wanted!r} is not installed. "
        f"Run:  ollama pull {wanted}\n"
        f"  Models present: {', '.join(installed) if installed else '(none)'}"
    )


def extract(transcript_text: str, cfg, business=None) -> dict:
    """Extract work-order fields. Long transcripts are chunked and merged."""
    text = (transcript_text or "").strip()
    if not text:
        return fields.empty()

    chunks = _chunk(text, cfg.chunk_chars)
    if len(chunks) == 1:
        return _reconcile(_extract_one(chunks[0], cfg, business, part=None))

    log.info("transcript is %d chars; extracting in %d parts", len(text), len(chunks))
    results = []
    for index, chunk in enumerate(chunks, start=1):
        results.append(_extract_one(chunk, cfg, business, part=(index, len(chunks))))
    return _reconcile(_merge(results))


def _reconcile(data: dict) -> dict:
    """Stop the work order contradicting itself.

    Models list a field under missing_info and then fill it in anyway, which
    produces a work order showing an address and, below it, "STILL NEEDED:
    service_address". Whether a value is present is a fact we hold, not
    something to be asked about -- so it is settled here rather than in the
    prompt. Field names are also mapped to their printed labels, since
    "callback_number" means nothing to whoever picks the job up.
    """
    captured: set[str] = set()
    for f in fields.FIELDS:
        if f.kind == "list":
            continue
        value = data.get(f.name)
        if value in (None, "", "unknown"):
            continue
        captured.add(f.name)
        captured.add(f.label.lower())

    cleaned: list[str] = []
    for item in data.get("missing_info") or []:
        raw = str(item).strip()
        if not raw:
            continue
        key = raw.lower().replace(" ", "_").replace("-", "_")
        if key in captured or raw.lower() in captured:
            continue
        field = fields.BY_NAME.get(key)
        label = field.label if field else raw
        if label not in cleaned:
            cleaned.append(label)

    # Add anything essential the model failed to flag. Left to itself a model
    # will report "nothing outstanding" on a call with no customer name on it,
    # which is worse than saying nothing -- it reads as a checked box.
    for f in fields.ESSENTIAL:
        if f.name not in captured and f.label not in cleaned:
            cleaned.append(f.label)

    data["missing_info"] = cleaned
    return data


# -- one round trip --------------------------------------------------------


# Ollama is a separate program with its own startup entry, so after a reboot it
# may still be coming up when the first call is processed -- or someone may have
# quit it from the tray. Waiting is far better than losing the extraction, since
# processing happens in the background where nobody is watching a clock.
CONNECT_RETRY_DELAYS = (5, 15, 30)


def _post(url: str, payload: dict, cfg):
    """POST to Ollama, waiting it out if the server is not up yet."""
    last: Exception | None = None

    for attempt, delay in enumerate((0, *CONNECT_RETRY_DELAYS)):
        if delay:
            log.warning(
                "Ollama is not answering at %s -- waiting %ds and trying again "
                "(%d of %d). If it was just started, it is probably still loading.",
                cfg.base_url, delay, attempt, len(CONNECT_RETRY_DELAYS),
            )
            time.sleep(delay)
        try:
            return requests.post(url, json=payload, timeout=cfg.timeout_s)
        except requests.Timeout as exc:
            # It answered, it is just slow. Retrying would only take longer.
            raise ExtractionError(
                f"the local model took longer than {cfg.timeout_s}s. Try a smaller "
                f"model in [extract] (gemma3:1b) or raise extract.timeout_s."
            ) from exc
        except requests.ConnectionError as exc:
            last = exc
        except requests.RequestException as exc:
            raise ExtractionError(f"could not reach Ollama at {cfg.base_url} ({exc})") from exc

    waited = sum(CONNECT_RETRY_DELAYS)
    raise ExtractionError(
        f"Ollama did not answer at {cfg.base_url} after waiting {waited}s ({last}). "
        f"Open Ollama from the Start menu -- it lives in the system tray -- and the "
        f"transcript of this call is still saved, so nothing was lost."
    )


def _extract_one(text: str, cfg, business, part: tuple[int, int] | None) -> dict:
    user = _user_prompt(text, business, part)
    payload = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "format": fields.json_schema(),
        "stream": False,
        "options": {
            "temperature": cfg.temperature,
            "num_ctx": cfg.num_ctx,
        },
    }
    if _THINK_SUPPORT.get(cfg.model, True):
        payload["think"] = bool(getattr(cfg, "think", False))

    url = f"{cfg.base_url}/api/chat"
    resp = _post(url, payload, cfg)

    # Older models have no reasoning mode and reject the parameter outright.
    if not resp.ok and "think" in resp.text.lower() and "think" in payload:
        log.debug("%s does not accept the think parameter; retrying without it", cfg.model)
        _THINK_SUPPORT[cfg.model] = False
        payload.pop("think")
        resp = _post(url, payload, cfg)

    if resp.status_code == 404:
        raise ExtractionError(
            f"Ollama does not have the model {cfg.model!r}. Run:  ollama pull {cfg.model}"
        )
    if not resp.ok:
        raise ExtractionError(f"Ollama returned HTTP {resp.status_code}: {resp.text[:400]}")

    try:
        content = resp.json()["message"]["content"]
    except (ValueError, KeyError) as exc:
        raise ExtractionError(f"unexpected reply from Ollama: {resp.text[:400]}") from exc

    try:
        raw = json.loads(_strip_reasoning(content))
    except ValueError as exc:
        # Structured output should make this impossible, but an older Ollama
        # that ignores `format` would land here.
        raise ExtractionError(
            f"the model did not return JSON. Update Ollama (0.5+ is required for "
            f"structured output). Reply began: {content[:200]!r}"
        ) from exc

    return _coerce(raw)


def _user_prompt(text: str, business, part: tuple[int, int] | None) -> str:
    blocks = []
    if part:
        index, total = part
        blocks.append(
            f"This is part {index} of {total} of one long call. Extract only what "
            f"appears in this part; leave anything else null."
        )
    if "SIDE A" in text or "SIDE B" in text:
        blocks.append(SPEAKER_NOTE)
    if business is not None and business.name:
        blocks.append(
            f"The business receiving the call is {business.name}. This call came "
            f"IN to them. Their own name, staff, equipment and address are never "
            f"the customer's details."
        )
    if business is not None and getattr(business, "staff", ""):
        blocks.append(
            f"These people answer the phone for the business: {business.staff}. "
            f"When one of them is the one speaking for the business, they are "
            f"staff, not the customer -- so a line like \"my name is X\" from the "
            f"side that answered is not the customer's name. This is a hint about "
            f"who is likely on which side, not a rule about the name itself: if "
            f"the person CALLING happens to share a name with one of them, they "
            f"are still the customer. Judge by what each side says and does, not "
            f"by the name alone. Anyone not on this list may be either side."
        )
    if business is not None and business.default_service_area:
        blocks.append(
            f"The business serves {business.default_service_area}. If the caller "
            f"gives a street with no city, you may record the street as stated -- "
            f"still do not invent a city, state or ZIP they did not say."
        )
    blocks.append("Extract these fields:\n" + fields.instructions())
    blocks.append("TRANSCRIPT:\n" + text)
    return "\n\n".join(blocks)


# -- long calls ------------------------------------------------------------


def _chunk(text: str, limit: int) -> list[str]:
    """Split on line boundaries so a chunk never cuts a sentence in half."""
    if limit <= 0 or len(text) <= limit:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for line in text.splitlines():
        # A single line longer than the limit is rare but must not loop forever.
        if len(line) > limit:
            if current:
                chunks.append("\n".join(current))
                current, size = [], 0
            for i in range(0, len(line), limit):
                chunks.append(line[i : i + limit])
            continue
        if size + len(line) + 1 > limit and current:
            chunks.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks or [text]


def _merge(results: list[dict]) -> dict:
    """Combine per-chunk extractions.

    First real answer wins for scalars, because callers state their name and
    address once, near the start. Lists are unioned -- follow-ups accumulate
    across the whole call.
    """
    merged = fields.empty()
    for result in results:
        for f in fields.FIELDS:
            value = result.get(f.name)
            if f.kind == "list":
                existing = merged[f.name]
                for item in value or []:
                    if item and item not in existing:
                        existing.append(item)
            elif f.kind == "enum":
                if merged[f.name] in (None, "unknown") and value not in (None, "unknown"):
                    merged[f.name] = value
            elif f.prefer == "last":
                if value:
                    merged[f.name] = value
            elif merged[f.name] in (None, "") and value:
                merged[f.name] = value
    return merged


def _coerce(raw: dict) -> dict:
    """Normalise whatever came back into exactly the expected shape."""
    out = fields.empty()
    if not isinstance(raw, dict):
        return out

    for f in fields.FIELDS:
        value = raw.get(f.name)
        if f.kind == "list":
            if isinstance(value, str):
                value = [value]
            if isinstance(value, list):
                # A model asked for an empty list sometimes writes ["None"]
                # instead, which printed as a bullet point reading "- None".
                out[f.name] = [
                    str(v).strip()
                    for v in value
                    if str(v).strip() and str(v).strip().lower().rstrip(".") not in _NULLISH
                ]
        elif f.kind == "enum":
            if isinstance(value, str) and value in f.choices:
                out[f.name] = value
        else:
            if value is None:
                continue
            text = str(value).strip()
            # Models like to write "N/A" or "not stated" instead of null.
            if text and text.lower() not in _NULLISH:
                out[f.name] = text
    return out


_NULLISH = {
    "n/a",
    "na",
    "none",
    "null",
    "unknown",
    "not stated",
    "not provided",
    "not mentioned",
    "not specified",
    "not given",
    "unspecified",
    "-",
    "--",
}
