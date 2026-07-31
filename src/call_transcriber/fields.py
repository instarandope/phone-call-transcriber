"""What we pull out of a call.

This list is the single place that defines the work order. The JSON schema
handed to the model, the extraction instructions, and the layout of the printed
work order are all generated from it -- so adding "warranty status" or
"referred by" to every future call is one entry here and nothing else.

Order matters: it is the order fields appear on the work order.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Field:
    name: str
    label: str
    prompt: str
    kind: str = "string"  # string | list | enum
    choices: tuple[str, ...] = ()
    # Long values print as an indented block instead of on the label line.
    block: bool = False
    # A job cannot be dispatched without this. If it comes back empty it is
    # listed under STILL NEEDED whatever the model thinks, because whether a
    # field is blank is something we know rather than something to ask about.
    essential: bool = False
    # When a long transcript is split, "first" keeps the earliest answer -- right
    # for a name or address, stated once near the start. "last" keeps the most
    # recent, which is what a negotiated time or price needs, since the earlier
    # mentions are the ones that got rejected.
    prefer: str = "first"


FIELDS: tuple[Field, ...] = (
    Field(
        name="caller_name",
        label="CUSTOMER",
        prompt="Full name of the person calling. If they give a business name, "
        "include it as 'Name (Business)'.",
        essential=True,
    ),
    Field(
        name="callback_number",
        label="PHONE",
        prompt="Best callback number, digits as spoken. If they say 'the number "
        "I'm calling from', return null rather than guessing.",
        essential=True,
    ),
    Field(
        name="service_address",
        label="ADDRESS",
        prompt="Street address where the work is needed, including unit/apt, "
        "city, state and ZIP if stated. This is where the tech drives, which is "
        "not always the caller's own address -- if they mention a rental, a "
        "relative's house, or a second property, use that one.",
        essential=True,
    ),
    Field(
        name="issue_summary",
        label="ISSUE",
        prompt="The problem in one short line a dispatcher can read at a glance. "
        "No pleasantries, no narrative.",
        essential=True,
    ),
    Field(
        name="issue_details",
        label="DETAILS",
        prompt="Concrete technical facts only: symptoms, when it started, what "
        "changed, what the caller already tried, any error codes or noises they "
        "described. Drop all small talk, apologies and repetition. Two or three "
        "sentences at most.",
        block=True,
    ),
    Field(
        name="equipment",
        label="EQUIPMENT",
        prompt="Appliance or system involved, with brand, model, and approximate "
        "age if mentioned.",
    ),
    Field(
        name="urgency",
        label="URGENCY",
        prompt="How fast this needs attention. 'emergency' only for active "
        "damage or a safety issue (water running, gas smell, no heat in freezing "
        "weather). 'urgent' for same-or-next-day. 'routine' otherwise.",
        kind="enum",
        choices=("emergency", "urgent", "routine", "unknown"),
    ),
    Field(
        name="access_notes",
        label="ACCESS",
        prompt="Anything the tech needs to get in and stay safe: gate codes, "
        "lockbox, where to park, dogs, which door, whether someone will be home.",
    ),
    Field(
        name="availability",
        label="AVAILABLE",
        prompt="When the caller said they can be there, in their own terms "
        "(e.g. 'weekday mornings', 'after 3pm Thursday'). This is their general "
        "availability, not the slot that was booked -- that goes in appointment.",
    ),
    Field(
        name="appointment",
        label="SCHEDULED",
        prompt="The date and time finally agreed on. Booking a job is usually a "
        "negotiation -- a time is offered, turned down, another is offered, and "
        "one is settled on. Record the LAST time both sides accepted, not the "
        "first one mentioned, and check nothing later in the call overrides it. "
        "Null if nothing was booked.",
        prefer="last",
    ),
    Field(
        name="existing_customer",
        label="EXISTING",
        prompt="Whether this caller says they have used the business before.",
        kind="enum",
        choices=("yes", "no", "unknown"),
    ),
    Field(
        name="price_discussed",
        label="PRICING",
        prompt="Money that was discussed, and WHOSE figure it is. A price the "
        "business quoted and a ceiling the caller set are different things and "
        "must not be confused -- label them: \"quoted $89 service call\" versus "
        "\"customer budget: up to $600\". If the caller says what they want to "
        "spend and no price was quoted, record only their budget, said as theirs. "
        "Quote figures exactly as spoken.",
        prefer="last",
    ),
    Field(
        name="follow_up",
        label="FOLLOW-UP",
        prompt="Things that were promised to the caller and now have to happen "
        "(send a quote, call back with a part price, confirm a window). One per "
        "item, imperative voice.",
        kind="list",
    ),
    Field(
        name="missing_info",
        label="STILL NEEDED",
        prompt="Which of the fields above the caller never actually provided, so "
        "whoever calls back knows what to ask for. Use the field labels. Do not "
        "list fields that simply don't apply to this job.",
        kind="list",
    ),
)

BY_NAME = {f.name: f for f in FIELDS}
ESSENTIAL = tuple(f for f in FIELDS if f.essential)
SCALAR_FIELDS = tuple(f for f in FIELDS if f.kind != "list")
LIST_FIELDS = tuple(f for f in FIELDS if f.kind == "list")


def json_schema() -> dict:
    """JSON schema for Ollama's structured-output mode.

    Every field is required, with null allowed. Making them required stops the
    model from quietly omitting a field it found nothing for; allowing null
    gives it a way to say so honestly instead of inventing a value.
    """
    props: dict[str, dict] = {}
    for f in FIELDS:
        if f.kind == "list":
            props[f.name] = {"type": "array", "items": {"type": "string"}}
        elif f.kind == "enum":
            props[f.name] = {"type": "string", "enum": list(f.choices)}
        else:
            props[f.name] = {"type": ["string", "null"]}
    return {
        "type": "object",
        "properties": props,
        "required": [f.name for f in FIELDS],
    }


def instructions() -> str:
    """The per-field extraction guide handed to the model."""
    lines = []
    for f in FIELDS:
        suffix = ""
        if f.kind == "enum":
            suffix = f" One of: {', '.join(f.choices)}."
        elif f.kind == "list":
            suffix = " An array of strings; empty array if none."
        lines.append(f"- {f.name}: {f.prompt}{suffix}")
    return "\n".join(lines)


def empty() -> dict:
    """A result with every field blank -- the fallback when extraction fails."""
    out: dict = {}
    for f in FIELDS:
        if f.kind == "list":
            out[f.name] = []
        elif f.kind == "enum":
            out[f.name] = "unknown" if "unknown" in f.choices else f.choices[0]
        else:
            out[f.name] = None
    return out
