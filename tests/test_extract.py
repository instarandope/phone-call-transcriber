import json

import pytest

from call_transcriber import extract, fields


# -- schema ----------------------------------------------------------------


def test_schema_requires_every_field():
    schema = fields.json_schema()
    assert set(schema["required"]) == {f.name for f in fields.FIELDS}


def test_scalar_fields_allow_null_so_the_model_can_say_it_dont_know():
    schema = fields.json_schema()
    assert schema["properties"]["caller_name"]["type"] == ["string", "null"]


def test_enum_fields_are_constrained():
    schema = fields.json_schema()
    assert schema["properties"]["urgency"]["enum"] == [
        "emergency", "urgent", "routine", "unknown"
    ]


def test_empty_has_the_right_shape_per_kind():
    blank = extract.empty_result()
    assert blank["caller_name"] is None
    assert blank["urgency"] == "unknown"
    assert blank["follow_up"] == []


# -- chunking --------------------------------------------------------------


def test_short_transcript_is_one_chunk():
    assert extract._chunk("hello there", 100) == ["hello there"]


def test_chunks_split_on_line_boundaries():
    text = "\n".join(f"line {i}" for i in range(50))
    chunks = extract._chunk(text, 60)
    assert len(chunks) > 1
    assert all(len(c) <= 60 for c in chunks)
    # No line may be cut in half.
    assert "\n".join(chunks).replace("\n", "") == text.replace("\n", "")


def test_a_single_overlong_line_still_terminates():
    chunks = extract._chunk("x" * 500, 100)
    assert len(chunks) == 5
    assert "".join(chunks) == "x" * 500


# -- coercion --------------------------------------------------------------


def test_coerce_strips_and_keeps_real_values():
    out = extract._coerce({"caller_name": "  Jane Doe  ", "urgency": "urgent"})
    assert out["caller_name"] == "Jane Doe"
    assert out["urgency"] == "urgent"


@pytest.mark.parametrize("value", ["N/A", "not stated", "none", "unknown", "-", ""])
def test_models_saying_nothing_becomes_null_not_a_string(value):
    assert extract._coerce({"caller_name": value})["caller_name"] is None


def test_invalid_enum_falls_back_to_unknown():
    assert extract._coerce({"urgency": "kind of urgent"})["urgency"] == "unknown"


def test_a_bare_string_for_a_list_field_is_wrapped():
    assert extract._coerce({"follow_up": "call back"})["follow_up"] == ["call back"]


def test_junk_response_yields_a_blank_result_rather_than_raising():
    assert extract._coerce("not a dict") == extract.empty_result()


# -- merging long calls ----------------------------------------------------


def test_first_real_answer_wins_for_scalars():
    merged = extract._merge([
        {**extract.empty_result(), "caller_name": "Jane"},
        {**extract.empty_result(), "caller_name": "Someone Else"},
    ])
    assert merged["caller_name"] == "Jane"


def test_a_later_chunk_fills_a_field_the_first_one_missed():
    merged = extract._merge([
        extract.empty_result(),
        {**extract.empty_result(), "service_address": "123 Main St"},
    ])
    assert merged["service_address"] == "123 Main St"


def test_lists_accumulate_across_chunks_without_duplicates():
    merged = extract._merge([
        {**extract.empty_result(), "follow_up": ["send quote", "call back"]},
        {**extract.empty_result(), "follow_up": ["call back", "order part"]},
    ])
    assert merged["follow_up"] == ["send quote", "call back", "order part"]


def test_unknown_enum_is_replaced_by_a_real_one():
    merged = extract._merge([
        extract.empty_result(),
        {**extract.empty_result(), "urgency": "emergency"},
    ])
    assert merged["urgency"] == "emergency"


# -- prompt ----------------------------------------------------------------


def test_speaker_note_only_appears_for_split_transcripts():
    plain = extract._user_prompt("hello", None, None)
    labelled = extract._user_prompt("SIDE A: hello", None, None)
    assert extract.SPEAKER_NOTE not in plain
    assert extract.SPEAKER_NOTE in labelled


def test_business_name_is_passed_so_it_isnt_mistaken_for_the_customer():
    class Business:
        name = "Acme Plumbing"
        default_service_area = ""

    prompt = extract._user_prompt("hello", Business(), None)
    assert "Acme Plumbing" in prompt


def test_empty_transcript_short_circuits_without_a_request():
    assert extract.extract("   ", cfg=None) == extract.empty_result()


# -- waiting for Ollama to come up -----------------------------------------


class _Cfg:
    base_url = "http://127.0.0.1:11434"
    model = "gemma3:4b"
    temperature = 0.0
    num_ctx = 8192
    chunk_chars = 12000
    timeout_s = 180


@pytest.fixture
def no_waiting(monkeypatch):
    """Keep the retry logic, drop the sleeps."""
    monkeypatch.setattr(extract.time, "sleep", lambda _s: None)


def test_a_server_that_comes_up_late_is_waited_for(monkeypatch, no_waiting):
    """The reboot case: the app is ready before Ollama has finished starting."""
    import requests

    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise requests.ConnectionError("connection refused")
        return "the response"

    monkeypatch.setattr(extract.requests, "post", flaky)

    assert extract._post("http://x/api/chat", {}, _Cfg()) == "the response"
    assert calls["n"] == 3


def test_giving_up_says_the_transcript_survived(monkeypatch, no_waiting):
    import requests

    monkeypatch.setattr(
        extract.requests, "post",
        lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("refused")),
    )

    with pytest.raises(extract.ExtractionError) as exc:
        extract._post("http://x/api/chat", {}, _Cfg())

    assert "system tray" in str(exc.value)
    assert "still saved" in str(exc.value)


def test_it_does_not_retry_forever(monkeypatch, no_waiting):
    import requests

    calls = {"n": 0}

    def always_refuse(*args, **kwargs):
        calls["n"] += 1
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(extract.requests, "post", always_refuse)

    with pytest.raises(extract.ExtractionError):
        extract._post("http://x/api/chat", {}, _Cfg())

    assert calls["n"] == len(extract.CONNECT_RETRY_DELAYS) + 1


def test_a_slow_model_is_not_retried(monkeypatch, no_waiting):
    """It answered, it is just slow -- retrying only makes the wait longer."""
    import requests

    calls = {"n": 0}

    def slow(*args, **kwargs):
        calls["n"] += 1
        raise requests.Timeout("too slow")

    monkeypatch.setattr(extract.requests, "post", slow)

    with pytest.raises(extract.ExtractionError, match="took longer than"):
        extract._post("http://x/api/chat", {}, _Cfg())

    assert calls["n"] == 1


# -- the work order must not contradict itself -----------------------------


def with_essentials(**overrides):
    """A result with the four dispatch-critical fields filled.

    Those are always added to missing_info when blank, so tests about anything
    else start from them being present -- otherwise every assertion carries
    four unrelated labels.
    """
    data = extract.empty_result()
    data.update({
        "caller_name": "Roomie",
        "callback_number": "204-962-2241",
        "service_address": "19 West Grove Bay, Mitchell, Manitoba",
        "issue_summary": "Door not closing",
    })
    data.update(overrides)
    return data


def test_a_field_that_was_captured_is_not_also_still_needed():
    """The reported bug: ADDRESS filled in, and listed under STILL NEEDED."""
    data = with_essentials(caller_name=None)
    data["missing_info"] = ["service_address", "callback_number", "caller_name"]

    out = extract._reconcile(data)

    assert out["missing_info"] == ["CUSTOMER"]


def test_field_names_become_the_labels_that_are_printed():
    data = with_essentials(callback_number=None, service_address=None)
    data["missing_info"] = ["callback_number", "service_address"]

    assert extract._reconcile(data)["missing_info"] == ["PHONE", "ADDRESS"]


def test_labels_the_model_already_got_right_are_left_alone():
    data = with_essentials(callback_number=None, service_address=None)
    data["missing_info"] = ["PHONE", "ADDRESS"]

    assert extract._reconcile(data)["missing_info"] == ["PHONE", "ADDRESS"]


def test_a_field_named_by_its_label_is_still_dropped_when_captured():
    data = with_essentials()
    data["missing_info"] = ["PHONE"]

    assert extract._reconcile(data)["missing_info"] == []


def test_an_unknown_enum_still_counts_as_missing():
    data = with_essentials()
    data["missing_info"] = ["urgency"]

    assert extract._reconcile(data)["missing_info"] == ["URGENCY"]


def test_free_text_the_model_invented_survives():
    data = with_essentials()
    data["missing_info"] = ["Whether the opener is under warranty"]

    assert extract._reconcile(data)["missing_info"] == [
        "Whether the opener is under warranty"
    ]


def test_duplicates_are_collapsed():
    data = with_essentials(caller_name=None)
    data["missing_info"] = ["caller_name", "CUSTOMER", "caller name"]

    assert extract._reconcile(data)["missing_info"] == ["CUSTOMER"]


def test_staff_names_reach_the_model():
    class Business:
        name = "Hanover Door Systems"
        staff = "Johan, Jeremy"
        default_service_area = ""

    prompt = extract._user_prompt("hello", Business(), None)
    assert "Johan, Jeremy" in prompt
    assert "staff, not the customer" in prompt


# -- reasoning models ------------------------------------------------------


def test_plain_json_is_untouched():
    assert extract._strip_reasoning('{"caller_name": "Roomie"}') == '{"caller_name": "Roomie"}'


def test_a_reasoning_block_is_removed():
    reply = '<think>The caller says their name early on.</think>\n{"caller_name": "Roomie"}'
    assert json.loads(extract._strip_reasoning(reply))["caller_name"] == "Roomie"


def test_the_pipe_tag_convention_is_handled_too():
    reply = '<|think|>Working through it.<|/think|> {"caller_name": "Roomie"}'
    assert json.loads(extract._strip_reasoning(reply))["caller_name"] == "Roomie"


def test_several_reasoning_blocks_are_all_removed():
    reply = '<think>one</think>\n<think>two</think>\n{"caller_name": "Roomie"}'
    assert json.loads(extract._strip_reasoning(reply))["caller_name"] == "Roomie"


def test_untagged_prose_before_the_json_still_parses():
    """Whatever tag convention a future model invents, braces still bound it."""
    reply = 'Let me work through this carefully.\n{"caller_name": "Roomie"}\nDone.'
    assert json.loads(extract._strip_reasoning(reply))["caller_name"] == "Roomie"


def test_something_with_no_json_at_all_is_left_alone_to_fail_loudly():
    assert extract._strip_reasoning("I cannot answer that") == "I cannot answer that"


# -- what is missing is arithmetic, not a question for the model -----------


def test_a_blank_customer_is_flagged_even_when_the_model_says_all_is_well():
    """gemma4 reported "nothing outstanding" on a call with no name on it."""
    data = with_essentials(caller_name=None)
    data["missing_info"] = []

    assert extract._reconcile(data)["missing_info"] == ["CUSTOMER"]


def test_every_essential_field_is_covered():
    out = extract._reconcile(extract.empty_result())
    assert out["missing_info"] == ["CUSTOMER", "PHONE", "ADDRESS", "ISSUE"]


def test_nothing_is_added_when_the_essentials_are_all_there():
    assert extract._reconcile(with_essentials())["missing_info"] == []


def test_an_essential_field_is_not_listed_twice():
    data = with_essentials(caller_name=None)
    data["missing_info"] = ["caller_name", "CUSTOMER"]

    assert extract._reconcile(data)["missing_info"] == ["CUSTOMER"]


def test_non_essential_gaps_are_still_left_to_the_model():
    """Not every blank matters -- an empty PRICING is usually just no quote."""
    assert "PRICING" not in extract._reconcile(with_essentials())["missing_info"]


# -- who is staff ----------------------------------------------------------


class _Business:
    name = "Hanover Door Systems"
    staff = "Johan, Derek"
    default_service_area = ""


def test_staff_are_named_to_the_model():
    prompt = extract._user_prompt("hello", _Business(), None)
    assert "Johan, Derek" in prompt


def test_a_caller_sharing_a_staff_name_is_still_the_customer():
    """Otherwise a customer called Johan would be silently dropped."""
    prompt = extract._user_prompt("hello", _Business(), None)
    assert "still the customer" in prompt
    assert "not by the name alone" in prompt


def test_someone_not_listed_is_not_ruled_out_as_staff():
    """A new hire answering the phone must not become the customer."""
    prompt = extract._user_prompt("hello", _Business(), None)
    assert "may be either side" in prompt


def test_the_standing_rule_covers_staff_nobody_listed():
    """Whoever answers by naming the business is staff, listed or not."""
    assert "naming the business" in extract.SYSTEM_PROMPT
    assert "STAFF" in extract.SYSTEM_PROMPT


def test_the_appointment_field_asks_for_the_final_agreed_time():
    """A booking is a negotiation; the first time offered is usually wrong."""
    prompt = fields.BY_NAME["appointment"].prompt
    assert "LAST time both sides accepted" in prompt
    assert "not the first one mentioned" in prompt
