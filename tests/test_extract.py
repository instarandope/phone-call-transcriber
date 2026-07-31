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
