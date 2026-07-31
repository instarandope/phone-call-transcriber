from call_transcriber import extract, workorder


def filled(**overrides):
    data = extract.empty_result()
    data.update(overrides)
    return data


def test_fields_the_caller_never_mentioned_are_omitted():
    text = workorder.render(filled(caller_name="Jane Doe"))
    assert "Jane Doe" in text
    assert "EQUIPMENT" not in text
    assert "PRICING" not in text


def test_still_needed_is_printed_even_when_empty():
    text = workorder.render(filled(caller_name="Jane Doe"))
    assert "STILL NEEDED" in text
    assert "nothing outstanding" in text


def test_still_needed_lists_what_to_ask_for():
    text = workorder.render(filled(missing_info=["ADDRESS", "PHONE"]))
    assert "- ADDRESS" in text
    assert "- PHONE" in text


def test_unknown_urgency_is_not_printed_as_a_value():
    assert "URGENCY" not in workorder.render(filled(caller_name="Jane"))


def test_known_urgency_is_uppercased():
    assert "URGENCY" in workorder.render(filled(urgency="emergency"))
    assert "EMERGENCY" in workorder.render(filled(urgency="emergency"))


def test_long_details_wrap_into_an_indented_block():
    detail = "The water heater started leaking last night. " * 6
    text = workorder.render(filled(issue_details=detail))
    assert "DETAILS" in text
    assert all(len(line) < 80 for line in text.splitlines())


def test_business_name_appears_in_the_header():
    class Business:
        name = "Acme Plumbing"
        default_service_area = ""

    text = workorder.render(filled(caller_name="Jane"), business=Business())
    assert text.splitlines()[0] == "WORK ORDER - Acme Plumbing"


def test_duration_is_shown_when_known():
    text = workorder.render(filled(caller_name="Jane"), duration_s=374)
    assert "6m 14s" in text


def test_an_empty_extraction_says_so_instead_of_printing_a_blank_form():
    text = workorder.render(extract.empty_result())
    assert "Nothing could be extracted" in text
    assert "transcript.txt" in text


def test_headline_survives_missing_fields():
    assert workorder.headline(extract.empty_result()) == "Unknown caller - no issue captured"
    assert workorder.headline(filled(caller_name="Jane", issue_summary="Leak")) == "Jane - Leak"


def test_slug_is_filesystem_safe():
    assert workorder.slug(filled(caller_name="Jane O'Doe-Smith")) == "jane-o-doe-smith"
    assert workorder.slug(extract.empty_result()) == "unknown"
    assert workorder.slug(filled(caller_name="!!!")) == "unknown"


def test_slug_is_bounded_for_a_very_long_name():
    assert len(workorder.slug(filled(caller_name="a" * 200))) <= 40
