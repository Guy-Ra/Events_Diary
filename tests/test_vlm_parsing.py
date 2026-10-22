"""Deterministic tests for independent VLM response parsing and provenance."""

import ast
import json
from pathlib import Path

import pytest

from events_diary.vlm import RawVLMResult, VLMRequest
from events_diary.vlm_parsing import (
    EXPECTED_VLM_FIELDS,
    VLM_FIELD_MAPPING,
    parse_vlm_result,
)

SOURCE = Path("data/videos/session.mp4")
CLIP = Path("data/debug/vlm/investigation_2.mp4")
MODEL = "gemini-flash-latest"
REVISION = None
PROMPT_VERSION = "fr24_vlm_v1"


def request(**overrides):
    """Build one valid prompt request for a continuous Investigation clip."""
    values = {
        "investigation_index": 2, "source_video": SOURCE, "video_clip_path": CLIP,
        "source_start_s": 12.0, "source_end_s": 20.4, "prompt": "Read this ticket.",
        "prompt_version": PROMPT_VERSION,
    }
    values.update(overrides)
    return VLMRequest(**values)


def raw_result(text=None, **overrides):
    """Build one raw runtime result with provenance matching ``request``."""
    values = {
        "execution_status": "success", "investigation_index": 2,
        "source_video": SOURCE, "video_clip_path": CLIP,
        "source_start_s": 12.0, "source_end_s": 20.4, "model_name": MODEL,
        "model_revision": REVISION, "prompt_version": PROMPT_VERSION, "raw_text": text,
    }
    values.update(overrides)
    return RawVLMResult(**values)


def full_payload(**overrides):
    """Return a realistic full production schema fixture with nullable remainder."""
    values = dict.fromkeys(EXPECTED_VLM_FIELDS)
    values.update({
        "airline": "Israir Airlines", "primary_flight_id": "ISR972",
        "alternate_flight_id": "6H972", "aircraft_code": "A332",
        "origin_airport": "JFK", "origin_city": "NEW YORK",
        "origin_timezone": "EDT (UTC -04:00)", "destination_airport": "TLV",
        "destination_city": "TEL AVIV", "destination_timezone": "IDT (UTC +03:00)",
        "scheduled_departure": "2:30 PM", "actual_departure": "4:49 PM",
        "scheduled_arrival": "8:45 AM", "estimated_or_actual_arrival": "9:31 AM",
        "aircraft_type": "Airbus A330-243", "registration": "4X-BAL",
        "serial_number": "1100", "age": "16 years", "aircraft_category": "Passenger",
        "track": "281°", "squawk": "5212", "scroll_observed": True,
        "same_aircraft_throughout": True,
    })
    values["visual_route"] = {
        "visual_flight_direction_sector": {"value": "ESE", "status": "observed"},
        "visual_flight_direction_text": {"value": "Generally east-southeast", "status": "inferred"},
        "visual_route_description": {"value": "Trail extends west of aircraft", "status": "observed"},
        "visual_departure_context": {"value": None, "status": "not_visible"},
    }
    values.update(overrides)
    return values


def parsed(payload, *, raw_text=None):
    """Parse a JSON mapping while allowing exact raw-response preservation tests."""
    text = json.dumps(payload, ensure_ascii=False) if raw_text is None else raw_text
    return parse_vlm_result(request(), raw_result(text))


def by_name(result):
    """Index parsed observations by their canonical field names."""
    return {item.field_name: item for item in result.observations}


def test_realistic_full_schema_strict_json_creates_one_observation_per_field():
    payload = full_payload()
    text = json.dumps(payload, ensure_ascii=False)
    result = parse_vlm_result(request(), raw_result(text))
    fields = by_name(result)
    assert len(result.observations) == len(VLM_FIELD_MAPPING)
    assert len(EXPECTED_VLM_FIELDS) == len(VLM_FIELD_MAPPING) + 1
    assert result.parse_diagnostics.status == "valid_json"
    assert result.parse_diagnostics.json_format == "strict_json"
    assert fields["airline_name"].raw_value == "Israir Airlines"
    assert fields["TRACK"].raw_value == "281°"
    assert fields["scroll_observed"].raw_value is True
    assert fields["same_aircraft_throughout"].raw_value is True
    assert result.visual_route.direction_sector.value == "ESE"
    assert result.visual_route.direction_sector.status == "observed"
    assert result.visual_route.direction_text.status == "inferred"
    assert result.visual_route.departure_context.value is None
    assert result.visual_route.departure_context.status == "not_visible"
    assert "origin_airport_code" in fields and fields["origin_airport_code"].raw_value == "JFK"
    assert all(item.status == "observed" for item in result.observations if item.raw_value is not None)


def test_full_schema_nulls_are_unavailable_and_all_fields_are_present():
    result = parsed(dict.fromkeys(EXPECTED_VLM_FIELDS))
    assert len(result.observations) == len(EXPECTED_VLM_FIELDS) - 1
    assert all(item.raw_value is None and item.status == "unavailable" for item in result.observations)


def test_canonical_key_mapping_reuses_ocr_names_and_keeps_time_slots_distinct():
    assert VLM_FIELD_MAPPING["airline"] == "airline_name"
    assert VLM_FIELD_MAPPING["origin_airport"] == "origin_airport_code"
    assert VLM_FIELD_MAPPING["destination_timezone"] == "destination_timezone_text"
    assert VLM_FIELD_MAPPING["scheduled_departure"] == "origin_scheduled_time"
    assert VLM_FIELD_MAPPING["scheduled_arrival"] == "destination_scheduled_time"
    assert VLM_FIELD_MAPPING["fir_uir"] == "FIR_UIR"


def test_raw_string_is_preserved_exactly_without_semantic_normalization():
    value = "  ISR972  "
    fields = by_name(parsed(full_payload(primary_flight_id=value)))
    assert fields["primary_flight_id"].raw_value == value
    assert fields["primary_flight_id"].status == "observed"


def test_missing_keys_become_unavailable_and_are_reported():
    result = parsed({"airline": "Israir Airlines"})
    fields = by_name(result)
    assert fields["airline_name"].raw_value == "Israir Airlines"
    assert fields["SERIAL_NUMBER"].status == "unavailable"
    assert result.parse_diagnostics.status == "missing_keys"
    assert "serial_number" in result.parse_diagnostics.missing_keys


def test_extra_keys_are_ignored_and_diagnosed_without_new_observations():
    payload = full_payload(unrecognized_value="keep only in raw text")
    result = parsed(payload)
    assert len(result.observations) == len(VLM_FIELD_MAPPING)
    assert result.parse_diagnostics.unknown_keys == ("unrecognized_value",)
    assert result.raw_result.raw_text is not None
    assert "unrecognized_value" in result.raw_result.raw_text


def test_standard_markdown_json_fence_is_recovered_without_changing_raw_text():
    original = "```json\n{" + '"serial_number":"738209"' + "}\n```"
    result = parse_vlm_result(request(), raw_result(original))
    assert result.parse_diagnostics.status == "missing_keys"
    assert result.parse_diagnostics.json_format == "markdown_fence"
    assert result.raw_result.raw_text == original
    assert by_name(result)["SERIAL_NUMBER"].raw_value == "738209"


@pytest.mark.parametrize("text,status", [
    ("{broken", "invalid_json"),
    ("The flight is ISR972. {\"airline\":\"Israir\"}", "invalid_json"),
    ("```json\n{broken}\n```", "invalid_json"),
    ("[1, 2]", "unsupported_top_level_shape"),
    (None, "invalid_json"),
])
def test_invalid_or_unsupported_semantic_response_returns_clean_empty_result(text, status):
    result = parse_vlm_result(request(), raw_result(text))
    assert result.execution_status == "success"
    assert result.observations == ()
    assert result.parse_diagnostics.status == status
    assert result.raw_result.raw_text == text


def test_prose_and_embedded_json_are_not_scraped():
    result = parse_vlm_result(request(), raw_result('Airline: Israir. {"airline":"Israir"}'))
    assert result.observations == ()
    assert result.parse_diagnostics.status == "invalid_json"


@pytest.mark.parametrize("value", [17, 4.9, False, [], {}, {"text": "JFK"}])
def test_unexpected_non_string_text_field_types_become_unavailable(value):
    result = parsed(full_payload(origin_airport=value))
    item = by_name(result)["origin_airport_code"]
    assert item.raw_value is None
    assert item.status == "unavailable"


@pytest.mark.parametrize("value", ["true", 1, [], {}, ""])
def test_context_boolean_fields_require_actual_json_booleans(value):
    item = by_name(parsed(full_payload(scroll_observed=value)))["scroll_observed"]
    assert item.raw_value is None
    assert item.status == "unavailable"


def test_null_context_boolean_is_unavailable_and_boolean_false_is_observed():
    fields = by_name(parsed(full_payload(scroll_observed=False, same_aircraft_throughout=None)))
    assert fields["scroll_observed"].raw_value is False
    assert fields["scroll_observed"].status == "observed"
    assert fields["same_aircraft_throughout"].status == "unavailable"


def test_calibration_wrong_association_is_preserved_without_reassignment():
    fields = by_name(parsed({"serial_number": "738209", "icao_address": None}))
    assert fields["SERIAL_NUMBER"].raw_value == "738209"
    assert fields["SERIAL_NUMBER"].status == "observed"
    assert fields["ICAO_ADDRESS"].raw_value is None
    assert fields["ICAO_ADDRESS"].status == "unavailable"


@pytest.mark.parametrize("status", ["timeout", "runtime_error"])
def test_failed_execution_bypasses_parsing_and_preserves_error(status):
    raw = raw_result(execution_status=status, raw_text='{"airline":"ignored"}', error="branch failed")
    result = parse_vlm_result(request(), raw)
    assert result.observations == ()
    assert result.error == "branch failed"
    assert result.parse_diagnostics.status == "not_attempted"


def test_observation_provenance_is_copied_and_optional_revision_stays_none():
    result = parse_vlm_result(request(), raw_result(json.dumps(full_payload())))
    assert result.observations
    for item in result.observations:
        assert item.investigation_index == 2
        assert item.source_video == SOURCE
        assert (item.source_start_s, item.source_end_s) == (12.0, 20.4)
        assert item.model_name == MODEL
        assert item.model_revision is None
        assert item.prompt_version == PROMPT_VERSION
        assert item.evidence_time_s is None


def test_mismatched_request_and_raw_provenance_is_rejected():
    with pytest.raises(ValueError, match="provenance"):
        parse_vlm_result(request(investigation_index=9), raw_result("{}"))


def test_parser_module_has_no_ocr_or_api_http_dependency():
    module_path = Path(__import__("events_diary.vlm_parsing", fromlist=["__file__"]).__file__)
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    assert not any(name == "events_diary.ocr" or name.startswith("events_diary.ocr_") for name in modules)
    assert not any(name in ("urllib", "urllib.request", "http", "requests") or name.startswith("http.")
        for name in modules)
    assert not any("fusion" in name for name in modules)
