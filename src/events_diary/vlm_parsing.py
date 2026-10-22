"""Parse raw FR24 VLM text into source-specific structured observations.

Only representation-level handling occurs here: outer response whitespace may
be removed for parsing, one standard JSON Markdown fence may be unwrapped, and
JSON null or missing values become unavailable. Decoded semantic strings remain
exactly as returned. No correction, inference, field reassignment, value
normalization, OCR access, API transport, or evidence fusion is performed.
"""

import json
import re
from types import MappingProxyType

from events_diary.vlm import (
    RawVLMResult,
    VLMFieldObservation,
    VLMInvestigationResult,
    VLMParseDiagnostics,
    VLMRequest,
    VLMVisualRouteClaim,
    VLMVisualRouteObservation,
)

# Prompt keys map onto existing OCR field slots when those names are compatible.
# Time slots remain distinct because OCR records their side/role separately.
VLM_FIELD_MAPPING = MappingProxyType({
    "airline": "airline_name",
    "primary_flight_id": "primary_flight_id",
    "alternate_flight_id": "alternate_flight_id",
    "aircraft_code": "aircraft_code",
    "origin_airport": "origin_airport_code",
    "origin_city": "origin_city",
    "origin_timezone": "origin_timezone_text",
    "destination_airport": "destination_airport_code",
    "destination_city": "destination_city",
    "destination_timezone": "destination_timezone_text",
    "scheduled_departure": "origin_scheduled_time",
    "actual_departure": "origin_actual_time",
    "scheduled_arrival": "destination_scheduled_time",
    "estimated_or_actual_arrival": "destination_estimated_or_actual_time",
    "aircraft_type": "AIRCRAFT_TYPE",
    "registration": "REGISTRATION",
    "serial_number": "SERIAL_NUMBER",
    "age": "AGE",
    "aircraft_category": "AIRCRAFT_CATEGORY",
    "barometric_altitude": "BAROMETRIC_ALTITUDE",
    "vertical_speed": "VERTICAL_SPEED",
    "gps_altitude": "GPS_ALTITUDE",
    "track": "TRACK",
    "ground_speed": "GROUND_SPEED",
    "true_airspeed": "TRUE_AIRSPEED",
    "indicated_airspeed": "INDICATED_AIRSPEED",
    "mach": "MACH",
    "wind": "WIND",
    "temperature": "TEMPERATURE",
    "fir_uir": "FIR_UIR",
    "icao_address": "ICAO_ADDRESS",
    "squawk": "SQUAWK",
    "latitude": "LATITUDE",
    "longitude": "LONGITUDE",
    "scroll_observed": "scroll_observed",
    "same_aircraft_throughout": "same_aircraft_throughout",
})
EXPECTED_VLM_FIELDS = (*VLM_FIELD_MAPPING, "visual_route")
_BOOLEAN_FIELDS = frozenset(("scroll_observed", "same_aircraft_throughout"))
_FENCED_JSON = re.compile(
    r"\A```json[ \t]*\r?\n(?P<payload>.*?)\r?\n```[ \t]*\Z",
    re.IGNORECASE | re.DOTALL,
)


def _decode_payload(raw_text: str | None) -> tuple[dict | None, VLMParseDiagnostics]:
    """Decode a strict JSON object or one clearly delimited JSON Markdown fence."""
    if raw_text is None:
        return None, VLMParseDiagnostics("invalid_json", error="No raw response text was provided.")
    text = raw_text.strip()
    json_format = "strict_json"
    payload_text = text
    if text.startswith("```"):
        match = _FENCED_JSON.fullmatch(text)
        if match is None:
            return None, VLMParseDiagnostics("invalid_json", error="Response is not a JSON object or a single JSON fence.")
        payload_text = match.group("payload").strip()
        json_format = "markdown_fence"
    try:
        payload = json.loads(payload_text)
    except (json.JSONDecodeError, TypeError):
        return None, VLMParseDiagnostics(
            "invalid_json", json_format=json_format, error="Response payload is not valid JSON."
        )
    if not isinstance(payload, dict):
        return None, VLMParseDiagnostics(
            "unsupported_top_level_shape", json_format=json_format,
            error="Expected one JSON object at the top level.",
        )
    missing = tuple(key for key in EXPECTED_VLM_FIELDS if key not in payload)
    unknown = tuple(sorted(key for key in payload if key not in (*VLM_FIELD_MAPPING, "visual_route")))
    status = "missing_keys" if missing else (
        "recovered_markdown_fence" if json_format == "markdown_fence" else "valid_json"
    )
    return payload, VLMParseDiagnostics(
        status, json_format=json_format, missing_keys=missing, unknown_keys=unknown
    )


def _observation_value(key: str, value: object) -> tuple[str | bool | None, str]:
    """Retain supported string/boolean values and abstain on unusable JSON types."""
    if value is None:
        return None, "unavailable"
    if key in _BOOLEAN_FIELDS:
        return (value, "observed") if isinstance(value, bool) else (None, "unavailable")
    if isinstance(value, str) and value.strip():
        return value, "observed"
    return None, "unavailable"


def _route_claim(payload: object, key: str) -> VLMVisualRouteClaim:
    """Parse one visual-route claim, conservatively abstaining on malformed data."""
    if not isinstance(payload, dict):
        return VLMVisualRouteClaim(None, "not_visible")
    item = payload.get(key)
    if not isinstance(item, dict):
        return VLMVisualRouteClaim(None, "not_visible")
    value, status = item.get("value"), item.get("status")
    if status not in ("observed", "inferred", "not_visible"):
        return VLMVisualRouteClaim(None, "not_visible")
    if status == "not_visible":
        return VLMVisualRouteClaim(None, "not_visible")
    if not isinstance(value, str) or not value.strip():
        return VLMVisualRouteClaim(None, "not_visible")
    if key == "visual_flight_direction_sector" and value.upper() not in {
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW",
        "SW", "WSW", "W", "WNW", "NW", "NNW",
    }:
        return VLMVisualRouteClaim(None, "not_visible")
    return VLMVisualRouteClaim(value, status)


def parse_vlm_result(request: VLMRequest, raw_result: RawVLMResult) -> VLMInvestigationResult:
    """Convert one raw runtime outcome into a provenance-bearing VLM result.

    Failed runtime outcomes bypass semantic parsing. Successful executions with
    malformed or unsupported model text remain valid results with no observations
    and explicit parse diagnostics; they do not raise or fabricate values.
    """
    if not isinstance(request, VLMRequest):
        raise TypeError("request must be VLMRequest.")
    if not isinstance(raw_result, RawVLMResult):
        raise TypeError("raw_result must be RawVLMResult.")
    if (
        request.investigation_index != raw_result.investigation_index
        or request.source_video != raw_result.source_video
        or request.video_clip_path != raw_result.video_clip_path
        or request.source_start_s != raw_result.source_start_s
        or request.source_end_s != raw_result.source_end_s
        or request.prompt_version != raw_result.prompt_version
    ):
        raise ValueError("Request and raw result provenance must match.")
    if raw_result.execution_status != "success":
        return VLMInvestigationResult(
            raw_result, parse_diagnostics=VLMParseDiagnostics("not_attempted")
        )
    payload, diagnostics = _decode_payload(raw_result.raw_text)
    if payload is None:
        return VLMInvestigationResult(raw_result, parse_diagnostics=diagnostics)
    observations = []
    for key, canonical_name in VLM_FIELD_MAPPING.items():
        value, status = _observation_value(key, payload.get(key))
        observations.append(VLMFieldObservation(
            field_name=canonical_name,
            raw_value=value,
            status=status,
            investigation_index=raw_result.investigation_index,
            source_video=raw_result.source_video,
            source_start_s=raw_result.source_start_s,
            source_end_s=raw_result.source_end_s,
            model_name=raw_result.model_name,
            model_revision=raw_result.model_revision,
            prompt_version=raw_result.prompt_version,
        ))
    route_payload = payload.get("visual_route")
    route = VLMVisualRouteObservation(
        direction_sector=_route_claim(route_payload, "visual_flight_direction_sector"),
        direction_text=_route_claim(route_payload, "visual_flight_direction_text"),
        route_description=_route_claim(route_payload, "visual_route_description"),
        departure_context=_route_claim(route_payload, "visual_departure_context"),
        investigation_index=raw_result.investigation_index,
        source_video=raw_result.source_video,
        source_start_s=raw_result.source_start_s,
        source_end_s=raw_result.source_end_s,
        model_name=raw_result.model_name,
        model_revision=raw_result.model_revision,
        prompt_version=raw_result.prompt_version,
    )
    return VLMInvestigationResult(raw_result, tuple(observations), diagnostics, route)
