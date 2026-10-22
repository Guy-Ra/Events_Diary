"""Versioned production prompts for the full-Investigation VLM branch.

Prompt content is kept separate from API transport so prompt versions can change
without altering the request serializer or runtime. This is the MVP P1-style
full-schema baseline, not an optimized prompt. `fr24_vlm_v1` is the only version
defined here; future versions can be added without changing VLM data models.
"""

import json

FR24_VLM_V1 = "fr24_vlm_v1"
_TEXT_FIELDS = (
    "airline", "primary_flight_id", "alternate_flight_id", "aircraft_code",
    "origin_airport", "origin_city", "origin_timezone", "destination_airport",
    "destination_city", "destination_timezone", "scheduled_departure",
    "actual_departure", "scheduled_arrival", "estimated_or_actual_arrival",
    "aircraft_type", "registration", "serial_number", "age", "aircraft_category",
    "barometric_altitude", "vertical_speed", "gps_altitude", "track", "ground_speed",
    "true_airspeed", "indicated_airspeed", "mach", "wind", "temperature", "fir_uir",
    "icao_address", "squawk", "latitude", "longitude",
)
_BOOLEAN_FIELDS = ("scroll_observed", "same_aircraft_throughout")
VLM_TEXT_FIELDS = _TEXT_FIELDS
VLM_BOOLEAN_FIELDS = _BOOLEAN_FIELDS
VLM_ROUTE_FIELDS = (
    "visual_flight_direction_sector", "visual_flight_direction_text",
    "visual_route_description", "visual_departure_context",
)


def build_vlm_prompt(prompt_version: str = FR24_VLM_V1) -> str:
    """Return the selected stable production prompt or reject unknown versions."""
    if prompt_version != FR24_VLM_V1:
        raise ValueError(f"Unsupported VLM prompt version: {prompt_version!r}.")
    schema = {**dict.fromkeys(_TEXT_FIELDS), **dict.fromkeys(_BOOLEAN_FIELDS)}
    route_schema = {name: {"value": None, "status": "not_visible"} for name in VLM_ROUTE_FIELDS}
    schema["visual_route"] = route_schema
    return f"""Watch the entire continuous video before answering. It contains one aircraft Investigation. The operator may scroll vertically through the same ticket. Information before and after scrolling belongs to the same Investigation unless the video clearly switches to another aircraft. Do not treat scrolling as an aircraft switch or copy values from another aircraft.

The ticket panel is on the left of the full screen. Within the ticket, the upper section contains flight identifiers, aircraft code, airline name, route and times. Origin is on the left and destination is on the right; keep each side's airport, city, timezone and scheduled/actual/estimated times separate. Lower sections contain aircraft and technical information.

Some technical rows contain two independent side-by-side fields. Associate each value with the label directly above it in the same visual column. SERIAL NUMBER (MSN) and AGE are separate fields. Parenthetical text in a label is context, not its value.

Use only visibly supported information. Do not use external aviation knowledge or infer a value from another field. Preserve visible text, numbers, units and codes. Do not normalize, repair or guess unreadable text.

Return one compact JSON object and nothing else. Text fields must be strings or null; the final two fields must be booleans or null. Use null for unreadable, unknown or not-visible values. Use N/A only when literally shown as that field's value.

Also analyze the visible map and selected aircraft trail. Keep these map-based claims separate from ticket fields: never use visual route analysis to fill or change origin_airport, origin_city, destination_airport, or destination_city. Do not infer departure context from flight identity, ticket values, or general aviation knowledge. For each visual route item below, return an object with `value` and `status`. Status is observed only when directly visible, inferred only when supported by the visible map/trail, and not_visible when unavailable (then value must be null). The direction sector must be one of N, NNE, NE, ENE, E, ESE, SE, SSE, S, SSW, SW, WSW, W, WNW, NW, NNW. Describe direction, historical trail, and departure/geographic context only as supported by the map.

Required JSON shape (null values are placeholders, not suggested answers):
{json.dumps(schema, separators=(",", ":"))}"""
