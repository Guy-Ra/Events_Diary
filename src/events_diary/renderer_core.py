"""Shared deterministic display helpers for future Events Diary renderers.

Phase 7.5 profiles select which canonical fields belong in an output. This
module defines their display labels and values without duplicating profile
membership or inspecting Phase 6 evidence. It does not render diary documents,
perform inference, or call OCR, VLM, or fusion code.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from decimal import ROUND_HALF_UP, Decimal
from numbers import Real
from types import MappingProxyType

from events_diary.canonical_events import (
    CANONICAL_RENDERER_FIELD_PATHS,
    CanonicalEvent,
)

OMISSION_MARKER = None
UNRESOLVED_DISPLAY = "Could not determine"
NOT_AVAILABLE_DISPLAY = "N/A"

FIELD_LABELS: Mapping[str, str] = MappingProxyType({
    "event_time_s": "Event time",
    "flight.primary_id": "Flight",
    "flight.alternate_id": "Alternate flight",
    "flight.airline": "Airline",
    "route.origin_airport_code": "Origin",
    "route.destination_airport_code": "Destination",
    "route.scheduled_departure": "Scheduled departure",
    "route.actual_departure": "Actual departure",
    "route.scheduled_arrival": "Scheduled arrival",
    "route.estimated_or_actual_arrival": "Estimated/actual arrival",
    "aircraft.registration": "Registration",
    "aircraft.code": "Aircraft code",
    "aircraft.type": "Aircraft type",
    "movement.track_degrees": "Track",
    "movement.direction": "Direction",
    "movement.visual_flight_direction": "Visual direction",
    "movement.visual_route_description": "Visual route",
    "movement.visual_departure_context": "Departure context",
    "investigation_id": "Investigation",
    "start_time_s": "Start time",
    "end_time_s": "End time",
    "data_quality.unresolved_fields": "Unresolved fields",
    "data_quality.not_available_fields": "Not available fields",
})

_TIME_PATHS = frozenset(("event_time_s", "start_time_s", "end_time_s"))
_METADATA_PATHS = frozenset(("investigation_id", *_TIME_PATHS))
_QUALITY_LIST_PATHS = frozenset((
    "data_quality.unresolved_fields", "data_quality.not_available_fields",
))

if set(FIELD_LABELS) != CANONICAL_RENDERER_FIELD_PATHS:
    raise RuntimeError("Renderer labels must cover every canonical renderer profile path exactly.")


def label_for_path(path: str) -> str:
    """Return a stable human-readable label for an approved canonical path."""
    try:
        return FIELD_LABELS[path]
    except (KeyError, TypeError) as exc:
        raise ValueError("path must identify an approved canonical renderer field.") from exc


def _seconds_to_tenths(seconds: Real) -> int:
    """Validate a nonnegative real timestamp and round deterministically to tenths."""
    if isinstance(seconds, bool) or not isinstance(seconds, Real):
        raise TypeError("video-relative time must be a real number, not bool.")
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("video-relative time must be finite and nonnegative.")
    return int(
        (Decimal(str(seconds)) * 10).quantize(Decimal(1), rounding=ROUND_HALF_UP),
    )


def format_video_timestamp(seconds: Real) -> str:
    """Format video-relative seconds as ``MM:SS.t`` without wall-clock conversion."""
    total_tenths = _seconds_to_tenths(seconds)
    minutes, remaining_tenths = divmod(total_tenths, 600)
    whole_seconds, tenths = divmod(remaining_tenths, 10)
    return f"{minutes:02d}:{whole_seconds:02d}.{tenths}"


def format_investigation_interval(start_s: Real, end_s: Real) -> str:
    """Format a video-relative Investigation interval using an en dash."""
    start_tenths = _seconds_to_tenths(start_s)
    end_tenths = _seconds_to_tenths(end_s)
    if end_tenths < start_tenths:
        raise ValueError("interval end must not precede interval start.")
    return f"{format_video_timestamp(start_s)}–{format_video_timestamp(end_s)}"


def _path_state(event: CanonicalEvent, path: str) -> str:
    """Read canonical field state while treating metadata and quality lists as values."""
    if path in _METADATA_PATHS or path in _QUALITY_LIST_PATHS:
        return "value"
    return event.field_state(path)


def _path_value(event: CanonicalEvent, path: str):
    """Read a field through the public canonical event data groups."""
    value: object = event
    for component in path.split("."):
        value = getattr(value, component)
    return value


def _format_value(path: str, value: object) -> str | None:
    """Format one present canonical value without changing its meaning."""
    if path in _TIME_PATHS:
        return format_video_timestamp(value)
    if path == "movement.track_degrees":
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("movement.track_degrees must be numeric when present.")
        return f"{format(value, '.15g')}°"
    if path in _QUALITY_LIST_PATHS:
        if not isinstance(value, (tuple, list)):
            raise TypeError(f"{path} must contain canonical field paths.")
        if not value:
            return OMISSION_MARKER
        return ", ".join(label_for_path(item) for item in value)
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (str, int, float)):
        return str(value)
    raise TypeError(f"Unsupported canonical display value type: {type(value).__name__}.")


def display_value(event: CanonicalEvent, path: str) -> str | None:
    """Format one selected canonical field according to its state and type.

    A value is presented as stored (with only time/degree display formatting),
    unresolved and unavailable states receive fixed human-readable labels, and
    absent fields return ``None`` for omission. Canonical paths and state names
    are never returned as user-facing text.
    """
    if not isinstance(event, CanonicalEvent):
        raise TypeError("event must be CanonicalEvent.")
    label_for_path(path)
    state = _path_state(event, path)
    if state == "unresolved":
        return UNRESOLVED_DISPLAY
    if state == "not_available":
        return NOT_AVAILABLE_DISPLAY
    if state == "absent":
        return OMISSION_MARKER
    if state != "value":
        raise ValueError(f"Unsupported canonical field state: {state!r}.")
    value = _path_value(event, path)
    if value is None:
        raise ValueError("A canonical field in the value state must contain a value.")
    return _format_value(path, value)


def compose_flight_identifiers(primary: str | None, alternate: str | None) -> str | None:
    """Join available primary and alternate flight identifiers consistently."""
    if primary and alternate:
        return f"{primary} / {alternate}"
    return primary or alternate


def compose_route_endpoints(
    origin: str | None,
    destination: str | None,
) -> tuple[tuple[str, str], ...]:
    """Format a route pair, preserving a lone endpoint's role when needed."""
    if origin and destination:
        return (("Route", f"{origin} → {destination}"),)
    if origin:
        return (("Origin", origin),)
    if destination:
        return (("Destination", destination),)
    return ()


def compose_aircraft_identity(aircraft_type: str | None, code: str | None) -> str | None:
    """Combine aircraft type and code without inventing either value."""
    if aircraft_type and code:
        return f"{aircraft_type} ({code})"
    return aircraft_type or code
