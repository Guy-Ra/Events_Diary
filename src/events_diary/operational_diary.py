"""Render a concise Operational Diary from a canonical session.

This renderer presents only fields selected by ``OPERATIONAL_DIARY_PROFILE``.
It uses the shared display helpers for field states and video-relative time
formatting, preserves CanonicalSession order, and performs no inference or
Phase 6 processing.
"""

from __future__ import annotations

from events_diary.canonical_events import (
    OPERATIONAL_DIARY_PROFILE,
    CanonicalEvent,
    CanonicalSession,
)
from events_diary.renderer_core import (
    compose_aircraft_identity,
    compose_flight_identifiers,
    compose_route_endpoints,
    display_value,
    format_investigation_interval,
)


def _usable_value(event: CanonicalEvent, path: str) -> str | None:
    """Return a profile-selected resolved value, omitting other field states."""
    if path not in OPERATIONAL_DIARY_PROFILE.field_paths:
        raise ValueError("Operational renderer requested a field outside its profile.")
    if event.field_state(path) != "value":
        return None
    value = display_value(event, path)
    if value is None or not value.strip():
        return None
    return value


def _flight_line(event: CanonicalEvent) -> str | None:
    """Compose available primary and alternate identifiers into one flight line."""
    primary = _usable_value(event, "flight.primary_id")
    alternate = _usable_value(event, "flight.alternate_id")
    value = compose_flight_identifiers(primary, alternate)
    return f"Flight: {value}" if value else None


def _route_lines(event: CanonicalEvent) -> list[str]:
    """Compose a complete route or label a single available endpoint plainly."""
    origin = _usable_value(event, "route.origin_airport_code")
    destination = _usable_value(event, "route.destination_airport_code")
    return [f"{label}: {value}" for label, value in compose_route_endpoints(
        origin, destination,
    )]


def _aircraft_line(event: CanonicalEvent) -> str | None:
    """Combine aircraft type and code when both are available."""
    aircraft_type = _usable_value(event, "aircraft.type")
    code = _usable_value(event, "aircraft.code")
    value = compose_aircraft_identity(aircraft_type, code)
    return f"Aircraft: {value}" if value else None


def _render_event(event: CanonicalEvent, number: int) -> str:
    """Render one canonical Investigation as one concise text block."""
    interval = format_investigation_interval(event.start_time_s, event.end_time_s)
    lines = [f"Event {number} — {interval}"]

    flight_line = _flight_line(event)
    if flight_line:
        lines.append(flight_line)

    airline = _usable_value(event, "flight.airline")
    if airline:
        lines.append(f"Airline: {airline}")

    lines.extend(_route_lines(event))

    for path, label in (
        ("route.scheduled_departure", "Scheduled departure"),
        ("route.actual_departure", "Actual departure"),
        ("route.scheduled_arrival", "Scheduled arrival"),
        ("route.estimated_or_actual_arrival", "Estimated/actual arrival"),
    ):
        value = _usable_value(event, path)
        if value:
            lines.append(f"{label}: {value}")

    aircraft_line = _aircraft_line(event)
    if aircraft_line:
        lines.append(aircraft_line)

    registration = _usable_value(event, "aircraft.registration")
    if registration:
        lines.append(f"Registration: {registration}")

    direction = _usable_value(event, "movement.direction")
    if direction:
        lines.append(f"Direction: {direction}")

    return "\n".join(lines)


def render_operational_diary(session: CanonicalSession) -> str:
    """Render a canonical session as deterministic Operational Diary text.

    Each event produces one block in the authoritative session order. Event
    headers use the Investigation interval, and unresolved, unavailable, and
    absent profile fields are omitted to keep this view concise. The function
    does not infer values or depend on Phase 6 internals.
    """
    if not isinstance(session, CanonicalSession):
        raise TypeError("session must be CanonicalSession.")
    blocks = [
        _render_event(event, number)
        for number, event in enumerate(session.events, start=1)
    ]
    return "\n\n".join(blocks)
