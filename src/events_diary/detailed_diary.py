"""Render richer, user-facing Detailed Diary text from a canonical session.

The renderer reads only ``CanonicalSession`` fields selected by
``DETAILED_DIARY_PROFILE`` and uses shared display/composition helpers. It keeps
deterministic TRACK information distinct from visual/map observations, displays
human-readable uncertainty, and preserves visual text without inference or
Phase 6 dependencies.
"""

from __future__ import annotations

from events_diary.canonical_events import (
    DETAILED_DIARY_PROFILE,
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


def _display(event: CanonicalEvent, path: str) -> str | None:
    """Return a state-aware value for a field selected by the Detailed profile."""
    if path not in DETAILED_DIARY_PROFILE.field_paths:
        raise ValueError("Detailed renderer requested a field outside its profile.")
    return display_value(event, path)


def _core_groups(event: CanonicalEvent) -> list[list[str]]:
    """Build the shared operational facts with Detailed missing-state display."""
    groups: list[list[str]] = []
    identity = []

    primary = _display(event, "flight.primary_id")
    alternate = _display(event, "flight.alternate_id")
    flight = compose_flight_identifiers(primary, alternate)
    if flight:
        identity.append(f"Flight: {flight}")

    airline = _display(event, "flight.airline")
    if airline:
        identity.append(f"Airline: {airline}")

    origin = _display(event, "route.origin_airport_code")
    destination = _display(event, "route.destination_airport_code")
    route = compose_route_endpoints(origin, destination)
    if route:
        identity.extend(f"{label}: {value}" for label, value in route)
    if identity:
        groups.append(identity)

    times = []
    for path, label in (
        ("route.scheduled_departure", "Scheduled departure"),
        ("route.actual_departure", "Actual departure"),
        ("route.scheduled_arrival", "Scheduled arrival"),
        ("route.estimated_or_actual_arrival", "Estimated/actual arrival"),
    ):
        value = _display(event, path)
        if value:
            times.append(f"{label}: {value}")
    if times:
        groups.append(times)

    aircraft_type = _display(event, "aircraft.type")
    code = _display(event, "aircraft.code")
    aircraft = compose_aircraft_identity(aircraft_type, code)
    if aircraft:
        aircraft_lines = [f"Aircraft: {aircraft}"]
    else:
        aircraft_lines = []

    registration = _display(event, "aircraft.registration")
    if registration:
        aircraft_lines.append(f"Registration: {registration}")
    if aircraft_lines:
        groups.append(aircraft_lines)
    return groups


def _movement_groups(event: CanonicalEvent) -> list[list[str]]:
    """Combine usable final TRACK fields and render visual observations separately."""
    groups: list[list[str]] = []
    movement = []
    track_state = event.field_state("movement.track_degrees")
    if track_state == "value":
        track = _display(event, "movement.track_degrees")
        direction = (
            _display(event, "movement.direction")
            if event.field_state("movement.direction") == "value"
            else None
        )
        if track:
            movement.append(f"Track: {track} — {direction}" if direction else f"Track: {track}")

    visual_direction = _display(event, "movement.visual_flight_direction")
    if visual_direction:
        movement.append(f"Visual direction: {visual_direction}")
    if movement:
        groups.append(movement)

    route_path = "movement.visual_route_description"
    departure_path = "movement.visual_departure_context"
    route_description = (
        _display(event, route_path)
        if event.field_state(route_path) == "value"
        else None
    )
    departure_context = (
        _display(event, departure_path)
        if event.field_state(departure_path) == "value"
        else None
    )
    route_lines = [
        value for value in (route_description, departure_context)
        if value is not None
    ]
    if route_lines:
        groups.append([f"Visual route:\n{chr(10).join(route_lines)}"])
    return groups


def _render_event(event: CanonicalEvent, number: int) -> str:
    """Render one Investigation event with a 1-based visible number."""
    interval = format_investigation_interval(event.start_time_s, event.end_time_s)
    groups = [[f"Event {number} — {interval}"]]
    groups.extend(_core_groups(event))
    groups.extend(_movement_groups(event))
    return "\n\n".join("\n".join(group) for group in groups)


def render_detailed_diary(session: CanonicalSession) -> str:
    """Render a canonical session as deterministic Detailed Diary text.

    Events retain session order and each receives one block. Profile-selected
    unresolved and unavailable values use shared human-readable wording;
    absent values are omitted. Visual direction, route, and departure text are
    displayed independently and source-faithfully. The renderer performs no
    inference and does not depend on Phase 6 internals.
    """
    if not isinstance(session, CanonicalSession):
        raise TypeError("session must be CanonicalSession.")
    return "\n\n".join(
        _render_event(event, number)
        for number, event in enumerate(session.events, start=1)
    )
