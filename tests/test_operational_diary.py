"""Focused tests for the concise Phase 8.2 Operational Diary renderer."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from events_diary.canonical_events import (
    CanonicalAircraft,
    CanonicalDataQuality,
    CanonicalEvent,
    CanonicalFlight,
    CanonicalMovement,
    CanonicalRoute,
    CanonicalSession,
    CanonicalSourceVideo,
)
from events_diary.operational_diary import render_operational_diary


def _event(
    event_id: int = 0,
    *,
    start: float = 1.6,
    end: float = 4.9,
    primary: str | None = "MEA251",
    alternate: str | None = "ME251",
    airline: str | None = "MEA",
    origin: str | None = "BEY",
    destination: str | None = "ATH",
    scheduled_departure: str | None = "07:35",
    actual_departure: str | None = "07:44",
    scheduled_arrival: str | None = "09:35",
    estimated_arrival: str | None = "09:18",
    aircraft_type: str | None = "Airbus A321-271NX",
    aircraft_code: str | None = "A321",
    registration: str | None = "T7-ME2",
    direction: str | None = None,
    unresolved: tuple[str, ...] = ("movement.direction",),
    unavailable: tuple[str, ...] = (),
) -> CanonicalEvent:
    """Build a canonical event with independently configurable display fields."""
    return CanonicalEvent(
        event_id=event_id,
        investigation_id=event_id,
        event_time_s=start,
        start_time_s=start,
        end_time_s=end,
        flight=CanonicalFlight(primary, alternate, airline),
        route=CanonicalRoute(
            origin, destination, scheduled_departure, actual_departure,
            scheduled_arrival, estimated_arrival,
        ),
        aircraft=CanonicalAircraft(registration, aircraft_code, aircraft_type),
        movement=CanonicalMovement(direction=direction),
        data_quality=CanonicalDataQuality(unresolved, unavailable),
    )


def _session(*events: CanonicalEvent) -> CanonicalSession:
    """Wrap events in a small canonical session for renderer tests."""
    return CanonicalSession(CanonicalSourceVideo("test.mp4", 20.0), events)


def test_one_event_renders_one_block_with_a_one_based_header():
    output = render_operational_diary(_session(_event()))
    assert output.startswith("Event 1 — 00:01.6–00:04.9\n")
    assert output.count("\n\n") == 0
    assert output.count("Event ") == 1


def test_multiple_events_preserve_session_order_and_number_from_one():
    first = _event(event_id=10, start=1.0, end=2.0, primary="FIRST")
    second = _event(event_id=11, start=8.0, end=9.0, primary="SECOND")
    output = render_operational_diary(_session(first, second))
    assert output.index("Flight: FIRST") < output.index("Flight: SECOND")
    assert output.startswith("Event 1 — 00:01.0–00:02.0")
    assert "\n\nEvent 2 — 00:08.0–00:09.0" in output


def test_header_uses_investigation_interval_and_deterministic_timestamps():
    event = _event(start=65.25, end=70.04)
    assert render_operational_diary(_session(event)).splitlines()[0] == (
        "Event 1 — 01:05.3–01:10.0"
    )


@pytest.mark.parametrize(
    ("primary", "alternate", "expected"),
    (("MEA251", "ME251", "Flight: MEA251 / ME251"),
     ("MEA251", None, "Flight: MEA251"),
     (None, "ME251", "Flight: ME251")),
)
def test_flight_identifiers_are_composed_conservatively(primary, alternate, expected):
    output = render_operational_diary(_session(_event(primary=primary, alternate=alternate)))
    assert expected in output
    assert output.count("Flight:") == 1


def test_flight_line_is_omitted_when_both_identifiers_are_absent():
    output = render_operational_diary(_session(_event(primary=None, alternate=None)))
    assert "Flight:" not in output


def test_route_endpoints_are_composed_or_shown_without_inventing_the_other_side():
    complete = render_operational_diary(_session(_event()))
    origin_only = render_operational_diary(_session(_event(destination=None)))
    destination_only = render_operational_diary(_session(_event(origin=None)))
    assert "Route: BEY → ATH" in complete
    assert "Origin: BEY" in origin_only
    assert "ATH" not in origin_only
    assert "Destination: ATH" in destination_only
    assert "BEY" not in destination_only


@pytest.mark.parametrize(
    ("aircraft_type", "aircraft_code", "expected"),
    (("Airbus A321-271NX", "A321", "Aircraft: Airbus A321-271NX (A321)"),
     ("Airbus A321-271NX", None, "Aircraft: Airbus A321-271NX"),
     (None, "A321", "Aircraft: A321")),
)
def test_aircraft_type_and_code_are_composed_when_available(
    aircraft_type, aircraft_code, expected,
):
    output = render_operational_diary(_session(_event(
        aircraft_type=aircraft_type, aircraft_code=aircraft_code,
    )))
    assert expected in output


@pytest.mark.parametrize(
    ("path", "kwargs"),
    (("flight.airline", {"airline": None}),
     ("route.scheduled_departure", {"scheduled_departure": None}),
     ("aircraft.registration", {"registration": None})),
)
def test_unresolved_profile_fields_are_omitted(path, kwargs):
    event = _event(**kwargs, unresolved=(path,))
    output = render_operational_diary(_session(event))
    assert {
        "flight.airline": "Airline:",
        "route.scheduled_departure": "Scheduled departure:",
        "aircraft.registration": "Registration:",
    }[path] not in output


def test_not_available_and_absent_profile_fields_are_omitted():
    unavailable = _event(
        airline=None,
        unresolved=(),
        unavailable=("flight.airline",),
    )
    absent = _event(airline=None, unresolved=(), unavailable=())
    assert "Airline:" not in render_operational_diary(_session(unavailable))
    assert "Airline:" not in render_operational_diary(_session(absent))


def test_resolved_direction_is_displayed_but_unresolved_direction_is_omitted():
    resolved = _event(direction="Eastbound", unresolved=())
    unresolved = _event(direction=None, unresolved=("movement.direction",))
    assert "Direction: Eastbound" in render_operational_diary(_session(resolved))
    assert "Direction:" not in render_operational_diary(_session(unresolved))
    assert "Could not determine" not in render_operational_diary(_session(unresolved))


def test_operational_output_hides_canonical_paths_and_quality_internals():
    output = render_operational_diary(_session(_event()))
    for internal in (
        "flight.primary_id", "movement.direction", "data_quality",
        "unresolved_fields", "not_available_fields", "investigation_id",
    ):
        assert internal not in output


def test_operational_renderer_import_does_not_load_phase6_modules():
    code = """
import sys
import events_diary.operational_diary
forbidden = {
    'events_diary.investigation_record', 'events_diary.fusion',
    'events_diary.ocr_fields', 'events_diary.vlm_review',
}
assert forbidden.isdisjoint(sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def _real_session() -> CanonicalSession:
    """Load the saved Phase 7.4 artifact using only canonical data contracts."""
    path = Path("data/debug/canonical_events/phase_7_4/canonical_session.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    events = tuple(
        CanonicalEvent(
            event_id=item["event_id"],
            investigation_id=item["investigation_id"],
            event_time_s=item["event_time_s"],
            start_time_s=item["start_time_s"],
            end_time_s=item["end_time_s"],
            flight=CanonicalFlight(**item["flight"]),
            route=CanonicalRoute(**item["route"]),
            aircraft=CanonicalAircraft(**item["aircraft"]),
            movement=CanonicalMovement(**item["movement"]),
            data_quality=CanonicalDataQuality(
                unresolved_fields=tuple(item["data_quality"]["unresolved_fields"]),
                not_available_fields=tuple(item["data_quality"]["not_available_fields"]),
            ),
        )
        for item in payload["events"]
    )
    return CanonicalSession(
        CanonicalSourceVideo(
            payload["source_video"]["name"], payload["source_video"]["duration_s"],
        ),
        events,
    )


def test_real_phase_7_4_session_renders_both_expected_investigations():
    output = render_operational_diary(_real_session())
    expected_fragments = (
        "Event 1 — 00:01.6–00:04.9", "Flight: MEA251 / ME251", "Airline: MEA",
        "Route: BEY → ATH", "Scheduled departure: 07:35", "Actual departure: 07:44",
        "Scheduled arrival: 09:35", "Estimated/actual arrival: 09:18",
        "Aircraft: Airbus A321-271NX (A321)", "Registration: T7-ME2",
        "Event 2 — 00:04.9–00:08.1", "Flight: ISR889 / 6H889",
        "Airline: Israir Airlines", "Route: TLV → BUS", "Scheduled departure: 09:10",
        "Actual departure: 09:40", "Scheduled arrival: 11:50",
        "Estimated/actual arrival: 11:59", "Aircraft: ATR 72-600 (AT76)",
        "Registration: 4X-ATI",
    )
    assert all(fragment in output for fragment in expected_fragments)
    assert output.count("Direction:") == 0
