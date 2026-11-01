"""Focused tests for the Phase 8.3 Detailed Diary renderer."""

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
from events_diary.detailed_diary import render_detailed_diary


def _event(
    event_id: int = 0,
    *,
    start: float = 1.6,
    end: float = 4.9,
    track: float | None = None,
    direction: str | None = None,
    visual_direction: str | None = "northwest",
    visual_route: str | None = "The aircraft is tracking northwest over open water.",
    departure_context: str | None = "Departed Beirut and proceeded northwest.",
    unresolved: tuple[str, ...] = ("movement.track_degrees", "movement.direction"),
    unavailable: tuple[str, ...] = (),
) -> CanonicalEvent:
    """Create a canonical event with configurable movement field states."""
    return CanonicalEvent(
        event_id=event_id,
        investigation_id=event_id,
        event_time_s=start,
        start_time_s=start,
        end_time_s=end,
        flight=CanonicalFlight("MEA251", "ME251", "MEA"),
        route=CanonicalRoute("BEY", "ATH", "07:35", "07:44", "09:35", "09:18"),
        aircraft=CanonicalAircraft("T7-ME2", "A321", "Airbus A321-271NX"),
        movement=CanonicalMovement(
            track_degrees=track,
            direction=direction,
            visual_flight_direction=visual_direction,
            visual_route_description=visual_route,
            visual_departure_context=departure_context,
        ),
        data_quality=CanonicalDataQuality(unresolved, unavailable),
    )


def _session(*events: CanonicalEvent) -> CanonicalSession:
    """Wrap events in a renderer-ready canonical session."""
    return CanonicalSession(CanonicalSourceVideo("test.mp4", 20.0), events)


def test_one_event_renders_one_block_with_one_based_header_and_interval():
    output = render_detailed_diary(_session(_event()))
    assert output.startswith("Event 1 — 00:01.6–00:04.9\n")
    assert output.count("Event ") == 1
    assert "\n\nEvent " not in output


def test_multiple_events_preserve_session_order_and_use_one_based_numbers():
    first = _event(event_id=8, start=1.0, end=2.0)
    second = _event(event_id=3, start=4.0, end=5.0)
    output = render_detailed_diary(_session(first, second))
    assert output.index("Event 1 — 00:01.0–00:02.0") < output.index(
        "Event 2 — 00:04.0–00:05.0"
    )


def test_core_operational_facts_are_included():
    output = render_detailed_diary(_session(_event()))
    for expected in (
        "Flight: MEA251 / ME251", "Airline: MEA", "Route: BEY → ATH",
        "Scheduled departure: 07:35", "Actual departure: 07:44",
        "Scheduled arrival: 09:35", "Estimated/actual arrival: 09:18",
        "Aircraft: Airbus A321-271NX (A321)", "Registration: T7-ME2",
    ):
        assert expected in output


def test_resolved_track_uses_shared_degree_formatting():
    event = _event(track=293.0, direction="West-Northwest", unresolved=())
    assert "Track: 293° — West-Northwest" in render_detailed_diary(_session(event))


def test_resolved_track_with_absent_direction_shows_track_without_inference():
    event = _event(track=293.0, direction=None, unresolved=())
    output = render_detailed_diary(_session(event))
    assert "Track: 293°" in output
    assert "Track: 293° —" not in output
    assert "Direction:" not in output


def test_unresolved_track_and_resolved_direction_are_both_omitted():
    event = _event(track=None, direction="West-Northwest", unresolved=("movement.track_degrees",))
    output = render_detailed_diary(_session(event))
    assert "Track:" not in output
    assert "Direction:" not in output


@pytest.mark.parametrize("state", ("not_available", "absent"))
def test_not_available_or_absent_track_is_omitted(state):
    if state == "not_available":
        event = _event(
            track=None,
            unresolved=("movement.direction",),
            unavailable=("movement.track_degrees",),
        )
    else:
        event = _event(track=None, unresolved=("movement.direction",))
    output = render_detailed_diary(_session(event))
    assert "Track:" not in output


def test_direction_is_combined_only_with_resolved_track():
    event = _event(track=90.0, direction="Eastbound", unresolved=())
    assert "Track: 90° — Eastbound" in render_detailed_diary(_session(event))


def test_visual_direction_remains_independent_from_unresolved_direction():
    output = render_detailed_diary(_session(_event()))
    assert "Track:" not in output
    assert "Direction:" not in output
    assert "Visual direction: northwest" in output


@pytest.mark.parametrize("visual_direction", ("northwest", "Heading north"))
def test_visual_direction_is_preserved_source_faithfully(visual_direction):
    event = _event(visual_direction=visual_direction)
    output = render_detailed_diary(_session(event))
    assert f"Visual direction: {visual_direction}" in output


def test_visual_route_text_is_preserved_source_faithfully():
    visual_route = "  Northbound claim; preserve punctuation and spacing.  "
    event = _event(visual_route=visual_route)
    output = render_detailed_diary(_session(event))
    assert f"Visual route:\n{visual_route}\nDeparted Beirut" in output
    assert output.count("Visual route:") == 1
    assert "Departure context:" not in output


def test_departure_context_is_preserved_source_faithfully():
    context = "Departed westbound (as displayed); no rewriting."
    event = _event(departure_context=context)
    output = render_detailed_diary(_session(event))
    assert f"Visual route:\nThe aircraft is tracking northwest over open water.\n{context}" in output
    assert "Departure context:" not in output


@pytest.mark.parametrize(
    ("path", "kwargs", "label"),
    (("movement.visual_flight_direction", {"visual_direction": None}, "Visual direction"),
     ("movement.visual_route_description", {"visual_route": None}, "Visual route"),
     ("movement.visual_departure_context", {"departure_context": None}, "Visual route")),
)
def test_unresolved_visual_context_does_not_replace_text_with_state_names(path, kwargs, label):
    event = _event(**kwargs, unresolved=(path, "movement.track_degrees", "movement.direction"))
    output = render_detailed_diary(_session(event))
    if label == "Visual direction":
        assert f"{label}: Could not determine" in output
    else:
        assert "Visual route:\n" in output
        assert "Could not determine" not in output
        if path == "movement.visual_route_description":
            assert "Departed Beirut and proceeded northwest." in output
        else:
            assert "The aircraft is tracking northwest over open water." in output


@pytest.mark.parametrize(
    ("path", "kwargs", "label"),
    (("movement.visual_flight_direction", {"visual_direction": None}, "Visual direction"),
     ("movement.visual_route_description", {"visual_route": None}, "Visual route"),
     ("movement.visual_departure_context", {"departure_context": None}, "Visual route")),
)
def test_unavailable_visual_context_does_not_expose_state_or_path(path, kwargs, label):
    event = _event(
        **kwargs,
        unresolved=("movement.track_degrees", "movement.direction"),
        unavailable=(path,),
    )
    output = render_detailed_diary(_session(event))
    if label == "Visual direction":
        assert f"{label}: N/A" in output
    else:
        assert "Visual route:\n" in output
        assert "N/A" not in output
        if path == "movement.visual_route_description":
            assert "Departed Beirut and proceeded northwest." in output
        else:
            assert "The aircraft is tracking northwest over open water." in output


def test_absent_visual_context_is_omitted_and_one_available_value_is_shown():
    event = _event(
        visual_route=None,
        departure_context=None,
        unresolved=("movement.track_degrees", "movement.direction"),
    )
    output = render_detailed_diary(_session(event))
    assert "Visual route:" not in output
    assert "Departure context:" not in output

    route_only = _event(departure_context=None, unresolved=(
        "movement.track_degrees", "movement.direction", "movement.visual_departure_context",
    ))
    assert "Visual route:\nThe aircraft is tracking northwest over open water." in (
        render_detailed_diary(_session(route_only))
    )

    context_only = _event(visual_route=None, unresolved=(
        "movement.track_degrees", "movement.direction", "movement.visual_route_description",
    ))
    assert "Visual route:\nDeparted Beirut and proceeded northwest." in (
        render_detailed_diary(_session(context_only))
    )


@pytest.mark.parametrize("quality_state", ("unresolved", "not_available"))
def test_visual_route_section_is_omitted_when_both_text_values_are_unusable(quality_state):
    paths = (
        "movement.visual_route_description", "movement.visual_departure_context",
    )
    quality = {quality_state: paths}
    event = _event(
        visual_route=None,
        departure_context=None,
        unresolved=("movement.track_degrees", "movement.direction", *quality.get("unresolved", ())),
        unavailable=quality.get("not_available", ()),
    )
    output = render_detailed_diary(_session(event))
    assert "Visual route:" not in output
    assert "visual_route_description" not in output
    assert "visual_departure_context" not in output


def test_detailed_output_hides_canonical_paths_ids_and_quality_lists():
    output = render_detailed_diary(_session(_event()))
    for internal in (
        "movement.track_degrees", "movement.direction", "movement.visual_",
        "investigation_id", "data_quality", "unresolved_fields",
        "not_available_fields",
    ):
        assert internal not in output


def test_detailed_renderer_import_does_not_load_phase6_modules():
    code = """
import sys
import events_diary.detailed_diary
forbidden = {
    'events_diary.investigation_record', 'events_diary.fusion',
    'events_diary.ocr_fields', 'events_diary.vlm_review',
}
assert forbidden.isdisjoint(sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def _real_session() -> CanonicalSession:
    """Load the saved Phase 7.4 session using canonical contract classes."""
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "data/debug/canonical_events/phase_7_4/canonical_session.json")
                         .read_text(encoding="utf-8"))
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
    source = payload["source_video"]
    return CanonicalSession(CanonicalSourceVideo(source["name"], source["duration_s"]), events)


def test_real_session_has_both_events_and_expected_independent_movement_values():
    output = render_detailed_diary(_real_session())
    for expected in (
        "Event 1 — 00:01.6–00:04.9", "Flight: MEA251 / ME251", "Airline: MEA",
        "Route: BEY → ATH", "Scheduled departure: 07:35", "Actual departure: 07:44",
        "Scheduled arrival: 09:35", "Estimated/actual arrival: 09:18",
        "Aircraft: Airbus A321-271NX (A321)", "Registration: T7-ME2",
        "Visual direction: northwest", "The aircraft is tracking northwest over the Mediterranean Sea",
        "Departed from Beirut on the coast of Lebanon, proceeding northwest over open water.",
        "Event 2 — 00:04.9–00:08.1", "Flight: ISR889 / 6H889",
        "Airline: Israir Airlines", "Route: TLV → BUS", "Scheduled departure: 09:10",
        "Actual departure: 09:40", "Scheduled arrival: 11:50",
        "Estimated/actual arrival: 11:59", "Aircraft: ATR 72-600 (AT76)",
        "Registration: 4X-ATI", "Visual direction: Heading north",
        "The aircraft is tracking northward over the eastern Mediterranean Sea",
        "Departed Tel Aviv and tracked north over the sea.",
    ):
        assert expected in output
    assert output.count("Event ") == 2
    assert output.count("Visual route:") == 2
    assert "Track:" not in output
    assert "Direction:" not in output
    assert "Departure context:" not in output
