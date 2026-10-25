"""Tests for the renderer-facing canonical field selection contract."""

import pytest

from events_diary.canonical_events import (
    CANONICAL_RENDERER_FIELD_PATHS,
    DETAILED_DIARY_PROFILE,
    OPERATIONAL_DIARY_PROFILE,
    CanonicalAircraft,
    CanonicalDataQuality,
    CanonicalEvent,
    CanonicalFieldProfile,
    CanonicalFlight,
    CanonicalMovement,
    CanonicalRoute,
    CanonicalSession,
    CanonicalSourceVideo,
)

OPERATIONAL_PATHS = (
    "event_time_s",
    "flight.primary_id", "flight.alternate_id", "flight.airline",
    "route.origin_airport_code", "route.destination_airport_code",
    "route.scheduled_departure", "route.actual_departure",
    "route.scheduled_arrival", "route.estimated_or_actual_arrival",
    "aircraft.registration", "aircraft.code", "aircraft.type",
    "movement.direction",
)
DETAILED_ADDITIONS = (
    "investigation_id", "start_time_s", "end_time_s",
    "movement.track_degrees", "movement.visual_flight_direction",
    "movement.visual_route_description", "movement.visual_departure_context",
    "data_quality.unresolved_fields", "data_quality.not_available_fields",
)


def _event(
    event_id: int = 0,
    investigation_id: int = 0,
    event_time_s: float = 1.0,
    *,
    direction: str | None = None,
    visual_direction: str | None = None,
    quality: CanonicalDataQuality | None = None,
) -> CanonicalEvent:
    """Create a compact canonical event for profile contract tests."""
    return CanonicalEvent(
        event_id=event_id,
        investigation_id=investigation_id,
        event_time_s=event_time_s,
        start_time_s=event_time_s,
        end_time_s=event_time_s + 1,
        flight=CanonicalFlight(primary_id="TEST1"),
        route=CanonicalRoute(origin_airport_code="AAA"),
        aircraft=CanonicalAircraft(registration="N123AA"),
        movement=CanonicalMovement(
            direction=direction,
            visual_flight_direction=visual_direction,
        ),
        data_quality=quality or CanonicalDataQuality(),
    )


def test_operational_profile_contains_exactly_approved_fields_in_order():
    assert OPERATIONAL_DIARY_PROFILE.field_paths == OPERATIONAL_PATHS


def test_detailed_profile_preserves_operational_fields_then_adds_approved_fields():
    assert DETAILED_DIARY_PROFILE.field_paths == (*OPERATIONAL_PATHS, *DETAILED_ADDITIONS)
    assert OPERATIONAL_DIARY_PROFILE.field_paths == DETAILED_DIARY_PROFILE.field_paths[
        :len(OPERATIONAL_PATHS)
    ]


def test_profiles_exclude_phase6_internals_and_validate_every_path():
    paths = (*OPERATIONAL_DIARY_PROFILE.field_paths, *DETAILED_DIARY_PROFILE.field_paths)
    assert all(path in CANONICAL_RENDERER_FIELD_PATHS for path in paths)
    assert not any(
        token in path.casefold()
        for path in paths
        for token in ("ocr", "vlm", "fusion", "provider", "candidate", "evidence")
    )


def test_profile_constructor_rejects_duplicates_and_noncanonical_or_stale_paths():
    with pytest.raises(ValueError, match="duplicates"):
        CanonicalFieldProfile("duplicate", ("event_time_s", "event_time_s"))
    for invalid_path in ("ocr.tokens", "flight.stale_name", "TRACK"):
        with pytest.raises(ValueError, match="noncanonical"):
            CanonicalFieldProfile("invalid", (invalid_path,))
    with pytest.raises(TypeError, match="ordered tuple"):
        CanonicalFieldProfile("unordered", ["event_time_s"])


def test_profiles_are_immutable_and_have_deterministic_order():
    assert isinstance(OPERATIONAL_DIARY_PROFILE.field_paths, tuple)
    assert isinstance(DETAILED_DIARY_PROFILE.field_paths, tuple)
    assert len(set(OPERATIONAL_DIARY_PROFILE.field_paths)) == len(
        OPERATIONAL_DIARY_PROFILE.field_paths
    )
    assert len(set(DETAILED_DIARY_PROFILE.field_paths)) == len(
        DETAILED_DIARY_PROFILE.field_paths
    )


def test_field_state_keeps_unresolved_not_available_and_absent_distinct():
    event = _event(quality=CanonicalDataQuality(
        unresolved_fields=("movement.direction",),
        not_available_fields=("movement.visual_route_description",),
    ))
    assert event.field_state("movement.direction") == "unresolved"
    assert event.field_state("movement.visual_route_description") == "not_available"
    assert event.field_state("movement.visual_departure_context") == "absent"


def test_track_direction_and_visual_direction_are_independent():
    event = _event(
        visual_direction="northwest",
        quality=CanonicalDataQuality(unresolved_fields=("movement.direction",)),
    )
    assert event.movement.direction is None
    assert event.field_state("movement.direction") == "unresolved"
    assert event.movement.visual_flight_direction == "northwest"
    assert event.field_state("movement.visual_flight_direction") == "value"


def test_session_is_renderer_ready_and_stays_chronologically_ordered():
    later = _event(event_id=1, investigation_id=1, event_time_s=4.9)
    earlier = _event(event_id=0, investigation_id=0, event_time_s=1.6)
    session = CanonicalSession(
        CanonicalSourceVideo("sample.mp4", 8.1), (later, earlier),
    )
    payload = session.to_dict()
    assert session.schema_version == "1.0"
    assert session.source_video.name == "sample.mp4"
    assert session.source_video.duration_s == 8.1
    assert session.events_count == 2
    assert [(item.event_id, item.investigation_id) for item in session.events] == [(0, 0), (1, 1)]
    assert [item["event_time_s"] for item in payload["events"]] == [1.6, 4.9]


def test_profile_metadata_does_not_change_canonical_serialization():
    event = _event()
    session = CanonicalSession(CanonicalSourceVideo("sample.mp4", 2.0), (event,))
    event_payload = event.to_dict()
    session_payload = session.to_dict()
    assert "field_state" not in event_payload
    assert "profiles" not in event_payload
    assert "field_profiles" not in session_payload
    assert "N/A" not in session.to_json()
    assert "Unresolved" not in session.to_json()
    assert "Could not determine" not in session.to_json()
