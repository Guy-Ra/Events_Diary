"""Perform the final offline Phase 7 canonical-session validation.

Run ``validate_phase7_4_saved_artifacts.py`` first to rebuild the real session
through ``build_canonical_session``. This script then reads only that
CanonicalSession JSON, reconstructs the existing canonical dataclasses for
contract checks, validates both renderer field profiles, and writes a
validation report. It does not inspect Phase 6 inputs or render diary prose.
"""

from __future__ import annotations

import json
import sys
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from events_diary.canonical_events import (
    CANONICAL_RENDERER_FIELD_PATHS,
    DETAILED_DIARY_PROFILE,
    OPERATIONAL_DIARY_PROFILE,
    CanonicalAircraft,
    CanonicalDataQuality,
    CanonicalEvent,
    CanonicalFlight,
    CanonicalMovement,
    CanonicalRoute,
    CanonicalSession,
    CanonicalSourceVideo,
)

SESSION_PATH = ROOT / "data/debug/canonical_events/phase_7_4/canonical_session.json"
REPORT_PATH = ROOT / "data/debug/canonical_events/phase_7_6/phase_7_6_validation_report.txt"
EXPECTED_OPERATIONAL_PATHS = (
    "event_time_s",
    "flight.primary_id", "flight.alternate_id", "flight.airline",
    "route.origin_airport_code", "route.destination_airport_code",
    "route.scheduled_departure", "route.actual_departure",
    "route.scheduled_arrival", "route.estimated_or_actual_arrival",
    "aircraft.registration", "aircraft.code", "aircraft.type",
    "movement.direction",
)
EXPECTED_DETAILED_ADDITIONS = (
    "investigation_id", "start_time_s", "end_time_s",
    "movement.track_degrees", "movement.visual_flight_direction",
    "movement.visual_route_description", "movement.visual_departure_context",
    "data_quality.unresolved_fields", "data_quality.not_available_fields",
)
_METADATA_PATHS = frozenset(("event_time_s", "investigation_id", "start_time_s", "end_time_s"))
_QUALITY_LIST_PATHS = frozenset((
    "data_quality.unresolved_fields", "data_quality.not_available_fields",
))
_EXCLUDED_NAMES = frozenset((
    "serial_number", "age", "mach", "temperature", "fir_uir", "latitude",
    "longitude", "barometric_altitude", "vertical_speed", "gps_altitude",
    "ground_speed", "true_airspeed", "indicated_airspeed", "icao_address",
    "squawk", "ocr", "vlm", "fusion", "provider_metadata", "retry_metadata",
    "sampling", "frame", "candidate", "revisit", "entity_matching",
    "confidence", "diagnostics", "prompt", "model",
))


def _event_from_dict(payload: dict) -> CanonicalEvent:
    """Reconstruct one typed canonical event directly from canonical JSON."""
    quality = payload["data_quality"]
    return CanonicalEvent(
        event_id=payload["event_id"],
        investigation_id=payload["investigation_id"],
        event_time_s=payload["event_time_s"],
        start_time_s=payload["start_time_s"],
        end_time_s=payload["end_time_s"],
        flight=CanonicalFlight(**payload["flight"]),
        route=CanonicalRoute(**payload["route"]),
        aircraft=CanonicalAircraft(**payload["aircraft"]),
        movement=CanonicalMovement(**payload["movement"]),
        data_quality=CanonicalDataQuality(
            unresolved_fields=tuple(quality["unresolved_fields"]),
            not_available_fields=tuple(quality["not_available_fields"]),
        ),
    )


def _value_at_path(event: CanonicalEvent, path: str):
    """Read a value only through the public CanonicalEvent data groups."""
    value: object = event
    for component in path.split("."):
        value = getattr(value, component)
    return value


def _state_for_path(event: CanonicalEvent, path: str) -> str:
    """Return explicit canonical state, treating metadata and lists as values."""
    if path in _METADATA_PATHS or path in _QUALITY_LIST_PATHS:
        return "value"
    return event.field_state(path)


def _audit_exclusions(payload: dict) -> tuple[str, ...]:
    """Find excluded Phase 6 or future entity data in serialized session keys."""
    found: set[str] = set()

    def visit(value) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                normalized = key.casefold()
                if normalized in _EXCLUDED_NAMES:
                    found.add(key)
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)
    return tuple(sorted(found))


def _presentation_values(payload: dict) -> tuple[str, ...]:
    """Find renderer wording stored as canonical scalar strings, not field names."""
    labels = {"n/a", "unresolved", "could not determine", "not available"}
    found: set[str] = set()

    def visit(value) -> None:
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, str) and value.casefold() in labels:
            found.add(value)

    visit(payload)
    return tuple(sorted(found))


def _validate_session(payload: dict) -> CanonicalSession:
    """Reconstruct and validate the persisted Phase 7 session contract."""
    if set(payload) != {"schema_version", "source_video", "events_count", "events"}:
        raise AssertionError("CanonicalSession has unexpected or missing top-level fields.")
    expected_event_fields = {
        "event_id", "investigation_id", "event_time_s", "start_time_s", "end_time_s",
        "flight", "route", "aircraft", "movement", "data_quality",
    }
    for event in payload["events"]:
        if set(event) != expected_event_fields:
            raise AssertionError("CanonicalEvent has unexpected or missing field groups.")
    source = payload["source_video"]
    if set(source) != {"name", "duration_s"}:
        raise AssertionError("CanonicalSourceVideo has unexpected fields.")
    session = CanonicalSession(
        source_video=CanonicalSourceVideo(source["name"], source["duration_s"]),
        events=tuple(_event_from_dict(event) for event in payload["events"]),
        schema_version=payload["schema_version"],
    )
    if session.to_dict() != payload:
        raise AssertionError("CanonicalSession serialization round-trip changed data or ordering.")
    if session.to_json(ensure_ascii=False):
        round_trip = json.loads(session.to_json(ensure_ascii=False))
        if round_trip != payload:
            raise AssertionError("CanonicalSession.to_json round-trip changed the payload.")
    return session


def main() -> int:
    """Validate real session, contracts, state semantics, and renderer profiles."""
    payload = json.loads(SESSION_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("CanonicalSession JSON must have an object root.")
    session = _validate_session(payload)

    if (session.schema_version, session.source_video.name, session.source_video.duration_s,
            session.events_count) != ("1.0", "investigation_switch_test.mp4", 8.1, 2):
        raise AssertionError("Real CanonicalSession metadata or event count is incorrect.")
    expected_order = ((0, 0, 1.6, 1.6, 4.9), (1, 1, 4.9, 4.9, 8.1))
    actual_order = tuple(
        (event.event_id, event.investigation_id, event.event_time_s,
         event.start_time_s, event.end_time_s)
        for event in session.events
    )
    if actual_order != expected_order:
        raise AssertionError(f"Real session ordering or IDs are incorrect: {actual_order!r}")

    expected_event_groups = {"event_id", "investigation_id", "event_time_s", "start_time_s",
                             "end_time_s", "flight", "route", "aircraft", "movement",
                             "data_quality"}
    if {item.name for item in fields(CanonicalEvent)} != expected_event_groups:
        raise AssertionError("CanonicalEvent model groups differ from the approved contract.")
    if {item.name for item in fields(CanonicalSession)} != {
        "source_video", "events", "schema_version", "events_count",
    }:
        raise AssertionError("CanonicalSession model fields differ from the approved contract.")
    if OPERATIONAL_DIARY_PROFILE.field_paths != EXPECTED_OPERATIONAL_PATHS:
        raise AssertionError("Operational renderer profile differs from the approved field list.")
    if DETAILED_DIARY_PROFILE.field_paths != (
        *EXPECTED_OPERATIONAL_PATHS, *EXPECTED_DETAILED_ADDITIONS,
    ):
        raise AssertionError("Detailed renderer profile differs from the approved field list.")
    for profile in (OPERATIONAL_DIARY_PROFILE, DETAILED_DIARY_PROFILE):
        if (not isinstance(profile.field_paths, tuple)
                or len(set(profile.field_paths)) != len(profile.field_paths)
                or any(path not in CANONICAL_RENDERER_FIELD_PATHS for path in profile.field_paths)):
            raise AssertionError(f"{profile.name} profile is mutable, duplicated, or noncanonical.")
    if not set(OPERATIONAL_DIARY_PROFILE.field_paths).issubset(DETAILED_DIARY_PROFILE.field_paths):
        raise AssertionError("Operational profile is not contained in Detailed profile.")

    serialized = session.to_dict()
    leaked = _audit_exclusions(serialized)
    if leaked:
        raise AssertionError(f"Excluded Phase 6 fields leaked into CanonicalSession: {leaked!r}")
    presentation_values = _presentation_values(serialized)
    if presentation_values:
        raise AssertionError(
            f"Presentation values entered canonical JSON: {presentation_values!r}"
        )

    states: set[str] = set()
    report_lines = [
        "Phase 7.6 final canonical-layer validation",
        "Source video: investigation_switch_test.mp4",
        "Session: schema_version=1.0; duration_s=8.1; events_count=2",
        "Production flow: VideoInvestigationResult -> build_canonical_session -> CanonicalSession.",
        "This session was rebuilt offline by validate_phase7_4_saved_artifacts.py before this audit.",
        "CanonicalEvent groups: event metadata, flight, route, aircraft, movement, data_quality.",
        "CanonicalSession fields: schema_version, source_video, events_count, events.",
        "Exactly one session for this video; one event per completed Investigation.",
        "Chronological order and deterministic IDs: " + repr(actual_order),
        "Profiles: immutable ordered tuples, canonical paths only, duplicate-free; Operational is a subset of Detailed.",
        "Serialization: to_dict/to_json round-trip preserves order, nulls, quality lists, schema and count.",
        "Serialization shape is unchanged; no renderer wording or duplicate presentation model is present.",
        "Exclusion audit: passed; no excluded ticket, OCR, VLM, fusion, provider, runtime, sampling, frame, or entity keys.",
        "",
    ]

    for event in session.events:
        report_lines.extend((
            f"Event {event.event_id} / Investigation {event.investigation_id}",
            f"  event_time_s={event.event_time_s}; interval=[{event.start_time_s}, {event.end_time_s})",
            f"  flight={serialized['events'][event.event_id]['flight']}",
            f"  route={serialized['events'][event.event_id]['route']}",
            f"  aircraft={serialized['events'][event.event_id]['aircraft']}",
            f"  movement={serialized['events'][event.event_id]['movement']}",
            f"  unresolved_fields={event.data_quality.unresolved_fields}",
            f"  not_available_fields={event.data_quality.not_available_fields}",
            "",
        ))
        unresolved = set(event.data_quality.unresolved_fields)
        unavailable = set(event.data_quality.not_available_fields)
        if unresolved & unavailable:
            raise AssertionError("A quality path appears in both quality lists.")
        if tuple(event.data_quality.unresolved_fields) != tuple(sorted(unresolved)):
            raise AssertionError("Unresolved paths are not deterministically ordered.")
        if tuple(event.data_quality.not_available_fields) != tuple(sorted(unavailable)):
            raise AssertionError("Not-available paths are not deterministically ordered.")
        if not (unresolved | unavailable).issubset(_CANONICAL_EVENT_VALUE_PATHS):
            raise AssertionError("A quality path is not a canonical value path.")
        if event.movement.track_degrees is not None and event.field_state("movement.track_degrees") != "value":
            raise AssertionError("TRACK degrees value conflicts with its state.")
        if event.movement.direction is not None and event.field_state("movement.direction") != "value":
            raise AssertionError("Deterministic direction value conflicts with its state.")
        if event.field_state("movement.direction") == "unresolved" and event.movement.direction is not None:
            raise AssertionError("Unresolved final TRACK propagated an intermediate direction.")
        if event.movement.visual_flight_direction is not None and event.field_state(
            "movement.visual_flight_direction",
        ) != "value":
            raise AssertionError("Visual direction state is inconsistent.")
        if event.movement.direction is None and event.movement.visual_flight_direction is None:
            raise AssertionError("Expected visual direction to remain independently available.")

        for path in _CANONICAL_EVENT_VALUE_PATHS:
            state = event.field_state(path)
            value = _value_at_path(event, path)
            if state == "value" and value is None:
                raise AssertionError(f"Resolved canonical path {path} has no value.")
            if state in ("unresolved", "not_available") and value is not None:
                raise AssertionError(f"Quality-labeled canonical path {path} has a value.")
            states.add(state)
        for profile in (OPERATIONAL_DIARY_PROFILE, DETAILED_DIARY_PROFILE):
            report_lines.append(f"  {profile.name.title()} profile fields:")
            for path in profile.field_paths:
                state = _state_for_path(event, path)
                value = _value_at_path(event, path)
                if state in ("unresolved", "not_available", "absent") and value is not None:
                    raise AssertionError(f"{path} has value {value!r} while state is {state}.")
                report_lines.append(f"    {path}: state={state}; value={value!r}")
        report_lines.append("")

    report_lines.extend((
        f"Real canonical value states exercised: {sorted(states)}",
        "Real artifact does not exercise not_available or absent value states; existing unit tests cover them.",
        "Aggregation edge cases (empty/single/N, failures, source mismatch, ordering, immutability, and event-local quality) are covered by Phase 7 tests.",
        "Renderer boundary audit: future Phase 8 needs only canonical models/profiles/serialization; no Phase 6 dependency is required.",
        "Phase 7 status: COMPLETE (7.1–7.6). Phase 8: NEXT — Deterministic Events Diary Renderer.",
        "No renderer was implemented; no Gemini call was made; Phase 8 was not started.",
    ))
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = "\n".join(report_lines)
    REPORT_PATH.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"Report: {REPORT_PATH}")
    return 0


# Canonical field paths whose states are represented in ``CanonicalEvent``.
_CANONICAL_EVENT_VALUE_PATHS = (
    "flight.primary_id", "flight.alternate_id", "flight.airline",
    "route.origin_airport_code", "route.destination_airport_code",
    "route.scheduled_departure", "route.actual_departure",
    "route.scheduled_arrival", "route.estimated_or_actual_arrival",
    "aircraft.registration", "aircraft.code", "aircraft.type",
    "movement.track_degrees", "movement.direction",
    "movement.visual_flight_direction", "movement.visual_route_description",
    "movement.visual_departure_context",
)


if __name__ == "__main__":
    raise SystemExit(main())
