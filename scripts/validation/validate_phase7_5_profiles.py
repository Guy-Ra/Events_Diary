"""Validate renderer field profiles against the saved Phase 7.4 session.

This offline validation reads only canonical session JSON and reconstructs the
existing canonical contracts. It does not read Phase 6 artifacts, inspect
evidence internals, or render diary prose. The report lists field paths, states,
and canonical values to demonstrate that both renderer profiles can consume
``CanonicalSession`` alone.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from events_diary.canonical_events import (
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
REPORT_PATH = ROOT / "data/debug/canonical_events/phase_7_5/profile_validation_report.txt"


def _event_from_dict(payload: dict) -> CanonicalEvent:
    """Reconstruct one typed canonical event from canonical JSON fields."""
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
    """Read a profile value from the public canonical event contract."""
    value: object = event
    for component in path.split("."):
        value = getattr(value, component)
    return value


def _field_state(event: CanonicalEvent, path: str) -> str:
    """Return the canonical state for a profile field, including metadata."""
    if path in ("event_time_s", "investigation_id", "start_time_s", "end_time_s"):
        return "value"
    if path in ("data_quality.unresolved_fields", "data_quality.not_available_fields"):
        return "value"
    return event.field_state(path)


def main() -> int:
    """Validate both ordered field profiles against the real saved session."""
    payload = json.loads(SESSION_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Canonical session JSON must have an object root.")
    source = payload["source_video"]
    session = CanonicalSession(
        source_video=CanonicalSourceVideo(source["name"], source["duration_s"]),
        events=tuple(_event_from_dict(item) for item in payload["events"]),
        schema_version=payload["schema_version"],
    )
    if session.to_dict() != payload:
        raise AssertionError("Typed CanonicalSession serialization differs from the saved contract.")
    if session.events_count != payload["events_count"] or session.events_count != 2:
        raise AssertionError("Expected exactly two saved canonical events.")
    if session.source_video.name != "investigation_switch_test.mp4" or session.source_video.duration_s != 8.1:
        raise AssertionError("Unexpected source video name or duration.")
    if [(item.event_id, item.investigation_id, item.event_time_s) for item in session.events] != [
        (0, 0, 1.6), (1, 1, 4.9),
    ]:
        raise AssertionError("Saved CanonicalSession event order or identity is incorrect.")
    if not set(OPERATIONAL_DIARY_PROFILE.field_paths).issubset(
        DETAILED_DIARY_PROFILE.field_paths,
    ):
        raise AssertionError("Operational profile is not contained in Detailed profile.")

    lines = [
        "Phase 7.5 renderer-profile validation",
        f"CanonicalSession source: {session.source_video.name}",
        f"Duration: {session.source_video.duration_s}",
        f"Schema version: {session.schema_version}",
        f"Events: {session.events_count} in chronological order",
        "Input used: CanonicalSession JSON only; no Phase 6 structures were read.",
        "Serialization round-trip: passed",
        "",
    ]
    for profile in (OPERATIONAL_DIARY_PROFILE, DETAILED_DIARY_PROFILE):
        lines.append(f"{profile.name.title()} profile ({len(profile.field_paths)} fields):")
        for event in session.events:
            lines.append(
                f"  Event {event.event_id} / Investigation {event.investigation_id}:"
            )
            for path in profile.field_paths:
                value = _value_at_path(event, path)
                state = _field_state(event, path)
                if state in ("unresolved", "not_available", "absent") and value is not None:
                    raise AssertionError(f"{path} has a value despite state {state}.")
                lines.append(f"    {path}: state={state}; value={value!r}")
        lines.append("")

    observed_states = {
        _field_state(event, path)
        for event in session.events
        for path in DETAILED_DIARY_PROFILE.field_paths
        if path not in (
            "data_quality.unresolved_fields", "data_quality.not_available_fields",
        )
    }
    lines.append(f"Observed canonical value states: {sorted(observed_states)}")
    lines.append(
        "Real-artifact coverage: not_available and absent value states are not present; "
        "unit tests cover both without fabricating real data."
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = "\n".join(lines)
    REPORT_PATH.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"Report: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
