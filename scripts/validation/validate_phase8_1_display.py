"""Validate Phase 8.1 field display using the saved canonical session only.

This offline check demonstrates labels, field states, and display values for a
small set of real CanonicalEvent fields. It does not render a diary or read any
Phase 6/OCR/VLM/fusion structures.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from events_diary.canonical_events import (
    CanonicalAircraft,
    CanonicalDataQuality,
    CanonicalEvent,
    CanonicalFlight,
    CanonicalMovement,
    CanonicalRoute,
)
from events_diary.renderer_core import display_value, label_for_path

SESSION_PATH = ROOT / "data/debug/canonical_events/phase_7_4/canonical_session.json"
REPORT_PATH = ROOT / "data/debug/events_diary/phase_8_1/display_validation.txt"
DISPLAY_PATHS = (
    "flight.primary_id", "route.origin_airport_code", "aircraft.registration",
    "movement.direction", "movement.visual_flight_direction",
)


def _event_from_dict(payload: dict) -> CanonicalEvent:
    """Build a canonical event directly from the serialized canonical schema."""
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


def main() -> int:
    """Check field-level display semantics for both saved canonical events."""
    session = json.loads(SESSION_PATH.read_text(encoding="utf-8"))
    if session["source_video"]["name"] != "investigation_switch_test.mp4":
        raise AssertionError("Unexpected canonical source video.")
    if session["events_count"] != 2:
        raise AssertionError("Expected two real canonical events.")

    lines = [
        "Phase 8.1 field display validation (not diary output)",
        f"Source: {session['source_video']['name']}; duration={session['source_video']['duration_s']}s",
    ]
    events = tuple(_event_from_dict(item) for item in session["events"])
    expected_visual = ("northwest", "Heading north")
    for index, event in enumerate(events):
        lines.append(f"Event {event.event_id} / Investigation {event.investigation_id}")
        for path in DISPLAY_PATHS:
            state = event.field_state(path)
            label = label_for_path(path)
            shown = display_value(event, path)
            if path == "movement.direction" and (
                state != "unresolved" or shown != "Could not determine"
            ):
                raise AssertionError("Real deterministic direction display is incorrect.")
            if path == "movement.visual_flight_direction" and shown != expected_visual[index]:
                raise AssertionError("Real reviewed visual direction was changed.")
            if (path in ("flight.primary_id", "route.origin_airport_code", "aircraft.registration")
                    and (shown is None or state != "value")):
                raise AssertionError(f"Expected real value for {path}.")
            lines.append(f"  {path}: state={state}; label={label}; display={shown!r}")
        lines.append("")

    report = "\n".join(lines)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"Report: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
