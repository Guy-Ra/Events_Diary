"""Render and validate the real Phase 7.4 session for Phase 8.3.

This offline validator reads only the saved CanonicalSession JSON artifact,
constructs canonical contract objects, calls the Detailed Diary renderer, and
saves the rendered text. It performs no model/provider calls and imports no
Phase 6 processing modules.
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
    CanonicalSession,
    CanonicalSourceVideo,
)
from events_diary.detailed_diary import render_detailed_diary

SESSION_PATH = ROOT / "data/debug/canonical_events/phase_7_4/canonical_session.json"
OUTPUT_PATH = ROOT / "data/debug/events_diary/phase_8_3/detailed_diary.txt"


def _load_session() -> CanonicalSession:
    """Reconstruct the canonical session from its saved JSON representation."""
    payload = json.loads(SESSION_PATH.read_text(encoding="utf-8"))
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


def main() -> int:
    """Check expected real diary facts and save the Detailed Diary artifact."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    session = _load_session()
    if session.source_video.name != "investigation_switch_test.mp4":
        raise AssertionError("Unexpected Phase 7.4 validation video.")
    diary = render_detailed_diary(session)
    expected = (
        "Event 1 — 00:01.6–00:04.9", "Flight: MEA251 / ME251", "Airline: MEA",
        "Route: BEY → ATH", "Scheduled departure: 07:35", "Actual departure: 07:44",
        "Scheduled arrival: 09:35", "Estimated/actual arrival: 09:18",
        "Aircraft: Airbus A321-271NX (A321)", "Registration: T7-ME2",
        "Visual direction: northwest",
        "Visual route:\nThe aircraft is tracking northwest over the Mediterranean Sea from the coast of Lebanon towards Greece.\nDeparted from Beirut on the coast of Lebanon, proceeding northwest over open water.",
        "Event 2 — 00:04.9–00:08.1", "Flight: ISR889 / 6H889",
        "Airline: Israir Airlines", "Route: TLV → BUS", "Scheduled departure: 09:10",
        "Actual departure: 09:40", "Scheduled arrival: 11:50",
        "Estimated/actual arrival: 11:59", "Aircraft: ATR 72-600 (AT76)",
        "Registration: 4X-ATI", "Visual direction: Heading north",
        "Visual route:\nThe aircraft is tracking northward over the eastern Mediterranean Sea off the coast of Lebanon.\nDeparted Tel Aviv and tracked north over the sea.",
    )
    missing = tuple(item for item in expected if item not in diary)
    if missing:
        raise AssertionError(f"Detailed Diary is missing expected content: {missing!r}.")
    if (
        diary.count("Event ") != 2
        or diary.count("Visual route:") != 2
        or "Track:" in diary
        or "Direction:" in diary
        or "Departure context:" in diary
        or "investigation_id" in diary
        or "data_quality" in diary
    ):
        raise AssertionError("Detailed Diary contains an unexpected event count or internal data.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(diary + "\n", encoding="utf-8")
    print(diary)
    print(f"Rendered Detailed Diary: {OUTPUT_PATH}")
    print("Validation: both events rendered; unresolved TRACK omitted; visual text grouped source-faithfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
