"""Validate Phase 7.4 aggregation from saved Phase 6 JSON snapshots, offline.

This validation-only adapter reconstructs the typed Phase 6 fields actually
consumed by ``build_canonical_session``. The saved artifacts are diagnostic JSON
summaries rather than a persisted ``InvestigationRecord`` graph, so unused OCR,
candidate, sampling, and frame-evidence objects are deliberately not rehydrated.
No production deserializer is added and no model/provider is called.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from events_diary.canonical_events import build_canonical_session
from events_diary.investigation_lifecycle import (
    InvestigationEndReason,
    InvestigationInterval,
    InvestigationStartReason,
)
from events_diary.investigation_record import (
    InvestigationRecord,
    VideoInvestigationResult,
)
from events_diary.video import VideoMetadata
from events_diary.vlm_review import VLMRouteReviewObservation

ARTIFACT_ROOT = ROOT / "data/debug/investigation_assembly/phase_6_6/latest_run"
OUTPUT_ROOT = ROOT / "data/debug/canonical_events/phase_7_4"
EXPECTED_INVESTIGATIONS = (0, 1)


def _load_json(path: Path) -> dict:
    """Load one existing validation snapshot and require an object root."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return payload


def _typed_route_reviews(payload: dict) -> dict[str, VLMRouteReviewObservation]:
    """Reconstruct final Pass 2 route-review contracts from saved values."""
    return {
        name: VLMRouteReviewObservation(
            field_name=item["field_name"],
            reviewed_value=item["reviewed_value"],
            review_status=item["review_status"],
            evidence_quality=item["evidence_quality"],
            observation_status=item["observation_status"],
            evidence_time_s=item.get("evidence_time_s"),
            notes=item.get("notes"),
        )
        for name, item in payload.items()
    }


def _typed_record(record_payload: dict, final_payload: dict, pass2_payload: dict) -> InvestigationRecord:
    """Build a validation-only typed record projection from source snapshots.

    Every attribute used by ``VideoInvestigationResult`` and the production
    canonical builder comes from the saved artifacts. The full Investigation
    evidence graph is not reconstructed because it is not required by this API.
    """
    investigation_id = record_payload["investigation_id"]
    start_s, end_s = record_payload["interval_s"]
    metadata = record_payload["event_metadata"]
    interval = InvestigationInterval(
        investigation_index=investigation_id,
        start_time_seconds=start_s,
        end_time_seconds=end_s,
        start_reason=InvestigationStartReason(metadata["start_reason"]),
        end_reason=InvestigationEndReason(metadata["end_reason"]),
    )
    final_fields = {
        name: SimpleNamespace(
            final_status=item["final_status"],
            final_value=item["final_value"],
        )
        for name, item in final_payload["fields"].items()
    }
    # The source snapshot has both a top-level final map and a flattened record
    # summary. Require them to agree before passing either representation on.
    for name, item in record_payload["final_fields"].items():
        final = final_fields[name]
        if (item["final_status"], item["final_value"]) != (
            final.final_status, final.final_value,
        ):
            raise ValueError(f"Saved final field representations disagree for {name}.")

    derived = (final_payload.get("direction_consistency") or {}).get("derived_direction")
    direction_consistency = SimpleNamespace(
        derived_direction=(SimpleNamespace(text=derived["text"]) if derived else None),
    )
    final_result = SimpleNamespace(
        fields=final_fields,
        direction_consistency=direction_consistency,
    )
    pass2 = SimpleNamespace(route_reviews=_typed_route_reviews(pass2_payload["route_reviews"]))

    # The existing builder intentionally accepts the Phase 6 record type. The
    # persisted report is a summary, not the full constructor payload; allocate
    # that type and populate exactly the validated attributes the boundary reads.
    record = object.__new__(InvestigationRecord)
    object.__setattr__(record, "investigation_id", investigation_id)
    object.__setattr__(record, "source_video", Path(record_payload["source_video"]))
    object.__setattr__(record, "interval", interval)
    object.__setattr__(record, "final_result", final_result)
    object.__setattr__(record, "vlm_pass2", pass2)
    return record


def _assert_no_internal_leakage(payload: dict) -> None:
    """Reject Phase 6 evidence/runtime keys from canonical session JSON."""
    forbidden = (
        "ocr", "vlm", "fusion", "provider", "diagnostic", "resolution_reason",
        "evidence", "candidate", "confidence", "prompt", "model",
    )

    def visit(value) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if any(token in key.casefold() for token in forbidden):
                    raise AssertionError(f"Internal field leaked into canonical JSON: {key}")
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)


def _event_report(event) -> str:
    """Format a validation diagnostic for one event, not diary output."""
    return "\n".join((
        f"Event {event.event_id} / Investigation {event.investigation_id}",
        f"  event_time_s={event.event_time_s}; interval=[{event.start_time_s}, {event.end_time_s})",
        f"  flight={asdict(event.flight)}",
        f"  route={asdict(event.route)}",
        f"  aircraft={asdict(event.aircraft)}",
        f"  movement={asdict(event.movement)}",
        f"  unresolved_fields={list(event.data_quality.unresolved_fields)}",
        f"  not_available_fields={list(event.data_quality.not_available_fields)}",
    ))


def main() -> int:
    """Reconstruct, aggregate, serialize, and report the saved two-event run."""
    session_payload = _load_json(ARTIFACT_ROOT / "session_result.json")
    video_data = session_payload["video_metadata"]
    metadata = VideoMetadata(
        path=Path(video_data["path"]),
        duration_seconds=video_data["duration_seconds"],
        fps=video_data["fps"],
        frame_count=video_data["frame_count"],
        width=video_data["width"],
        height=video_data["height"],
    )
    if session_payload["source_video"] != str(metadata.path):
        raise ValueError("Saved session and video metadata source paths disagree.")
    completed = session_payload["completed_records"]
    if tuple(item["investigation_id"] for item in completed) != EXPECTED_INVESTIGATIONS:
        raise ValueError("Saved run does not contain the expected two completed records.")

    records = []
    source_final_payloads = []
    source_pass2_payloads = []
    for item in completed:
        record_dir = ARTIFACT_ROOT / f"investigation_{item['investigation_id']:03}"
        record_payload = _load_json(record_dir / "investigation_record.json")
        final_payload = _load_json(record_dir / "final_result.json")
        pass2_payload = _load_json(record_dir / "vlm_pass2_result.json")
        records.append(_typed_record(record_payload, final_payload, pass2_payload))
        source_final_payloads.append(final_payload)
        source_pass2_payloads.append(pass2_payload)

    video_result = VideoInvestigationResult(
        source_video=metadata.path,
        video_metadata=metadata,
        investigations=tuple(records),
    )
    session = build_canonical_session(video_result)
    payload = session.to_dict()
    json_text = session.to_json(indent=2, ensure_ascii=False) + "\n"
    round_trip = json.loads(json_text)
    if round_trip != payload:
        raise AssertionError("to_dict and to_json outputs do not agree.")
    _assert_no_internal_leakage(payload)

    if (session.schema_version != "1.0" or session.source_video.name != metadata.path.name
            or session.source_video.duration_s != metadata.duration_seconds
            or session.events_count != 2):
        raise AssertionError("Canonical session header or event count is incorrect.")
    expected_starts = (1.6, 4.9)
    expected_investigation_ids = (0, 1)
    for event_id, (event, expected_start, expected_investigation) in enumerate(
        zip(session.events, expected_starts, expected_investigation_ids, strict=True),
    ):
        if (event.event_id, event.investigation_id, event.event_time_s) != (
            event_id, expected_investigation, expected_start,
        ):
            raise AssertionError("Canonical event ordering or identity is incorrect.")
        if event.event_time_s != event.start_time_s:
            raise AssertionError("event_time_s must equal the Investigation start.")
        source_record = video_result.investigations[event_id]
        if (event.start_time_s, event.end_time_s) != (
            source_record.start_time_s, source_record.end_time_s,
        ):
            raise AssertionError("Canonical event interval differs from saved Phase 6 bounds.")
        source_track = source_final_payloads[event_id]["fields"]["TRACK"]
        intermediate_direction = (
            source_final_payloads[event_id].get("direction_consistency") or {}
        ).get("derived_direction")
        if source_track["final_status"] != "unresolved" or source_track["final_value"] is not None:
            raise AssertionError("Expected each saved final TRACK to be unresolved with a null value.")
        if event.movement.track_degrees is not None or event.movement.direction is not None:
            raise AssertionError("Unresolved final TRACK must not populate canonical direction.")
        if not {"movement.track_degrees", "movement.direction"}.issubset(
            event.data_quality.unresolved_fields,
        ):
            raise AssertionError("Unresolved final TRACK must mark both canonical paths unresolved.")
        if intermediate_direction is None or not intermediate_direction.get("text"):
            raise AssertionError("Expected saved intermediate derived direction is missing.")

        route_reviews = source_pass2_payloads[event_id]["route_reviews"]
        expected_visual_direction = next(
            (route_reviews[name]["reviewed_value"]
             for name in ("visual_flight_direction_text", "visual_flight_direction_sector")
             if name in route_reviews
             and route_reviews[name]["review_status"] in ("confirmed", "corrected")
             and route_reviews[name]["observation_status"] != "not_visible"
             and route_reviews[name]["reviewed_value"] is not None),
            None,
        )
        if event.movement.visual_flight_direction != expected_visual_direction:
            raise AssertionError("Visual direction differs from the final Pass 2 review.")
        value_by_path = {
            "flight.primary_id": event.flight.primary_id,
            "flight.alternate_id": event.flight.alternate_id,
            "flight.airline": event.flight.airline,
            "route.origin_airport_code": event.route.origin_airport_code,
            "route.destination_airport_code": event.route.destination_airport_code,
            "route.scheduled_departure": event.route.scheduled_departure,
            "route.actual_departure": event.route.actual_departure,
            "route.scheduled_arrival": event.route.scheduled_arrival,
            "route.estimated_or_actual_arrival": event.route.estimated_or_actual_arrival,
            "aircraft.registration": event.aircraft.registration,
            "aircraft.code": event.aircraft.code,
            "aircraft.type": event.aircraft.type,
            "movement.track_degrees": event.movement.track_degrees,
            "movement.direction": event.movement.direction,
            "movement.visual_flight_direction": event.movement.visual_flight_direction,
            "movement.visual_route_description": event.movement.visual_route_description,
            "movement.visual_departure_context": event.movement.visual_departure_context,
        }
        for path in (*event.data_quality.unresolved_fields,
                     *event.data_quality.not_available_fields):
            if value_by_path[path] is not None:
                raise AssertionError(f"Quality-labeled canonical value is not null: {path}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_ROOT / "canonical_session.json"
    report_path = OUTPUT_ROOT / "validation_report.txt"
    json_path.write_text(json_text, encoding="utf-8")

    expected_snapshots = (
        ("MEA251", "ME251", "BEY", "ATH", "T7-ME2", "West-Northwest"),
        ("ISR889", "6H889", "TLV", "BUS", "4X-ATI", "North"),
    )
    snapshot_notes = []
    for event, expected in zip(session.events, expected_snapshots, strict=True):
        actual = (
            event.flight.primary_id, event.flight.alternate_id,
            event.route.origin_airport_code, event.route.destination_airport_code,
            event.aircraft.registration,
        )
        if actual != expected[:-1]:
            raise AssertionError(
                f"Investigation {event.investigation_id}: canonical={actual!r}; "
                f"snapshot={expected[:-1]!r}"
            )
        source_track = source_final_payloads[event.investigation_id]["fields"]["TRACK"]
        derived = source_final_payloads[event.investigation_id]["direction_consistency"][
            "derived_direction"
        ]
        if derived["text"] != expected[-1]:
            raise AssertionError(
                f"Investigation {event.investigation_id}: intermediate derived direction "
                f"{derived['text']!r} does not match the expected saved value {expected[-1]!r}."
            )
        snapshot_notes.append(
            f"  Investigation {event.investigation_id}: intermediate OCR-derived "
            f"direction={derived['text']!r} from TRACK {derived['degrees']}; "
            f"final TRACK status={source_track['final_status']!r}, "
            f"value={source_track['final_value']!r}; canonical movement.direction="
            f"{event.movement.direction!r} (unresolved)."
        )

    report_lines = [
        "Phase 7.4 saved-artifact aggregation validation",
        f"Source video: {session.source_video.name}",
        f"Duration: {session.source_video.duration_s}",
        f"Input typed record projections: {len(video_result.investigations)}",
        f"Output CanonicalEvents: {session.events_count}",
        f"Schema version: {session.schema_version}",
        "Event ordering: " + repr([
            (event.event_id, event.investigation_id, event.event_time_s)
            for event in session.events
        ]),
        "Typed reconstruction uses source-saved attributes consumed by the production builder.",
        "The InvestigationRecord initializer was bypassed because the saved JSON is a summary, not a complete evidence graph.",
        "VideoInvestigationResult validation still checked source identity, IDs, and chronological bounds.",
        "Unused OCR/candidate/sampling/frame objects were not reconstructed; no provider calls were made.",
        "Historical snapshot comparison: representative flight, route, and aircraft fields match.",
        "Direction semantics: derived_direction is intermediate evidence from an OCR TRACK observation.",
        "It is not propagated when final Phase 6 TRACK is unresolved; both canonical TRACK and direction remain null/unresolved.",
        "Visual flight direction is separate and comes from final Pass 2 visual review.",
        *snapshot_notes,
        "",
        *(_event_report(event) for event in session.events),
        "",
        "JSON serialization and internal-field leakage checks: passed",
        f"Canonical JSON: {json_path}",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("\n".join(report_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
