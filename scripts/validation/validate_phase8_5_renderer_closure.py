"""Validate Phase 8 renderer output using the real saved CanonicalSession.

This offline diagnostic reconstructs the Phase 7.4 CanonicalSession for
``investigation_switch_test.mp4`` and calls only the two production renderers.
It checks the exact expected diary strings against the prior Phase 8.2/8.3
artifacts and the existing Phase 8.5 output files, along with determinism,
profile selection, no-inference content, runtime import boundaries, UTF-8
string safety, empty-session behavior, and experiment workspace policy.

The script may write validation copies of the renderer strings to
``experiment/output`` for inspection. That file writing belongs only to this
validation script; it is not a reusable production output API. This diagnostic
does not run video analysis, call OCR/VLM/Gemini, or clean historical data.
"""

from __future__ import annotations

import ast
import json
import subprocess
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
from events_diary.detailed_diary import render_detailed_diary
from events_diary.operational_diary import render_operational_diary

SESSION_PATH = ROOT / "data/debug/canonical_events/phase_7_4/canonical_session.json"
OPERATIONAL_REFERENCE = ROOT / "data/debug/events_diary/phase_8_2/operational_diary.txt"
DETAILED_REFERENCE = ROOT / "data/debug/events_diary/phase_8_3/detailed_diary.txt"
OUTPUT_DIR = ROOT / "experiment/output"
REPORT_PATH = ROOT / "data/debug/events_diary/phase_8_5/phase_8_5_validation_report.txt"
RENDERER_MODULES = ("renderer_core.py", "operational_diary.py", "detailed_diary.py")
ALLOWED_IMPORTS = {
    "events_diary.canonical_events",
    "events_diary.renderer_core",
    "events_diary.operational_diary",
    "events_diary.detailed_diary",
}


def _load_session() -> CanonicalSession:
    """Reconstruct the saved real session with canonical contract objects."""
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
    session = CanonicalSession(
        CanonicalSourceVideo(source["name"], source["duration_s"]),
        events,
        schema_version=payload["schema_version"],
    )
    if payload["events_count"] != session.events_count:
        raise AssertionError("Saved event count differs from reconstructed session.")
    return session


def _read_reference(path: Path) -> str:
    """Read a prior diary artifact while excluding its final newline."""
    return path.read_text(encoding="utf-8").removesuffix("\n")


def _require_fragments(text: str, fragments: tuple[str, ...], label: str) -> None:
    """Require every expected display fragment in a rendered diary string."""
    missing = tuple(fragment for fragment in fragments if fragment not in text)
    if missing:
        raise AssertionError(f"{label} is missing expected content: {missing!r}.")


def _audit_runtime_imports() -> None:
    """Check renderer modules import only canonical contracts and renderer helpers."""
    for filename in RENDERER_MODULES:
        tree = ast.parse((ROOT / "src/events_diary" / filename).read_text(encoding="utf-8"))
        imports = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.module.startswith("events_diary.")
        }
        imports.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
            if alias.name.startswith("events_diary.")
        )
        unexpected = imports - ALLOWED_IMPORTS
        if unexpected:
            raise AssertionError(f"{filename} imports unsupported modules: {sorted(unexpected)}")


def _check_workspace() -> None:
    """Verify the two workspace placeholders and ignored validation outputs."""
    input_dir = ROOT / "experiment/input"
    if {path.name for path in input_dir.iterdir()} != {".gitkeep"}:
        raise AssertionError("experiment/input must contain only its .gitkeep placeholder.")
    if {path.name for path in OUTPUT_DIR.iterdir()} != {
        ".gitkeep", "operational_diary.txt", "detailed_diary.txt",
    }:
        raise AssertionError("experiment/output contains unexpected workspace files.")
    for placeholder in (input_dir / ".gitkeep", OUTPUT_DIR / ".gitkeep"):
        result = subprocess.run(
            ["git", "check-ignore", "-q", str(placeholder)], cwd=ROOT, check=False,
        )
        if result.returncode == 0:
            raise AssertionError("Experiment .gitkeep placeholders must remain trackable.")
    for name in ("operational_diary.txt", "detailed_diary.txt"):
        result = subprocess.run(
            ["git", "check-ignore", "-q", str(OUTPUT_DIR / name)], cwd=ROOT, check=False,
        )
        if result.returncode != 0:
            raise AssertionError(f"Generated output must remain ignored: {name}.")


def _validation_copy(name: str, content: str) -> Path:
    """Write one renderer string directly as a validation-only UTF-8 artifact."""
    path = OUTPUT_DIR / name
    path.write_text(content, encoding="utf-8")
    if path.read_text(encoding="utf-8") != content:
        raise AssertionError(f"Validation artifact did not round-trip as UTF-8: {path}.")
    return path


def _build_report(session: CanonicalSession, operational: str, detailed: str) -> str:
    """Build the final Phase 8.5 renderer-only validation report."""
    return "\n".join((
        "Phase 8.5 — Real End-to-End Renderer Validation",
        "",
        f"Canonical source: {session.source_video.name}",
        f"Schema version: {session.schema_version}",
        f"Duration: {session.source_video.duration_s:.1f} seconds",
        f"Events: {session.events_count}",
        "Intervals: 1.6–4.9 seconds; 4.9–8.1 seconds",
        "",
        "Operational Diary: PASS — expected flight, route, time and aircraft values; unresolved direction and internal/debug content omitted.",
        "Detailed Diary: PASS — expected fields and verbatim visual route/departure text; unresolved TRACK/direction and separate departure heading omitted.",
        f"Operational profile: PASS — {len(OPERATIONAL_DIARY_PROFILE.field_paths)} approved paths.",
        f"Detailed profile: PASS — {len(DETAILED_DIARY_PROFILE.field_paths)} approved paths.",
        "Determinism and prior-content parity: PASS — both exact strings match prior Phase 8.2/8.3 artifacts and prior Phase 8.5 outputs.",
        "Runtime boundary: PASS — renderers depend only on canonical contracts and renderer helpers; no Phase 6 runtime dependency.",
        "No inference: PASS — exact comparison with prior accepted text; presentation-only field composition.",
        "UTF-8 string safety: PASS — both outputs encode/decode as UTF-8 without changing content.",
        f"Operational validation copy: experiment/output/operational_diary.txt ({len(operational)} characters)",
        f"Detailed validation copy: experiment/output/detailed_diary.txt ({len(detailed)} characters)",
        "Empty session: PASS — both renderers continue returning empty strings.",
        "Workspace: PASS — fixed input/output folders, ignored generated contents, trackable .gitkeep placeholders.",
        "Historical data/: not cleaned.",
        "",
        "Phase 8 status: COMPLETE (8.1–8.5 COMPLETE).",
        "Full-system integration remains the next major task and was not started.",
        "Gemini was not called; OCR/VLM mode configuration was not implemented.",
        "",
    ))


def main() -> int:
    """Validate both production renderers and write validation-only artifacts."""
    session = _load_session()
    if (
        session.schema_version != "1.0"
        or session.source_video.name != "investigation_switch_test.mp4"
        or session.source_video.duration_s != 8.1
        or session.events_count != 2
        or tuple((event.start_time_s, event.end_time_s) for event in session.events)
        != ((1.6, 4.9), (4.9, 8.1))
    ):
        raise AssertionError("CanonicalSession does not match the Phase 8.5 real fixture.")

    operational = render_operational_diary(session)
    detailed = render_detailed_diary(session)
    if operational != render_operational_diary(session):
        raise AssertionError("Operational rendering is not deterministic.")
    if detailed != render_detailed_diary(session):
        raise AssertionError("Detailed rendering is not deterministic.")

    _require_fragments(operational, (
        "Event 1 — 00:01.6–00:04.9", "Flight: MEA251 / ME251", "Airline: MEA",
        "Route: BEY → ATH", "Scheduled departure: 07:35", "Actual departure: 07:44",
        "Scheduled arrival: 09:35", "Estimated/actual arrival: 09:18",
        "Aircraft: Airbus A321-271NX (A321)", "Registration: T7-ME2",
        "Event 2 — 00:04.9–00:08.1", "Flight: ISR889 / 6H889",
        "Airline: Israir Airlines", "Route: TLV → BUS", "Scheduled departure: 09:10",
        "Actual departure: 09:40", "Scheduled arrival: 11:50",
        "Estimated/actual arrival: 11:59", "Aircraft: ATR 72-600 (AT76)",
        "Registration: 4X-ATI",
    ), "Operational Diary")
    if any(token in operational for token in (
        "Direction:", "Could not determine", "N/A", "data_quality", "investigation_id",
        "movement.", "flight.", "route.", "aircraft.", "TRACK", "Phase 6", "inferred",
    )):
        raise AssertionError("Operational output exposes excluded or internal content.")

    _require_fragments(detailed, (
        "Event 1 — 00:01.6–00:04.9", "Flight: MEA251 / ME251", "Airline: MEA",
        "Route: BEY → ATH", "Scheduled departure: 07:35", "Actual departure: 07:44",
        "Scheduled arrival: 09:35", "Estimated/actual arrival: 09:18",
        "Aircraft: Airbus A321-271NX (A321)", "Registration: T7-ME2",
        "Visual direction: northwest", "Visual route:\nThe aircraft is tracking northwest over the Mediterranean Sea from the coast of Lebanon towards Greece.\nDeparted from Beirut on the coast of Lebanon, proceeding northwest over open water.",
        "Event 2 — 00:04.9–00:08.1", "Flight: ISR889 / 6H889",
        "Airline: Israir Airlines", "Route: TLV → BUS", "Scheduled departure: 09:10",
        "Actual departure: 09:40", "Scheduled arrival: 11:50",
        "Estimated/actual arrival: 11:59", "Aircraft: ATR 72-600 (AT76)",
        "Registration: 4X-ATI", "Visual direction: Heading north",
        "Visual route:\nThe aircraft is tracking northward over the eastern Mediterranean Sea off the coast of Lebanon.\nDeparted Tel Aviv and tracked north over the sea.",
    ), "Detailed Diary")
    if any(token in detailed for token in (
        "Track:", "Direction:", "Departure context:", "Could not determine", "data_quality",
        "investigation_id", "movement.", "flight.", "route.", "aircraft.", "Phase 6", "inferred",
    )):
        raise AssertionError("Detailed output exposes excluded or internal content.")

    if operational != _read_reference(OPERATIONAL_REFERENCE):
        raise AssertionError("Operational diary changed from its accepted Phase 8.2 content.")
    if detailed != _read_reference(DETAILED_REFERENCE):
        raise AssertionError("Detailed diary changed from its accepted Phase 8.3 content.")

    old_operational = (OUTPUT_DIR / "operational_diary.txt").read_text(encoding="utf-8")
    old_detailed = (OUTPUT_DIR / "detailed_diary.txt").read_text(encoding="utf-8")
    if (operational, detailed) != (old_operational, old_detailed):
        raise AssertionError("Diary text differs from the previous Phase 8.5 output files.")
    if operational.encode("utf-8").decode("utf-8") != operational:
        raise AssertionError("Operational string is not UTF-8 safe.")
    if detailed.encode("utf-8").decode("utf-8") != detailed:
        raise AssertionError("Detailed string is not UTF-8 safe.")

    empty = CanonicalSession(CanonicalSourceVideo("empty.mp4", 0.0), ())
    if render_operational_diary(empty) != "" or render_detailed_diary(empty) != "":
        raise AssertionError("Empty-session renderer behavior changed.")

    _audit_runtime_imports()
    _check_workspace()
    _validation_copy("operational_diary.txt", operational)
    _validation_copy("detailed_diary.txt", detailed)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(_build_report(session, operational, detailed), encoding="utf-8")

    print(f"Canonical session: {session.source_video.name}, {session.source_video.duration_s:.1f}s, {session.events_count} events")
    print(f"Operational Diary: unchanged ({len(operational)} characters)")
    print(f"Detailed Diary: unchanged ({len(detailed)} characters)")
    print(f"Validation report: {REPORT_PATH}")
    print("Phase 8.5 renderer-only validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
