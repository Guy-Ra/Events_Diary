"""Phase 3.3 offline temporal validation on three existing manual recordings.

Reuse the candidate evaluator's sequential frames -> presence -> pair signals ->
production candidates pipeline. Convert semantic rows to CandidateTimelineEntry
and call the production seconds-based confirmation API without parameter tuning.
No CV or candidate logic is duplicated here. The output contains one row per
non-NONE candidate, not another large raw timeline. Original candidate time and
look-ahead confirmation time remain distinct. Reports expose supersession and
insufficient future context, including the close/reopen accidental-click scenario.

All paths resolve from the repository. CSVs and report under investigation_temporal
replace matching local artifacts only. Nominal source times assume constant FPS;
canonical layout and all existing upstream provisional thresholds are retained.
An unassociated presence reversal with full coverage propagates the temporal API's
ambiguity error for review. No threshold adjustment, lifecycle, startup-active
initialization, identity, confidence, temporal merging, or streaming logic occurs.
"""

import csv
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from evaluate_investigation_candidates import SOURCES, _evaluate

from events_diary.config import UIConfig
from events_diary.investigation_detection import (
    InvestigationCandidate,
    InvestigationCandidateKind,
)
from events_diary.investigation_temporal import (
    CONFIRMATION_WINDOW_SECONDS,
    CandidateTimelineEntry,
    confirm_investigation_candidates,
)
from events_diary.video import read_video_metadata

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/debug/map_motion/investigation_temporal"


def _entries(rows):
    """Convert measured candidate rows to minimal immutable semantic entries."""
    return [CandidateTimelineEntry(
        row["frame_a"], row["time_a"], row["time_b"], row["present_before"], row["present_after"],
        InvestigationCandidate(InvestigationCandidateKind(row["candidate_kind"]),
                               row["candidate_reason"])) for row in rows]


def _evaluate_video(name, config):
    """Measure a recording once, confirm its semantic candidates, and write CSV.

    Returns report lines. Confirmation errors propagate rather than inventing an
    outcome. The CSV retains before/after presence and original debugging reasons.
    """
    path = ROOT / "data/videos" / f"{name}_test.mp4"
    print(f"Input: {path}", flush=True)
    metadata = read_video_metadata(path)
    measured = _evaluate(path, metadata, config)
    entries = _entries(measured)
    events, results = confirm_investigation_candidates(entries)
    by_index = {entry.pair_index: entry for entry in entries}
    rows = []
    for result in results:
        entry = by_index[result.candidate_pair_index]
        row = asdict(result)
        row["candidate_kind"] = result.candidate_kind.value
        row["status"] = result.status.value
        row.update(present_before=entry.present_before, present_after=entry.present_after,
                   candidate_reason=entry.candidate.reason)
        rows.append(row)
    csv_path = OUTPUT / f"{name}_temporal.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "candidate_pair_index", "candidate_kind", "candidate_time_seconds", "status",
            "confirmed_at_seconds", "superseded_by_pair_index", "present_before",
            "present_after", "candidate_reason"])
        writer.writeheader()
        writer.writerows(rows)
    lines = [f"\n{name}: {path}",
             (f"{metadata.width}x{metadata.height}; fps={metadata.fps}; frames={metadata.frame_count}; "
              f"duration={metadata.duration_seconds}s; pairs={len(entries)}"),
             f"Non-NONE candidates={len(results)}; statuses={dict(Counter(r.status.value for r in results))}"]
    for result in results:
        entry = by_index[result.candidate_pair_index]
        lines.append(f"pair {entry.pair_index}, {entry.time_before_seconds:.1f}-"
                     f"{entry.time_after_seconds:.1f}s: {result.candidate_kind.value}; "
                     f"presence={entry.present_before}->{entry.present_after}; {result.status.value}; "
                     f"event_time={result.candidate_time_seconds:.1f}s; "
                     f"confirmed_at={result.confirmed_at_seconds}; "
                     f"superseded_by_pair={result.superseded_by_pair_index}; reason={entry.candidate.reason}")
    lines.append("Confirmed events (candidate time is event estimate; confirmation is later):")
    for event in events:
        lines.append(f"  {event.kind.value}, pair {event.candidate_pair_index}: "
                     f"event_time={event.candidate_time_seconds:.1f}s, "
                     f"confirmed_at={event.confirmed_at_seconds:.1f}s")
    lines.append(f"Startup present={entries[0].present_before}; no synthetic startup OPEN is added.")
    if name == "investigation_close_reopen":
        lines.append("Key scenario: inspect CLOSE pair 37, accidental OPEN pair 70, and final SWITCH "
                     "pair 75 above. Expected: confirmed, superseded by 75, confirmed respectively. "
                     "This expectation is a report annotation, never a detection rule.")
    print(f"CSV: {csv_path}", flush=True)
    return lines


def main():
    """Run all three recordings and write confirmation outcomes without lifecycle."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lines = ["Phase 3.3 offline Investigation temporal confirmation",
             f"Provisional confirmation window: {CONFIRMATION_WINDOW_SECONDS}s",
             "ANY later non-NONE candidate through the inclusive horizon supersedes the earlier one.",
             "NONE is neutral. Expected presence: OPEN/SWITCH=True, CLOSE=False."]
    for name in SOURCES:
        lines.extend(_evaluate_video(name, UIConfig()))
    lines += [("\nLimitations: three manual canonical-layout recordings and provisional upstream "
               "thresholds/window, not held-out validation. Supersession does not merge equivalent "
               "candidates. Sampled presence cannot establish unsampled behavior. Unassociated "
               "presence reversal with full coverage is an explicit ambiguity error. Early endings "
               "cannot confirm candidates; event time remains the original candidate time. "
               "No unexpected outcomes are automatically suppressed. No lifecycle or startup-active "
               "initialization is implemented.")]
    report = "\n".join(lines) + "\n"
    report_path = OUTPUT / "investigation_temporal_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
