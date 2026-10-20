"""Phase 3 closure validation of the complete existing semantic pipeline.

Decode every consecutive pair of all four manual validation recordings through
the existing candidate evaluator (presence -> signals -> candidates). Reuse the
temporal entry adapter and production confirmation/lifecycle APIs. No detection
logic or thresholds are duplicated, tuned, or inferred from golden expectations.
Expected observations below are scenario assertions applied AFTER actual video
measurement. Any mismatch produces FAIL and a nonzero exit status for review.

Check presence/candidate/confirmation/lifecycle consistency, candidate boundary
provenance, NONE/superseded exclusions, interval order, map-only gaps, and anomalies.
As a counterfactual check, alter only confirmed_at metadata and verify that existing
lifecycle outputs remain identical. This validates timestamp independence without
reimplementing the state machine. Checks do not correct or suppress any output.

Write a concise report and one interval CSV under data/debug/map_motion. Source
timing assumes constant FPS and canonical layout. Video end uses full metadata
duration; temporal coverage ends at the last decoded sample. These four recordings
provide representative manual validation, not a benchmark. Only matching closure
outputs are replaced; detailed existing evaluation artifacts remain untouched.
No Phase 4 sampling/content extraction, identity, OCR, or Jira writes occur here.
"""

import csv
import math
import sys
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

# Allow direct execution and explicit imports across script directories.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from events_diary.config import UIConfig
from events_diary.investigation_detection import InvestigationCandidateKind
from events_diary.investigation_lifecycle import (
    InvestigationEndReason,
    InvestigationStartReason,
    build_investigation_lifecycle,
)
from events_diary.investigation_temporal import (
    CONFIRMATION_WINDOW_SECONDS,
    ConfirmationStatus,
    confirm_investigation_candidates,
)
from events_diary.video import read_video_metadata
from scripts.evaluation.evaluate_investigation_candidates import _evaluate
from scripts.evaluation.evaluate_investigation_temporal import _entries

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/debug/map_motion"
EXPECTED = {
    "combined_interaction": {
        "initial": True,
        "presence_runs": ((0, 185, True),),
        "candidates": ((146, "switch", 14.7),),
        "results": ((146, "confirmed", None),),
        "intervals": ((0, 14.7, "video_started_active", "switch"),
                      (14.7, 18.6, "switch", "video_end")),
    },
    "investigation_switch": {
        "initial": False,
        "presence_runs": ((0, 15, False), (16, 80, True)),
        "candidates": ((15, "open", 1.6), (48, "switch", 4.9)),
        "results": ((15, "confirmed", None), (48, "confirmed", None)),
        "intervals": ((1.6, 4.9, "open", "switch"), (4.9, 8.1, "switch", "video_end")),
    },
    "investigation_started_active": {
        "initial": True,
        "presence_runs": ((0, 46, True),),
        "candidates": ((23, "switch", 2.4),),
        "results": ((23, "confirmed", None),),
        "intervals": ((0, 2.4, "video_started_active", "switch"),
                      (2.4, 4.7, "switch", "video_end")),
    },
    "investigation_close_reopen": {
        "initial": True,
        "presence_runs": ((0, 37, True), (38, 70, False), (71, 119, True)),
        "candidates": ((37, "close", 3.8), (70, "open", 7.1), (75, "switch", 7.6)),
        "results": ((37, "confirmed", None), (70, "superseded", 75), (75, "confirmed", None)),
        "intervals": ((0, 3.8, "video_started_active", "close"),
                      (7.6, 12, "switch", "video_end")),
    },
}


def _same_time(a, b):
    """Compare validation observations with a small floating-point display tolerance."""
    return math.isclose(a, b, rel_tol=0, abs_tol=1e-9)


def _boundary_provenance(intervals, events, video_end):
    """Check each output boundary against its confirmed candidate-time source.

    Startup/video-end boundaries instead use explicit video bounds. Replacement
    reasons map to their corresponding event kinds without recomputing lifecycle.
    """
    def matches(time, kind):
        """Find a matching confirmed kind at the original candidate timestamp."""
        return any(event.kind is kind and _same_time(time, event.candidate_time_seconds)
                   for event in events)

    for interval in intervals:
        if interval.start_reason is InvestigationStartReason.VIDEO_STARTED_ACTIVE:
            start_ok = _same_time(interval.start_time_seconds, 0)
        else:
            start_ok = matches(interval.start_time_seconds,
                               InvestigationCandidateKind(interval.start_reason.value))
        if interval.end_reason is InvestigationEndReason.VIDEO_END:
            end_ok = _same_time(interval.end_time_seconds, video_end)
        else:
            kind = (InvestigationCandidateKind.OPEN
                    if interval.end_reason is InvestigationEndReason.OPEN_REPLACEMENT
                    else InvestigationCandidateKind(interval.end_reason.value))
            end_ok = matches(interval.end_time_seconds, kind)
        if not start_ok or not end_ok:
            return False
    return True


def _validate_video(name, expected):
    """Run actual CV/semantic APIs and compare observations/invariants afterward.

    Return PASS flag, concise report lines, and interval CSV rows. Production
    failures propagate; scenario mismatches remain FAIL rather than being tuned.
    """
    path = ROOT / "data/videos" / f"{name}_test.mp4"
    print(f"Input: {path}", flush=True)
    metadata = read_video_metadata(path)
    measured = _evaluate(path, metadata, UIConfig())
    entries = _entries(measured)
    events, results = confirm_investigation_candidates(entries)
    initial = entries[0].present_before
    intervals, anomalies = build_investigation_lifecycle(
        initial_present=initial, video_start_time_seconds=0,
        video_end_time_seconds=metadata.duration_seconds, confirmed_events=events)
    states = [initial] + [entry.present_after for entry in entries]
    candidates = [entry for entry in entries
                  if entry.candidate.kind is not InvestigationCandidateKind.NONE]
    observed_candidates = [(entry.pair_index, entry.candidate.kind.value, entry.time_after_seconds)
                           for entry in candidates]
    candidate_match = len(observed_candidates) == len(expected["candidates"]) and all(
        a[:2] == b[:2] and _same_time(a[2], b[2])
        for a, b in zip(observed_candidates, expected["candidates"]))
    observed_results = tuple((r.candidate_pair_index, r.status.value, r.superseded_by_pair_index)
                             for r in results)
    interval_match = len(intervals) == len(expected["intervals"]) and all(
        _same_time(interval.start_time_seconds, golden[0])
        and _same_time(interval.end_time_seconds, golden[1])
        and (interval.start_reason.value, interval.end_reason.value) == golden[2:]
        for interval, golden in zip(intervals, expected["intervals"]))
    by_index = {entry.pair_index: entry for entry in entries}
    confirmed_indices = {r.candidate_pair_index for r in results
                         if r.status is ConfirmationStatus.CONFIRMED}
    superseded_indices = {r.candidate_pair_index for r in results
                          if r.status is ConfirmationStatus.SUPERSEDED}
    event_indices = {event.candidate_pair_index for event in events}
    altered_events = [replace(event, confirmed_at_seconds=event.confirmed_at_seconds + 1000)
                      for event in events]
    altered_lifecycle = build_investigation_lifecycle(
        initial_present=initial, video_start_time_seconds=0,
        video_end_time_seconds=metadata.duration_seconds, confirmed_events=altered_events)
    checks = [
        ("all consecutive frame pairs analyzed", len(entries) == metadata.frame_count - 1),
        ("initial presence matches scenario", initial is expected["initial"]),
        ("effective presence timeline matches scenario", len(states) == metadata.frame_count
         and sum(end - start + 1 for start, end, _ in expected["presence_runs"]) == len(states)
         and all(all(state is present for state in states[start:end + 1])
                 for start, end, present in expected["presence_runs"])),
        ("candidate timeline matches scenario without bursts", candidate_match),
        ("one confirmation result per non-NONE candidate", [r.candidate_pair_index for r in results]
         == [entry.pair_index for entry in candidates]),
        ("temporal statuses/superseding relationships match scenario",
         observed_results == expected["results"]),
        ("confirmed events correspond exactly to confirmed results", event_indices == confirmed_indices
         and len(events) == len(confirmed_indices)),
        ("NONE never becomes an event", all(event.kind is not InvestigationCandidateKind.NONE
                                           for event in events)),
        ("superseded candidates excluded from lifecycle input", not event_indices & superseded_indices),
        ("events preserve candidate kind/time and confirmation horizon", all(
            event.kind is by_index[event.candidate_pair_index].candidate.kind
            and _same_time(event.candidate_time_seconds,
                           by_index[event.candidate_pair_index].time_after_seconds)
            and _same_time(event.confirmed_at_seconds,
                           event.candidate_time_seconds + CONFIRMATION_WINDOW_SECONDS) for event in events)),
        ("intervals match expected lifecycle", interval_match),
        ("every boundary has candidate-time/video-bound provenance",
         _boundary_provenance(intervals, events, metadata.duration_seconds)),
        ("confirmation-time metadata cannot alter lifecycle boundaries",
         altered_lifecycle == (intervals, anomalies)),
        ("intervals have valid durations and chronological contiguous indexes",
         [i.investigation_index for i in intervals] == list(range(len(intervals)))
         and all(i.start_time_seconds <= i.end_time_seconds for i in intervals)
         and all(a.end_time_seconds <= b.start_time_seconds for a, b in pairwise(intervals))),
        ("stable both-absent pairs do not overlap Investigation intervals", all(
            max(entry.time_before_seconds, interval.start_time_seconds)
            >= min(entry.time_after_seconds, interval.end_time_seconds)
            for entry in entries if not entry.present_before and not entry.present_after
            for interval in intervals)),
        ("no lifecycle anomalies", not anomalies),
        ("no unexpected insufficient-future-context cases", all(
            r.status is not ConfirmationStatus.INSUFFICIENT_FUTURE_CONTEXT for r in results)),
    ]
    if initial:
        checks.append(("startup presence does not synthesize OPEN",
                       all(entry.candidate.kind is not InvestigationCandidateKind.OPEN
                           for entry in candidates if entry.time_after_seconds <= 0.7)))
    if name == "investigation_close_reopen":
        checks.append(("accidental OPEN creates no lifecycle interval",
                       70 in superseded_indices and all(
                           not _same_time(interval.start_time_seconds, 7.1) for interval in intervals)))
        checks.append(("closed-to-final-switch gap remains inactive", all(
            max(interval.start_time_seconds, 3.8) >= min(interval.end_time_seconds, 7.6)
            for interval in intervals)))
    if name == "combined_interaction":
        checks.append(("only loading frame146 is bridged; raw evidence retained",
                       [row["frame_b"] for row in measured
                        if row["raw_present_after"] != row["present_after"]] == [146]))
    passed = all(ok for _, ok in checks)
    lines = [f"\n{name}: {'PASS' if passed else 'FAIL'}",
             (f"  {metadata.width}x{metadata.height}, {metadata.fps:g} FPS, "
              f"{metadata.frame_count} frames, {metadata.duration_seconds:g}s; pairs={len(entries)}; "
              f"initial_present={initial}"),
             "  Candidates and temporal results:"]
    for result in results:
        entry = by_index[result.candidate_pair_index]
        confirmation = (f"{result.confirmed_at_seconds:.1f}s"
                        if result.confirmed_at_seconds is not None else "none")
        lines.append(f"    pair {entry.pair_index}, {entry.time_before_seconds:.1f}-"
                     f"{entry.time_after_seconds:.1f}s {entry.candidate.kind.value}: "
                     f"{result.status.value}; confirmed_at={confirmation}; "
                     f"superseded_by={result.superseded_by_pair_index}")
    lines.append(f"  Confirmed events: {len(events)}; lifecycle anomalies: {len(anomalies)}")
    for event in events:
        lines.append(f"    {event.kind.value}: event_time={event.candidate_time_seconds:.1f}s, "
                     f"confirmed_at={event.confirmed_at_seconds:.1f}s")
    csv_rows = []
    for interval in intervals:
        duration = interval.end_time_seconds - interval.start_time_seconds
        lines.append(f"  Investigation {interval.investigation_index}: "
                     f"{interval.start_time_seconds:.1f}-{interval.end_time_seconds:.1f}s "
                     f"({duration:.1f}s), {interval.start_reason.value}->{interval.end_reason.value}")
        csv_rows.append({"video_name": path.name, "investigation_index": interval.investigation_index,
                         "start_time_seconds": interval.start_time_seconds,
                         "end_time_seconds": interval.end_time_seconds, "duration_seconds": duration,
                         "start_reason": interval.start_reason.value, "end_reason": interval.end_reason.value})
    lines.extend(f"  Anomaly: {anomaly}" for anomaly in anomalies)
    if name == "combined_interaction":
        lines.append("  Raw/effective presence around replacement (raw fractions unchanged):")
        for row in measured:
            if 14.4 <= row["time_b"] <= 14.9:
                lines.append(f"    {row['time_b']:.1f}s: raw={row['raw_present_after']}; "
                             f"effective={row['present_after']}; dark={row['after_bottom_actions_dark_fraction']}; "
                             f"edges={row['after_ticket_body_edge_fraction']}; light={row['after_ticket_body_light_fraction']}")
    lines.append("  Invariant/scenario checks:")
    lines.extend(f"    {'PASS' if ok else 'FAIL'}: {label}" for label, ok in checks)
    print(f"{name}: {'PASS' if passed else 'FAIL'}", flush=True)
    return passed, lines, csv_rows


def main():
    """Run closure validation, write concise artifacts, and return nonzero on FAIL."""
    for name in EXPECTED:
        path = ROOT / "data/videos" / f"{name}_test.mp4"
        if not path.is_file():
            raise FileNotFoundError(f"Required Phase 3 recording missing: {path}")
    lines = ["Phase 3 real-video closure validation",
             "Full existing production pipeline was executed; no thresholds/rules were tuned.",
             "Representative manual validation, not an accuracy benchmark.",
             f"Provisional offline confirmation window={CONFIRMATION_WINDOW_SECONDS}s.",
             "Event boundaries use candidate times; VIDEO_END uses full nominal metadata duration."]
    overall, rows = True, []
    for name, expected in EXPECTED.items():
        passed, section, intervals = _validate_video(name, expected)
        overall = overall and passed
        lines.extend(section)
        rows.extend(intervals)
    lines += [f"\nOVERALL: {'PASS' if overall else 'FAIL'}",
              ("Limits: four canonical-layout recordings, constant-FPS source timing, provisional "
               "visual thresholds and window; no held-out benchmark, identity, OCR, or Phase 4 extraction.")]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT / "phase3_validation_report.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    csv_path = OUTPUT / "phase3_validation_intervals.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["video_name", "investigation_index",
            "start_time_seconds", "end_time_seconds", "duration_seconds", "start_reason", "end_reason"])
        writer.writeheader()
        writer.writerows(rows)
    print("\n".join(lines))
    print(f"Report: {report_path}\nIntervals: {csv_path}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
