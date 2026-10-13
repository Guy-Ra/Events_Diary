"""Phase 3.4 manual validation of the complete offline Investigation pipeline.

Reuse the existing sequential candidate evaluator and temporal entry adapter:
video -> initial single-frame presence -> pair signals/candidates -> confirmed
events -> production lifecycle intervals. No CV, candidate, confirmation, or
lifecycle algorithms are duplicated. Superseded candidates never enter lifecycle.
Startup-active initialization uses the actual first-frame presence, not a synthetic
OPEN. Interval boundaries use candidate times, not later confirmation timestamps.

Evaluate the three existing recordings in their canonical UIConfig layout. Video
start is 0 seconds; video end uses metadata.duration_seconds (full nominal source
duration), while temporal coverage ends at the last sampled timestamp. This
distinction is explicit in the report. Source timing assumes constant FPS.

Write one CSV row per interval under investigation_lifecycle, plus a report of
metadata, initial presence, confirmed events, durations/reasons, and anomalies.
Matching outputs overwrite on rerun; unrelated artifacts are preserved. Errors
propagate; no thresholds, semantic rules, or events are silently corrected. This
is validation on three manual recordings, not a benchmark or identity extraction.
No OCR, frame selection, Phase 4, final JSON, or online processing is introduced.
"""

import csv
from dataclasses import asdict
from pathlib import Path

from evaluate_investigation_candidates import SOURCES, _evaluate
from evaluate_investigation_temporal import _entries

from events_diary.config import UIConfig
from events_diary.investigation_lifecycle import build_investigation_lifecycle
from events_diary.investigation_temporal import confirm_investigation_candidates
from events_diary.video import read_video_metadata

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/debug/map_motion/investigation_lifecycle"


def _evaluate_video(name, config):
    """Run existing production stages, write interval CSV, and return report lines.

    The candidate evaluator already measures first-frame presence and preserves
    it in its first row. No second interpretation or invented startup event occurs.
    confirmed_at metadata remains visible alongside candidate boundary times.
    """
    path = ROOT / "data/videos" / f"{name}_test.mp4"
    print(f"Input: {path}", flush=True)
    metadata = read_video_metadata(path)
    measured = _evaluate(path, metadata, config)
    entries = _entries(measured)
    events, results = confirm_investigation_candidates(entries)
    initial_present = entries[0].present_before
    intervals, anomalies = build_investigation_lifecycle(
        initial_present=initial_present, video_start_time_seconds=0,
        video_end_time_seconds=metadata.duration_seconds, confirmed_events=events)
    rows = []
    for interval in intervals:
        row = asdict(interval)
        row.update(start_reason=interval.start_reason.value, end_reason=interval.end_reason.value,
                   duration_seconds=interval.end_time_seconds - interval.start_time_seconds)
        rows.append(row)
    csv_path = OUTPUT / f"{name}_lifecycle.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["investigation_index", "start_time_seconds",
            "end_time_seconds", "duration_seconds", "start_reason", "end_reason"])
        writer.writeheader()
        writer.writerows(rows)
    lines = [f"\n{name}: {path}",
             (f"{metadata.width}x{metadata.height}; fps={metadata.fps}; frames={metadata.frame_count}; "
              f"duration={metadata.duration_seconds}s; analyzed pairs={len(entries)}"),
             (f"initial_present={initial_present}; video_start=0; video_end={metadata.duration_seconds}s; "
              f"last sampled time={entries[-1].time_after_seconds}s"),
             "Confirmed event timeline (candidate time is lifecycle boundary):"]
    for event in events:
        lines.append(f"  {event.kind.value}, pair={event.candidate_pair_index}; "
                     f"candidate_time={event.candidate_time_seconds:.1f}s; "
                     f"confirmed_at={event.confirmed_at_seconds:.1f}s")
    lines.append(f"Intervals: {len(intervals)}")
    for interval in intervals:
        lines.append(f"  Investigation {interval.investigation_index}: "
                     f"{interval.start_time_seconds:.1f}-{interval.end_time_seconds:.1f}s; "
                     f"duration={interval.end_time_seconds - interval.start_time_seconds:.1f}s; "
                     f"{interval.start_reason.value}->{interval.end_reason.value}")
    lines.append(f"Anomalies: {len(anomalies)}")
    for anomaly in anomalies:
        lines.append(f"  event_index={anomaly.event_index}; kind={anomaly.event_kind.value}; "
                     f"time={anomaly.event_time_seconds}; state_before={anomaly.state_before.value}; "
                     f"reason={anomaly.reason}")
    excluded = [result for result in results if result.confirmed_at_seconds is None]
    lines.append("Unconfirmed candidates excluded from lifecycle: " + (", ".join(
        f"{r.candidate_kind.value} at {r.candidate_time_seconds:.1f}s ({r.status.value})"
        for r in excluded) or "none"))
    lines.append("Boundaries use candidate_time_seconds and nominal video duration, "
                 "never confirmed_at_seconds. No synthetic startup OPEN.")
    print(f"CSV: {csv_path}", flush=True)
    return lines


def main():
    """Write three lifecycle CSVs and one report without changing production rules."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lines = ["Phase 3.4 offline Investigation lifecycle validation",
             "Lifecycle consumes confirmed events only; presence initializes the first state."]
    for name in SOURCES:
        lines.extend(_evaluate_video(name, UIConfig()))
    lines += [("\nLimitations: three manual recordings, provisional upstream evidence/confirmation, "
               "canonical layout and constant FPS assumed. Intervals describe semantic lifecycle, "
               "not aircraft identity. NO + SWITCH is valid; ACTIVE + OPEN replaces an interval. "
               "Equal timestamps may produce zero-duration intervals. No corrections, OCR, "
               "frame extraction, Phase 4 content processing, or online state machine is implemented.")]
    report = "\n".join(lines) + "\n"
    report_path = OUTPUT / "investigation_lifecycle_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
