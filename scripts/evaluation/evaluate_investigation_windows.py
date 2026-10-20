"""Phase 4.1 fixed-context evaluation using validated real-video lifecycle output.

Reuse data/debug/map_motion/phase3_validation_intervals.csv from the successful
Phase 3 closure run. Reconstruct existing InvestigationInterval records, then call
the production build_investigation_windows API with requested 0.5/0.5 seconds.
No lifecycle rules are duplicated and no frames are decoded or signals analyzed.
This utility opens only existing CSV/report artifacts, never the video files.

The cached Phase 3 pipeline starts these recordings at 0 seconds and closes their
last active interval with VIDEO_END at full nominal metadata duration. Therefore
each final VIDEO_END interval supplies the recorded video end bound. If that
evidence or the successful closure report is absent, fail explicitly rather than
guessing duration or hardcoding example boundaries. This is validation against
recorded lifecycle output, not a fresh rerun of CV or a metadata reinspection.

Preserve interval order and all overlapping expanded windows. Report requested
context, derived actual clipped context, and clipping independently. Save only a
concise report and one CSV under investigation_windows, overwriting matching
outputs but preserving other artifacts. CSV values retain Python float precision;
normal binary floating-point arithmetic may affect the last decimal places.
No adaptive sampling, activity levels, evidence images, OCR, or Phase 4.2 logic
is introduced. Input/schema/production/I/O errors propagate.
"""

import csv
import sys
from pathlib import Path

# Allow direct execution and explicit imports across script directories.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from events_diary.investigation_lifecycle import (
    InvestigationEndReason,
    InvestigationInterval,
    InvestigationStartReason,
)
from events_diary.investigation_window import build_investigation_windows

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/debug/map_motion/phase3_validation_intervals.csv"
CLOSURE_REPORT = ROOT / "data/debug/map_motion/phase3_validation_report.txt"
OUTPUT = ROOT / "data/debug/map_motion/investigation_windows"
VIDEO_NAMES = ("investigation_switch_test.mp4", "investigation_started_active_test.mp4",
               "investigation_close_reopen_test.mp4")


def _read_lifecycle():
    """Load the existing validated intervals in source order, without video access.

    Require the closure PASS report and final VIDEO_END evidence for each source.
    Missing artifacts or malformed data raise explicitly. No substitute timeline
    or duration is invented; all core bounds are copied from the existing CSV.
    """
    if "OVERALL: PASS" not in CLOSURE_REPORT.read_text(encoding="utf-8"):
        raise ValueError("Phase 3 closure must PASS before reusing its lifecycle output.")
    investigations = {name: [] for name in VIDEO_NAMES}
    with SOURCE.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            name = row["video_name"]
            if name not in investigations:
                continue
            investigations[name].append(InvestigationInterval(
                int(row["investigation_index"]), float(row["start_time_seconds"]),
                float(row["end_time_seconds"]), InvestigationStartReason(row["start_reason"]),
                InvestigationEndReason(row["end_reason"])))
    for name, intervals in investigations.items():
        if not intervals or intervals[-1].end_reason is not InvestigationEndReason.VIDEO_END:
            raise ValueError(f"No recorded VIDEO_END bound for {name}; do not guess video duration.")
    return investigations


def main():
    """Build and report fixed windows for the three cached real-video lifecycles."""
    investigations = _read_lifecycle()
    lines = ["Phase 4.1 Investigation window evaluation",
             f"Source lifecycle CSV: {SOURCE}",
             "Uses existing successful Phase 3 real-video output; no video/frame decoding.",
             "Requested context: 0.5s before / 0.5s after (provisional). Overlaps stay separate."]
    rows = []
    for name, intervals in investigations.items():
        video_start, video_end = 0.0, intervals[-1].end_time_seconds
        windows = build_investigation_windows(
            intervals, video_start_time_seconds=video_start, video_end_time_seconds=video_end,
            context_before_seconds=0.5, context_after_seconds=0.5)
        lines.append(f"\n{name}: recorded video bounds={video_start!r}->{video_end!r}s")
        for window in windows:
            start, end = (window.investigation_start_time_seconds,
                          window.investigation_end_time_seconds)
            window_start, window_end = window.window_start_time_seconds, window.window_end_time_seconds
            actual_before, actual_after = start - window_start, window_end - end
            clipped_before = start - window.context_before_seconds < video_start
            clipped_after = end + window.context_after_seconds > video_end
            rows.append({
                "video_name": name, "investigation_index": window.investigation_index,
                "investigation_start_time_seconds": start, "investigation_end_time_seconds": end,
                "window_start_time_seconds": window_start, "window_end_time_seconds": window_end,
                "requested_context_before_seconds": window.context_before_seconds,
                "requested_context_after_seconds": window.context_after_seconds,
                "actual_context_before_seconds": actual_before,
                "actual_context_after_seconds": actual_after,
                "clipped_before": clipped_before, "clipped_after": clipped_after,
            })
            lines.append(f"  Investigation {window.investigation_index}: core={start!r}->{end!r}s; "
                         f"window={window_start!r}->{window_end!r}s; "
                         f"requested={window.context_before_seconds!r}/{window.context_after_seconds!r}s; "
                         f"actual={actual_before!r}/{actual_after!r}s; "
                         f"clipped_before={clipped_before}, clipped_after={clipped_after}")
    lines.append("\nOnly temporal windows were constructed. No merging, frames, signals, or Phase 4.2 logic.")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT / "investigation_windows.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report_path = OUTPUT / "investigation_windows_report.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"CSV: {csv_path}\nReport: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
