"""Phase 4.3b real-video validation of production stable-span annotations.

Reuse the shared hardened Phase 3 pipeline and production scroll interpretation,
lifecycle, InvestigationWindow, stable detector and unchanged adaptive planner.
No production logic is duplicated. All five recordings use actual lifecycle cores;
the isolated scroll video already has a real startup-active Investigation and
therefore needs no synthetic context. Only pairs wholly inside a core contribute
to detection; crossing pairs are excluded, never clipped. Both effective endpoints
must be present. Header/image diagnostics are not stable-condition inputs.

Write one report and CSV, no images. Preserve dense during-scroll evidence and
original semantic records. Report each span's requested timestamp coverage without
requiring a frame exactly at confirmation. A final None end is right-censored when
last observed pair precedes the known Investigation end; report observed duration
as a lower bound in that case. Constant source FPS/canonical geometry are assumed.
No OCR, ranking, best-frame selection, semantic deduplication or Phase 4.4/5 occurs.
"""

import csv
from dataclasses import asdict
from pathlib import Path

from evaluate_investigation_candidates import _measure_timeline

from events_diary.adaptive_sampling import (
    SamplingSignalEntry,
    build_adaptive_sampling_plan,
)
from events_diary.config import UIConfig
from events_diary.investigation_lifecycle import build_investigation_lifecycle
from events_diary.investigation_temporal import confirm_investigation_candidates
from events_diary.investigation_window import build_investigation_windows
from events_diary.ticket_scroll import analyze_ticket_scroll_sequence
from events_diary.ticket_stable_views import (
    TicketStableViewEntry,
    detect_ticket_stable_views,
)
from events_diary.video import read_video_metadata

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/debug/map_motion/ticket_stable_views"
SOURCES = ("ticket_scroll", "combined_interaction", "investigation_switch",
           "investigation_started_active", "investigation_close_reopen")


def _build_contexts(name, *, include_phase3=False):
    """Prepare upstream model objects once for shared Phase 4.3/4.4 validation.

    This existing real-video pipeline owns decoding and upstream API calls;
    downstream composition consumes only its windows, plans and stable views.
    Return metadata, lifecycle anomalies and immutable per-core context tuples.
    Closure validation can additionally request original Phase 3 records, without
    a second CV pass or changing the existing three-item return for other callers.
    """
    metadata = read_video_metadata(ROOT / "data/videos" / f"{name}_test.mp4")
    print(f"Input: {metadata.path}", flush=True)
    presences, signals, timeline = _measure_timeline(metadata.path, metadata, UIConfig())
    states = analyze_ticket_scroll_sequence([s.ticket_scroll_evidence for s in signals])
    events, confirmations = confirm_investigation_candidates(timeline)
    intervals, anomalies = build_investigation_lifecycle(initial_present=timeline[0].present_before,
        video_start_time_seconds=0, video_end_time_seconds=metadata.duration_seconds, confirmed_events=events)
    windows = build_investigation_windows(intervals, video_start_time_seconds=0,
                                           video_end_time_seconds=metadata.duration_seconds)
    contexts = []
    for interval, window in zip(intervals, windows):
        core = [e for e in timeline if interval.start_time_seconds <= e.time_before_seconds
                and e.time_after_seconds <= interval.end_time_seconds]
        entries = [TicketStableViewEntry(e.pair_index, e.time_before_seconds, e.time_after_seconds,
                   e.present_before and e.present_after, signals[e.pair_index].ticket_scroll_detected,
                   signals[e.pair_index].ticket_body_change, states[e.pair_index]) for e in core]
        snapshots = tuple(entries)
        selected = [e for e in timeline if window.window_start_time_seconds <= e.time_before_seconds
                    and e.time_after_seconds <= window.window_end_time_seconds]
        sampling_entries = [SamplingSignalEntry(e.pair_index, e.time_before_seconds, e.time_after_seconds,
                                               signals[e.pair_index]) for e in selected]
        plan_before = build_adaptive_sampling_plan(window=window, signal_entries=sampling_entries,
                                                   source_fps=metadata.fps)
        views = detect_ticket_stable_views(investigation_index=interval.investigation_index, entries=entries,
            investigation_start_time_seconds=interval.start_time_seconds,
            investigation_end_time_seconds=interval.end_time_seconds)
        plan_after = build_adaptive_sampling_plan(window=window, signal_entries=sampling_entries,
                                                  source_fps=metadata.fps)
        if tuple(entries) != snapshots or plan_before != plan_after:
            raise AssertionError("Stable annotations must not mutate inputs or adaptive sampling.")
        contexts.append((window, plan_before, views, core[-1].time_after_seconds if core else None))
    if include_phase3:
        return metadata, anomalies, tuple(contexts), {
            "raw_presence": tuple(presences), "signals": tuple(signals), "timeline": tuple(timeline),
            "events": tuple(events), "confirmations": tuple(confirmations), "intervals": tuple(intervals)}
    return metadata, anomalies, tuple(contexts)


def _evaluate(name):
    """Format the shared upstream contexts without altering Phase 4.3 diagnostics."""
    metadata, anomalies, contexts = _build_contexts(name)
    lines = [(f"\n{name}: {metadata.duration_seconds:g}s, {metadata.fps:g} FPS, {metadata.frame_count} frames, "
              f"{metadata.width}x{metadata.height}; actual lifecycle; anomalies={anomalies}")]
    rows = []
    for window, plan_before, views, observed_through in contexts:
        lines.append(f"  Investigation {window.investigation_index}: {window.investigation_start_time_seconds:g}-"
                     f"{window.investigation_end_time_seconds:g}s; views={len(views)}; last observed="
                     f"{observed_through}; sampling unchanged=True")
        for index, view in enumerate(views):
            observed_end = (view.stable_end_time_seconds if view.stable_end_time_seconds is not None
                            else observed_through)
            requests = [t for t in plan_before.requested_timestamps_seconds
                        if view.stable_start_time_seconds - 1e-9 <= t <= observed_end + 1e-9]
            dense = any(d.start_time_seconds - 1e-9 <= view.stable_confirmed_time_seconds <= d.end_time_seconds + 1e-9
                        for d in plan_before.dense_intervals)
            row = {"video_name": metadata.path.name, "stable_view_index": index, **asdict(view),
                   "stable_duration_seconds": (observed_end - view.stable_start_time_seconds
                        if view.stable_end_time_seconds is not None else None),
                   "observed_through_seconds": observed_end,
                   "observed_stable_duration_seconds": observed_end - view.stable_start_time_seconds,
                   "sampling_request_count_in_span": len(requests), "confirmation_in_dense": dense,
                   "confirmation_requested": any(abs(t - view.stable_confirmed_time_seconds) <= 1e-9 for t in requests)}
            row["prior_scroll_direction"] = view.prior_scroll_direction.value if view.prior_scroll_direction else None
            rows.append(row)
            lines.append("    " + str(row))
    return lines, rows


def main():
    """Write the five-source production annotation report without selecting frames."""
    lines = ["Phase 4.3b Production Stable Ticket Views",
             "Default dwell=0.5 seconds; effective presence AND no scroll AND body LOW.",
             "Confirmation=start+dwell after sufficient coverage, including between sampled timestamps.",
             "end=None means ongoing at end of observed input; observed duration is a lower bound.",
             "Prior scroll consumed by its first confirmed pause; short pauses do not split direction provenance.",
             "All sampling plans/entries are checked unchanged across stable detection."]
    rows = []
    for name in SOURCES:
        section, measured = _evaluate(name)
        lines.extend(section)
        rows.extend(measured)
    lines += ["\nA continuous stable span includes post-pan ticket stability without emitting duplicate views.",
              "Final spans are right-censored when analysis ends one source step before metadata/video end.",
              "Existing samples inside stable spans remain valid evidence even without an exact confirmation sample.",
              "No moving evidence removed, no OCR, ranking, content deduplication or Phase4.4/5."]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = OUTPUT / "ticket_stable_views_report.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (OUTPUT / "ticket_stable_views.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("\n".join(lines))
    print(f"Report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
