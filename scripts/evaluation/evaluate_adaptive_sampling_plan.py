"""Phase 4.2b real-video evaluation of the production request planner.

Decode seven sources consecutively with existing validation; compute production
presence, InvestigationSignals, candidates, temporal confirmation, and lifecycle.
The Investigation and combined recordings use actual lifecycle-derived Phase 4.1 windows.
Pan/zoom/scroll use explicitly labeled evaluation-only full-video windows, regardless
of their incidental lifecycle results, to test isolated sampling behavior. No
synthetic-window fallback exists in production. No algorithms/thresholds are tuned.

SamplingSignalEntry retains original already-computed InvestigationSignals. Each
plan receives only pairs entirely within its expanded window; crossing pairs are
reported as excluded, not clipped or relabeled. The planner never decodes pixels.
Report reasons, merge counts, duration, anchors, requests, and reduction versus
nominal source-frame density plus a fair full-rate request plan with the same anchors.
Requested times may resolve to duplicate physical frames later, especially at
off-grid anchors/endpoints; this evaluator does not seek/select/save evidence.

Outputs are one dense-interval CSV and one report under adaptive_sampling_plan.
Source index/FPS timing assumes constant FPS and canonical geometry. Full metadata
duration bounds lifecycle/window end, while observed pairs end at the last sample.
No scroll-stop retention, per-frame evidence metadata, Phase 4.3/4.4/5, OCR, VLM,
or route geometry is implemented. Existing artifacts outside this output are preserved.
"""

import csv
import math
import sys
from collections import Counter
from pathlib import Path

# Allow direct execution and explicit imports across script directories.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from events_diary.adaptive_sampling import (
    DENSE_REASON_ORDER,
    SamplingSignalEntry,
    build_adaptive_sampling_plan,
)
from events_diary.config import UIConfig
from events_diary.investigation_lifecycle import build_investigation_lifecycle
from events_diary.investigation_signals import (
    SignalStrength,
)
from events_diary.investigation_temporal import (
    confirm_investigation_candidates,
)
from events_diary.investigation_window import (
    InvestigationWindow,
    build_investigation_windows,
)
from events_diary.video import read_video_metadata
from scripts.evaluation.evaluate_investigation_candidates import _measure_timeline

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/debug/map_motion/adaptive_sampling_plan"
STEMS = ("investigation_switch", "investigation_started_active", "investigation_close_reopen",
         "ticket_scroll", "map_pan", "map_zoom", "combined_interaction")


def _measure(metadata, config):
    """Reuse Phase 3's raw/effective presence pipeline and original pair signals."""
    _, measured, candidates = _measure_timeline(metadata.path, metadata, config)
    signals = [SamplingSignalEntry(entry.pair_index, entry.time_before_seconds,
                                   entry.time_after_seconds, observations)
               for entry, observations in zip(candidates, measured)]
    return signals, candidates


def _pair_reasons(entry, source_fps):
    """Obtain a pair's default trigger reasons from the production planner itself.

    Zero padding avoids needing a duplicate reason-classifier in this evaluator;
    testing each pair in its own zero-context window provides the pre-merge count.
    """
    window = InvestigationWindow(0, entry.time_before_seconds, entry.time_after_seconds,
                                  entry.time_before_seconds, entry.time_after_seconds, 0, 0)
    plan = build_adaptive_sampling_plan(window=window, signal_entries=[entry], source_fps=source_fps,
                                      pre_padding_seconds=0, post_padding_seconds=0)
    return plan.dense_intervals[0].reasons if plan.dense_intervals else ()


def _combined_diagnostics(entries, reasons):
    """Report every observed pair, including quiet gaps, without reclassifying signals.

    These development diagnostics expose trigger timing, ticket strengths and map
    magnitudes so pauses and cross-talk can be inspected in the existing report.
    No new evidence artifact or production decision is created.
    """
    lines = ["  Combined source timeline (un-padded pair bounds; '-' means no dense trigger):"]
    for entry in entries:
        signals = entry.signals
        motion = signals.map_motion
        magnitude = (math.hypot(motion.dx, motion.dy) if motion.valid
                     and motion.dx is not None and motion.dy is not None else None)
        lines.append(f"    {entry.time_before_seconds:.3f}-{entry.time_after_seconds:.3f}s: "
                     f"reasons={','.join(reasons[entry.pair_index]) or '-'}; "
                     f"ticket H/I/B={signals.header_change.strength.value}/"
                     f"{signals.aircraft_image_change.strength.value}/{signals.ticket_body_change.strength.value}; "
                     f"translation_px={magnitude}; scale={motion.scale}; map_valid={motion.valid}")
    return lines


def _evaluate_video(stem):
    """Run production lifecycle/windows/planner and return report lines and dense rows."""
    path = ROOT / "data/videos" / f"{stem}_test.mp4"
    if not path.is_file():
        return [f"\nMISSING: {path}"], []
    print(f"Input: {path}", flush=True)
    metadata = read_video_metadata(path)
    signals, candidates = _measure(metadata, UIConfig())
    confirmed, statuses = confirm_investigation_candidates(candidates)
    intervals, anomalies = build_investigation_lifecycle(initial_present=candidates[0].present_before,
        video_start_time_seconds=0, video_end_time_seconds=metadata.duration_seconds, confirmed_events=confirmed)
    synthetic = stem in ("ticket_scroll", "map_pan", "map_zoom")
    windows = ((InvestigationWindow(0, 0, metadata.duration_seconds, 0, metadata.duration_seconds, 0, 0),)
               if synthetic else build_investigation_windows(intervals,
               video_start_time_seconds=0, video_end_time_seconds=metadata.duration_seconds))
    lines = [(f"\n{path.name}: {metadata.width}x{metadata.height}, {metadata.fps:g} FPS, "
             f"{metadata.duration_seconds:g}s; frames={metadata.frame_count}; measured pairs={len(signals)}"),
             (f"Phase 3 intervals={len(intervals)}, anomalies={anomalies}; "
             f"confirmation statuses={dict(Counter(r.status.value for r in statuses))}"),
             "Window provenance: " + ("EVALUATION-ONLY synthetic full-video window" if synthetic
                                       else "actual production lifecycle + InvestigationWindow")]
    for interval_index, interval in enumerate(intervals):
        lines.append(f"  Lifecycle #{interval_index}: {interval.start_time_seconds:g}-"
                     f"{interval.end_time_seconds:g}s; start_reason={interval.start_reason.value}; "
                     f"end_reason={interval.end_reason.value}")
    reasons = {entry.pair_index: _pair_reasons(entry, metadata.fps) for entry in signals}
    counts = Counter(reason for values in reasons.values() for reason in values)
    lines.append(f"  Whole-video trigger counts: { {reason: counts[reason] for reason in DENSE_REASON_ORDER} }")
    rows = []
    for window_index, window in enumerate(windows):
        selected = [entry for entry in signals if window.window_start_time_seconds <= entry.time_before_seconds
                    and entry.time_after_seconds <= window.window_end_time_seconds]
        selected_indexes = {entry.pair_index for entry in selected}
        crossing = sum(entry.time_before_seconds < window.window_end_time_seconds
                       and entry.time_after_seconds > window.window_start_time_seconds
                       and entry.pair_index not in selected_indexes for entry in signals)
        plan = build_adaptive_sampling_plan(window=window, signal_entries=selected, source_fps=metadata.fps)
        baseline = build_adaptive_sampling_plan(window=window, signal_entries=[], source_fps=metadata.fps,
                                                sparse_sampling_fps=metadata.fps)
        duration = window.window_end_time_seconds - window.window_start_time_seconds
        dense_duration = sum(d.end_time_seconds - d.start_time_seconds for d in plan.dense_intervals)
        nominal_source = max(1, math.ceil(duration * metadata.fps - 1e-9))
        count = len(plan.requested_timestamps_seconds)
        reduction = 1 - count / len(baseline.requested_timestamps_seconds)
        before_merge = sum(bool(reasons[entry.pair_index]) for entry in selected)
        anchors = (window.window_start_time_seconds, window.investigation_start_time_seconds,
                   window.investigation_end_time_seconds, window.window_end_time_seconds)
        lines += [(f"  Window {window_index}: {window.window_start_time_seconds!r}-"
                  f"{window.window_end_time_seconds!r}s; core={window.investigation_start_time_seconds!r}-"
                  f"{window.investigation_end_time_seconds!r}s; pairs={len(selected)}; crossing pairs excluded={crossing}"),
                  (f"    dense={len(plan.dense_intervals)} intervals from {before_merge} triggering pairs; "
                  f"overlap/touch merging={before_merge > len(plan.dense_intervals)}"),
                  (f"    dense_duration={dense_duration:.6g}s; sparse_duration={duration - dense_duration:.6g}s; "
                  f"requested={count}; nominal full-source frames~{nominal_source}; "
                  f"full-rate requests with same anchors={len(baseline.requested_timestamps_seconds)}; "
                  f"request reduction={reduction:.2%}; nominal-frame comparison={1 - count / nominal_source:.2%}"),
                  f"    exact anchors={anchors}; preserved={set(anchors).issubset(plan.requested_timestamps_seconds)}"]
        for dense_index, dense in enumerate(plan.dense_intervals):
            lines.append(f"    dense #{dense_index}: {dense.start_time_seconds!r}-{dense.end_time_seconds!r}s; "
                         f"duration={dense.end_time_seconds - dense.start_time_seconds:.6g}s; reasons={dense.reasons}")
            rows.append({"video_name": path.name, "window_index": window_index,
                         "dense_interval_index": dense_index, "start_time_seconds": dense.start_time_seconds,
                         "end_time_seconds": dense.end_time_seconds,
                         "duration_seconds": dense.end_time_seconds - dense.start_time_seconds,
                         "reasons": ";".join(dense.reasons)})
            if stem == "combined_interaction":
                contributors = [entry for entry in selected if reasons[entry.pair_index]
                    and max(window.window_start_time_seconds,
                            entry.time_before_seconds - plan.pre_padding_seconds) <= dense.end_time_seconds
                    and min(window.window_end_time_seconds,
                            entry.time_after_seconds + plan.post_padding_seconds) >= dense.start_time_seconds]
                for reason in dense.reasons:
                    contributing = [entry for entry in contributors if reason in reasons[entry.pair_index]]
                    lines.append(f"      {reason}: {len(contributing)} pairs; first="
                                 f"{contributing[0].time_before_seconds:g}-{contributing[0].time_after_seconds:g}s; "
                                 f"last={contributing[-1].time_before_seconds:g}-{contributing[-1].time_after_seconds:g}s; "
                                 f"padded tail={min(window.window_end_time_seconds, contributing[-1].time_after_seconds + plan.post_padding_seconds):g}s")
        if stem == "combined_interaction":
            counts = Counter(reason for entry in selected for reason in reasons[entry.pair_index])
            lines.append(f"    Window trigger counts: { {reason: counts[reason] for reason in DENSE_REASON_ORDER} }")
            lines.append(f"    Requested timestamps seconds={plan.requested_timestamps_seconds}")
        image_only = sum(entry.signals.aircraft_image_change.strength in (SignalStrength.MEDIUM, SignalStrength.HIGH)
                         and entry.signals.header_change.strength is SignalStrength.LOW
                         and entry.signals.ticket_body_change.strength is SignalStrength.LOW
                         for entry in selected)
        route_positive = sum(entry.signals.route_selection_change for entry in selected)
        lines.append(f"    contextual trigger counts: route_visual_change={route_positive}; image-only={image_only}. "
                     "Route can cross-trigger during map interaction; hover/image changes can add dense coverage.")
    if stem == "investigation_close_reopen":
        lines.append("Lifecycle-derived windows omit most of the inactive 3.8-7.6s gap. "
                     "Its padded closing/reopening context can be dense; it is not a full-video sparse plan. "
                     "The superseded accidental OPEN's trailing visual activity can be sampled as context.")
    if stem == "combined_interaction":
        lines.extend(_combined_diagnostics(signals, reasons))
    return lines, rows


def main():
    """Evaluate all seven videos and write planner diagnostics without tuning."""
    lines = ["Phase 4.2b Adaptive Sampling Planner real evaluation",
             "Provisional defaults: pre/post=.5s; sparse=1 FPS; dense=source FPS; map >1px / >.01 scale delta.",
             "Requested timestamps are not unique decoded frames. Full-rate comparison retains identical anchors."]
    rows = []
    for stem in STEMS:
        section, dense_rows = _evaluate_video(stem)
        lines.extend(section)
        rows.extend(dense_rows)
    lines += [("\nCross-talk/limits: route_visual_change can keep map windows dense beyond large-motion fits; "
              "image hover changes add density without new ticket identity. No calibrated parameters were tuned. "
              "Grid phase resets, interval endpoints, and off-grid anchors may add requests compared with "
              "nominal source-frame counts. Dense plans need actual frame-resolution/dedup evaluation later."),
              "Combined recording uses real lifecycle windows; its un-padded signal timeline exposes pauses and cross-talk.",
              "No Phase 4.3 scroll stops, Phase 4.4 evidence metadata, Phase 5 selection, OCR, or VLM."]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT / "adaptive_sampling_plan_report.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (OUTPUT / "adaptive_sampling_plan_intervals.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["video_name", "window_index", "dense_interval_index",
            "start_time_seconds", "end_time_seconds", "duration_seconds", "reasons"])
        writer.writeheader()
        writer.writerows(rows)
    print("\n".join(lines))
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
