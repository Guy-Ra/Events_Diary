"""Phase 4.2a empirical trigger/padding/rate calibration, not an adaptive sampler.

Sequentially measure every pair in six existing manual recordings through the
production analyze_investigation_pair API. Retain existing ticket strengths/CPR,
route and scroll decisions, and map validity/transform/support diagnostics. No CV
logic is duplicated and no production thresholds or models are changed.

The evaluation-only 1px translation and 0.01 scale-delta probes were chosen AFTER
inspection of prior Phase 2/3 measurements: controls had subpixel translation and
near-unit scale, while deliberate motion was much larger. Fresh distributions and
control activations are reported to audit those proposals; they are not automatic
optimizations or intent labels. Zoom also produces translation in a B-to-A affine
transform, so map reasons may overlap. Rejected fits remain unavailable, not quiet.
Generic map residual never triggers. MEDIUM/HIGH are explicit meaningful labels;
scroll context is not a veto and no ActivityLevel abstraction is introduced.

Report contiguous observed trigger runs and manually annotated event neighborhoods
without implementing padded dense intervals or merging. Recommend provisional
0.5/0.5 seconds, sparse 1 FPS, and dense source FPS only with stated limitations.
Compare candidate rates using arithmetic sample-count estimates and phase-dependent
coverage bounds. No requested timestamps, selected frames, AdaptiveSamplingPlan,
scroll stops, or evidence timeline are created. No images are dumped.

The full source videos are analyzed, including closed-ticket context; triggers are
not gated by Investigation presence. Cached Phase 4.1 window CSV supplies measured
window durations for rate estimates when available. Missing videos/windows are
reported and available sources continue. Decode/layout/production errors propagate.
Paths resolve from the repository; only matching report/CSV outputs overwrite.
Times assume constant source FPS and canonical layout. This small isolated set
supports provisional engineering recommendations, not broad accuracy claims.
The optional --report-only mode reuses this script's measured CSV for analysis;
it verifies source count/index/FPS coverage but does not prove pixel-level freshness.
"""

import argparse
import csv
import math
from collections import Counter
from pathlib import Path

import numpy as np
from validate_phase2 import _sequential_frames

from events_diary.config import UIConfig
from events_diary.investigation_signals import (
    SignalStrength,
    analyze_investigation_pair,
)
from events_diary.video import read_video_metadata

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/debug/map_motion/adaptive_sampling_calibration"
VIDEO_STEMS = ("investigation_switch", "investigation_started_active", "investigation_close_reopen",
               "ticket_scroll", "map_pan", "map_zoom")
# Evaluation probes supported by prior measured control/motion separation, not production defaults.
TRANSLATION_PROBE_PX = 1.0
SCALE_DELTA_PROBE = 0.01
REASONS = ("ticket_scroll", "route_selection_change", "header_change", "aircraft_image_change",
           "ticket_body_change", "map_translation", "map_scale_change")
# Manually known neighborhoods: report annotations only, never signal classification.
LANDMARKS = {
    "investigation_switch": (("initial opening/loading", 1.6, 1.2, 1.8),
                             ("switch", 4.9, 4.6, 5.1),
                             ("ordinary traffic", 7.0, 6.8, 7.2)),
    "investigation_started_active": (("switch/loading", 2.4, 2.0, 2.7),),
    "investigation_close_reopen": (("close/recovery", 3.8, 3.6, 4.4),
                                   ("accidental opening", 7.1, 6.6, 7.3),
                                   ("final replacement", 7.6, 7.3, 7.9)),
    "ticket_scroll": (("upward scroll", None, 0.6, 2.8),
                      ("downward scroll", None, 4.3, 6.1)),
}


def _measure_video(stem, metadata):
    """Return existing pair signals and evaluation-only trigger reasons at source density."""
    rows, previous = [], None
    config = UIConfig()
    for index, frame in _sequential_frames(metadata.path, metadata, config):
        if previous is not None:
            signals = analyze_investigation_pair(previous, frame, config)
            row = {"video_name": metadata.path.name, "pair_index": index - 1,
                   "time_before_seconds": (index - 1) / metadata.fps,
                   "time_after_seconds": index / metadata.fps}
            reasons = []
            for name, reason, signal in (
                ("header", "header_change", signals.header_change),
                ("image", "aircraft_image_change", signals.aircraft_image_change),
                ("body", "ticket_body_change", signals.ticket_body_change),
                ("map_residual", None, signals.map_residual_change),
            ):
                row[f"{name}_strength"] = signal.strength.value
                row[f"{name}_cpr"] = signal.cpr
                if reason and signal.strength in (SignalStrength.MEDIUM, SignalStrength.HIGH):
                    reasons.append(reason)
            route, scroll, motion = (signals.route_selection_evidence,
                                     signals.ticket_scroll_evidence, signals.map_motion)
            translation = (math.hypot(motion.dx, motion.dy)
                           if motion.dx is not None and motion.dy is not None else None)
            delta = abs(motion.scale - 1) if motion.scale is not None else None
            row.update(route_selection_change=signals.route_selection_change,
                       route_evidence_valid=route.valid, route_evidence_reason=route.reason,
                       route_added_fraction=route.added_fraction,
                       route_removed_fraction=route.removed_fraction,
                       route_changed_fraction=route.changed_fraction,
                       ticket_scroll_detected=signals.ticket_scroll_detected,
                       scroll_dx=scroll.dx, scroll_dy=scroll.dy, scroll_response=scroll.motion_response,
                       map_motion_valid=motion.valid, map_motion_reason=motion.reason,
                       map_dx=motion.dx, map_dy=motion.dy,
                       map_translation_magnitude=translation, map_scale=motion.scale,
                       map_scale_delta=delta, map_rotation_degrees=motion.rotation_degrees,
                       map_keypoints_a=motion.keypoints_a, map_keypoints_b=motion.keypoints_b,
                       map_good_matches=motion.good_matches, map_inliers=motion.inliers,
                       map_inlier_ratio=motion.inlier_ratio)
            if signals.ticket_scroll_detected:
                reasons.append("ticket_scroll")
            if signals.route_selection_change:
                reasons.append("route_selection_change")
            if motion.valid and translation is not None and translation > TRANSLATION_PROBE_PX:
                reasons.append("map_translation")
            if motion.valid and delta is not None and delta > SCALE_DELTA_PROBE:
                reasons.append("map_scale_change")
            row["trigger_reasons"] = ";".join(reason for reason in REASONS if reason in reasons)
            rows.append(row)
            if index % 25 == 0:
                print(f"  {stem}: {index}/{metadata.frame_count - 1} pairs", flush=True)
        previous = frame
    return rows


def _stats(rows, field):
    """Summarize finite numeric observations using linear-interpolated percentiles."""
    values = [row[field] for row in rows if row[field] is not None and np.isfinite(row[field])]
    if not values:
        return "available=0"
    return f"n={len(values)}; " + ", ".join(
        f"{label}={value:.9g}" for label, value in zip(
            ("min", "median", "p90", "p95", "max"), np.percentile(values, [0, 50, 90, 95, 100])))


def _runs(rows, reason=None):
    """Describe contiguous observed active pairs, without padding or merging gaps.

    Active means an existing reason or any reason. Runs are descriptive summaries,
    not production dense intervals; start/end retain the full affected pair span.
    """
    runs, start = [], None
    for index in range(len(rows) + 1):
        active = index < len(rows) and (bool(rows[index]["trigger_reasons"]) if reason is None
                  else reason in rows[index]["trigger_reasons"].split(";"))
        if active and start is None:
            start = index
        if not active and start is not None:
            runs.append((rows[start]["time_before_seconds"], rows[index - 1]["time_after_seconds"],
                         index - start))
            start = None
    return runs


def _rate_comparison(lines, metadata, rows):
    """Estimate time-based observation counts only; never generate sampling timestamps."""
    rates = (1, 2, 4, 8, metadata.fps)
    lines.append("  Sampling comparison (counts approximate, capped by source FPS):")
    for rate in dict.fromkeys(rates):
        effective = min(rate, metadata.fps)
        lines.append(f"    requested={rate:g} FPS; effective={effective:g}; "
                     f"whole-video observations~{math.ceil(metadata.duration_seconds * effective)}; "
                     f"observations per 0.1s change~{0.1 * effective:g}; "
                     f"per 0.5s accidental-ticket interval~{0.5 * effective:g}")
    runs = _runs(rows)
    if runs:
        lines.append("  Observed affected-run duration: " + _stats(
            [{"duration": end - start} for start, end, _ in runs], "duration"))
        for rate in dict.fromkeys((4, 8, metadata.fps)):
            effective = min(rate, metadata.fps)
            lower = sum(math.floor((end - start) * effective + 1e-9) for start, end, _ in runs)
            upper = sum(math.ceil((end - start) * effective - 1e-9) + 1 for start, end, _ in runs)
            lines.append(f"    {effective:g} FPS phase-dependent observations across affected runs "
                         f"approximately {lower}..{upper}; endpoints/alignment matter")


def _window_counts(lines, metadata):
    """Compare sparse/dense counts in existing real Investigation cores/windows if available."""
    path = ROOT / "data/debug/map_motion/investigation_windows/investigation_windows.csv"
    if not path.is_file():
        lines.append("\nWindow-rate estimates unavailable: Phase 4.1 CSV missing.")
        return
    lines.append("\nRates in measured Phase 4.1 Investigation cores/windows (no timestamps generated):")
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            stem = row["video_name"].removesuffix("_test.mp4")
            if stem not in metadata:
                continue
            core = float(row["investigation_end_time_seconds"]) - float(row["investigation_start_time_seconds"])
            window = float(row["window_end_time_seconds"]) - float(row["window_start_time_seconds"])
            counts = []
            for rate in dict.fromkeys((1, 2, 4, 8, metadata[stem].fps)):
                effective = min(rate, metadata[stem].fps)
                counts.append(f"{rate:g}FPS core~{math.floor(core * effective) + 1}/"
                              f"window~{math.floor(window * effective) + 1}")
            lines.append(f"  {row['video_name']} #{row['investigation_index']}: " + "; ".join(counts))


def _load_timeline(metadata):
    """Reload this utility's measured CSV for report-only analysis, validating source coverage.

    This explicit mode does not rerun CV. Count/index/FPS checks reject a mismatched
    cached timeline; matching metadata alone cannot prove unchanged video pixels.
    """
    path = OUTPUT / "adaptive_sampling_signal_timeline.csv"
    timelines = {stem: [] for stem in metadata}
    strings = {"video_name", "header_strength", "image_strength", "body_strength",
               "map_residual_strength", "route_evidence_reason", "map_motion_reason", "trigger_reasons"}
    booleans = {"route_selection_change", "route_evidence_valid", "ticket_scroll_detected", "map_motion_valid"}
    integers = {"pair_index", "map_keypoints_a", "map_keypoints_b", "map_good_matches", "map_inliers"}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            stem = row["video_name"].removesuffix("_test.mp4")
            if stem not in timelines:
                continue
            for key, value in row.items():
                if key in strings:
                    continue
                if key in booleans:
                    if value not in ("True", "False"):
                        raise ValueError(f"Invalid boolean {key} in cached timeline.")
                    row[key] = value == "True"
                else:
                    row[key] = None if not value else (int(value) if key in integers else float(value))
            timelines[stem].append(row)
    for stem, rows in timelines.items():
        meta = metadata[stem]
        if len(rows) != meta.frame_count - 1 or any(
            row["pair_index"] != index
            or not math.isclose(row["time_before_seconds"], index / meta.fps, abs_tol=1e-9)
            or not math.isclose(row["time_after_seconds"], (index + 1) / meta.fps, abs_tol=1e-9)
            for index, row in enumerate(rows)
        ):
            raise ValueError(f"Cached observations do not match {stem}'s source coverage.")
    return timelines


def _report(timelines, metadata, missing):
    """Separate measured observations from provisional engineering recommendations.

    Include every occurrence via the compact CSV's pair/reasons/measured-value
    fields. Text summarizes counts/runs, map controls, annotated padding regions,
    and rate estimates. No automatic optimization or production plan is computed.
    """
    lines = ["Phase 4.2a trigger and sampling calibration",
             "MEASURED OBSERVATIONS (full existing signal API at source-frame density)",
             f"Missing videos: {missing or 'none'}",
             ("Every trigger occurrence is in adaptive_sampling_signal_timeline.csv: video, pair index, "
             "pair interval, trigger_reasons, corresponding CPR/boolean/motion values.")]
    controls, route_cross_talk = [], []
    for stem, rows in timelines.items():
        meta = metadata[stem]
        valid = [r for r in rows if r["map_motion_valid"]]
        if stem not in ("map_pan", "map_zoom"):
            controls.extend(valid)
        lines += [(f"\n{meta.path.name}: {meta.width}x{meta.height}, FPS={meta.fps:g}, "
                  f"duration={meta.duration_seconds:g}s, frames={meta.frame_count}, pairs={len(rows)}"),
                  f"  Map valid={len(valid)}, invalid={len(rows) - len(valid)}",
                  f"  Valid translation: {_stats(valid, 'map_translation_magnitude')}",
                  f"  Valid abs(scale-1): {_stats(valid, 'map_scale_delta')}",
                  f"  Valid abs(dx): {_stats([{'value': abs(r['map_dx'])} for r in valid], 'value')}",
                  f"  Valid abs(dy): {_stats([{'value': abs(r['map_dy'])} for r in valid], 'value')}"]
        invalid = Counter(r["map_motion_reason"] for r in rows if not r["map_motion_valid"])
        lines.append(f"  Rejected map reasons: {dict(invalid)}")
        for label in ("header", "image", "body", "map_residual"):
            lines.append(f"  {label}: strengths={dict(Counter(r[label + '_strength'] for r in rows))}; "
                         f"CPR {_stats(rows, label + '_cpr')}")
        for reason in REASONS:
            active = [r for r in rows if reason in r["trigger_reasons"].split(";")]
            lines.append(f"  Trigger {reason}: {len(active)}/{len(rows)}; observed runs="
                         + ", ".join(f"{a:.1f}-{b:.1f}s ({n} pairs)" for a, b, n in _runs(rows, reason)))
        if stem in ("map_pan", "map_zoom"):
            no_motion_routes = [r for r in rows if r["route_selection_change"]
                                and "map_translation" not in r["trigger_reasons"].split(";")
                                and "map_scale_change" not in r["trigger_reasons"].split(";")]
            lines.append(f"  Route cross-talk: {len(no_motion_routes)} route-positive pairs without "
                         "either meaningful-map probe. These are not proven selection changes.")
            route_count = sum(r["route_selection_change"] for r in rows)
            route_cross_talk.append(f"{stem}: {route_count} route positives, "
                                    f"{len(no_motion_routes)} without either map probe")
            for reason, field in (("map_translation", "map_translation_magnitude"),
                                  ("map_scale_change", "map_scale_delta")):
                active = [r for r in valid if reason in r["trigger_reasons"].split(";")]
                lines.append(f"  Probe-active {field}: {_stats(active, field)}")
        if stem == "investigation_close_reopen":
            lines.append("  Image-only triggers at 9.1-9.2s (CPR 0.0722474) and 9.5-9.6s "
                         "(CPR 0.0677910) occur outside the selection landmarks. Visual crop "
                         "inspection shows the same photograph with cursor/hover rendering changes, "
                         "not a new aircraft ticket. Sampling them is explainable but redundant.")
        lines.append(f"  Any trigger: {sum(bool(r['trigger_reasons']) for r in rows)}/{len(rows)}")
        _rate_comparison(lines, meta, rows)
        for label, event_time, start, end in LANDMARKS.get(stem, ()):
            active = [r for r in rows if r["time_before_seconds"] >= start
                      and r["time_after_seconds"] <= end and r["trigger_reasons"]]
            lines.append(f"  Padding neighborhood {label} ({start}-{end}s): " + (", ".join(
                f"{r['time_before_seconds']:.1f}-{r['time_after_seconds']:.1f}[{r['trigger_reasons']}]"
                for r in active) or "no trigger"))
            if active and event_time is not None:
                lines.append(f"    annotated event={event_time}s; affected lead="
                             f"{max(0, event_time - active[0]['time_before_seconds']):.3g}s; tail="
                             f"{max(0, active[-1]['time_after_seconds'] - event_time):.3g}s")
            runs = _runs([r for r in rows if r["time_before_seconds"] >= start
                          and r["time_after_seconds"] <= end])
            if len(runs) > 1:
                gaps = [runs[i + 1][0] - runs[i][1] for i in range(len(runs) - 1)]
                lines.append(f"    internal trigger-free gaps: {gaps}; max={max(gaps):.3g}s")
    lines += ["\nNegative controls (Investigation videos + ticket-scroll, valid estimates only):",
              f"  Translation: {_stats(controls, 'map_translation_magnitude')}",
              f"  Scale delta: {_stats(controls, 'map_scale_delta')}"]
    translation_supported = controls and max(r["map_translation_magnitude"] for r in controls) < TRANSLATION_PROBE_PX
    scale_supported = controls and max(r["map_scale_delta"] for r in controls) < SCALE_DELTA_PROBE
    pan_scale_controls = [r for r in timelines.get("map_pan", []) if r["map_motion_valid"]]
    if pan_scale_controls:
        lines.append("  Pure-pan scale control: " + _stats(pan_scale_controls, "map_scale_delta"))
        scale_supported = scale_supported and max(r["map_scale_delta"] for r in pan_scale_controls) < SCALE_DELTA_PROBE
    for reason, supported in (("map_translation", translation_supported), ("map_scale_change", scale_supported)):
        lines.append(f"  {reason} probe separation from available controls: {bool(supported)}")
    _window_counts(lines, metadata)
    lines += ["\nPROVISIONAL ENGINEERING RECOMMENDATIONS (not production parameters)",
              ("Dense reasons: existing ticket_scroll_detected, route_selection_change, explicit "
              "MEDIUM/HIGH header/image/body. Preserve reasons; context is not a veto."),
              (f"Map translation probe: valid AND hypot(dx,dy) > {TRANSLATION_PROBE_PX}px. "
              f"Control separation supported={bool(translation_supported)}; do not adopt if False."),
              (f"Map scale probe: valid AND abs(scale-1) > {SCALE_DELTA_PROBE}. "
              f"Control separation supported={bool(scale_supported)}; do not adopt if False."),
              "Map residual alone: NO. Ordinary traffic creates residual change without map interaction.",
              ("Translation and scale reasons are not mutually exclusive: zoom includes affine translation. "
              "Control separation supports provisional noise margins, not clean intent classification "
              "or guaranteed detection of very slow pan/zoom onset. Invalid fits cannot be classified."),
              ("pre_padding_seconds=0.5; post_padding_seconds=0.5: provisional symmetric context, "
              "covering a stable predecessor and observed subsecond loading/settling neighborhoods. "
              "Long scroll/pan/zoom bursts require renewed triggers, not one isolated padded event. "
              "Quiet gaps listed above require further planner validation; no merging is done here."),
              ("sparse_sampling_fps=1: several representative observations in the measured multi-second "
              "stable cores/windows, with fewer near-duplicates than 2/4 FPS. This does not discover "
              "fast triggers; trigger analysis still requires dense consecutive source observations."),
              ("dense_sampling_fps=source FPS (10 here): source-adjacent observations preserve the "
              "observed 0.1s changes. 4/8 FPS can miss source frames depending on sample phase. "
              "Never upsample beyond source FPS; higher/lower/VFR sources need calibration. "
              "No sampling timestamp schedule was generated."),
              ("Cross-talk: ticket-body changes during scroll are expected; initial loading/close "
              "also cause ticket triggers. Measured route/motion cross-talk: "
              f"{'; '.join(route_cross_talk) or 'map recordings unavailable'}. "
              "Route is not a clean selection-specific sampling signal. Potential mask/traffic/alignment causes remain "
              "unproven; no thresholds are tuned. These are sampling reasons, "
              "not Investigation events. Generic map residual is deliberately excluded."),
              ("Confidence/limits: small isolated, canonical, constant-10-FPS recordings; manual "
              "landmarks, no held-out benchmark or exact intent labels. Candidate-rate estimates "
              "do not validate readability or scroll-stop retention. Sparse evidence selection "
              "and source-density detection are different responsibilities. Padding is provisional, "
              "not a measured universal settling duration. More loading styles, slow motion, "
              "different source FPS/layouts, longer pauses, and simultaneous interactions are needed."),
              ("Recommend ONE combined validation: active ticket -> stable -> scroll -> stable stop "
              "-> map pan -> stable -> aircraft switch -> stable. It tests cross-talk and quiet gaps "
              "which isolated recordings cannot establish. No new video is required for this task."),
              ("No production sampler, ActivityLevel, AdaptiveSamplingPlan, padded interval model, "
              "merging, requested timestamps, scroll-stop selection, or Phase 4.2b was implemented.")]
    return "\n".join(lines) + "\n"


def main():
    """Measure available recordings and write one reusable timeline and calibration report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-only", action="store_true",
                        help="Reuse this script's existing measured CSV; do not rerun CV.")
    args = parser.parse_args()
    timelines, metadata, missing = {}, {}, []
    for stem in VIDEO_STEMS:
        path = ROOT / "data/videos" / f"{stem}_test.mp4"
        if not path.is_file():
            missing.append(path.name)
            print(f"Missing: {path}", flush=True)
            continue
        print(f"Input: {path}", flush=True)
        meta = read_video_metadata(path)
        metadata[stem] = meta
        if not args.report_only:
            timelines[stem] = _measure_video(stem, meta)
    if args.report_only:
        timelines = _load_timeline(metadata)
    if not timelines or not any(timelines.values()):
        raise ValueError("No available consecutive-pair observations.")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT / "adaptive_sampling_signal_timeline.csv"
    rows = [row for video_rows in timelines.values() for row in video_rows]
    if not args.report_only:
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    report = _report(timelines, metadata, missing)
    report_path = OUTPUT / "adaptive_sampling_calibration_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Timeline: {csv_path}\nReport: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
