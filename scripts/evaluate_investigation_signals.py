"""Phase 3.1b-2 manual, single-recording Investigation signal calibration.

Decode every consecutive pair of investigation_switch_test.mp4 using the existing
sequential decoder and canonical layout checks. Production APIs supply ROI MAD/CPR,
aligned map residuals, route replacement, and pair-level ticket-scroll evidence.
No algorithms, production configuration, or specialized thresholds are changed.
No Investigation candidates, temporal states, lifecycle, or identity are inferred.

The provisional CPR boundaries imported from investigation_signals were chosen after inspecting the prior
evaluation of this recording. They are calibration-derived proposals, not optimized or final.
Ticket regions show a quiet/transition separation; generic map residual does not
separate traffic from the switch. Its conservative HIGH boundary is an unvalidated
proposal, since this recording contains no separated large-map-change population.
Existing analyze_investigation_pair supplies all evidence and strength previews.

Paths are repository-relative. Nominal timestamps assume constant source FPS.
Landmark windows only organize the report; they never affect strength decisions.
Unavailable aligned map residuals stay unavailable, with raw values retained as
diagnostics rather than substituted into the residual signal. Matching CSV/report
outputs are overwritten on rerun; other local artifacts are preserved. Decode,
layout, API, and I/O errors propagate. This is manual calibration on ONE recording,
not a benchmark or validation of scrolling, other layouts, or other recordings.
"""

import csv
from collections import Counter
from pathlib import Path

import numpy as np
from evaluate_map_motion_video import _validate_layout
from validate_phase2 import _sequential_frames

from events_diary.config import UIConfig
from events_diary.investigation_signals import (
    AIRCRAFT_IMAGE_HIGH_CPR,
    AIRCRAFT_IMAGE_MEDIUM_CPR,
    HEADER_HIGH_CPR,
    HEADER_MEDIUM_CPR,
    MAP_RESIDUAL_HIGH_CPR,
    MAP_RESIDUAL_MEDIUM_CPR,
    TICKET_BODY_HIGH_CPR,
    TICKET_BODY_MEDIUM_CPR,
    analyze_investigation_pair,
)
from events_diary.video import read_video_metadata

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "data/videos/investigation_switch_test.mp4"
OUTPUT = ROOT / "data/debug/map_motion/investigation_switch"
THRESHOLDS = {
    "header": (HEADER_MEDIUM_CPR, HEADER_HIGH_CPR),
    "aircraft_image": (AIRCRAFT_IMAGE_MEDIUM_CPR, AIRCRAFT_IMAGE_HIGH_CPR),
    "ticket_body": (TICKET_BODY_MEDIUM_CPR, TICKET_BODY_HIGH_CPR),
    "map": (MAP_RESIDUAL_MEDIUM_CPR, MAP_RESIDUAL_HIGH_CPR),
}
RATIONALES = {
    "header": "Quiet CPR is zero; 0.02 includes partial transitions (about 0.038-0.252); "
              "0.4 separates these from large opening/switch changes (about 0.443-0.573).",
    "aircraft_image": "Quiet/pre-opening fluctuation is at most about 0.0098; 0.02 admits "
                      "partial changes (0.038-0.348); 0.6 separates full changes (0.834-0.891).",
    "ticket_body": "Small updates/early loading stay below 0.01; 0.02 separates them from "
                   "partial loading (0.382); 0.6 separates full replacements (0.830-0.855). "
                   "Body strength alone cannot distinguish scrolling from replacement.",
    "map": "Traffic peaks at about 0.00967, above the switch's 0.00716. There is no semantic "
           "separation. 0.01 is a rounded boundary above observed routine changes; 0.02 is "
           "a conservative two-percent HIGH proposal, NOT an empirically separated cutoff. "
           "All observed map pairs remain LOW. More recordings are required to calibrate HIGH.",
}
WINDOWS = {
    "quiet before opening": (0.0, 1.2),
    "opening landmarks": (1.4, 1.6),
    "loading/transition context": (1.2, 1.7),
    "quiet before switch": (1.7, 4.7),
    "switch landmarks": (4.7, 4.9),
    "quiet after switch, before traffic": (4.9, 6.9),
    "ordinary traffic": (6.9, 7.0),
    "quiet after traffic": (7.0, 8.0),
}


def _evaluate(config, metadata):
    """Return consecutive-pair rows using unchanged production measurement APIs.

    Only the previous full frame is retained. Motion is B-to-A; ticket scroll is
    A-to-B. Rejected alignment produces unavailable generic residual evidence.
    The aggregation API delegates to the existing route, scroll, and motion APIs.
    """
    rows, previous = [], None
    for index, frame in _sequential_frames(VIDEO, metadata, config):
        if previous is not None:
            row = {"frame_a": index - 1, "frame_b": index,
                   "time_a": (index - 1) / metadata.fps, "time_b": index / metadata.fps}
            signals = analyze_investigation_pair(previous, frame, config)
            for name, signal in (("header", signals.header_change),
                                 ("aircraft_image", signals.aircraft_image_change),
                                 ("ticket_body", signals.ticket_body_change)):
                row[f"{name}_mad"] = signal.mad
                row[f"{name}_cpr"] = signal.cpr
            motion = signals.map_motion
            row.update(map_mad=signals.map_residual_change.mad,
                       map_cpr=signals.map_residual_change.cpr,
                       map_motion_valid=motion.valid, map_motion_reason=motion.reason,
                       map_dx=motion.dx, map_dy=motion.dy, map_scale=motion.scale,
                       map_rotation_degrees=motion.rotation_degrees,
                       map_raw_mad=motion.raw_mad, map_raw_cpr=motion.raw_cpr,
                       map_inliers=motion.inliers, map_inlier_ratio=motion.inlier_ratio)
            route = signals.route_selection_evidence
            row.update(route_evidence_valid=route.valid, route_evidence_reason=route.reason,
                       route_added_fraction=route.added_fraction,
                       route_removed_fraction=route.removed_fraction,
                       route_changed_fraction=route.changed_fraction,
                       route_selection_change=signals.route_selection_change)
            row["ticket_scroll_detected"] = signals.ticket_scroll_detected
            for name, signal in (("header", signals.header_change),
                                 ("aircraft_image", signals.aircraft_image_change),
                                 ("ticket_body", signals.ticket_body_change),
                                 ("map_residual", signals.map_residual_change)):
                row[f"{name}_strength"] = signal.strength.value
            rows.append(row)
        previous = frame
    if not rows:
        raise ValueError("Need at least one consecutive frame pair.")
    return rows


def _stats(rows, field):
    """Summarize finite observations with NumPy linear-interpolated percentiles."""
    values = [row[field] for row in rows
              if row[field] is not None and np.isfinite(row[field])]
    if not values:
        return "available=0"
    quantiles = np.percentile(values, [0, 50, 90, 95, 99, 100])
    return f"available={len(values)}/{len(rows)}; " + ", ".join(
        f"{name}={value:.9g}" for name, value in
        zip(("min", "median", "p90", "p95", "p99", "max"), quantiles))


def _interval(row):
    """Format nominal pair times without assigning semantic event boundaries."""
    return f"{row['time_a']:.1f}-{row['time_b']:.1f}s"


def _report(rows, metadata):
    """Build statistics, manually labeled scenario comparisons, and limitations.

    Windows are descriptive annotations only. Detailed landmark rows retain MAD,
    CPR, strengths, and specialized context; CSV preserves full precision for all.
    """
    lines = ["Phase 3.1b-2: SINGLE-RECORDING provisional calibration",
             f"Source: {VIDEO}",
             (f"{metadata.width}x{metadata.height}; fps={metadata.fps}; "
              f"frames={metadata.frame_count}; duration={metadata.duration_seconds}s; pairs={len(rows)}"),
             "No Investigation candidates or temporal decisions. CPR fractions, not percentages."]
    for name, (medium, high) in THRESHOLDS.items():
        lines += [f"\n{name}: CPR {_stats(rows, name + '_cpr')}",
                  f"MAD {_stats(rows, name + '_mad')}",
                  f"Proposed medium={medium}, high={high}. {RATIONALES[name]}",
                  "Top five CPR intervals (ties retain chronological order):"]
        ranked = sorted((r for r in rows if r[f"{name}_cpr"] is not None),
                        key=lambda r: r[f"{name}_cpr"], reverse=True)[:5]
        lines += [f"  {_interval(r)}: {r[name + '_cpr']:.9g}" for r in ranked]
    for label, (start, end) in WINDOWS.items():
        subset = [r for r in rows if r["time_a"] >= start and r["time_b"] <= end]
        lines.append(f"\n{label} ({start}-{end}s), {len(subset)} pairs:")
        for name in THRESHOLDS:
            strength = "map_residual_strength" if name == "map" else f"{name}_strength"
            lines.append(f"  {name}: {_stats(subset, name + '_cpr')}; "
                         f"strengths={dict(Counter(r[strength] for r in subset))}")
        if not label.startswith("quiet"):
            for row in subset:
                lines.append(f"  {_interval(row)}: " + "; ".join(
                    f"{key}={value}" for key, value in row.items()
                    if key not in ("frame_a", "frame_b", "time_a", "time_b")))
    for field in ("route_selection_change", "ticket_scroll_detected"):
        positives = [r for r in rows if r[field]]
        lines.append(f"\n{field}: {len(positives)}/{len(rows)}; "
                     + ", ".join(_interval(r) for r in positives))
    valid = [r for r in rows if r["map_motion_valid"]]
    lines.append(f"\nMap motion valid: {len(valid)}/{len(rows)}; B-to-A convention.")
    for field in ("map_dx", "map_dy", "map_scale", "map_rotation_degrees", "map_inlier_ratio"):
        lines.append(f"{field}: {_stats(valid, field)}")
    lines += ["\nUnexpected behavior / interpretation:",
              ("Generic map CPR ranks ordinary traffic above the actual switch; it cannot separate "
               "those cases here. Route replacement provides different, specialized evidence."),
              "Opening and loading transitions can be strong. They are not suppressed by calibration.",
              "Quiet refers to ticket/interaction landmarks; map traffic still moves during quiet windows.",
              ("Limitations: ONE recording, manually annotated windows, no held-out benchmark. "
              "No positive scrolling examples: body thresholds are not validated on scrolling. "
              "Map HIGH cutoff lacks empirical positive support. Thresholds are proposals only. "
              "Canonical layout and constant FPS assumed. Alignment/interpolation, compression, "
              "loading, and color-mask errors can affect evidence. Strength is visual change, "
              "not identity, confidence, or an Investigation decision.")]
    return "\n".join(lines) + "\n"


def main():
    """Evaluate the source recording and write only this utility's CSV/text outputs."""
    print(f"Input: {VIDEO}", flush=True)
    config = UIConfig()
    metadata = read_video_metadata(VIDEO)
    _validate_layout(metadata.width, metadata.height, config)
    rows = _evaluate(config, metadata)
    report = _report(rows, metadata)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT / "investigation_signal_calibration.csv"
    report_path = OUTPUT / "investigation_signal_calibration_report.txt"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"CSV: {csv_path}\nReport: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
