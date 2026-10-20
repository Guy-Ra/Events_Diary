"""Phase 3.1 exploratory aircraft-switch evaluation, not Investigation detection.

Study selection/route visual changes in data/videos/investigation_switch_test.mp4
at consecutive-source-frame density. Reuse Phase 2 metadata/layout/ROI, generic
change metrics, map-motion estimation, and ticket-scroll evidence/state APIs. No
production code, parameters, algorithms, or thresholds are modified. This utility
does not extract routes, associate aircraft, create event objects, or classify a
selection switch. Rankings are observations, not production triggers.
The production route_selection measurement API also supplies overlap-normalized
route-color occupancy/addition/removal fractions. Its pair-level classifier is
recorded as route_selection_change using the supplied provisional defaults;
this does not create an Investigation decision or temporal state.

Evaluation-only color measurements compare BGR directly: mean absolute difference
over all pixels/channels and the fraction of pixels whose maximum channel
difference exceeds the exploratory intensity value 10. With a valid map estimate,
apply its rotation-free B-to-A scale/translation and linear warp; an identically
warped float support image excludes padding/interpolation footprints outside B.
This rendering follows the existing estimator contract without refitting motion.
Rejected estimates use explicitly labeled raw_unaligned color measurements;
these must not be interpreted as compensated residuals. No HSV segmentation,
color-specific route masks, or final detector is introduced.

Sequential decoding, canonical validation, and metadata are reused from the
closure utility. Nominal times are frame_index/source FPS (constant FPS assumed).
Console output ranks the strongest changes and reports source metadata, motion,
and temporal scroll intervals. Full-precision timeline CSV is saved under
data/debug/map_motion/investigation_switch/investigation_switch_video_evaluation.csv.
A second sequential
pass saves at most five representative pairs: the strongest header pair, the
strongest aircraft-image pair with its before/after neighbors, and the strongest
aligned color pair if different. Images under
data/debug/map_motion/investigation_switch/ show full frames, aligned/raw map A/B,
unamplified absolute BGR difference, and the exploratory mask. Excluded overlap
pixels are black in difference/mask displays and omitted from color statistics.

Paths resolve from the repository. Read/count/layout failures stop explicitly;
CSV is written after successful evaluation. Local outputs overwrite matching files
on reruns without removing unrelated artifacts. This is manual exploratory evidence
from one recording, not a benchmark or a calibrated semantic route-change signal.
"""

import csv
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import median

# Allow direct execution and explicit imports across script directories.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np

from events_diary.change_metrics import compute_changed_pixel_ratio, compute_mad
from events_diary.config import UIConfig
from events_diary.map_motion import MapMotionEstimate, estimate_map_motion
from events_diary.route_selection import (
    estimate_route_selection_change,
    is_route_selection_change,
)
from events_diary.ticket_scroll import (
    analyze_ticket_scroll_sequence,
    estimate_ticket_scroll_evidence,
    is_vertical_scroll_evidence,
)
from events_diary.video import read_video_metadata
from scripts.evaluation.evaluate_map_motion_video import _format_value, _validate_layout
from scripts.validation.validate_phase2 import _intervals, _sequential_frames

ROOT = Path(__file__).resolve().parents[2]
VIDEO_PATH = ROOT / "data/videos/investigation_switch_test.mp4"
OUTPUT_DIR = ROOT / "data/debug/map_motion/investigation_switch"
CSV_PATH = OUTPUT_DIR / "investigation_switch_video_evaluation.csv"
EXPLORATORY_COLOR_THRESHOLD = 10


def _color_residual(image_a, image_b, motion):
    """Return evaluation-only BGR residual measurements and display arrays.

    Args: image_a/B are matching canonical BGR map crops; motion is the unchanged
        estimator result with B-to-A components. Valid motion omits rotation exactly
        as production alignment does; invalid motion compares original crops.
    Returns: (mean BGR difference, max-channel changed fraction, domain, aligned B,
        absolute BGR difference, binary changed mask). domain labels aligned_overlap
        or raw_unaligned. Excluded pixels do not enter statistics.
    Assumes valid transforms have nonempty supported overlap. ValueError is raised
    otherwise; OpenCV errors propagate. No input pixels or production metric logic
    are modified. The max-channel threshold is exploratory, not semantic masking.
    """
    aligned = image_b
    overlap = np.ones(image_a.shape[:2], dtype=bool)
    domain = "raw_unaligned"
    if motion.valid:
        warp = np.float64([[motion.scale, 0, motion.dx], [0, motion.scale, motion.dy]])
        size = (image_a.shape[1], image_a.shape[0])
        aligned = cv2.warpAffine(image_b, warp, size, flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        support = cv2.warpAffine(np.ones(image_b.shape[:2], np.float32), warp, size,
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        overlap = support == 1.0
        domain = "aligned_overlap"
    if not overlap.any():
        raise ValueError("No supported map overlap for exploratory color measurement.")
    difference = cv2.absdiff(image_a, aligned)
    changed = difference.max(axis=2) > EXPLORATORY_COLOR_THRESHOLD
    mean_difference = float(difference[overlap].mean())
    changed_ratio = float(changed[overlap].mean())
    display = difference.copy()
    display[~overlap] = 0
    mask = np.uint8(changed & overlap) * 255
    return mean_difference, changed_ratio, domain, aligned, display, mask


def _evaluate(config, metadata):
    """Read every consecutive pair and return measurement rows and scroll states.

    config supplies canonical ROIs; metadata is the existing public video record.
    Returns rows with full-precision timing/ticket/map/color data, plus temporal
    states computed from buffered existing evidence. Assumes source constant FPS.
    Decode/layout/production failures propagate; no skipping/downsampling occurs.
    Only a previous frame is retained, with small measurements buffered for summary.
    """
    rows, evidences, previous, decoded = [], [], None, 0
    for index, frame in _sequential_frames(VIDEO_PATH, metadata, config):
        decoded += 1
        if previous is not None:
            row = {"pair_index": index - 1, "frame_a_index": index - 1,
                   "frame_b_index": index, "time_a": (index - 1) / metadata.fps,
                   "time_b": index / metadata.fps}
            for name in ("ticket_header", "aircraft_image", "ticket_scroll_body"):
                region = getattr(config, name)
                a, b = region.crop(previous), region.crop(frame)
                row[f"{name}_mad"] = compute_mad(a, b)
                row[f"{name}_cpr"] = compute_changed_pixel_ratio(a, b)
            a, b = (config.map_analysis_region.crop(image) for image in (previous, frame))
            motion = estimate_map_motion(a, b, ui_config=config)
            row.update({f"map_{name}": value for name, value in asdict(motion).items()})
            color_mean, color_ratio, domain, *_ = _color_residual(a, b, motion)
            row.update(color_mean_bgr_absdiff=color_mean, color_max_channel_ratio=color_ratio,
                       color_domain=domain)
            route = estimate_route_selection_change(previous, frame, config)
            row.update(route_evidence_valid=route.valid,
                       route_fraction_a=route.route_fraction_a,
                       route_fraction_b=route.route_fraction_b,
                       route_added_fraction=route.added_fraction,
                       route_removed_fraction=route.removed_fraction,
                       route_changed_fraction=route.changed_fraction,
                       route_evidence_reason=route.reason,
                       route_selection_change=is_route_selection_change(route))
            evidence = estimate_ticket_scroll_evidence(previous, frame, config)
            evidences.append(evidence)
            row["vertical_scroll_evidence"] = is_vertical_scroll_evidence(evidence)
            rows.append(row)
        previous = frame
    states = analyze_ticket_scroll_sequence(evidences)
    for row, state in zip(rows, states):
        row["ticket_scroll_state"] = state.value
    return rows, decoded


def _rank(rows, field, count=3):
    """Return up to count largest available values for an observation field.

    rows are timeline dictionaries; field is a numeric measurement key; count is
    a positive display limit. None values are omitted and ties preserve time order.
    No significance threshold or selection classifier is applied. Key errors
    propagate; an empty available subset returns [].
    """
    return sorted((row for row in rows if row[field] is not None),
                  key=lambda row: row[field], reverse=True)[:count]


def _save_debug(rows, config, metadata):
    """Save at most five representative pairs in a second sequential video pass.

    rows supply observed rankings, config supplies ROIs, metadata supplies decoding
    validation. Returns saved pair indices. Choose peak header CPR, peak image CPR
    plus its neighbors, and strongest valid aligned color pair. Display selection is not a
    switch detector. Reuse stored transform measurements, without recomputing SIFT.
    Decode errors or failed image writes raise; no random seeking or frame dumps.
    """
    peak = _rank(rows, "aircraft_image_cpr", 1)[0]["pair_index"]
    targets = {i for i in (peak - 1, peak, peak + 1) if 0 <= i < len(rows)}
    targets.add(_rank(rows, "ticket_header_cpr", 1)[0]["pair_index"])
    aligned_rows = [row for row in rows if row["map_valid"]]
    if aligned_rows:
        targets.add(_rank(aligned_rows, "color_max_channel_ratio", 1)[0]["pair_index"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    previous = None
    for index, frame in _sequential_frames(VIDEO_PATH, metadata, config):
        if previous is not None and index - 1 in targets:
            row = rows[index - 1]
            # Use the public result record solely to replay its existing display warp.
            motion = MapMotionEstimate(**{name[4:]: value for name, value in row.items()
                                          if name.startswith("map_")})
            a, b = (config.map_analysis_region.crop(image) for image in (previous, frame))
            _, _, domain, aligned, difference, mask = _color_residual(a, b, motion)
            prefix = f"pair_{index - 1:03d}_{row['time_a']:.1f}_{row['time_b']:.1f}"
            for suffix, image in (("frames_ab", np.hstack((previous, frame))),
                                  (f"map_ab_{domain}", np.hstack((a, aligned))),
                                  ("bgr_absdiff", difference), ("color_change_mask", mask)):
                path = OUTPUT_DIR / f"{prefix}_{suffix}.png"
                if not cv2.imwrite(str(path), image):
                    raise OSError(f"Could not write debug image: {path}")
        previous = frame
    return sorted(targets)


def _print_summary(rows, metadata, decoded):
    """Print metadata, ranked changes, motion context, and temporal scroll intervals.

    rows contain complete evaluated measurements, metadata describes the source,
    decoded counts successful frames. Returns None. Rankings are not event labels;
    display filters translation>1px and |scale-1|>0.01 are descriptive only.
    Assumes at least one pair; output/key errors propagate. Rejected motion values
    remain diagnostics and do not enter valid-only medians/ranges.
    """
    print(f"\nSource: {metadata.width}x{metadata.height}, FPS={metadata.fps:g}, "
          f"duration={metadata.duration_seconds:g}s; decoded={decoded}; pairs={len(rows)}")
    for field in ("ticket_header_cpr", "ticket_header_mad", "aircraft_image_cpr",
                  "aircraft_image_mad", "ticket_scroll_body_cpr", "map_raw_cpr",
                  "map_raw_mad", "map_aligned_cpr", "map_aligned_mad",
                  "color_max_channel_ratio", "color_mean_bgr_absdiff",
                  "route_changed_fraction", "route_added_fraction", "route_removed_fraction"):
        print(f"Strongest {field}:")
        for row in _rank(rows, field):
            print(f"  pair {row['pair_index']}, {row['time_a']:.1f}-{row['time_b']:.1f}s: "
                  f"{row[field]:.6f}; map_valid={row['map_valid']}; color={row['color_domain']}")
    valid = [row for row in rows if row["map_valid"]]
    print(f"Map estimates valid={len(valid)}/{len(rows)}")
    if valid:
        for field in ("map_dx", "map_dy", "map_scale", "map_rotation_degrees", "map_inlier_ratio"):
            values = [row[field] for row in valid]
            print(f"  {field}: median={median(values):.6f}; range={min(values):.6f}..{max(values):.6f}")
        print("Descriptive motion pairs (translation>1px or |scale-1|>0.01; not classifiers):")
        for row in valid:
            if np.hypot(row["map_dx"], row["map_dy"]) > 1 or abs(row["map_scale"] - 1) > 0.01:
                print(f"  {row['time_a']:.1f}-{row['time_b']:.1f}: dx={row['map_dx']:.4f}, "
                      f"dy={row['map_dy']:.4f}, scale={row['map_scale']:.4f}")
    print(f"Pair scroll positives: {sum(row['vertical_scroll_evidence'] for row in rows)}/{len(rows)}")
    print("Temporal ticket-scroll intervals:")
    timing = [{"start": row["time_a"], "end": row["time_b"]} for row in rows]
    for start, end, state in _intervals(timing, [row["ticket_scroll_state"] for row in rows]):
        print(f"  {start:.1f}-{end:.1f} {state}")


def main():
    """Run the repository switch recording, write CSV, and save selected debug pairs.

    No parameters. Returns zero on success; input/decode/API/I/O failures propagate
    clearly and do not produce a success summary. Assumes canonical constant-FPS
    recording. Raw timeline stays observable; interpretation is manual and no
    production detector is created. CSV overwrites its previous matching artifact.
    """
    print(f"Input: {VIDEO_PATH}", flush=True)
    config = UIConfig()
    metadata = read_video_metadata(VIDEO_PATH)
    _validate_layout(metadata.width, metadata.height, config)
    if metadata.frame_count < 2:
        raise ValueError("Need at least two frames for consecutive-pair evaluation.")
    rows, decoded = _evaluate(config, metadata)
    if not rows:
        raise ValueError("No consecutive pairs decoded.")
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(" | ".join(rows[0]))
    for row in rows:
        print(" | ".join(_format_value(value) for value in row.values()))
    _print_summary(rows, metadata, decoded)
    print(f"Debug pair indices: {_save_debug(rows, config, metadata)}")
    print(f"CSV: {CSV_PATH}\nDebug images: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
