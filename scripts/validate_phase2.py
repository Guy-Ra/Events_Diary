"""Manual Phase 2 closure validation using existing production components.

Decode the pan, zoom, and ticket-scroll recordings sequentially at source FPS.
Run unchanged map-motion and ticket-scroll measurement/interpretation APIs on
each consecutive pair for real-video regression and cross-phenomenon sanity
checks. This script changes no production behavior, thresholds, geometry, or
algorithms. It is not a benchmark, training tool, or new pan/zoom classifier.

Print four report sections and save the same report text to
data/debug/map_motion/phase2_validation_report.txt. Input/output paths resolve
from the repository. Canonical layout validation and metadata APIs are reused;
there is no resizing, random seeking, or downsampling. Times are frame index/FPS
and assume constant frame rate. No frame dumps, event objects, or accuracy metrics
are produced. Test/lint results are handled separately; closure remains for review.

For descriptive map summaries only, show valid pairs with translation magnitude
above one pixel or |scale-1| above 0.01. These round-number display filters are not
production classifications or tuned thresholds. Candidate zoom intervals are
printed with their scale values, including rejected fits. Review finite estimates
in context: geometric validity does not guarantee correct registration. Known
scroll periods are approximate manual observations, not benchmark labels.
"""

import math
import sys
from pathlib import Path
from statistics import median

import cv2
from evaluate_map_motion_video import _format_value, _validate_layout

from events_diary.config import UIConfig
from events_diary.map_motion import estimate_map_motion
from events_diary.ticket_scroll import (
    TicketScrollState,
    analyze_ticket_scroll_sequence,
    estimate_ticket_scroll_evidence,
    is_vertical_scroll_evidence,
)
from events_diary.video import read_video_metadata

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "data/debug/map_motion/phase2_validation_report.txt"


def _sequential_frames(path, metadata, config):
    """Yield (zero-based index, full frame) without seeking or skipping failures.

    Parameters are input Path, public metadata, and canonical UIConfig. Returns
    an iterator over decoded BGR frames; timestamps are derived by the caller.
    Assumes container frame count is accurate and constant FPS. Raises ValueError
    for opening, premature EOF/count mismatch, or incompatible decoded dimensions;
    OpenCV errors propagate. Decoder closes even when iteration is interrupted.
    """
    capture = cv2.VideoCapture(str(path))
    index = 0
    try:
        if not capture.isOpened():
            raise ValueError(f"Unable to open video: {path}")
        while True:
            success, frame = capture.read()
            if not success or frame is None:
                if index != metadata.frame_count:
                    raise ValueError(f"{path.name}: read failed at {index}; "
                                     f"metadata reports {metadata.frame_count} frames.")
                break
            height, width = frame.shape[:2]
            _validate_layout(width, height, config)
            yield index, frame
            index += 1
    finally:
        capture.release()


def _evaluate_video(name, config):
    """Measure both existing pipelines once per consecutive pair in one recording.

    name is the recording stem; config supplies canonical ROIs. Returns a dictionary
    containing metadata, decoded count, pair measurements, and temporal states.
    Assumes constant source FPS; no algorithm/parameter overrides are supplied.
    Metadata, decode, crop, and production exceptions propagate rather than silently
    reporting successful partial validation. Only one previous frame is retained.
    """
    path = ROOT / "data/videos" / f"{name}.mp4"
    metadata = read_video_metadata(path)
    _validate_layout(metadata.width, metadata.height, config)
    if metadata.frame_count < 2:
        raise ValueError(f"{name}: need at least two reported frames.")
    rows, evidences, previous, decoded = [], [], None, 0
    for index, frame in _sequential_frames(path, metadata, config):
        decoded += 1
        if previous is not None:
            motion = estimate_map_motion(config.map_analysis_region.crop(previous),
                                         config.map_analysis_region.crop(frame), ui_config=config)
            evidence = estimate_ticket_scroll_evidence(previous, frame, config)
            evidences.append(evidence)
            rows.append({"start": (index - 1) / metadata.fps, "end": index / metadata.fps,
                         "map": motion, "scroll": is_vertical_scroll_evidence(evidence)})
        previous = frame
    return {"metadata": metadata, "decoded": decoded, "rows": rows,
            "states": analyze_ticket_scroll_sequence(evidences)}


def _intervals(rows, labels):
    """Return contiguous (start seconds, end seconds, label) runs for pair labels.

    rows supplies nominal pair start/end times; labels is a same-length sequence.
    Returns [] for empty input. Assumes ordered adjacent pairs and comparable labels;
    length mismatch raises ValueError. This formats observations, not event objects.
    """
    if len(rows) != len(labels):
        raise ValueError("Interval labels must match pair count.")
    intervals = []
    start = 0
    for index in range(1, len(rows) + 1):
        if index == len(rows) or labels[index] != labels[start]:
            intervals.append((rows[start]["start"], rows[index - 1]["end"], labels[start]))
            start = index
    return intervals


def _map_summary(data, lines, descriptive_subset=False):
    """Append source counts and map medians/ranges to report lines.

    data is an evaluated recording; lines is the output string list; optional
    descriptive_subset selects valid translation>1px or |scale-1|>0.01 pairs.
    Returns None. Assumes valid motion fields are finite/non-None. Empty subsets
    print unavailable medians. No pan/zoom decision or production threshold is added.
    """
    metadata, rows = data["metadata"], data["rows"]
    valid = [row["map"] for row in rows if row["map"].valid]
    lines.append(f"Source: {metadata.path}; {metadata.width}x{metadata.height}, "
                 f"FPS={metadata.fps:g}, duration={metadata.duration_seconds:g}s, "
                 f"reported/decoded frames={metadata.frame_count}/{data['decoded']}")
    lines.append(f"Pairs={len(rows)}, map valid={len(valid)}, invalid={len(rows)-len(valid)}, "
                 f"valid={100*len(valid)/len(rows):.2f}%")
    subset = [m for m in valid if math.hypot(m.dx, m.dy) > 1 or abs(m.scale - 1) > 0.01]
    selected = subset if descriptive_subset else valid
    lines.append(f"Median domain: {'descriptive motion subset' if descriptive_subset else 'all valid pairs'} "
                 f"({len(selected)} pairs). Display filters only: translation>1px or |scale-1|>0.01.")
    for field in ("dx", "dy", "scale", "rotation_degrees", "inlier_ratio", "raw_cpr", "aligned_cpr"):
        values = [getattr(m, field) for m in selected]
        lines.append(f"  {field}: median={_format_value(median(values) if values else None)}")
    if valid:
        lines.append(f"Valid scale range={min(m.scale for m in valid):.6f}..{max(m.scale for m in valid):.6f}; "
                     f"max |rotation|={max(abs(m.rotation_degrees) for m in valid):.6f} degrees; "
                     f"inlier ratio range={min(m.inlier_ratio for m in valid):.4f}.."
                     f"{max(m.inlier_ratio for m in valid):.4f}")
        lines.append(f"Valid max translation magnitude={max(math.hypot(m.dx,m.dy) for m in valid):.4f}px")


def _scroll_summary(data, lines, cross=False):
    """Append pair positives, temporal counts, and contiguous state intervals.

    data holds measured rows/states, lines collects text, cross flags pan/zoom
    controls. Returns None. Uses existing classifier/state outputs only; no new
    temporal decisions or accuracy labels. Known periods are approximate context.
    """
    rows, states = data["rows"], data["states"]
    positives = sum(row["scroll"] for row in rows)
    lines.append(f"Decoded={data['decoded']}, pairs={len(rows)}, pair scroll positives="
                 f"{positives}/{len(rows)} ({100*positives/len(rows):.2f}%)")
    for state in TicketScrollState:
        lines.append(f"  {state.value}: {states.count(state)}")
    lines.append("Contiguous states (seconds):")
    intervals = _intervals(rows, [state.value for state in states])
    for start, end, label in intervals:
        lines.append(f"  {start:.1f}-{end:.1f} {label}")
    active = [(a, b, label) for a, b, label in intervals if label != "quiet"]
    if cross:
        lines.append("Potential false scroll intervals: " + ("none" if not active else
                     "; ".join(f"{a:.1f}-{b:.1f} {label}" for a, b, label in active)))
    else:
        outside = [(a, b, label) for a, b, label in active
                   if not (label == "scrolling_up" and a >= 0.6 and b <= 2.8)
                   and not (label == "scrolling_down" and a >= 4.3 and b <= 6.1)]
        lines.append(f"Active intervals outside known approximate direction periods: {len(outside)}")
        lines.append("Known visual periods: up 0.6-2.8; quiet 2.8-4.3; down 4.3-6.1; quiet after 6.1.")


def main() -> int:
    """Run all three recordings and write the four-section closure review report.

    No parameters; uses repository paths and unchanged production defaults. Returns
    zero when all recordings evaluate, one on expected input/decode/API failures.
    Writes identical essential text to stdout/file; failures produce an explicitly
    INCOMPLETE report. File I/O/unexpected errors propagate. Completion of this
    utility does not declare Phase 2 complete or approve production behavior.
    """
    lines = ["Phase 2 closure validation -- observations for review only"]
    videos = {}
    try:
        for name in ("map_pan_test", "map_zoom_test", "ticket_scroll_test"):
            print(f"Evaluating {name} at source FPS with both existing pipelines...", flush=True)
            videos[name] = _evaluate_video(name, UIConfig())
    except (OSError, ValueError, TypeError, cv2.error) as error:
        lines.append(f"INCOMPLETE: {error}")
        status = 1
    else:
        status = 0
        pan, zoom, ticket = (videos[name] for name in ("map_pan_test", "map_zoom_test", "ticket_scroll_test"))
        lines.append("\n1. MAP PAN VALIDATION")
        _map_summary(pan, lines, descriptive_subset=True)
        lines.append("Inspect nonzero translation together with near-unit scale; no pan detector is applied.")
        lines.append("\n2. MAP ZOOM VALIDATION")
        _map_summary(zoom, lines, descriptive_subset=True)
        lines.append("Candidate scale-change runs (|scale-1|>0.01, display filter only):")
        labels = [row["map"].scale is not None and abs(row["map"].scale - 1) > 0.01 for row in zoom["rows"]]
        for start, end, selected in _intervals(zoom["rows"], labels):
            if selected:
                values = [f"{row['map'].scale:.4f}{'' if row['map'].valid else ' (rejected)'}"
                          for row in zoom["rows"] if row["start"] >= start and row["end"] <= end]
                lines.append(f"  {start:.1f}-{end:.1f}: " + ", ".join(values))
        lines.append("\n3. TICKET SCROLL VALIDATION")
        _scroll_summary(ticket, lines)
        lines.append("\n4. CROSS-PHENOMENON SANITY CHECK")
        for name, data in (("pan", pan), ("zoom", zoom)):
            lines.append(f"Ticket-scroll perspective on {name}:")
            _scroll_summary(data, lines, cross=True)
        lines.append("Map-motion perspective on ticket-scroll recording:")
        _map_summary(ticket, lines)
        lines.append("\nPhase 2 validation observations:")
        lines.append("Tests/lint status handled separately; production defaults were unchanged.")
        pan_valid = [row["map"] for row in pan["rows"] if row["map"].valid]
        lines.append(f"Pan: nonzero translation is present in the displayed motion subset; "
                     f"valid scale stays in {min(m.scale for m in pan_valid):.6f}.."
                     f"{max(m.scale for m in pan_valid):.6f}. Inspect this as translation with near-unit scale.")
        zoom_valid = [row["map"] for row in zoom["rows"] if row["map"].valid]
        lines.append(f"Zoom: {sum(m.scale < 0.99 for m in zoom_valid)} valid displayed pairs "
                     f"have scale below 0.99 and {sum(m.scale > 1.01 for m in zoom_valid)} above 1.01; "
                     "ordered scale runs above expose coherent direction and pauses without a zoom classifier.")
        lines.append("Ticket: active-direction intervals are listed against the known periods; "
                     "quiet gaps are preserved, so these states do not cover whole human interactions.")
        lines.append(f"Cross-check: pan/zoom ticket-positive counts are "
                     f"{sum(row['scroll'] for row in pan['rows'])}/"
                     f"{sum(row['scroll'] for row in zoom['rows'])}; "
                     "any temporal false intervals are explicitly listed above.")
        ticket_valid = [row["map"] for row in ticket["rows"] if row["map"].valid]
        lines.append(f"Ticket-video map motion: maximum valid displacement="
                     f"{max(math.hypot(m.dx, m.dy) for m in ticket_valid):.4f}px; "
                     "inspect alongside the near-unit scale range above for cross-talk.")
        lines.append("Known limitation: one-pair gap smoothing leaves longer pauses QUIET; zoom fits may "
                     "be rejected and pixel residuals need not improve. Review any control-video scroll "
                     "interval or sustained ticket-video map displacement before closure.")
        lines.append("Phase 2 completion is NOT automatically declared; final decision remains for review.")
    report = "\n".join(lines) + "\n"
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"Report saved: {REPORT_PATH}")
    return status


if __name__ == "__main__":
    sys.exit(main())
