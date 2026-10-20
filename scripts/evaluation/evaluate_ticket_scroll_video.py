"""Phase 2.4 manual temporal evaluation of ticket scrolling, not a detector.

Read every consecutive decoded frame from data/videos/ticket_scroll_test.mp4 and
compare existing UIConfig ticket_header, aircraft_image, ticket_scroll_body, and
map_analysis_region crops using the public MAD and default CPR APIs. The purpose
is to inspect whether the scroll body changes while surrounding regions stay
relatively stable. Measurements now reuse the production ticket_scroll APIs,
with provisional pair-level vertical_scroll_evidence and summary counts. These
booleans do not classify temporal events. Offline sequence interpretation adds
quiet/up/down states using existing evidence, without creating event objects.

Production pair measurement uses cv2.phaseCorrelate on float32 grayscale
images with a Hann window to reduce edge discontinuities. dx/dy describe A-to-B
content motion in pixels: positive dy is downward. Horizontal motion is diagnostic
only. The returned response is reported without acceptance thresholds; it is not
a calibrated probability of correct motion. Phase correlation assumes a dominant
translation, is periodic/ambiguous for large shifts, and can be unreliable with
changing content, repeated rows, fixed overlays, or little texture. No alignment,
optical flow, or temporal event objects are introduced.

Reuse project metadata, canonical layout validation, ROI cropping, and metrics.
OpenCV decoding is sequential without seeking/downsampling; nominal timestamps
are zero-based decoded frame index divided by actual source FPS, assuming constant
frame rate. The input defaults to the repository video above; an optional CLI path
resolves against cwd. Resolved input/output paths are printed before loading.

Print the complete timeline and per-ROI median/maximum metrics plus motion summary.
For descriptive reporting only, count |dy| > 1 pixel; this is an explicitly
exploratory summary value, not a production threshold or a scroll decision. Save
full-precision results using standard-library CSV to
data/debug/map_motion/ticket_scroll_video_evaluation.csv, overwriting that file
on each run. Raw non-finite phase-correlation values are retained diagnostically
and cannot pass the production pair-level interpretation helper.
The output directory is created as needed; no new dependencies are required.

Read failure before the reported frame count stops processing with a nonzero exit
and an INCOMPLETE summary. A read failure after exactly the reported frame count
is EOF. Container count inaccuracies cannot be distinguished from decode failures.
Partial CSVs are explicitly identified. No frame data is retained beyond a pair. Measurement records are buffered for
centered offline temporal analysis; timeline rows/CSV are emitted after decoding.
Partial runs interpret only their decoded prefix and remain labeled INCOMPLETE.
"""

import argparse
import csv
import math
import sys
from pathlib import Path
from statistics import median

# Allow direct execution and explicit imports across script directories.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2

from events_diary.config import UIConfig
from events_diary.ticket_scroll import (
    TicketScrollState,
    analyze_ticket_scroll_sequence,
    estimate_ticket_scroll_evidence,
    is_vertical_scroll_evidence,
)
from events_diary.video import VideoMetadata, read_video_metadata
from scripts.evaluation.evaluate_map_motion_video import _format_value, _validate_layout

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "data/debug/map_motion/ticket_scroll_video_evaluation.csv"
REGION_NAMES = ("ticket_header", "aircraft_image", "ticket_scroll_body", "map_analysis_region")
FIELDS = ("pair_index", "frame_a_index", "frame_b_index", "time_a", "time_b") + tuple(
    f"{region}_{metric}" for region in REGION_NAMES for metric in ("mad", "cpr")
) + ("estimated_dx", "estimated_dy", "motion_response", "vertical_scroll_evidence", "ticket_scroll_state")
EXPLORATORY_DISPLACEMENT_PIXELS = 1.0


def _measure_pair(evidence, index: int, fps: float) -> dict:
    """Measure one sequential full-frame pair through the production evidence APIs.

    Args: evidence is an existing production pair record; index is current zero-based
        source index; fps is positive source FPS. Measurements are not recomputed.
    Returns: A timeline dictionary with raw measurements and one pair-level bool.
    Assumes no skipped frames, nominal constant-FPS times, and no temporal state.
    Production validation/OpenCV errors propagate; no local metric or motion
    implementation is retained and temporal interpretation is applied separately after decoding.
    """
    row = {"pair_index": index - 1, "frame_a_index": index - 1, "frame_b_index": index,
           "time_a": (index - 1) / fps, "time_b": index / fps}
    for region, prefix in zip(REGION_NAMES, ("header", "aircraft_image", "body", "map")):
        for metric in ("mad", "cpr"):
            row[f"{region}_{metric}"] = getattr(evidence, f"{prefix}_{metric}")
    row.update(estimated_dx=evidence.dx, estimated_dy=evidence.dy,
               motion_response=evidence.motion_response,
               vertical_scroll_evidence=is_vertical_scroll_evidence(evidence))
    return row


def _print_summary(metadata: VideoMetadata, decoded: int, rows: list[dict],
                   complete: bool) -> None:
    """Print metadata, per-region metric statistics, and descriptive displacement.

    Args: metadata is the public video record; decoded counts successful reads;
        rows contains measurements; complete means EOF matches metadata count.
    Returns: None, writing stdout. Assumes rows follow FIELDS and coordinates are
        canonical crop-local pixels. Empty series print '-'; unavailable motion
        values are excluded from medians and their count is reported separately.
    |dy| > 1 pixel is descriptive only; percentages use finite dy pairs. Missing
    dictionary fields or output failures propagate. Pair-level decisions are counted
    separately; contiguous temporal states are summarized without event objects.
    """
    print(f"\nSummary: {'COMPLETE' if complete else 'INCOMPLETE'}")
    print(f"Source: {metadata.width}x{metadata.height}, FPS={metadata.fps:.6f}, "
          f"duration={metadata.duration_seconds:.6f}s, metadata frames={metadata.frame_count}")
    print(f"Decoded frames={decoded}, analyzed consecutive pairs={len(rows)}")
    print("ROI | Median MAD | Median CPR | Maximum MAD | Maximum CPR")
    for region in REGION_NAMES:
        mad = [row[f"{region}_mad"] for row in rows]
        cpr = [row[f"{region}_cpr"] for row in rows]
        values = (median(mad), median(cpr), max(mad), max(cpr)) if rows else (None,) * 4
        print(f"{region} | " + " | ".join(_format_value(value) for value in values))
    dx = [row["estimated_dx"] for row in rows if math.isfinite(row["estimated_dx"])]
    dy = [row["estimated_dy"] for row in rows if math.isfinite(row["estimated_dy"])]
    nontrivial = sum(abs(value) > EXPLORATORY_DISPLACEMENT_PIXELS for value in dy)
    percentage = 100 * nontrivial / len(dy) if dy else None
    print(f"Body motion median dx={_format_value(median(dx) if dx else None)}, "
          f"median dy={_format_value(median(dy) if dy else None)}, "
          f"maximum absolute dy={_format_value(max(map(abs, dy)) if dy else None)}")
    print(f"Exploratory |dy| > {EXPLORATORY_DISPLACEMENT_PIXELS:.1f} pixel: "
          f"{nontrivial}/{len(dy)} finite pairs ({_format_value(percentage)}%); "
          f"unavailable dy={len(rows) - len(dy)}. Not a production threshold.")

    classified = sum(row["vertical_scroll_evidence"] for row in rows)
    classified_percentage = 100 * classified / len(rows) if rows else None
    print(f"Pair-level vertical-scroll evidence: {classified}/{len(rows)} "
          f"({_format_value(classified_percentage)}%). Provisional defaults; no temporal events.")

    for state in TicketScrollState:
        print(f"Temporal {state.value} pairs: "
              f"{sum(row['ticket_scroll_state'] == state.value for row in rows)}")
    print("Contiguous state intervals (nominal source seconds):")
    start = 0
    for index in range(1, len(rows) + 1):
        if index == len(rows) or rows[index]["ticket_scroll_state"] != rows[start]["ticket_scroll_state"]:
            print(f"{rows[start]['time_a']:.1f}-{rows[index - 1]['time_b']:.1f} "
                  f"{rows[start]['ticket_scroll_state']}")
            start = index


def main() -> int:
    """Sequentially evaluate the optional CLI input, defaulting to the ticket video.

    Takes no Python parameters; argparse reads an optional video path from argv.
    Returns zero on complete evaluation or one on metadata/layout/read/OpenCV
    failures. Invalid CLI arguments exit with status 2. Assumes constant source FPS
    and exact canonical dimensions. Existing metadata handles missing/unreadable
    input and invalid FPS. Partial results are labeled and the decoder always closes;
    CSV I/O failures and unexpected programming errors propagate. No temporal
    state is maintained and no frames are resized or silently skipped.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_path", nargs="?", type=Path,
                        default=ROOT / "data/videos/ticket_scroll_test.mp4")
    video_path = parser.parse_args().video_path.resolve()
    print(f"Resolved input video: {video_path}", flush=True)
    print(f"Timeline CSV output: {CSV_PATH}", flush=True)
    config = UIConfig()
    try:
        metadata = read_video_metadata(video_path)
        _validate_layout(metadata.width, metadata.height, config)
        if metadata.frame_count < 2:
            raise ValueError(f"Need at least two frames; reported {metadata.frame_count}.")
    except (OSError, ValueError, cv2.error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    capture = cv2.VideoCapture(str(video_path))
    decoded, rows, previous, complete = 0, [], None, False
    evidences = []
    try:
        if not capture.isOpened():
            raise ValueError(f"Unable to open video for decoding: {video_path}")
        CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CSV_PATH.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=FIELDS)
            writer.writeheader()
            print(" | ".join(FIELDS), flush=True)
            try:
                while True:
                    success, frame = capture.read()
                    if not success or frame is None:
                        if decoded != metadata.frame_count:
                            raise ValueError(
                                f"Read failed at frame {decoded}; decoded {decoded} "
                                f"of {metadata.frame_count} reported frames."
                            )
                        complete = True
                        break
                    index = decoded
                    decoded += 1
                    height, width = frame.shape[:2]
                    _validate_layout(width, height, config)
                    current = frame
                    if previous is not None:
                        evidence = estimate_ticket_scroll_evidence(previous, current, config)
                        evidences.append(evidence)
                        row = _measure_pair(evidence, index, metadata.fps)
                        rows.append(row)
                        if any(not math.isfinite(row[name]) for name in ("estimated_dx", "estimated_dy", "motion_response")):
                            print(f"WARNING: non-finite phase correlation at pair {index - 1}.",
                                  file=sys.stderr)
                    previous = current
            except (ValueError, TypeError, cv2.error) as error:
                print(f"ERROR: evaluation stopped: {error}", file=sys.stderr)
            states = analyze_ticket_scroll_sequence(evidences)
            for row, state in zip(rows, states):
                row["ticket_scroll_state"] = state.value
                writer.writerow(row)
                print(" | ".join(_format_value(row[name]) for name in FIELDS), flush=True)
    except (ValueError, cv2.error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
    finally:
        capture.release()
    _print_summary(metadata, decoded, rows, complete)
    print(f"{'Complete' if complete else 'Partial'} timeline CSV: {CSV_PATH}")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
