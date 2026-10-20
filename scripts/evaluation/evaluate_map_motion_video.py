"""Phase 2.3 manual temporal evaluation of deterministic map-motion estimation.

Evaluate every consecutive decoded source-frame pair in a supplied recording:
uv run python scripts/evaluation/evaluate_map_motion_video.py data/videos/map_zoom_test.mp4
Temporally adjacent frames may retain
more geographic overlap than the earlier distant screenshot pair, allowing the
existing estimator's evidence to be inspected at source temporal density. This is
an observation utility, not an event detector or a pan/no-pan classifier. It does
not alter production SIFT, matching, RANSAC, thresholds, alignment, or metrics.

Reuse public video metadata, layout classification, UIConfig ROI cropping, and
map-motion APIs. Decode sequentially using a local OpenCV VideoCapture because
the existing sampler seeks and skips read failures. Keep only the previous crop
and the current frame, plus small result dictionaries. Canonical dimensions are
required; there is no resizing, scaling, or layout recovery. The existing static
UI exclusions affect SIFT detection only through the supplied UIConfig.

Zero-based decoded frame indices are assigned in sequential order. Timestamps
are index/source FPS consistently, assuming a constant-frame-rate recording;
these are nominal source times, not variable-frame-rate presentation timestamps.
The script prints each pair's complete measurements and a summary with valid-only
medians and strict CPR improvement counts. Standard-library CSV also writes the
timeline to data/debug/map_motion/<input_stem>_video_evaluation.csv, dropping a
trailing '_test' from the input stem (map_zoom_test.mp4 becomes
map_zoom_video_evaluation.csv). That file is overwritten on each run. Empty
numeric cells represent unavailable values in CSV; console output uses '-'. The
required input argument resolves relative to cwd; output is repository-relative.
The resolved input and output paths are printed before metadata loading.

An unsuccessful read is normal EOF only after the metadata frame count has been
decoded. Earlier failure stops evaluation, reports the next frame index, labels
the summary INCOMPLETE, preserves the partial CSV, and returns a nonzero exit code.
Container frame-count inaccuracies cannot be distinguished from decoding failures
by VideoCapture; count disagreement is therefore reported rather than ignored.
Summary values are experimental observations, not calibrated production decisions.
"""

import argparse
import csv
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import median

# Allow direct execution and explicit imports across script directories.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2

from events_diary.config import UIConfig
from events_diary.layout import LayoutKind, classify_layout
from events_diary.map_motion import estimate_map_motion
from events_diary.video import VideoMetadata, read_video_metadata

ROOT = Path(__file__).resolve().parents[2]
FIELDS = (
    "pair_index", "frame_a_index", "frame_b_index", "time_a", "time_b", "valid",
    "dx", "dy", "scale", "rotation_degrees", "good_matches", "inliers",
    "inlier_ratio", "raw_mad", "aligned_mad", "raw_cpr", "aligned_cpr", "reason",
)


def _validate_layout(width: int, height: int, config: UIConfig) -> None:
    """Require canonical frame geometry before any cropping.

    Args: width/height are decoded or metadata dimensions; config supplies the
        canonical reference geometry. No proportional scaling is accepted.
    Returns: None when compatible.
    Raises: ValueError for nonpositive or incompatible dimensions, including actual
        and expected resolution in the message. Layout classification is reused.
    """
    if (width <= 0 or height <= 0
            or classify_layout(config, width, height) != LayoutKind.EXACT_REFERENCE):
        raise ValueError(
            f"Incompatible resolution {width}x{height}; expected canonical "
            f"{config.reference_width}x{config.reference_height}. No resizing allowed."
        )


def _format_value(value: object) -> str:
    """Format a timeline/summary value without changing the measurement.

    Args: value is a scalar diagnostic or None; floats use four decimal places.
    Returns: Display string, with '-' for unavailable values.
    Assumes ordinary Python scalars; exceptions from custom string conversions
    propagate. This affects console presentation only, not CSV precision.
    """
    if value is None:
        return "-"
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def _print_pair(row: dict) -> None:
    """Print a complete pair row in the fixed timeline column order.

    Args: row contains all FIELDS; indices and times refer to consecutive source
        frames, and motion components map B into A.
    Returns: None; writes stdout. Missing keys raise KeyError; output errors
        propagate. No classification or interpretation is added.
    """
    print(" | ".join(_format_value(row[name]) for name in FIELDS), flush=True)


def _print_summary(metadata: VideoMetadata, decoded: int, rows: list[dict],
                   complete: bool) -> None:
    """Print source metadata, processed counts, and valid-only statistics.

    Args: metadata is the existing public metadata record; decoded is successful
        reads; rows holds analyzed pairs; complete indicates verified EOF/count.
    Returns: None; writes stdout. Valid result fields are assumed non-None; malformed
        rows raise KeyError/TypeError. Output errors propagate. Empty denominators
        print '-' and no medians are computed when there are no valid results.
    Medians cover the entire processed timeline, including stationary segments;
    CPR improvement means strictly aligned_cpr < raw_cpr, without tuning.
    """
    valid = [row for row in rows if row["valid"]]
    improved = sum(row["aligned_cpr"] < row["raw_cpr"] for row in valid)
    valid_percentage = 100 * len(valid) / len(rows) if rows else None
    improvement_percentage = 100 * improved / len(valid) if valid else None
    print(f"\nSummary: {'COMPLETE' if complete else 'INCOMPLETE'}")
    print(f"Source: {metadata.width}x{metadata.height}, FPS={metadata.fps:.6f}, "
          f"duration={metadata.duration_seconds:.6f}s, metadata frames={metadata.frame_count}")
    print(f"Decoded frames={decoded}, analyzed consecutive pairs={len(rows)}, "
          f"valid={len(valid)}, invalid={len(rows) - len(valid)}, "
          f"valid percentage={_format_value(valid_percentage)}%")
    for name in ("dx", "dy", "scale", "rotation_degrees", "inlier_ratio",
                 "raw_cpr", "aligned_cpr"):
        value = median(row[name] for row in valid) if valid else None
        print(f"Valid-only median {name}: {_format_value(value)}")
    print(f"Valid pairs with strictly improved CPR: {improved}/{len(valid)} "
          f"({_format_value(improvement_percentage)}%)")


def main() -> int:
    """Evaluate the command-line video sequentially, saving and printing all pairs.

    Takes no Python parameters; argparse reads a required video path from sys.argv.
    Relative inputs resolve against cwd; CSV naming uses the supplied file stem,
    removing only a trailing '_test'. Assumes constant FPS and canonical placement.
    Invalid/missing CLI arguments exit with argparse's status 2.
    Returns zero on complete evaluation, one on metadata/read/layout/evaluation
    failures. Expected file, ValueError, TypeError, and OpenCV errors are reported
    on stderr; partial processing is summarized explicitly. Unexpected programming
    errors propagate. Metadata loading delegates to read_video_metadata, including
    missing/unreadable input and FPS validation. CSV I/O errors propagate after
    releasing the decoder. No production code or configuration is modified.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_path", type=Path, help="Input video; relative to the current directory.")
    arguments = parser.parse_args()
    video_path = arguments.video_path.resolve()
    output_stem = video_path.stem.removesuffix("_test")
    csv_path = ROOT / "data/debug/map_motion" / f"{output_stem}_video_evaluation.csv"
    print(f"Resolved input video: {video_path}", flush=True)
    print(f"Timeline CSV output: {csv_path}", flush=True)
    config = UIConfig()
    try:
        metadata = read_video_metadata(video_path)
        _validate_layout(metadata.width, metadata.height, config)
        if metadata.frame_count < 2:
            raise ValueError(f"Need at least two source frames; reported {metadata.frame_count}.")
    except (OSError, ValueError, cv2.error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    capture = cv2.VideoCapture(str(video_path))
    decoded, rows, previous, complete = 0, [], None, False
    try:
        if not capture.isOpened():
            raise ValueError(f"Unable to open video for sequential decoding: {video_path}")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=FIELDS)
            writer.writeheader()
            print(" | ".join(FIELDS), flush=True)
            try:
                while True:
                    success, frame = capture.read()
                    if not success or frame is None:
                        if decoded != metadata.frame_count:
                            raise ValueError(
                                f"Frame read failed at index {decoded} "
                                f"(nominal {decoded / metadata.fps:.6f}s); decoded "
                                f"{decoded} of {metadata.frame_count} reported frames."
                            )
                        complete = True
                        break
                    frame_index = decoded
                    decoded += 1
                    height, width = frame.shape[:2]
                    _validate_layout(width, height, config)
                    current = config.map_analysis_region.crop(frame)
                    if previous is not None:
                        result = estimate_map_motion(previous, current, ui_config=config)
                        values = asdict(result)
                        row = {name: values[name] for name in FIELDS[5:]}
                        row.update(pair_index=len(rows), frame_a_index=frame_index - 1,
                                   frame_b_index=frame_index,
                                   time_a=(frame_index - 1) / metadata.fps,
                                   time_b=frame_index / metadata.fps)
                        rows.append(row)
                        writer.writerow(row)
                        _print_pair(row)
                    previous = current
            except (ValueError, TypeError, cv2.error) as error:
                print(f"ERROR: evaluation stopped: {error}", file=sys.stderr)
    except (ValueError, cv2.error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
    finally:
        capture.release()
    _print_summary(metadata, decoded, rows, complete)
    print(f"{'Complete' if complete else 'Partial'} timeline CSV: {csv_path}")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
