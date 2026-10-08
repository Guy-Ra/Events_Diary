"""Manual Phase 2.2 evaluation/calibration of basic visual change metrics.

Compare Mean Absolute Difference (MAD) and Changed Pixel Ratio (CPR) across
multiple configured ROIs in known Flightradar24 screenshot pairs. The dataset
is intentionally small: seven scenarios under data/sanity_frames/change_metrics/,
with no aircraft-image-load pair collected yet. Paths resolve relative to the
repository, independently of the working directory.

Screenshots must have matching dimensions and the canonical UIConfig reference
resolution. No resizing or layout recovery is performed. CPR thresholds 5, 10,
15, and 20 are experimental comparison values only; this utility does not
select event thresholds or determine production behavior. Metric calculations
reuse the public image-array APIs without adding preprocessing or interpretation.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

from events_diary.change_metrics import compute_changed_pixel_ratio, compute_mad
from events_diary.config import UIConfig
from events_diary.layout import LayoutKind, classify_layout, map_roi_to_frame

DATA_DIR = Path(__file__).resolve().parents[1] / "data/sanity_frames/change_metrics"
SCENARIOS = {
    "no_change": (DATA_DIR / "no_change_01_a.png", DATA_DIR / "no_change_01_b.png"),
    "small_update": (DATA_DIR / "small_update_01_a.png", DATA_DIR / "small_update_01_b.png"),
    "ticket_scroll": (DATA_DIR / "ticket_scroll_01_a.png", DATA_DIR / "ticket_scroll_01_b.png"),
    "aircraft_change": (DATA_DIR / "aircraft_change_01_a.png", DATA_DIR / "aircraft_change_01_b.png"),
    "map_pan": (DATA_DIR / "map_pan_01_a.png", DATA_DIR / "map_pan_01_b.png"),
    "map_zoom": (DATA_DIR / "map_zoom_01_a.png", DATA_DIR / "map_zoom_01_b.png"),
    "ticket_open": (DATA_DIR / "ticket_open_01_a.png", DATA_DIR / "ticket_open_01_b.png"),
}
PIXEL_THRESHOLDS = (5, 10, 15, 20)


def load_image_pair(
    scenario: str,
    path_a: Path,
    path_b: Path,
    config: UIConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate a known screenshot pair against canonical dimensions.

    Args:
        scenario: Scenario name used in error messages.
        path_a: Path to the first screenshot.
        path_b: Path to the second screenshot.
        config: Canonical UI configuration providing the reference dimensions.

    Returns:
        Two matching uint8 BGR images loaded with OpenCV's color-image mode.

    Raises:
        FileNotFoundError: If an image file is missing.
        ValueError: If an image cannot be decoded, pair dimensions differ, or
            either screenshot does not use the exact reference resolution.

    Assumes canonical UI placement; dimensions alone do not verify UI content.
    No image is resized or geometrically adjusted.
    """
    images = []
    for path in (path_a, path_b):
        if not path.is_file():
            raise FileNotFoundError(f"{scenario}: missing image file: {path}")
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"{scenario}: unreadable image: {path}")
        images.append(image)

    image_a, image_b = images
    if image_a.shape != image_b.shape:
        raise ValueError(
            f"{scenario}: mismatched pair dimensions: "
            f"{path_a} has shape {image_a.shape}; {path_b} has shape {image_b.shape}."
        )

    height, width = image_a.shape[:2]
    if classify_layout(config, width, height) != LayoutKind.EXACT_REFERENCE:
        raise ValueError(
            f"{scenario}: unexpected reference resolution for {path_a} and {path_b}: "
            f"{width}x{height}; expected {config.reference_width}x{config.reference_height}."
        )
    return image_a, image_b


def print_table(rows: list[tuple[str, str, float, tuple[float, ...]]]) -> None:
    """Print scenario/ROI measurements using standard-library formatting.

    Args:
        rows: Scenario name, ROI name, MAD, and CPR values in threshold order.

    Returns:
        None; writes the comparison table to standard output.

    Assumes each row supplies CPR values for PIXEL_THRESHOLDS. No data loading
    or validation is performed here; normal printing errors propagate.
    """
    header = f"{'Scenario':<17} {'ROI':<21} {'MAD':>9}"
    header += "".join(f"{'CPR@' + str(threshold):>10}" for threshold in PIXEL_THRESHOLDS)
    print(header)
    print("-" * len(header))
    for scenario, roi_name, mad, ratios in rows:
        values = "".join(f"{ratio:>10.4f}" for ratio in ratios)
        print(f"{scenario:<17} {roi_name:<21} {mad:>9.3f}{values}")


def main() -> int:
    """Evaluate all configured screenshot pairs and print the comparison table.

    Takes no parameters. Uses the explicit SCENARIOS mapping, four canonical
    UIConfig ROIs, and evaluation-only PIXEL_THRESHOLDS. Strict crops feed the
    existing public MAD and CPR functions without additional preprocessing.

    Returns:
        Zero if all scenarios succeed, otherwise one. Invalid pairs are
        identified on standard error and other scenarios are still evaluated.

    Expected file, dimension, and metric validation failures are reported;
    unexpected programming errors propagate. Successful scenario rows are
    printed only after all four ROIs have been evaluated.
    """
    config = UIConfig()
    regions = {
        "ticket_header": config.ticket_header,
        "aircraft_image": config.aircraft_image,
        "ticket_scroll_body": config.ticket_scroll_body,
        "map_analysis_region": config.map_analysis_region,
    }
    rows = []
    processed = 0
    for scenario, (path_a, path_b) in SCENARIOS.items():
        try:
            image_a, image_b = load_image_pair(scenario, path_a, path_b, config)
            height, width = image_a.shape[:2]
            scenario_rows = []
            for roi_name, reference_roi in regions.items():
                roi = map_roi_to_frame(reference_roi, config, width, height)
                crop_a, crop_b = roi.crop(image_a), roi.crop(image_b)
                mad = compute_mad(crop_a, crop_b)
                ratios = tuple(
                    compute_changed_pixel_ratio(crop_a, crop_b, threshold)
                    for threshold in PIXEL_THRESHOLDS
                )
                scenario_rows.append((scenario, roi_name, mad, ratios))
        except (FileNotFoundError, ValueError, TypeError, cv2.error) as error:
            print(f"ERROR [{scenario}]: {error}", file=sys.stderr)
            continue
        rows.extend(scenario_rows)
        processed += 1

    print_table(rows)
    print(f"\nProcessed {processed}/{len(SCENARIOS)} scenario pairs; {len(rows)} ROI rows.")
    print("CPR thresholds are experimental; no production threshold selected.")
    return 0 if processed == len(SCENARIOS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
