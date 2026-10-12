"""Manually evaluate Phase 2.3 map motion on canonical screenshot pairs.

Paths resolve from the repository. UIConfig coordinates and existing screenshot
validation are reused; no layout recovery or resizing occurs. Results describe
experimental geometric evidence, not semantic events or production decisions.
Canonical UI exclusions affect SIFT detection only; all metric pixels remain.
"""

import sys

import cv2
from evaluate_change_metrics import DATA_DIR, load_image_pair

from events_diary.config import UIConfig
from events_diary.map_motion import estimate_map_motion


def main() -> int:
    """Print pan, zoom, no-change, and aircraft-change map measurements.

    Takes no parameters and assumes canonical screenshots in the existing dataset.
    Returns zero if all pairs load and evaluate, otherwise one. Expected file,
    image-contract, and OpenCV errors are printed to stderr; unexpected errors
    propagate. An invalid motion estimate is an evaluation result, not a script
    failure. Only the configured map region is measured; no interpretation occurs.
    """
    config = UIConfig()
    columns = ("Scenario", "Valid", "dx", "dy", "scale", "rotation", "Good Matches",
               "Inliers", "Inlier Ratio", "Raw MAD", "Aligned MAD", "Raw CPR",
               "Aligned CPR", "Reason")
    print(" | ".join(columns))
    failed = False
    for scenario in ("map_pan", "map_zoom", "no_change", "aircraft_change"):
        try:
            image_a, image_b = load_image_pair(
                scenario, DATA_DIR / f"{scenario}_01_a.png",
                DATA_DIR / f"{scenario}_01_b.png", config,
            )
            result = estimate_map_motion(config.map_analysis_region.crop(image_a),
                                         config.map_analysis_region.crop(image_b),
                                         ui_config=config)
        except (FileNotFoundError, ValueError, TypeError, cv2.error) as error:
            print(f"ERROR [{scenario}]: {error}", file=sys.stderr)
            failed = True
            continue
        values = (scenario, result.valid, result.dx, result.dy, result.scale,
                  result.rotation_degrees, result.good_matches, result.inliers,
                  result.inlier_ratio, result.raw_mad, result.aligned_mad,
                  result.raw_cpr, result.aligned_cpr, result.reason)
        print(" | ".join(_format(value) for value in values))
    print("Working thresholds are experimental. dx/dy/scale map B into A.")
    return int(failed)


def _format(value: object) -> str:
    """Format optional diagnostic values compactly without dependencies."""
    if value is None:
        return "-"
    return f"{value:.4f}" if isinstance(value, float) else str(value)


if __name__ == "__main__":
    raise SystemExit(main())
