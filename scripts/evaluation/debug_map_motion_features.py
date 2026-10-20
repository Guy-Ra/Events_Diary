"""Visualize the current Phase 2.3 SIFT/Lowe/RANSAC evidence for development.

This manual diagnostic tool loads canonical pan and zoom screenshots, crops only
UIConfig.map_analysis_region, and calls the existing map-motion estimator. Scoped
OpenCV wrappers forward every argument unchanged and capture the actual keypoints,
KNN results, and RANSAC mask from that call. They are restored before drawing or
saving. No production source, threshold, SIFT/RANSAC parameter, transform model,
coordinate, or estimation behavior is changed; no second fit is performed.
The canonical UIConfig is passed to the estimator, so its SIFT-only exclusion
mask is reflected in every captured visualization. A binary mask PNG is also
saved (white allowed, black excluded); metric pixels are unaffected.

Outputs retain the entire map crop, including static UI overlays. Keypoint images
show SIFT scale/orientation. Match images show A on the left and B on the right:
the upper row contains endpoint circles without connecting lines so geographic
content, aircraft/labels/routes, and UI controls remain visible; the lower row
shows the same matches with connecting lines. Yellow denotes all Lowe-accepted
matches; green denotes only the original RANSAC inliers. No features are sampled
or removed for display. Feature locations are evidence for human inspection, not
automatic semantic classification or new ignore masks.

Paths resolve from this repository, independently of the working directory.
Existing canonical screenshot validation is reused. PNGs are written under
data/debug/map_motion/ and overwritten on subsequent manual runs. This script's
temporary wrappers are suitable only for standalone, single-threaded development
use, not concurrent production execution. No new dependencies are required.
"""

import sys
from pathlib import Path
from unittest.mock import patch

# Allow direct execution and explicit imports across script directories.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np

from events_diary import map_motion
from events_diary.config import UIConfig
from scripts.calibration.evaluate_change_metrics import DATA_DIR, load_image_pair

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data/debug/map_motion"


def _capture_estimate(
    image_a: np.ndarray, image_b: np.ndarray, config: UIConfig,
) -> tuple:
    """Run the unchanged estimator and return its result and display evidence.

    Args:
        image_a, image_b: Matching canonical map crops, passed through unchanged.
        config: Canonical full-frame UI geometry used for SIFT-only exclusions.
    Returns:
        Result, A/B keypoint sequences, Lowe-accepted B-to-A matches, and the
        subset selected by the actual RANSAC mask. Empty stages yield empty lists.
    Raises:
        Estimator validation/OpenCV errors or RuntimeError for inconsistent capture.
    Wrappers delegate to original OpenCV objects and are restored even on errors.
    Lowe filtering is replayed only on captured distances using the current module
    constant, preserving order; detection, matching, and fitting each run once.
    """
    detections, pairs, masks = [], [], []
    original_sift = cv2.SIFT_create
    original_matcher = cv2.BFMatcher
    original_fit = cv2.estimateAffinePartial2D

    class SiftCapture:
        """Delegate SIFT detection while retaining returned keypoints."""

        def detectAndCompute(self, *args, **kwargs):
            """Forward detection arguments unchanged and capture its output."""
            output = sift.detectAndCompute(*args, **kwargs)
            detections.append(output[0])
            return output

    class MatcherCapture:
        """Delegate descriptor matching while retaining returned KNN pairs."""

        def knnMatch(self, *args, **kwargs):
            """Forward KNN arguments unchanged and capture its output."""
            output = matcher.knnMatch(*args, **kwargs)
            pairs.extend(output)
            return output

    def create_sift(*args, **kwargs):
        """Create the original SIFT object using exactly the estimator arguments."""
        nonlocal sift
        sift = original_sift(*args, **kwargs)
        return SiftCapture()

    def create_matcher(*args, **kwargs):
        """Create the original BF matcher using exactly the estimator arguments."""
        nonlocal matcher
        matcher = original_matcher(*args, **kwargs)
        return MatcherCapture()

    def fit(*args, **kwargs):
        """Return the original fit unchanged and retain its RANSAC inlier mask."""
        output = original_fit(*args, **kwargs)
        masks.append(output[1])
        return output

    sift = matcher = None
    with patch.object(cv2, "SIFT_create", create_sift), \
            patch.object(cv2, "BFMatcher", create_matcher), \
            patch.object(cv2, "estimateAffinePartial2D", fit):
        result = map_motion.estimate_map_motion(image_a, image_b, ui_config=config)
    points_a, points_b = detections
    good = [pair[0] for pair in pairs if len(pair) == 2
            and pair[0].distance < map_motion.LOWE_RATIO * pair[1].distance]
    inliers = []
    if masks and masks[0] is not None:
        mask = masks[0].ravel()
        if len(mask) != len(good):
            raise RuntimeError("Captured RANSAC mask does not match Lowe match count.")
        inliers = [match for match, keep in zip(good, mask) if keep]
    if (len(points_a), len(points_b), len(good), len(inliers)) != (
        result.keypoints_a, result.keypoints_b, result.good_matches, result.inliers
    ):
        raise RuntimeError("Captured evidence counts differ from estimator result.")
    return result, points_a, points_b, good, inliers


def _title(image: np.ndarray, text: str) -> np.ndarray:
    """Return a new image with a title band, preserving all crop pixels.

    image is a BGR visualization; text describes its scenario and evidence.
    OpenCV/array errors propagate. No coordinates in the displayed crop change.
    """
    band = np.zeros((44, image.shape[1], 3), dtype=np.uint8)
    cv2.putText(band, text, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack((band, image))


def _match_view(image_a, image_b, points_a, points_b, matches, color, label):
    """Return full-resolution endpoint and connection panels for B-to-A matches.

    Args are BGR crops, their captured keypoints, B-query/A-train DMatches, a BGR
    drawing color, and title text. Returns vertically stacked labeled A-left/B-right
    panels. OpenCV errors propagate. No semantic regions are inferred or masked.
    Matches are reversed only for drawing, never for estimation.
    """
    display_matches = [cv2.DMatch(m.trainIdx, m.queryIdx, m.distance) for m in matches]
    endpoints = np.hstack((image_a.copy(), image_b.copy()))
    for match in matches:
        for point, offset in ((points_a[match.trainIdx].pt, 0),
                              (points_b[match.queryIdx].pt, image_a.shape[1])):
            center = (round(point[0]) + offset, round(point[1]))
            cv2.circle(endpoints, center, 4, color, 1, cv2.LINE_AA)
    connections = cv2.drawMatches(
        image_a, points_a, image_b, points_b, display_matches, None,
        matchColor=color, singlePointColor=color,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return np.vstack((
        _title(endpoints, f"{label}: {len(matches)} endpoints | A left / B right"),
        _title(connections, f"{label}: same matches with connections | A left / B right"),
    ))


def _save(path: Path, image: np.ndarray) -> None:
    """Write a PNG visualization, raising OSError if OpenCV cannot save it.

    path is the destination and image is the rendered array. Returns None; existing
    files are overwritten. OpenCV codec errors propagate. Parent must already exist.
    """
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write debug image: {path}")


def main() -> int:
    """Generate eight evidence PNGs plus a SIFT mask and print pan/zoom counts.

    Takes no parameters; uses repository paths and canonical UIConfig crops.
    Returns zero on success. File, validation, capture, or drawing failures propagate
    with a traceback instead of silently skipping evidence. No production state is
    persisted, and output visualizations do not alter the estimator's input arrays.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = UIConfig()
    region = config.map_analysis_region
    _save(OUTPUT_DIR / "sift_detection_mask.png",
          map_motion._build_sift_mask(config, (region.height, region.width)))
    print("Scenario | Keypoints A | Keypoints B | Lowe matches | RANSAC inliers | Estimate")
    for scenario in ("pan", "zoom"):
        image_a, image_b = load_image_pair(
            scenario, DATA_DIR / f"map_{scenario}_01_a.png",
            DATA_DIR / f"map_{scenario}_01_b.png", config,
        )
        image_a = config.map_analysis_region.crop(image_a)
        image_b = config.map_analysis_region.crop(image_b)
        result, points_a, points_b, good, inliers = _capture_estimate(image_a, image_b, config)
        for suffix, image, points in (("a", image_a, points_a), ("b", image_b, points_b)):
            drawn = cv2.drawKeypoints(
                image, points, None, color=(0, 255, 255),
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            )
            _save(OUTPUT_DIR / f"{scenario}_keypoints_{suffix}.png",
                  _title(drawn, f"{scenario} Map {suffix.upper()}: {len(points)} SIFT keypoints"))
        for suffix, matches, color in (("matches", good, (0, 255, 255)),
                                        ("inliers", inliers, (0, 255, 0))):
            _save(OUTPUT_DIR / f"{scenario}_{suffix}.png", _match_view(
                image_a, image_b, points_a, points_b, matches, color,
                f"{scenario} {suffix}",
            ))
        print(f"{scenario} | {len(points_a)} | {len(points_b)} | {len(good)} | "
              f"{len(inliers)} | valid={result.valid}, dx={result.dx:.4f}, "
              f"dy={result.dy:.4f}, scale={result.scale:.4f}")
    print(f"Saved diagnostic PNGs to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
