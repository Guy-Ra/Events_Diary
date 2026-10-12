"""Estimate global map motion without interpreting semantic events.

Raw pixel-change metrics alone cannot distinguish pan/zoom from content changes:
moving an unchanged map changes many pixel locations. SIFT features, KNN descriptor
matching, Lowe's ratio test, and RANSAC partial-affine fitting provide geometric
evidence for global motion. The physical model is translation plus uniform scale
on a north-up map. Estimated rotation is only a validity/sanity check and is
excluded from the alignment warp, even when accepted.

Aligned residual MAD/CPR describe visual change remaining after compensation on
valid overlapping pixels. They are not semantic event interpretation or a decision
that motion explains a scene. All thresholds below are experimental Phase 2.3
working values, not calibrated production constants. Dynamic overlays, repetitive
features, interpolation, and newly rendered zoom details can leave residuals.

When UIConfig is explicitly supplied, canonical static UI exclusion rectangles
are translated into map-local coordinates and used only as a SIFT detection mask.
Image pixels and raw/aligned metric domains are unchanged. Generic image-array
calls without UIConfig remain unmasked; canonical geometry is never guessed.
"""

import math
from dataclasses import dataclass, replace

import cv2
import numpy as np

from events_diary.change_metrics import compute_changed_pixel_ratio, compute_mad
from events_diary.config import UIConfig

LOWE_RATIO = 0.75
MIN_GOOD_MATCHES = 10
MIN_INLIERS = 8
MIN_INLIER_RATIO = 0.50
MIN_SCALE = 0.5
MAX_SCALE = 2.0
ROTATION_TOLERANCE_DEGREES = 3.0
RANSAC_REPROJECTION_THRESHOLD = 3.0
RANSAC_MAX_ITERS = 2000
RANSAC_CONFIDENCE = 0.99
RANSAC_REFINE_ITERS = 10


@dataclass(frozen=True)
class MapMotionEstimate:
    """Immutable geometric evidence and grayscale change measurements.

    Attributes:
        valid: Whether geometric checks and nonempty overlap passed.
        dx, dy: Translation in pixels from B coordinates into A coordinates.
        scale: Uniform B-to-A scale, about the image coordinate origin.
        rotation_degrees: Original fitted rotation, retained only diagnostically.
        keypoints_a, keypoints_b: Detected SIFT keypoint counts.
        good_matches: Descriptor matches passing Lowe's ratio test.
        inliers, inlier_ratio: RANSAC support count and fraction of good matches.
        raw_mad, raw_cpr: Whole-image grayscale metrics before alignment.
        aligned_mad, aligned_cpr: Metrics on fully supported overlap after the
            rotation-free warp, or None for rejected estimates.
        reason: Rejection explanation, or None on success.

    Unavailable transform components are None. Rejected finite transforms retain
    their components for inspection. This record neither classifies events nor
    guarantees correct registration; small or unrepresentative overlap can mislead.
    """

    valid: bool = False
    dx: float | None = None
    dy: float | None = None
    scale: float | None = None
    rotation_degrees: float | None = None
    keypoints_a: int = 0
    keypoints_b: int = 0
    good_matches: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0
    raw_mad: float = 0.0
    raw_cpr: float = 0.0
    aligned_mad: float | None = None
    aligned_cpr: float | None = None
    reason: str | None = None


def estimate_map_motion(
    image_a: np.ndarray,
    image_b: np.ndarray,
    *,
    pixel_threshold: int = 10,
    ui_config: UIConfig | None = None,
) -> MapMotionEstimate:
    """Fit map B to map A and measure residual change on valid overlap.

    Args:
        image_a: Nonempty uint8 NumPy image, 2D grayscale or three-channel BGR.
        image_b: Image with identical shape and dtype requirements as image_a.
        pixel_threshold: Integer CPR threshold in [0, 255], excluding bool;
            defaults to the existing metric's experimental default of 10.
        ui_config: Optional canonical layout. If supplied, inputs must be exact
            map_analysis_region crops at reference resolution. Its full-frame
            map_ignore_masks exclude feature centers from SIFT detection only.
            None preserves generic, unmasked image-array estimation.

    Returns:
        Frozen evidence record. Weak or implausible fits return valid=False,
        a reason, raw metrics, and no aligned metrics. dx/dy/scale describe B-to-A
        motion, the inverse of a transform used to synthesize B from A.

    Raises:
        TypeError: Invalid input array/dtype or CPR threshold type.
        ValueError: Empty/unsupported/mismatched images or out-of-range threshold.
            Also raised for config/crop dimension mismatch or exclusions outside
            map_analysis_region. Masks are not resized or clipped silently.
        cv2.error: Unexpected OpenCV feature, matching, or warp failures.

    Assumes already cropped compatible north-up map regions. BGR is converted to
    grayscale for both features and metrics; no resizing or layout recovery occurs.
    Inputs are unchanged. BF L2 KNN(k=2) and experimental constants govern fitting.
    Partial-affine rotation is checked but discarded for warping. Linear warp
    interpolation can leave residuals; only pixels with full source support enter
    aligned metrics. Raw and aligned metrics therefore have different pixel domains.
    No fallback, semantic interpretation, or parameter tuning is performed.
    """
    raw_mad = compute_mad(image_a, image_b)
    raw_cpr = compute_changed_pixel_ratio(image_a, image_b, pixel_threshold)
    gray_a, gray_b = (_gray(image) for image in (image_a, image_b))
    detection_mask = (
        _build_sift_mask(ui_config, gray_a.shape) if ui_config is not None else None
    )
    sift = cv2.SIFT_create()
    points_a, descriptors_a = sift.detectAndCompute(gray_a, detection_mask)
    points_b, descriptors_b = sift.detectAndCompute(gray_b, detection_mask)
    result = MapMotionEstimate(
        raw_mad=raw_mad, raw_cpr=raw_cpr,
        keypoints_a=len(points_a), keypoints_b=len(points_b),
    )
    if descriptors_a is None or descriptors_b is None:
        return replace(result, reason="No SIFT descriptors in one or both images.")
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(descriptors_b, descriptors_a, k=2)
    good = [pair[0] for pair in pairs
            if len(pair) == 2 and pair[0].distance < LOWE_RATIO * pair[1].distance]
    result = replace(result, good_matches=len(good))
    if len(good) < MIN_GOOD_MATCHES:
        return replace(result, reason="Too few good matches (minimum 10).")
    source = np.float32([points_b[m.queryIdx].pt for m in good])
    destination = np.float32([points_a[m.trainIdx].pt for m in good])
    matrix, mask = cv2.estimateAffinePartial2D(
        source, destination, method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_REPROJECTION_THRESHOLD,
        maxIters=RANSAC_MAX_ITERS, confidence=RANSAC_CONFIDENCE,
        refineIters=RANSAC_REFINE_ITERS,
    )
    if matrix is None or mask is None:
        return replace(result, reason="Transform estimation failed.")
    if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        return replace(result, reason="Transform has malformed or non-finite values.")
    a, b = float(matrix[0, 0]), float(matrix[1, 0])
    inliers = int(np.count_nonzero(mask))
    result = replace(
        result, dx=float(matrix[0, 2]), dy=float(matrix[1, 2]),
        scale=math.hypot(a, b), rotation_degrees=math.degrees(math.atan2(b, a)),
        inliers=inliers, inlier_ratio=inliers / len(good),
    )
    if inliers < MIN_INLIERS:
        return replace(result, reason="Too few RANSAC inliers (minimum 8).")
    if result.inlier_ratio < MIN_INLIER_RATIO:
        return replace(result, reason="Inlier ratio below working minimum 0.50.")
    if not MIN_SCALE <= result.scale <= MAX_SCALE:
        return replace(result, reason="Scale outside working sanity range [0.5, 2.0].")
    if abs(result.rotation_degrees) > ROTATION_TOLERANCE_DEGREES:
        return replace(result, reason="Rotation exceeds working tolerance 3 degrees.")
    warp = np.array([[result.scale, 0, result.dx], [0, result.scale, result.dy]])
    size = (gray_a.shape[1], gray_a.shape[0])
    aligned = cv2.warpAffine(gray_b, warp, size, flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    support = cv2.warpAffine(np.ones(gray_b.shape, dtype=np.float32), warp, size,
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    overlap = support == 1.0
    if not np.any(overlap):
        return replace(result, reason="No fully supported overlap after alignment.")
    # Nx1 grayscale arrays preserve the existing public metrics' image contract.
    pixels_a = gray_a[overlap].reshape(-1, 1)
    pixels_b = aligned[overlap].reshape(-1, 1)
    return replace(
        result, valid=True, aligned_mad=compute_mad(pixels_a, pixels_b),
        aligned_cpr=compute_changed_pixel_ratio(pixels_a, pixels_b, pixel_threshold),
    )


def _gray(image: np.ndarray) -> np.ndarray:
    """Convert already validated BGR arrays, leaving grayscale arrays unchanged."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def _build_sift_mask(config: UIConfig, image_shape: tuple[int, int]) -> np.ndarray:
    """Translate canonical exclusions into a binary map-local SIFT mask.

    Args:
        config: UIConfig with map crop and ignore rectangles in canonical
            full-frame reference pixels; no scaled or recovered layout is assumed.
        image_shape: Grayscale (height, width) matching map_analysis_region.
    Returns:
        uint8 array of that shape: 255 allows detection and 0 excludes centers.
        Local origins subtract map_analysis_region.x/y; right/bottom are exclusive.
    Raises:
        ValueError: Crop dimensions differ or an exclusion is outside the map.
    Only feature-center detection is masked. Descriptor neighborhoods can extend
    across exclusion edges; pixels and change-metric domains are not altered.
    """
    region = config.map_analysis_region
    if image_shape != (region.height, region.width):
        raise ValueError("UIConfig requires an exact canonical map_analysis_region crop.")
    mask = np.full(image_shape, 255, dtype=np.uint8)
    for roi in config.map_ignore_masks:
        if (roi.x < region.x or roi.y < region.y
                or roi.right > region.right or roi.bottom > region.bottom):
            raise ValueError("map_ignore_masks must remain inside map_analysis_region.")
        x, y = roi.x - region.x, roi.y - region.y
        mask[y:y + roi.height, x:x + roi.width] = 0
    return mask
