"""Measure route-colored pattern change without semantic selection detection.

For preferably consecutive full frames, reuse map-motion estimation to compensate
global motion before comparing HSV color masks on valid overlap only. Map B is
aligned to A with accepted uniform scale and translation; diagnostic rotation is
not applied. Padding and partially supported interpolation pixels are excluded.

The HSV range H=60..131, S=118..255, V=167..255 is provisional, derived from one
manual recording's blue/cyan/green trail cores. OpenCV hue uses 0..179, not degrees.
Red highlight is intentionally excluded because hover can resemble selection.
Color masks do not establish route connectivity, ownership, selected-aircraft
identity, or Investigation change. Blue map icons, boundaries, and labels remain
possible false positives. Measurement performs no classification, temporal logic,
or route extraction. Fractions use overlap area, never route area, as denominator.
A separate pair-level classifier requires bidirectional mask replacement using
provisional working defaults. This is not Investigation detection or temporal logic.
"""

import math
from dataclasses import dataclass
from numbers import Real

import cv2
import numpy as np

from events_diary.config import UIConfig
from events_diary.layout import LayoutKind, classify_layout
from events_diary.map_motion import MapMotionEstimate, estimate_map_motion

ROUTE_HSV_LOWER = (60, 118, 167)
ROUTE_HSV_UPPER = (131, 255, 255)


@dataclass(frozen=True)
class RouteSelectionChangeEvidence:
    """Immutable overlap-normalized route-color measurements for one frame pair.

    Attributes:
        valid: Whether motion compensation and a nonempty overlap are available.
        reason: Rejection explanation, or None for valid measurements.
        route_fraction_a, route_fraction_b: Route-colored occupancy of A and
            aligned B, respectively, divided by the same valid-overlap area.
        added_fraction: Non-route A becoming route-colored aligned B.
        removed_fraction: Route-colored A becoming non-route aligned B.
        changed_fraction: Union of added and removed pixels; equals their sum
            within floating-point tolerance because those sets are disjoint.

    Constructor parameters correspond to these attributes; no semantic inference
    or constructor validation occurs. The measurement API returns all fractions
    as None when invalid. These observations are not a selection-change decision.
    """

    valid: bool
    reason: str | None
    route_fraction_a: float | None = None
    route_fraction_b: float | None = None
    added_fraction: float | None = None
    removed_fraction: float | None = None
    changed_fraction: float | None = None


def estimate_route_selection_change(
    frame_a: np.ndarray, frame_b: np.ndarray, ui_config: UIConfig,
) -> RouteSelectionChangeEvidence:
    """Measure route-color additions/removals after B-to-A map compensation.

    Args:
        frame_a, frame_b: Nonempty matching uint8 NumPy full frames with exactly
            three BGR channels. Color is required; grayscale is not accepted.
        ui_config: UIConfig whose reference resolution matches the frames and
            whose map_analysis_region supplies the canonical crop.
    Returns:
        Frozen evidence. If map motion is rejected or supported overlap is empty,
        valid is False, reason explains rejection, and all fractions are None.
        Otherwise every fraction lies in [0,1] and uses count(valid overlap).
    Raises:
        TypeError: Non-array/uint8 input or non-UIConfig configuration.
        ValueError: Empty, non-BGR, mismatched/noncanonical frames or invalid ROI.
        cv2.error: Unexpected estimation/conversion/warp errors.
    Assumes canonical UI placement and dominant translation/scale; no resizing or
    layout recovery occurs. Existing estimate_map_motion receives ui_config for
    SIFT-only static UI exclusions, with unchanged defaults. Accepted rotation
    remains unused. BGR is warped BEFORE conversion/masking, so interpolation and
    threshold crossings can cause small residual mask changes. Input arrays are
    unchanged. No red mask, ownership inference, detector, or temporal logic exists.
    """
    _validate_frames(frame_a, frame_b, ui_config)
    image_a = ui_config.map_analysis_region.crop(frame_a)
    image_b = ui_config.map_analysis_region.crop(frame_b)
    motion = estimate_map_motion(image_a, image_b, ui_config=ui_config)
    if not motion.valid:
        return RouteSelectionChangeEvidence(False, f"Map motion rejected: {motion.reason}")
    aligned, overlap = _align_map(image_b, motion)
    denominator = int(np.count_nonzero(overlap))
    if denominator == 0:
        return RouteSelectionChangeEvidence(False, "No fully supported overlap after alignment.")
    mask_a, mask_b = _build_route_mask(image_a), _build_route_mask(aligned)
    added = ~mask_a & mask_b & overlap
    removed = mask_a & ~mask_b & overlap
    return RouteSelectionChangeEvidence(
        True, None,
        float(np.count_nonzero(mask_a & overlap) / denominator),
        float(np.count_nonzero(mask_b & overlap) / denominator),
        float(np.count_nonzero(added) / denominator),
        float(np.count_nonzero(removed) / denominator),
        float(np.count_nonzero(added | removed) / denominator),
    )


def is_route_selection_change(
    evidence: RouteSelectionChangeEvidence, *, min_changed_fraction: float = 0.0003,
    min_added_fraction: float = 0.0001, min_removed_fraction: float = 0.0001,
) -> bool:
    """Check one pair for bidirectional route-colored pattern replacement.

    Args:
        evidence: Existing overlap-normalized measurements for one frame pair.
        min_changed_fraction: Inclusive minimum changed overlap fraction (0.0003).
        min_added_fraction: Inclusive minimum added overlap fraction (0.0001).
        min_removed_fraction: Inclusive minimum removed overlap fraction (0.0001).
    Returns:
        True only for valid evidence meeting all three thresholds. Invalid evidence
        or missing/non-finite required fractions returns False. Occupancy fractions
        do not enter the decision; evidence is never mutated.
    Raises:
        TypeError: Non-real/bool threshold or non-RouteSelectionChangeEvidence input.
        ValueError: Non-finite/out-of-range threshold (must be in [0,1]).
    Thresholds are validated before checking unavailable evidence. Defaults are
    provisional working values derived from one investigation-switch recording,
    requiring additions AND removals to distinguish replacement from appearance.
    These are pair-level color-pattern evidence, not proof of route ownership,
    selection identity, a new Investigation, or a temporal event. No tuning,
    coordinate conversion, or new measurement occurs here.
    """
    for name, value in (("min_changed_fraction", min_changed_fraction),
                        ("min_added_fraction", min_added_fraction),
                        ("min_removed_fraction", min_removed_fraction)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real number, not a boolean.")
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be finite and in [0, 1].")
    if not isinstance(evidence, RouteSelectionChangeEvidence):
        raise TypeError("evidence must be RouteSelectionChangeEvidence.")
    if not evidence.valid:
        return False
    fractions = (evidence.changed_fraction, evidence.added_fraction, evidence.removed_fraction)
    if any(value is None or not math.isfinite(value) for value in fractions):
        return False
    return bool(evidence.changed_fraction >= min_changed_fraction
                and evidence.added_fraction >= min_added_fraction
                and evidence.removed_fraction >= min_removed_fraction)


def _build_route_mask(image: np.ndarray) -> np.ndarray:
    """Return a boolean route-color mask for a validated BGR map crop.

    image uses map-local pixels. Returns same-height/width boolean membership in
    inclusive provisional HSV bounds; OpenCV H=0..179, S/V=0..255. Red is excluded.
    Assumes uint8 three-channel BGR; OpenCV errors propagate. No morphology, route
    geometry, or spatial classification is performed and input pixels are unchanged.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.uint8(ROUTE_HSV_LOWER), np.uint8(ROUTE_HSV_UPPER)) != 0


def _align_map(image: np.ndarray, motion: MapMotionEstimate) -> tuple[np.ndarray, np.ndarray]:
    """Return B aligned into A and a fully supported map-local overlap mask.

    image is a BGR Map B crop; motion is a valid existing B-to-A estimate. Returns
    aligned B and boolean support for an equal-sized Map A. No rotation is applied.
    Uses linear interpolation and constant-zero borders; an identically warped
    float32 all-ones image must equal 1 to indicate full source support.
    This small local helper is necessary because map_motion exposes measurements
    only and has no reusable warp/support helper. It preserves those semantics
    without refactoring production estimation. cv2 errors propagate; assumes valid
    finite motion components and matching map dimensions. Inputs are not mutated.
    """
    matrix = np.float64([[motion.scale, 0, motion.dx], [0, motion.scale, motion.dy]])
    size = (image.shape[1], image.shape[0])
    aligned = cv2.warpAffine(image, matrix, size, flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    support = cv2.warpAffine(np.ones(image.shape[:2], np.float32), matrix, size,
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return aligned, support == 1.0


def _validate_frames(frame_a, frame_b, config):
    """Validate canonical full-frame BGR inputs before cropping.

    frame_a/B are candidate arrays and config is UIConfig. Returns None; TypeError
    rejects array/dtype/config violations, ValueError rejects empty/non-BGR,
    mismatched, or noncanonical dimensions. Coordinates remain reference pixels;
    existing ROI.crop handles bounds. Color is required for HSV evidence.
    """
    if not isinstance(config, UIConfig):
        raise TypeError("ui_config must be UIConfig.")
    for name, frame in (("frame_a", frame_a), ("frame_b", frame_b)):
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
            raise TypeError(f"{name} must be a NumPy array with dtype np.uint8.")
        if frame.size == 0:
            raise ValueError(f"{name} must not be empty.")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"{name} must have exactly three BGR channels.")
    if frame_a.shape != frame_b.shape:
        raise ValueError("Frames must have identical shapes.")
    height, width = frame_a.shape[:2]
    if classify_layout(config, width, height) != LayoutKind.EXACT_REFERENCE:
        raise ValueError(f"Expected canonical {config.reference_width}x{config.reference_height} "
                         f"full frames; got {width}x{height}.")
