"""
Layout utilities for mapping canonical Flightradar24 UI regions to video frames.

The project defines its UI regions against a canonical 1920x1080 reference
layout. This module provides the geometry and classification logic needed to
adapt those canonical regions to frames with different resolutions.

It currently supports:
- Exact reference resolution mapping.
- Proportional scaling for frames that preserve the reference aspect ratio.
- Detection of frames that are likely cropped or otherwise do not match the
  canonical layout.

This module does not perform UI detection, OCR, visual analysis, or VLM-based
layout recovery. Those capabilities are handled by later stages of the
pipeline.
"""

import math
from dataclasses import dataclass
from enum import Enum

from events_diary.config import ROI, UIConfig


class LayoutKind(str, Enum):
    """Supported layout relationships to the canonical UI reference."""

    EXACT_REFERENCE = "exact_reference"
    PROPORTIONAL_SCALE = "proportional_scale"
    PARTIAL_OR_CROPPED = "partial_or_cropped"


@dataclass(frozen=True)
class LayoutTransform:
    """A supplied reference-to-frame transform; performs no layout estimation."""

    scale_x: float = 1.0
    scale_y: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0

    def __post_init__(self) -> None:
        for name in ("scale_x", "scale_y"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in ("offset_x", "offset_y"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite.")


@dataclass(frozen=True)
class TransformedBounds:
    """Unrounded, possibly negative frame-space edges; not a crop-able ROI."""

    left: float
    top: float
    right: float
    bottom: float


class ROIVisibility(str, Enum):
    """Visibility of the complete unrounded region within a frame."""

    FULL = "full"
    PARTIAL = "partial"
    OUTSIDE = "outside"


@dataclass(frozen=True)
class MappedROI:
    """Preserve original geometry and visibility alongside the visible crop."""

    transformed_bounds: TransformedBounds
    visibility: ROIVisibility
    visible_roi: ROI | None


def _transform_bounds(roi: ROI, transform: LayoutTransform) -> TransformedBounds:
    bounds = TransformedBounds(
        left=roi.x * transform.scale_x + transform.offset_x,
        top=roi.y * transform.scale_y + transform.offset_y,
        right=roi.right * transform.scale_x + transform.offset_x,
        bottom=roi.bottom * transform.scale_y + transform.offset_y,
    )
    if not all(math.isfinite(edge) for edge in (
        bounds.left, bounds.top, bounds.right, bounds.bottom,
    )):
        raise ValueError("Transformed bounds must be finite.")
    if bounds.right <= bounds.left or bounds.bottom <= bounds.top:
        raise ValueError("Transformed bounds collapsed to zero area.")
    return bounds


def _bounds_to_roi(bounds: TransformedBounds) -> ROI:
    # Shared edges use the same nearest-integer rounding (ties to even).
    left, top = round(bounds.left), round(bounds.top)
    right, bottom = round(bounds.right), round(bounds.bottom)
    if right <= left or bottom <= top:
        raise ValueError("Visible geometry collapsed to zero pixel area after rounding.")
    return ROI(left, top, right - left, bottom - top)


def map_roi_with_transform(
    roi: ROI,
    transform: LayoutTransform,
    frame_width: int,
    frame_height: int,
) -> MappedROI:
    """Map supplied geometry, preserving visibility before pixel rounding.

    Raises ValueError for invalid dimensions or visible geometry that rounds
    to zero pixel area. Edge-only contact is OUTSIDE, with no visible ROI.
    """
    if (not math.isfinite(frame_width) or not math.isfinite(frame_height)
            or frame_width <= 0 or frame_height <= 0):
        raise ValueError("Frame dimensions must be positive and finite.")

    bounds = _transform_bounds(roi, transform)
    intersection = TransformedBounds(
        left=max(0.0, bounds.left),
        top=max(0.0, bounds.top),
        right=min(frame_width, bounds.right),
        bottom=min(frame_height, bounds.bottom),
    )
    if intersection.right <= intersection.left or intersection.bottom <= intersection.top:
        return MappedROI(bounds, ROIVisibility.OUTSIDE, None)

    full = (bounds.left >= 0 and bounds.top >= 0
            and bounds.right <= frame_width and bounds.bottom <= frame_height)
    visibility = ROIVisibility.FULL if full else ROIVisibility.PARTIAL
    return MappedROI(bounds, visibility, _bounds_to_roi(intersection))


def scale_roi(
    roi: ROI,
    scale_x: float,
    scale_y: float,
) -> ROI:
    """Scale an ROI from reference coordinates to another frame size."""

    return _bounds_to_roi(_transform_bounds(roi, LayoutTransform(scale_x, scale_y)))


def map_roi_to_frame(
    roi: ROI,
    config: UIConfig,
    frame_width: int,
    frame_height: int,
) -> ROI:
    """Map a canonical ROI to a frame with a compatible layout."""

    layout_kind = classify_layout(
        config=config,
        frame_width=frame_width,
        frame_height=frame_height,
    )

    if layout_kind == LayoutKind.PARTIAL_OR_CROPPED:
        raise ValueError(
            "Cannot safely map ROI using proportional scaling for a "
            f"partial or cropped frame: {frame_width}x{frame_height}."
        )

    scale_x = frame_width / config.reference_width
    scale_y = frame_height / config.reference_height

    return scale_roi(
        roi=roi,
        scale_x=scale_x,
        scale_y=scale_y,
    )

def classify_layout(
    config: UIConfig,
    frame_width: int,
    frame_height: int,
) -> LayoutKind:
    """Classify a frame relative to the canonical UI layout."""

    if (not math.isfinite(frame_width) or not math.isfinite(frame_height)
            or frame_width <= 0 or frame_height <= 0):
        raise ValueError("Frame dimensions must be positive and finite.")
    if (not math.isfinite(config.reference_width)
            or not math.isfinite(config.reference_height)
            or config.reference_width <= 0 or config.reference_height <= 0):
        raise ValueError("Reference dimensions must be positive and finite.")

    if (
        frame_width == config.reference_width
        and frame_height == config.reference_height
    ):
        return LayoutKind.EXACT_REFERENCE

    same_aspect_ratio = (
        frame_width * config.reference_height
        == frame_height * config.reference_width
    )

    if same_aspect_ratio:
        return LayoutKind.PROPORTIONAL_SCALE

    return LayoutKind.PARTIAL_OR_CROPPED
