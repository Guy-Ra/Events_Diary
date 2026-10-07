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

from enum import Enum

from events_diary.config import ROI, UIConfig


class LayoutKind(str, Enum):
    """Supported layout relationships to the canonical UI reference."""

    EXACT_REFERENCE = "exact_reference"
    PROPORTIONAL_SCALE = "proportional_scale"
    PARTIAL_OR_CROPPED = "partial_or_cropped"


def scale_roi(
    roi: ROI,
    scale_x: float,
    scale_y: float,
) -> ROI:
    """Scale an ROI from reference coordinates to another frame size."""

    return ROI(
        x=round(roi.x * scale_x),
        y=round(roi.y * scale_y),
        width=round(roi.width * scale_x),
        height=round(roi.height * scale_y),
    )


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

    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("Frame dimensions must be positive.")
    if config.reference_width <= 0 or config.reference_height <= 0:
        raise ValueError("Reference dimensions must be positive.")

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
