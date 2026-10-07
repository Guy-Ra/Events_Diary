"""
Debug visualization utilities for Flightradar24 UI regions.

This module draws configured regions of interest on top of video frames so the
canonical UI layout can be visually validated during development.

It is intended for debugging and calibration only and is not part of the core
analysis pipeline.
"""

from collections.abc import Mapping

import cv2
import numpy as np

from events_diary.config import ROI


def draw_rois(
    frame: np.ndarray,
    regions: Mapping[str, ROI],
) -> np.ndarray:
    """Draw named ROI rectangles on a copy of a frame."""

    output = frame.copy()

    for name, roi in regions.items():
        cv2.rectangle(
            output,
            (roi.x, roi.y),
            (roi.right - 1, roi.bottom - 1),
            color=(0, 255, 0),
            thickness=2,
        )

        cv2.putText(
            output,
            name,
            (roi.x + 5, roi.y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color=(0, 255, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    return output
