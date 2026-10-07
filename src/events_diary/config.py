"""Application configuration definitions.

This module contains configuration objects used by Events Diary.
Configuration values are kept separate from runtime behavior so that
settings can be managed centrally and extended as the project grows.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AppConfig:
    """Application-level configuration.

    Attributes:
        log_level:
            Logging level used by the application, such as
            "DEBUG", "INFO", "WARNING", or "ERROR".

    The configuration is immutable after creation to prevent accidental
    modification during application execution.
    """

    log_level: str = "INFO"


@dataclass(frozen=True)
class ROI:
    """A rectangular region of interest in image pixel coordinates."""

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        for name in ("x", "y", "width", "height"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"ROI {name} must be an integer.")
        if self.x < 0:
            raise ValueError("ROI x coordinate must be non-negative.")
        if self.y < 0:
            raise ValueError("ROI y coordinate must be non-negative.")
        if self.width <= 0:
            raise ValueError("ROI width must be positive.")
        if self.height <= 0:
            raise ValueError("ROI height must be positive.")

    @property
    def right(self) -> int:
        """Return the exclusive right boundary of the ROI."""
        return self.x + self.width

    @property
    def bottom(self) -> int:
        """Return the exclusive bottom boundary of the ROI."""
        return self.y + self.height

    def crop(self, frame: np.ndarray) -> np.ndarray:
        """Return the ROI crop from a frame.

        Raises:
            ValueError: If the ROI extends beyond the frame boundaries.
        """
        frame_height, frame_width = frame.shape[:2]

        if self.right > frame_width or self.bottom > frame_height:
            raise ValueError(
                "ROI extends beyond frame boundaries: "
                f"roi=({self.x}, {self.y}, {self.width}, {self.height}), "
                f"frame={frame_width}x{frame_height}."
            )

        return frame[self.y : self.bottom, self.x : self.right]

@dataclass(frozen=True)
class UIConfig:
    """Canonical Flightradar24 UI layout configuration."""

    reference_width: int = 1920
    reference_height: int = 1080

    browser_chrome: ROI = ROI(0, 0, 1920, 121)
    app_region: ROI = ROI(0, 121, 1920, 919)

    ticket_zone: ROI = ROI(16, 121, 336, 887)
    ticket_header: ROI = ROI(16, 121, 336, 76)
    aircraft_image: ROI = ROI(16, 197, 336, 189)
    ticket_scroll_body: ROI = ROI(16, 386, 336, 566)
    ticket_bottom_actions: ROI = ROI(16, 952, 336, 56)

    map_analysis_region: ROI = ROI(352, 121, 1568, 919)

    system_taskbar: ROI = ROI(0, 1040, 1920, 40)

    map_ignore_masks: tuple[ROI, ...] = ()
