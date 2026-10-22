"""Application configuration definitions.

This module contains configuration objects used by Events Diary.
Configuration values are kept separate from runtime behavior so that
settings can be managed centrally and extended as the project grows.
"""

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from dotenv import load_dotenv


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
class VLMAPISettings:
    """Immutable provider/model settings; the API key is hidden from repr output."""

    provider: str
    api_key: str = field(repr=False)
    base_url: str
    api_version: str
    model_name: str
    request_timeout_s: float

    @property
    def endpoint_url(self) -> str:
        """Build the Gemini GenerateContent endpoint from configured components."""
        return f"{self.base_url.rstrip('/')}/{self.api_version.strip('/')}/models/{self.model_name}:generateContent"

    @property
    def api_key_configured(self) -> bool:
        """Report whether credentials are present without exposing their value."""
        return bool(self.api_key.strip())


def load_vlm_api_settings(*, env_file: Path | None = None, override: bool = False) -> VLMAPISettings:
    """Load VLM settings once at the project configuration boundary.

    Existing process environment variables retain precedence by default. Passing
    an explicit dotenv path is useful for isolated tests and project tooling.
    """
    dotenv_path = env_file or Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(dotenv_path=dotenv_path, override=override)

    def get(name: str, default: str) -> str:
        """Read one already-loaded setting, using its safe project default."""
        return os.environ.get(name, default)

    def number(name: str, default: str, kind):
        """Parse a numeric setting and raise a non-sensitive configuration error."""
        try:
            value = kind(get(name, default))
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be a valid {kind.__name__}.") from error
        return value

    settings = VLMAPISettings(
        provider=get("EVENTS_DIARY_VLM_PROVIDER", "gemini").strip().lower(),
        api_key=get("EVENTS_DIARY_VLM_API_KEY", ""),
        base_url=get("EVENTS_DIARY_VLM_API_BASE_URL", "https://generativelanguage.googleapis.com").strip(),
        api_version=get("EVENTS_DIARY_VLM_API_VERSION", "v1beta").strip(),
        model_name=get("EVENTS_DIARY_VLM_MODEL", "gemini-flash-latest").strip(),
        request_timeout_s=number("EVENTS_DIARY_VLM_REQUEST_TIMEOUT_S", "180", float),
    )
    if settings.provider != "gemini":
        raise ValueError("Only the 'gemini' VLM provider is currently supported.")
    if not settings.base_url or not settings.api_version or not settings.model_name:
        raise ValueError("VLM API base URL, API version, and model name must be nonempty.")
    if not math.isfinite(settings.request_timeout_s) or settings.request_timeout_s <= 0:
        raise ValueError("EVENTS_DIARY_VLM_REQUEST_TIMEOUT_S must be positive.")
    return settings


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

    # Canonical full-frame rectangles: SIFT detection exclusions only.
    # Slight padding covers overlay edges; these are not metric masks.
    map_ignore_masks: tuple[ROI, ...] = (
        ROI(352, 121, 1568, 60),   # Persistent top FR24 strip.
        ROI(1856, 284, 60, 218),   # Zoom, fullscreen, navigation controls.
        ROI(746, 958, 428, 70),    # Settings/weather/filters/widgets/playback.
        ROI(1480, 1024, 440, 16), # Attribution and scale footer.
    )
