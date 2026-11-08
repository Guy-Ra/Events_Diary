"""Application configuration definitions.

This module contains configuration objects used by Events Diary.
Configuration values are kept separate from runtime behavior so that
settings can be managed centrally and extended as the project grows.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from enum import Enum
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


class UnderstandingMode(str, Enum):
    """One authoritative selection for OCR/VLM Investigation understanding."""

    OCR_ONLY = "ocr_only"
    VLM_ONLY = "vlm_only"
    OCR_VLM = "ocr_vlm"

    @property
    def uses_vlm(self) -> bool:
        """Return whether the selected mode requires configured VLM settings."""
        return self in (UnderstandingMode.VLM_ONLY, UnderstandingMode.OCR_VLM)


@dataclass(frozen=True)
class RuntimeConfig:
    """Centralized immutable runtime choices for one MVP video run."""

    understanding_mode: UnderstandingMode
    input_dir: Path
    output_dir: Path
    vlm_api_settings: VLMAPISettings | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Reject inconsistent mode, path, and VLM configuration contracts."""
        if not isinstance(self.understanding_mode, UnderstandingMode):
            raise TypeError("understanding_mode must be an UnderstandingMode.")
        if not isinstance(self.input_dir, Path) or not isinstance(self.output_dir, Path):
            raise TypeError("input_dir and output_dir must be pathlib.Path values.")
        if self.understanding_mode.uses_vlm and not isinstance(
            self.vlm_api_settings, VLMAPISettings,
        ):
            raise ValueError("The selected understanding mode requires centralized VLM settings.")
        if not self.understanding_mode.uses_vlm and self.vlm_api_settings is not None:
            raise ValueError("ocr_only must not load or receive VLM settings.")


@dataclass(frozen=True)
class VLMAPISettings:
    """Immutable provider/model settings; the API key is hidden from repr output."""

    provider: str
    api_key: str = field(repr=False)
    base_url: str
    api_version: str
    model_name: str
    request_timeout_s: float
    max_attempts: int = 3
    retry_base_delay_s: float = 5.0
    retry_max_delay_s: float = 60.0
    retry_jitter_s: float = 1.0

    def __post_init__(self) -> None:
        """Validate the bounded direct-provider retry configuration."""
        if isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, int):
            raise TypeError("max_attempts must be an integer.")
        if not 1 <= self.max_attempts <= 3:
            raise ValueError("max_attempts must be between 1 and 3.")
        for name in ("retry_base_delay_s", "retry_max_delay_s", "retry_jitter_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric.")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if self.retry_max_delay_s < self.retry_base_delay_s:
            raise ValueError("retry_max_delay_s must be at least retry_base_delay_s.")

    @property
    def endpoint_url(self) -> str:
        """Build the Gemini GenerateContent endpoint from configured components."""
        return f"{self.base_url.rstrip('/')}/{self.api_version.strip('/')}/models/{self.model_name}:generateContent"

    @property
    def api_key_configured(self) -> bool:
        """Report whether credentials are present without exposing their value."""
        return bool(self.api_key.strip())


def load_runtime_config(
    *,
    project_root: Path | None = None,
    env_file: Path | None = None,
    override: bool = False,
) -> RuntimeConfig:
    """Load understanding mode, fixed workspaces, and existing VLM settings.

    The mode is read from ``EVENTS_DIARY_UNDERSTANDING_MODE`` and defaults to
    ``ocr_vlm``. Workspace paths are rooted at the repository unless a project
    root is explicitly supplied by a caller or test. VLM credentials and
    provider details continue to be loaded only through
    :func:`load_vlm_api_settings`.
    """
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    dotenv_path = env_file if env_file is not None else root / ".env"
    load_dotenv(dotenv_path=dotenv_path, override=override)
    raw_mode = os.environ.get("EVENTS_DIARY_UNDERSTANDING_MODE", "ocr_vlm").strip().lower()
    try:
        mode = UnderstandingMode(raw_mode)
    except ValueError as error:
        allowed = ", ".join(item.value for item in UnderstandingMode)
        raise ValueError(
            f"EVENTS_DIARY_UNDERSTANDING_MODE must be one of: {allowed}."
        ) from error

    vlm_settings = (
        load_vlm_api_settings(env_file=dotenv_path, override=override)
        if mode.uses_vlm
        else None
    )
    return RuntimeConfig(
        understanding_mode=mode,
        input_dir=root / "experiment" / "input",
        output_dir=root / "experiment" / "output",
        vlm_api_settings=vlm_settings,
    )


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

    max_attempts = number("EVENTS_DIARY_VLM_MAX_ATTEMPTS", "3", int)

    settings = VLMAPISettings(
        provider=get("EVENTS_DIARY_VLM_PROVIDER", "gemini").strip().lower(),
        api_key=get("EVENTS_DIARY_VLM_API_KEY", ""),
        base_url=get("EVENTS_DIARY_VLM_API_BASE_URL", "https://generativelanguage.googleapis.com").strip(),
        api_version=get("EVENTS_DIARY_VLM_API_VERSION", "v1beta").strip(),
        model_name=get("EVENTS_DIARY_VLM_MODEL", "gemini-flash-latest").strip(),
        request_timeout_s=number("EVENTS_DIARY_VLM_REQUEST_TIMEOUT_S", "180", float),
        max_attempts=max_attempts,
        retry_base_delay_s=number("EVENTS_DIARY_VLM_RETRY_BASE_DELAY_S", "5", float),
        retry_max_delay_s=number("EVENTS_DIARY_VLM_RETRY_MAX_DELAY_S", "60", float),
        retry_jitter_s=number("EVENTS_DIARY_VLM_RETRY_JITTER_S", "1", float),
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
