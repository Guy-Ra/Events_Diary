"""Deterministically derive compass direction from an OCR TRACK observation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from events_diary.ocr_fields import OCRFieldObservation

COMPASS_SECTORS = (
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
)
COMPASS_NAMES = (
    "North", "North-Northeast", "Northeast", "East-Northeast", "East",
    "East-Southeast", "Southeast", "South-Southeast", "South",
    "South-Southwest", "Southwest", "West-Southwest", "West",
    "West-Northwest", "Northwest", "North-Northwest",
)
_TRACK_PATTERN = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))[\s\u05b2]*(?:°|º|deg(?:rees?)?)?[\s\u05b2]*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DerivedTrackDirection:
    """A compass direction derived from, and linked to, one OCR TRACK record."""

    source_observation: OCRFieldObservation
    degrees: float
    sector: str
    text: str
    provenance: Literal["deterministic_derived_from_ocr_track"] = (
        "deterministic_derived_from_ocr_track"
    )

    def __post_init__(self) -> None:
        """Ensure the derived bearing agrees with its retained OCR source."""
        from events_diary.ocr_fields import OCRFieldObservation

        if not isinstance(self.source_observation, OCRFieldObservation):
            raise TypeError("source_observation must be OCRFieldObservation.")
        if self.source_observation.canonical_label != "TRACK" or self.source_observation.status != "extracted":
            raise ValueError("source_observation must be an extracted OCR TRACK field.")
        sector, text = compass_direction(self.degrees)
        if (self.sector, self.text) != (sector, text):
            raise ValueError("Derived compass values must match the source bearing.")
        if self.provenance != "deterministic_derived_from_ocr_track":
            raise ValueError("Unsupported track-direction provenance.")


@dataclass(frozen=True)
class DirectionConsistency:
    """Compare OCR-derived and VLM-observed sectors without choosing a winner."""

    status: Literal["consistent", "inconsistent", "unavailable"]
    derived_direction: DerivedTrackDirection | None
    visual_direction_sector: str | None
    visual_direction_status: Literal["observed", "inferred", "not_visible"]

    def __post_init__(self) -> None:
        """Validate status and ensure unavailable comparisons do not claim agreement."""
        if self.status not in ("consistent", "inconsistent", "unavailable"):
            raise ValueError("Unsupported direction-consistency status.")
        if self.visual_direction_status not in ("observed", "inferred", "not_visible"):
            raise ValueError("Unsupported visual direction status.")
        available = self.derived_direction is not None and self.visual_direction_sector is not None
        if (self.status != "unavailable") != available:
            raise ValueError("Direction-consistency status must match available inputs.")
        if self.visual_direction_sector is None and self.visual_direction_status != "not_visible":
            raise ValueError("A missing visual direction must use not_visible status.")
        if self.visual_direction_sector is not None and self.visual_direction_status == "not_visible":
            raise ValueError("not_visible visual direction must not carry a sector.")


def compass_direction(degrees: float) -> tuple[str, str]:
    """Map a finite heading in [0, 360] to its nearest 16-point compass sector.

    Each sector is centered on its compass bearing and spans 22.5 degrees.
    Exact halfway boundaries round clockwise; 360 degrees wraps to North.
    """
    if isinstance(degrees, bool) or not isinstance(degrees, Real):
        raise TypeError("degrees must be a real number, not bool.")
    if not math.isfinite(degrees) or not 0 <= degrees <= 360:
        raise ValueError("degrees must be finite and within [0, 360].")
    index = math.floor((float(degrees) + 11.25) / 22.5) % 16
    return COMPASS_SECTORS[index], COMPASS_NAMES[index]


def derive_track_direction(observation: OCRFieldObservation) -> DerivedTrackDirection | None:
    """Return deterministic direction for a usable OCR TRACK observation.

    Missing, unselected, malformed, non-finite, or out-of-range OCR text returns
    None. The original immutable OCR observation remains attached as provenance.
    """
    from events_diary.ocr_fields import OCRFieldObservation

    if not isinstance(observation, OCRFieldObservation):
        raise TypeError("observation must be OCRFieldObservation.")
    if observation.canonical_label != "TRACK" or observation.status != "extracted":
        return None
    raw = observation.raw_value_text
    match = _TRACK_PATTERN.fullmatch(raw or "")
    if match is None:
        return None
    degrees = float(match.group(1))
    if not math.isfinite(degrees) or not 0 <= degrees <= 360:
        return None
    sector, text = compass_direction(degrees)
    return DerivedTrackDirection(observation, degrees, sector, text)


def compare_track_direction(
    derived: DerivedTrackDirection | None,
    visual_direction_sector: str | None,
    *,
    visual_direction_status: Literal["observed", "inferred", "not_visible"] | None = None,
) -> DirectionConsistency:
    """Report sector agreement while preserving both evidence inputs."""
    if derived is not None and not isinstance(derived, DerivedTrackDirection):
        raise TypeError("derived must be DerivedTrackDirection or None.")
    if visual_direction_sector is not None:
        if not isinstance(visual_direction_sector, str):
            raise TypeError("visual_direction_sector must be a string or None.")
        visual_direction_sector = visual_direction_sector.upper()
        if visual_direction_sector not in COMPASS_SECTORS:
            raise ValueError("visual_direction_sector must be a 16-point compass sector.")
    if visual_direction_status is None:
        visual_direction_status = "observed" if visual_direction_sector is not None else "not_visible"
    if visual_direction_status not in ("observed", "inferred", "not_visible"):
        raise ValueError("Unsupported visual_direction_status.")
    if visual_direction_status == "not_visible":
        visual_direction_sector = None
    status = "unavailable" if derived is None or visual_direction_sector is None else (
        "consistent" if derived.sector == visual_direction_sector else "inconsistent"
    )
    return DirectionConsistency(status, derived, visual_direction_sector, visual_direction_status)
