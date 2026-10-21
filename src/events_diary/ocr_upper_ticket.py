"""FR24-specific upper-ticket observations and complete per-frame OCR output.

The known MVP layout has a pinned identifier strip, an airline row, a photo and
two airport blocks above the SCHEDULED row. All gates are fractions of the actual
ticket crop; geometry comes from unchanged Phase 6.2 observations. Header token
shapes disambiguate positional roles, never look up or correct an identity. Route
slots are relative to each side's SCHEDULED label so a partially scrolled view
can retain a timezone while its code/city are unavailable. The photo/content
boundary prevents assigning photo text to a scrolled-out airport slot.

These explicit working layout gates support the current FR24 full-ticket MVP,
not arbitrary layouts. Missing detections, multiple candidates, overlapping
columns and inconsistent anchors abstain; no external airline/airport database,
fuzzy value repair, cross-frame propagation or image arrays are used. A route
requires both extracted airport codes; cities and timezones are independent
raw observations, not inferred from those codes. Left means origin and right
means destination solely by the approved FR24 positional convention.

The combined API retains existing Phase 6.3b fields exactly and wraps time
occurrences with explicit side metadata without changing their label, selected
value, geometry or scores. Frozen observation references preserve original raw
text, polygons, indexes and engine confidence. Layout methods are not confidence
probabilities. This is OCR-only: no VLM, fusion, multi-view ticket assembly or
final Investigation object is created.

Three separated strip detections use left-to-right primary/alternate/aircraft
roles with shape checks. In incomplete/crowded strips, bounded left-edge slots
prevent an aircraft code such as AT76 or a flight token such as J222 shifting
into another role. Unsupported shapes/layouts remain unavailable or ambiguous.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

from events_diary.ocr_fields import (
    OCRFieldObservation,
    OCRFieldProvenance,
    extract_ocr_fields,
)
from events_diary.ocr_spatial import SpatialOCRObservation, build_spatial_observations
from events_diary.ticket_ocr import OCRTicketResult

HEADER_FIELDS = ("airline_name", "primary_flight_id", "alternate_flight_id", "aircraft_code")
AIRPORT_FIELDS = tuple(f"{side}_{part}" for side in ("origin", "destination")
                       for part in ("airport_code", "city", "timezone_text"))
_PRIMARY = re.compile(r"[A-Z]{3}\d{1,4}[A-Z]?")
_ALTERNATE = re.compile(r"[A-Z0-9]{2}\d{1,4}[A-Z]?")
_AIRCRAFT = re.compile(r"[A-Z][A-Z0-9]\d[A-Z0-9]")
_AIRPORT = re.compile(r"[A-Z]{3}")
_TIMEZONE = re.compile(r"[A-Z0-9+\-]+\s*\(\s*UTC\s*[+\-]\d{2}:\d{2}\s*\)")


@dataclass(frozen=True)
class FR24TextObservation:
    """One semantic slot with exact raw evidence or an explicit local abstention.

    evidence contains the selected original observation when extracted, all
    competing observations when ambiguous, or no observations when unavailable.
    No normalized replacement text or combined semantic confidence is stored.
    """

    field_name: str
    status: str
    evidence: tuple[SpatialOCRObservation, ...]
    method: str
    reason: str | None

    @property
    def raw_text(self) -> str | None:
        """Return exact selected engine text, never a guessed value for abstentions."""
        return self.evidence[0].text if self.status == "extracted" else None


@dataclass(frozen=True)
class FR24RouteObservation:
    """Positional origin/destination code evidence from one physical frame only."""

    origin: FR24TextObservation
    destination: FR24TextObservation

    @property
    def status(self) -> str:
        """Require both selected codes; preserve ambiguity before unavailability."""
        if "ambiguous" in (self.origin.status, self.destination.status):
            return "ambiguous"
        return "extracted" if self.origin.raw_text and self.destination.raw_text else "unavailable"

    @property
    def route_text(self) -> str | None:
        """Format the two literal code strings, adding only the directional arrow."""
        if self.status != "extracted":
            return None
        return f"{self.origin.raw_text} → {self.destination.raw_text}"


@dataclass(frozen=True)
class FR24UpperTicketObservation:
    """Ten explicit upper slots and route semantics with shared physical provenance."""

    provenance: OCRFieldProvenance
    fields: tuple[FR24TextObservation, ...]
    route: FR24RouteObservation

    def get(self, field_name: str) -> FR24TextObservation:
        """Retrieve a supported slot; unknown semantic names raise KeyError."""
        for field in self.fields:
            if field.field_name == field_name:
                return field
        raise KeyError(field_name)


@dataclass(frozen=True)
class FR24TimeFieldObservation:
    """An unchanged Phase 6.3b occurrence with explicit positional role metadata.

    visual_side is left/right or unknown for a box crossing the crop midpoint.
    route_role is origin/destination by FR24 convention, not inferred travel data.
    """

    field: OCRFieldObservation
    visual_side: str
    route_role: str | None


@dataclass(frozen=True)
class FR24TicketObservation:
    """One frame's upper observations, original lower fields and positioned times."""

    upper: FR24UpperTicketObservation
    fields: tuple[OCRFieldObservation, ...]
    time_fields: tuple[FR24TimeFieldObservation, ...]


def _side(observation: SpatialOCRObservation) -> str:
    """Assign a whole source box to a crop half; crossing boxes remain unknown."""
    bounds = observation.normalized_bounds
    if bounds.left >= 0 and bounds.right <= .5:
        return "left"
    if bounds.left >= .5 and bounds.right <= 1:
        return "right"
    return "unknown"


def _slot(name, candidates, method, reason="not_visible_or_not_detected"):
    """Select only a unique layout-compatible raw detection; retain competing evidence."""
    candidates = tuple(sorted(candidates, key=lambda o: o.detection_index))
    status = "extracted" if len(candidates) == 1 else ("ambiguous" if candidates else "unavailable")
    return FR24TextObservation(name, status, candidates, method,
                               None if status == "extracted" else ("multiple_candidates" if candidates else reason))


def _upper(result, observations, fields):
    """Interpret fixed header slots and SCHEDULED-relative airport regions independently."""
    provenance = OCRFieldProvenance(result.investigation_index, result.candidate_index,
        result.frame_index, result.actual_timestamp_seconds, result.engine_name,
        result.preprocessing_name, result.resolution_status)
    strip = [o for o in observations if .025 <= o.normalized_bounds.center_y <= .060
             and 0 <= o.normalized_bounds.left and o.normalized_bounds.right <= .70]
    ordered_strip = sorted(strip, key=lambda o: o.normalized_bounds.left)
    patterns = (_PRIMARY, _ALTERNATE, _AIRCRAFT)
    positional_strip = (len(ordered_strip) == 3
        and all(pattern.fullmatch(o.text.strip()) for pattern, o in zip(patterns, ordered_strip))
        and all(a.normalized_bounds.right < b.normalized_bounds.left
                for a, b in pairwise(ordered_strip)))
    slots = []
    for index, (name, pattern) in enumerate((("primary_flight_id", _PRIMARY), ("alternate_flight_id", _ALTERNATE),
                                           ("aircraft_code", _AIRCRAFT))):
        left_min, left_max = ((0, .10), (.20, .36), (.36, .64))[index]
        candidates = [ordered_strip[index]] if positional_strip else [o for o in strip
            if pattern.fullmatch(o.text.strip())
            and left_min <= o.normalized_bounds.left < left_max]
        slots.append(_slot(name, candidates,
                           "fr24_identifier_strip_and_token_shape"))
    airline = [o for o in observations if .060 < o.normalized_bounds.center_y <= .088
               and 0 <= o.normalized_bounds.left and o.normalized_bounds.right <= .70
               and o.text.strip() and any(c.isalpha() for c in o.text)]
    slots.append(_slot("airline_name", airline, "fr24_airline_row"))
    anchors = {side: [f.label_match.observation for f in fields
                     if f.canonical_label == "SCHEDULED" and _side(f.label_match.observation) == side]
               for side in ("left", "right")}
    inconsistent = (len(anchors["left"]) == len(anchors["right"]) == 1 and
        abs(anchors["left"][0].normalized_bounds.center_y -
            anchors["right"][0].normalized_bounds.center_y) > .02)
    for side, role in (("left", "origin"), ("right", "destination")):
        anchor = anchors[side]
        for part in ("airport_code", "city", "timezone_text"):
            name = f"{role}_{part}"
            method = "fr24_" + side + "_block_above_scheduled"
            if inconsistent or len(anchor) > 1:
                slots.append(FR24TextObservation(name, "ambiguous", (), method, "ambiguous_route_anchor"))
                continue
            if not anchor:
                slots.append(_slot(name, (), method, "scheduled_anchor_unavailable"))
                continue
            ay = anchor[0].normalized_bounds.center_y
            candidates = []
            for o in observations:
                b = o.normalized_bounds
                if _side(o) != side or b.top < .315:
                    continue  # Fixed photo/content boundary; never read aircraft-photo text as route.
                delta = ay - b.center_y
                raw = o.text.strip()
                if part == "airport_code":
                    accepted = .075 <= delta <= .115 and .025 <= b.height <= .055 and _AIRPORT.fullmatch(raw)
                elif part == "city":
                    accepted = (.043 <= delta <= .073 and .008 <= b.height <= .027
                                and any(c.isalpha() for c in raw)
                                and all(c.isalpha() or c in " -.'’" for c in raw))
                else:
                    accepted = .012 <= delta <= .043 and _TIMEZONE.fullmatch(raw)
                if accepted:
                    candidates.append(o)
            slots.append(_slot(name, candidates, method))
    ordered = tuple(next(s for s in slots if s.field_name == name) for name in HEADER_FIELDS + AIRPORT_FIELDS)
    by_name = {s.field_name: s for s in ordered}
    return FR24UpperTicketObservation(provenance, ordered,
        FR24RouteObservation(by_name["origin_airport_code"], by_name["destination_airport_code"]))


def extract_fr24_ticket(ocr_result: OCRTicketResult) -> FR24TicketObservation:
    """Add upper semantics to unchanged lower extraction for one resolved OCR result.

    Existing lower-field defaults/validation are delegated unchanged. Results never
    copy missing route/header values from another representative or Investigation.
    Time occurrences remain separate, with side metadata even if a route is absent.
    """
    fields = extract_ocr_fields(ocr_result)
    observations = build_spatial_observations(ocr_result)
    upper = _upper(ocr_result, observations, fields)
    times = []
    for field in fields:
        if field.canonical_label in ("SCHEDULED", "ACTUAL", "ESTIMATED"):
            side = _side(field.label_match.observation)
            times.append(FR24TimeFieldObservation(field, side,
                {"left": "origin", "right": "destination"}.get(side)))
    return FR24TicketObservation(upper, fields, tuple(times))


def extract_fr24_upper_ticket(ocr_result: OCRTicketResult) -> FR24UpperTicketObservation:
    """Return just the provenance-bearing upper section through the same production path."""
    return extract_fr24_ticket(ocr_result).upper


def extract_fr24_tickets_batch(ocr_results: Sequence[OCRTicketResult]) -> tuple[FR24TicketObservation, ...]:
    """Preserve caller representative/Investigation order without cross-view merging."""
    if not isinstance(ocr_results, Sequence) or isinstance(ocr_results, (str, bytes)):
        raise TypeError("ocr_results must be a sequence of OCRTicketResult records.")
    return tuple(extract_fr24_ticket(result) for result in ocr_results)
