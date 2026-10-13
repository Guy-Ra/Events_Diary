"""Phase 3 single-frame presence and consecutive-pair Investigation signals.

Single-frame presence asks "What exists now?" using dark bottom-action occupancy
and body edges. Pair signals ask "What changed?" independently. Presence does not
prove aircraft identity, finished loading, ownership, an opening, or a transition.
The inclusive presence defaults (0.50 dark, 0.05 edges) are provisional values
from one exploration dataset; body light occupancy is diagnostic, not a condition.
Canonical layout/theme assumptions and limited negative examples constrain them.

ChangeSignal describes only generic ROI MAD/CPR visual changes: header, aircraft
image, ticket body, or generic compensated map residual. Route-selection,
map-motion, and ticket-scroll evidence retain their existing dedicated types.
InvestigationSignals aggregates these records and caller-provided pair decisions
without duplicating measurement or classification algorithms.

NOT_AVAILABLE means measurement could not be obtained (None or non-finite),
whereas LOW is an available measurement with small visual change. Raw MAD/CPR
values are preserved, including unavailable/non-finite values, for diagnostics.
Strength uses CPR alone; MAD does not vote. The generic interpreter takes caller
thresholds. analyze_investigation_pair orchestrates existing measurement and
classification APIs using provisional constants derived from one recording.
Map residual thresholds are conservative and poorly calibrated; that signal
remains supporting/context evidence. No low-level algorithm is duplicated.

This is pair-level aggregation, not Investigation event detection. There is no
candidate score/rule, temporal confirmation,
TicketScrollState integration, event lifecycle, identity, or timestamps. Strength
labels are visual-change categories, not confidence or proof of a new Investigation.
"""

import math
from dataclasses import dataclass
from enum import Enum
from numbers import Real

import cv2
import numpy as np

from events_diary.change_metrics import compute_changed_pixel_ratio, compute_mad
from events_diary.config import UIConfig
from events_diary.layout import LayoutKind, classify_layout
from events_diary.map_motion import MapMotionEstimate, estimate_map_motion
from events_diary.route_selection import (
    RouteSelectionChangeEvidence,
    estimate_route_selection_change,
    is_route_selection_change,
)
from events_diary.ticket_scroll import (
    TicketScrollEvidence,
    estimate_ticket_scroll_evidence,
    is_vertical_scroll_evidence,
)

# Provisional, single-recording calibration; not final production truth.
HEADER_MEDIUM_CPR = 0.02
HEADER_HIGH_CPR = 0.40
AIRCRAFT_IMAGE_MEDIUM_CPR = 0.02
AIRCRAFT_IMAGE_HIGH_CPR = 0.60
TICKET_BODY_MEDIUM_CPR = 0.02
TICKET_BODY_HIGH_CPR = 0.60
# Supporting/context only: no empirically separated HIGH population observed.
MAP_RESIDUAL_MEDIUM_CPR = 0.01
MAP_RESIDUAL_HIGH_CPR = 0.02


@dataclass(frozen=True)
class InvestigationPresenceEvidence:
    """Single-frame fractions, with no identity, confidence, or lifecycle state.

    bottom_actions_dark_fraction counts grayscale <64 in ticket_bottom_actions.
    ticket_body_edge_fraction counts Canny(50,150) edges in ticket_scroll_body.
    ticket_body_light_fraction counts grayscale >=192 in that body, diagnostically.
    Each uses its own ROI area. Measurement returns finite fractions in [0,1].
    Construction stores values without validation; the classifier rejects malformed
    or unavailable fractions. No timestamps, frame indices, or change evidence exist.
    """

    bottom_actions_dark_fraction: float
    ticket_body_edge_fraction: float
    ticket_body_light_fraction: float


def estimate_investigation_presence(
    frame: np.ndarray, ui_config: UIConfig,
) -> InvestigationPresenceEvidence:
    """Measure ticket structure from one canonical uint8 grayscale or BGR frame.

    Use unchanged bottom-actions/body ROIs with no resizing or layout recovery.
    BGR crops convert with OpenCV COLOR_BGR2GRAY; grayscale crops are used directly.
    Dark is strictly <64; light is >=192. Canny uses the exploration's thresholds
    50/150 and default apertureSize=3, L2gradient=False. Count nonzero edge pixels
    and divide each count by its ROI area. Input pixels are not mutated.

    Raises TypeError for invalid config, array, or dtype; ValueError for empty,
    unsupported-channel, noncanonical frames or out-of-bounds ROI. Existing layout,
    crop, and OpenCV failures propagate. No pair comparison or semantic inference
    is performed. Returned evidence does not imply content has finished loading.
    """
    if not isinstance(ui_config, UIConfig):
        raise TypeError("ui_config must be UIConfig.")
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
        raise TypeError("frame must be a NumPy array with dtype np.uint8.")
    if frame.size == 0:
        raise ValueError("frame must not be empty.")
    if frame.ndim not in (2, 3) or (frame.ndim == 3 and frame.shape[2] != 3):
        raise ValueError("frame must be 2D grayscale or exactly 3-channel BGR.")
    height, width = frame.shape[:2]
    if classify_layout(ui_config, width, height) != LayoutKind.EXACT_REFERENCE:
        raise ValueError(f"Expected canonical {ui_config.reference_width}x"
                         f"{ui_config.reference_height} full frame; got {width}x{height}.")
    bottom = ui_config.ticket_bottom_actions.crop(frame)
    body = ui_config.ticket_scroll_body.crop(frame)
    if frame.ndim == 3:
        bottom = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)
        body = cv2.cvtColor(body, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(body, 50, 150)
    return InvestigationPresenceEvidence(
        float(np.count_nonzero(bottom < 64) / bottom.size),
        float(np.count_nonzero(edges) / body.size),
        float(np.count_nonzero(body >= 192) / body.size))


def is_investigation_present(
    evidence: InvestigationPresenceEvidence, *, min_bottom_dark_fraction: float = 0.50,
    min_body_edge_fraction: float = 0.05,
) -> bool:
    """Check whether one frame visually resembles an open FR24 aircraft ticket.

    Both dark-bottom and body-edge fractions must meet inclusive provisional
    minima. Light occupancy does not vote, allowing dimmed/loading tickets.
    All stored fractions must nevertheless be available, real, finite, and in
    [0,1]; malformed evidence returns False. Evidence is never changed.
    Thresholds validate first: TypeError for bool/non-real; ValueError for
    non-finite/out-of-range. A wrong evidence object type raises TypeError.
    Defaults come from one exploration dataset, not broad classifier validation.
    True establishes no identity, loading completion, route ownership, new
    Investigation, transition, confidence, temporal state, or lifecycle decision.
    """
    _validate_cpr_threshold("min_bottom_dark_fraction", min_bottom_dark_fraction)
    _validate_cpr_threshold("min_body_edge_fraction", min_body_edge_fraction)
    if not isinstance(evidence, InvestigationPresenceEvidence):
        raise TypeError("evidence must be InvestigationPresenceEvidence.")
    for value in (evidence.bottom_actions_dark_fraction, evidence.ticket_body_edge_fraction,
                  evidence.ticket_body_light_fraction):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not math.isfinite(value) or not 0 <= value <= 1):
            return False
    return bool(evidence.bottom_actions_dark_fraction >= min_bottom_dark_fraction
                and evidence.ticket_body_edge_fraction >= min_body_edge_fraction)


class SignalStrength(Enum):
    """Availability and generic visual-change magnitude, not event confidence.

    NOT_AVAILABLE means an absent/non-finite measurement; LOW means available
    small change; MEDIUM means meaningful change; HIGH means strong change.
    Exact cutoffs belong to callers. Enum construction returns the corresponding
    label; it performs no measurement or Investigation interpretation.
    """

    NOT_AVAILABLE = "not_available"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class ChangeSignal:
    """Immutable generic ROI strength plus unchanged MAD/CPR evidence.

    Parameters/attributes:
        strength: SignalStrength visual-change category or NOT_AVAILABLE.
        mad: Mean absolute grayscale difference in intensity units, or None.
        cpr: Changed Pixel Ratio fraction, or None.
    This record stores caller values without constructor validation. The public
    interpretation helper preserves raw non-finite measurements diagnostically.
    Assumes metric provenance/ROI association is managed by the consumer; no
    timestamps, coordinates, motion, or semantic event decision are added.
    Dedicated route-selection and motion evidence must not use this type.
    """

    strength: SignalStrength
    mad: float | None
    cpr: float | None


@dataclass(frozen=True)
class InvestigationSignals:
    """Aggregate existing visual/context evidence associated with ONE frame pair.

    Parameters/attributes:
        header_change: Generic ticket-header MAD/CPR signal.
        aircraft_image_change: Generic aircraft-image MAD/CPR signal.
        ticket_body_change: Generic ticket-scroll-body MAD/CPR signal.
        map_residual_change: Generic aligned map-residual signal, unavailable
            when that measurement could not be obtained.
        route_selection_evidence: Existing dedicated route-color measurements.
        route_selection_change: Caller-provided pair-level route-change decision.
        map_motion: Existing dedicated B-to-A map-motion record.
        ticket_scroll_evidence: Existing dedicated A-to-B body-motion record.
        ticket_scroll_detected: Caller-provided pair-level scroll decision.
    Construction stores the supplied immutable records without running algorithms,
    validating cross-field consistency, or changing their conventions. Assumes all
    records refer to the same pair; analyze_investigation_pair supplies these records.
    The booleans are not Investigation decisions. No TicketScrollState, temporal
    state, frame indices, timestamps, aircraft identity, score, or confidence exists.
    """

    header_change: ChangeSignal
    aircraft_image_change: ChangeSignal
    ticket_body_change: ChangeSignal
    map_residual_change: ChangeSignal
    route_selection_evidence: RouteSelectionChangeEvidence
    route_selection_change: bool
    map_motion: MapMotionEstimate
    ticket_scroll_evidence: TicketScrollEvidence
    ticket_scroll_detected: bool


def analyze_investigation_pair(
    frame_a: np.ndarray, frame_b: np.ndarray, ui_config: UIConfig,
) -> InvestigationSignals:
    """Aggregate existing evidence for two consecutive canonical BGR full frames.

    Inputs must be matching nonempty uint8 three-channel arrays in the reference
    layout of ui_config. The route measurement API validates that shared contract
    first; existing TypeError/ValueError and subsystem errors propagate unchanged.
    Input frames are not mutated. This function supplies no layout recovery.

    Ticket ROI MAD/CPR use existing metrics and provisional one-recording cutoffs.
    Map motion retains B-to-A semantics; only available aligned metrics contribute
    to map_residual_change. Rejected alignment yields NOT_AVAILABLE, never raw
    fallback. Missing/non-finite aligned values are handled by the interpreter.
    The original motion record retains all diagnostics. Map strength is supporting
    context; its conservative thresholds remain poorly calibrated.

    Route and scroll records are preserved with their existing pair classifiers'
    decisions. Scroll motion remains A-to-B. Public measurement APIs may repeat
    internal metric/motion work; no low-level implementation is duplicated here.
    Returns an immutable InvestigationSignals, without events, identity, temporal
    confirmation, TicketScrollState, scores, timestamps, or lifecycle decisions.
    """
    route = estimate_route_selection_change(frame_a, frame_b, ui_config)
    changes = []
    for roi, medium, high in (
        (ui_config.ticket_header, HEADER_MEDIUM_CPR, HEADER_HIGH_CPR),
        (ui_config.aircraft_image, AIRCRAFT_IMAGE_MEDIUM_CPR, AIRCRAFT_IMAGE_HIGH_CPR),
        (ui_config.ticket_scroll_body, TICKET_BODY_MEDIUM_CPR, TICKET_BODY_HIGH_CPR),
    ):
        a, b = roi.crop(frame_a), roi.crop(frame_b)
        changes.append(interpret_change_signal(
            compute_mad(a, b), compute_changed_pixel_ratio(a, b),
            medium_cpr=medium, high_cpr=high))
    roi = ui_config.map_analysis_region
    motion = estimate_map_motion(roi.crop(frame_a), roi.crop(frame_b), ui_config=ui_config)
    residual = interpret_change_signal(
        motion.aligned_mad if motion.valid else None,
        motion.aligned_cpr if motion.valid else None,
        medium_cpr=MAP_RESIDUAL_MEDIUM_CPR, high_cpr=MAP_RESIDUAL_HIGH_CPR)
    scroll = estimate_ticket_scroll_evidence(frame_a, frame_b, ui_config)
    return InvestigationSignals(
        header_change=changes[0], aircraft_image_change=changes[1],
        ticket_body_change=changes[2], map_residual_change=residual,
        route_selection_evidence=route, route_selection_change=is_route_selection_change(route),
        map_motion=motion, ticket_scroll_evidence=scroll,
        ticket_scroll_detected=is_vertical_scroll_evidence(scroll))


def interpret_change_signal(
    mad: float | None, cpr: float | None, *, medium_cpr: float, high_cpr: float,
) -> ChangeSignal:
    """Interpret generic ROI change using caller-supplied inclusive CPR boundaries.

    Args:
        mad: Raw mean absolute grayscale difference, or None if unavailable.
        cpr: Raw Changed Pixel Ratio, or None if unavailable.
        medium_cpr: Finite real CPR boundary in [0,1], excluding bool.
        high_cpr: Finite real CPR boundary in [0,1], excluding bool, >= medium_cpr.
    Returns:
        Frozen ChangeSignal preserving mad/cpr exactly. None or non-finite in either
        measurement gives NOT_AVAILABLE. Otherwise cpr < medium_cpr gives LOW,
        medium_cpr <= cpr < high_cpr gives MEDIUM, and cpr >= high_cpr gives HIGH.
        Equal thresholds are allowed; equality then gives HIGH (no MEDIUM interval).
        Valid zero measurements are available, with strength determined by cutoffs.
    Raises:
        TypeError: Non-real/bool threshold; unsupported measurement types may also
            raise TypeError through math.isfinite (measurements should be float/None).
        ValueError: Non-finite/out-of-range threshold or medium_cpr > high_cpr.
    Thresholds are validated even for missing measurements. Assumes raw values
    come from existing metric APIs; no measurement range normalization or clipping
    occurs. MAD is diagnostic only and does not affect strength beyond availability.
    No ROI-specific calibration, coordinate conversion, or event detection occurs.
    """
    _validate_cpr_threshold("medium_cpr", medium_cpr)
    _validate_cpr_threshold("high_cpr", high_cpr)
    if medium_cpr > high_cpr:
        raise ValueError("medium_cpr must not exceed high_cpr.")
    if mad is None or cpr is None or not math.isfinite(mad) or not math.isfinite(cpr):
        strength = SignalStrength.NOT_AVAILABLE
    elif cpr >= high_cpr:
        strength = SignalStrength.HIGH
    elif cpr >= medium_cpr:
        strength = SignalStrength.MEDIUM
    else:
        strength = SignalStrength.LOW
    return ChangeSignal(strength, mad, cpr)


def _validate_cpr_threshold(name: str, value: float) -> None:
    """Validate a finite real CPR threshold in [0,1], without boolean coercion.

    name labels errors and value is a caller-provided threshold. Returns None on
    success; TypeError rejects bool/non-real values, ValueError rejects non-finite
    or out-of-range values. Assumes CPR fraction units; no calibration or clamping.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, not a boolean.")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and in [0, 1].")
