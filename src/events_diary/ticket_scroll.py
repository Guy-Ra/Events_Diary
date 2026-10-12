"""Measure ticket-scroll pairs and separately interpret adjacent-frame sequences.

Existing change_metrics public APIs supply grayscale MAD and default CPR for the
ticket header, aircraft image, scroll body, and map. Phase correlation estimates
A-to-B translation only inside ticket_scroll_body, using float32 grayscale crops
and a Hann window. Positive dx/dy means content moves right/down; negative means
left/up. Response is an uncalibrated diagnostic, with no reliability gate. Input
pixels are never modified, motion is not clipped, and no alignment is performed.

The separate interpretation helper checks body activity, vertical displacement,
vertical dominance, and surrounding stability for ONE pair. Its defaults are
provisional round-number working values informed by one evaluation recording,
not calibrated production truth. A True result neither confirms a temporal scroll
sequence nor starts/stops an event, changes temporal state, or suppresses an
Investigation. No semantic interpretation, OCR, or content extraction occurs.

Phase correlation assumes dominant translation and adequate texture. Repeated
rows, fixed elements, new content, large shifts, and periodic ambiguity can distort
estimates; even finite values and high response do not guarantee correct motion.
Textureless images can return meaningless displacement. Measurements are retained
as supplied by OpenCV; the interpretation helper rejects non-finite decision
inputs but applies no response threshold. Canonical full frames are required for
the supplied UIConfig; there is no layout recovery or implicit resizing.

Temporal analysis consumes existing TicketScrollEvidence with provisional short
context and gap parameters. It returns one quiet/up/down state per input pair,
without recomputing measurements or creating scroll events or Investigation
decisions. Pair-level measurement remains independent of temporal interpretation.

SCROLLING_UP and SCROLLING_DOWN represent detected active scrolling motion in
the current temporal context, not necessarily the entire human interaction from
first touch/drag until the user mentally finishes scrolling. Short or longer
pauses may legitimately appear as QUIET between active bursts. A future "recently
scrolling" cooldown or broader interaction context belongs to a higher-level
consumer, such as Phase 3 Investigation Detection, rather than artificially
extending TicketScrollState. Temporal interpretation creates no scroll events,
start/end timestamps, or Investigation decisions.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from numbers import Real

import cv2
import numpy as np

from events_diary.change_metrics import compute_changed_pixel_ratio, compute_mad
from events_diary.config import UIConfig
from events_diary.layout import LayoutKind, classify_layout

# Pair measurement is independent of offline temporal interpretation below.


@dataclass(frozen=True)
class TicketScrollEvidence:
    """Immutable measurements for one full-frame pair, with no temporal state.

    Parameters/attributes:
        header_mad, header_cpr: Ticket-header grayscale change measurements.
        aircraft_image_mad, aircraft_image_cpr: Aircraft-image measurements.
        body_mad, body_cpr: Ticket-scroll-body measurements.
        map_mad, map_cpr: Map-region measurements, without map compensation.
        dx, dy: Unclipped A-to-B body translation in pixels (right/down positive).
        motion_response: Raw, uncalibrated OpenCV phase-correlation response.

    Constructing this record returns measurement storage only and performs no
    validation or classification. MAD is in intensity units; CPR is a fraction.
    Meaningful motion assumes textured content with dominant translation. Low
    response and non-finite estimates remain diagnostics, not confidence scores.
    """

    header_mad: float
    header_cpr: float
    aircraft_image_mad: float
    aircraft_image_cpr: float
    body_mad: float
    body_cpr: float
    map_mad: float
    map_cpr: float
    dx: float
    dy: float
    motion_response: float


class TicketScrollState(Enum):
    """Small per-pair temporal state, with no events, timestamps, or decisions.

    SCROLLING_UP and SCROLLING_DOWN represent detected active scrolling motion in
    the current temporal context (negative/positive A-to-B dy). They do not
    necessarily cover the whole human interaction from first touch/drag until
    the user mentally finishes. QUIET indicates insufficient or ambiguous active
    motion evidence; short or longer pauses can legitimately be QUIET between
    active bursts, even within one human scrolling interaction.

    Any future "recently scrolling" cooldown or broader interaction context is
    the responsibility of a higher-level consumer such as Phase 3 Investigation
    Detection, not a reason to artificially extend these states. Enum values are
    output labels, not scroll events, start/end timestamps, or Investigation
    decisions.
    """

    QUIET = "quiet"
    SCROLLING_UP = "scrolling_up"
    SCROLLING_DOWN = "scrolling_down"


def analyze_ticket_scroll_sequence(
    evidences: Sequence[TicketScrollEvidence], *, window_size: int = 5,
    min_positive_pairs: int = 2, max_gap_pairs: int = 1,
) -> list[TicketScrollState]:
    """Interpret adjacent-frame evidence using bounded offline temporal context.

    Args:
        evidences: Ordered pair measurements at consecutive source-frame positions.
            No timestamps are needed; parameter units are pairs, not seconds.
        window_size: Positive integer context length, provisional default 5.
        min_positive_pairs: Positive integer support count, <= window_size;
            provisional default 2. Short sequences use available support only.
        max_gap_pairs: Nonnegative integer number of consecutive False pairs that
            can be bridged between positives, provisional default 1.
    Returns:
        Same-length list of states without modifying evidences. Empty input returns
        []. Default isolated positives remain QUIET. This is offline, not causal:
        a centered window includes future pairs. For even sizes the extra pair is
        on the right. Windows are clipped rather than shifted at boundaries.
    Raises:
        TypeError: Bool/non-integer temporal parameter or invalid evidence item.
        ValueError: Invalid parameter range, including support > window size.
    Positive decisions reuse is_vertical_scroll_evidence with unchanged defaults.
    Candidate spans join positives separated by at most max_gap_pairs negatives;
    longer gaps split spans and are QUIET. No leading/trailing negatives are filled.
    For each position inside a span, clip its centered window to that span; require
    min_positive_pairs and choose the majority dy sign of positive pairs ONLY.
    Negative dy votes up, positive dy votes down; ties return QUIET. response does
    not vote. Support cannot cross long gaps. Direction switches may include a
    quiet tie; short gaps between opposing directions need not remain scrolling.
    Defaults are provisional and frame-rate dependent. No crops, pixel metrics,
    phase correlation, event objects, or Investigation decisions are computed.

    This function temporally interprets pair-level evidence as active-motion
    states in the current context; it does not reconstruct an entire human
    scrolling interaction or create scroll events or start/end timestamps.
    Short or longer pauses may legitimately remain QUIET between active bursts.
    A future "recently scrolling" cooldown or broader interaction context belongs
    to a higher-level consumer, such as Phase 3 Investigation Detection, rather
    than artificial extension of TicketScrollState here.
    """
    for name, value, minimum in (("window_size", window_size, 1),
                                  ("min_positive_pairs", min_positive_pairs, 1),
                                  ("max_gap_pairs", max_gap_pairs, 0)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer, not a boolean.")
        if value < minimum:
            raise ValueError(f"{name} must be >= {minimum}.")
    if min_positive_pairs > window_size:
        raise ValueError("min_positive_pairs must not exceed window_size.")
    positive = [is_vertical_scroll_evidence(evidence) for evidence in evidences]
    states = [TicketScrollState.QUIET] * len(positive)
    indices = [i for i, accepted in enumerate(positive) if accepted]
    spans = []
    for index in indices:
        if not spans or index - spans[-1][1] - 1 > max_gap_pairs:
            spans.append((index, index))
        else:
            spans[-1] = (spans[-1][0], index)
    left, right = (window_size - 1) // 2, window_size // 2
    for start, end in spans:
        for index in range(start, end + 1):
            context = range(max(start, index - left), min(end, index + right) + 1)
            up = down = 0
            for position in context:
                if positive[position]:
                    up += evidences[position].dy < 0
                    down += evidences[position].dy > 0
            if up + down >= min_positive_pairs and up != down:
                states[index] = (TicketScrollState.SCROLLING_UP if up > down
                                 else TicketScrollState.SCROLLING_DOWN)
    return states


def estimate_ticket_scroll_evidence(
    frame_a: np.ndarray, frame_b: np.ndarray, ui_config: UIConfig,
) -> TicketScrollEvidence:
    """Measure four canonical ROIs and A-to-B scroll-body translation.

    Args:
        frame_a, frame_b: Nonempty uint8 NumPy full frames of identical shape,
            either 2D grayscale or 3-channel BGR. Dimensions must exactly match
            ui_config's reference resolution; no implicit conversion/resizing.
        ui_config: UIConfig providing existing canonical ROIs in full-frame pixels.
    Returns:
        Frozen TicketScrollEvidence with public MAD/default CPR measurements and
        raw phase-correlation dx, dy, response. Positive dx is right, dy is down.
    Raises:
        TypeError: Invalid array/dtype or non-UIConfig configuration.
        ValueError: Empty/unsupported/mismatched frames, incompatible resolution,
            out-of-frame ROI, or body crop smaller than two pixels in either axis.
        cv2.error: OpenCV conversion, windowing, or phase-correlation failure.
    Assumes canonical UI placement and preferably adjacent frames. BGR crops are
    converted to grayscale; temporary float32 body arrays prevent Hann windowing
    from mutating inputs. No motion/response gate or temporal interpretation occurs.
    Flat or repetitive content can produce unreliable or non-finite measurements.
    """
    _validate_frames(frame_a, frame_b, ui_config)
    metrics = []
    for region in (ui_config.ticket_header, ui_config.aircraft_image,
                   ui_config.ticket_scroll_body, ui_config.map_analysis_region):
        crop_a, crop_b = region.crop(frame_a), region.crop(frame_b)
        metrics.extend((compute_mad(crop_a, crop_b),
                        compute_changed_pixel_ratio(crop_a, crop_b)))
    body_a = ui_config.ticket_scroll_body.crop(frame_a)
    body_b = ui_config.ticket_scroll_body.crop(frame_b)
    gray_a, gray_b = (_body_float(image) for image in (body_a, body_b))
    if min(gray_a.shape) < 2:
        raise ValueError("ticket_scroll_body must be at least 2x2 for Hann windowing.")
    window = cv2.createHanningWindow((gray_a.shape[1], gray_a.shape[0]), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(gray_a, gray_b, window)
    return TicketScrollEvidence(*metrics, float(dx), float(dy), float(response))


def is_vertical_scroll_evidence(
    evidence: TicketScrollEvidence, *, min_body_cpr: float = 0.10,
    min_vertical_shift_px: float = 2.0, min_vertical_dominance_ratio: float = 3.0,
    max_surrounding_cpr: float = 0.01,
) -> bool:
    """Check whether one pair is consistent with predominantly vertical scrolling.

    Args:
        evidence: Pair measurements, with A-to-B dx/dy in pixels. Either vertical
            sign is accepted; only magnitudes enter these comparisons.
        min_body_cpr: Inclusive minimum body activity fraction, finite in [0,1].
        min_vertical_shift_px: Inclusive minimum |dy| in pixels, finite and >=0.
        min_vertical_dominance_ratio: Required |dy|/|dx| factor, finite and >0;
            implemented by multiplication so zero dx requires no special division.
        max_surrounding_cpr: Inclusive maximum for EACH header/image/map CPR,
            finite in [0,1]. All four defaults are provisional working values.
    Returns:
        Python bool requiring all explicit conditions. Non-finite decision inputs
        return False. MAD and motion_response are not decision inputs or gates.
    Raises:
        TypeError: Non-real/bool threshold or non-TicketScrollEvidence input.
        ValueError: Non-finite or out-of-range threshold.
    Defaults leave margins around one video's strong body CPR (roughly 0.22+),
    clear shifts (roughly 6+ pixels), and surrounding CPR (<0.0067). Two pixels
    avoids subpixel jitter; 3x dominance is a conservative directional requirement
    rather than a ratio calibrated by this almost purely vertical recording.
    True is pair evidence only, not a sequence, start/stop signal, event, or reason
    to suppress an Investigation. Caller-created records are not range-normalized.
    """
    for name, value, lower, upper, strict in (
        ("min_body_cpr", min_body_cpr, 0, 1, False),
        ("min_vertical_shift_px", min_vertical_shift_px, 0, None, False),
        ("min_vertical_dominance_ratio", min_vertical_dominance_ratio, 0, None, True),
        ("max_surrounding_cpr", max_surrounding_cpr, 0, 1, False),
    ):
        _validate_threshold(name, value, lower, upper, strict)
    if not isinstance(evidence, TicketScrollEvidence):
        raise TypeError("evidence must be TicketScrollEvidence.")
    if not all(math.isfinite(value) for value in (
        evidence.body_cpr, evidence.dx, evidence.dy, evidence.header_cpr,
        evidence.aircraft_image_cpr, evidence.map_cpr,
    )):
        return False
    return bool(
        evidence.body_cpr >= min_body_cpr
        and abs(evidence.dy) >= min_vertical_shift_px
        and abs(evidence.dy) >= abs(evidence.dx) * min_vertical_dominance_ratio
        and evidence.header_cpr <= max_surrounding_cpr
        and evidence.aircraft_image_cpr <= max_surrounding_cpr
        and evidence.map_cpr <= max_surrounding_cpr
    )


def _validate_frames(frame_a: np.ndarray, frame_b: np.ndarray, config: UIConfig) -> None:
    """Validate matching uint8 full frames against the supplied canonical layout.

    Args: frame_a/frame_b are grayscale or BGR arrays; config supplies reference
        dimensions. Coordinates are full-frame canonical pixels, never scaled.
    Returns: None. Raises TypeError for array/dtype/config violations and ValueError
        for empty, unsupported, mismatched, or noncanonical geometry. ROI bounds
        are subsequently checked by existing ROI.crop rather than duplicated here.
    """
    if not isinstance(config, UIConfig):
        raise TypeError("ui_config must be UIConfig.")
    for name, frame in (("frame_a", frame_a), ("frame_b", frame_b)):
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
            raise TypeError(f"{name} must be a NumPy array with dtype np.uint8.")
        if frame.size == 0:
            raise ValueError(f"{name} must not be empty.")
        if frame.ndim not in (2, 3) or (frame.ndim == 3 and frame.shape[2] != 3):
            raise ValueError(f"{name} must be 2D grayscale or exactly 3-channel BGR.")
    if frame_a.shape != frame_b.shape:
        raise ValueError("Frames must have identical shapes.")
    height, width = frame_a.shape[:2]
    if classify_layout(config, width, height) != LayoutKind.EXACT_REFERENCE:
        raise ValueError(f"Expected canonical {config.reference_width}x{config.reference_height} "
                         f"full frames; got {width}x{height}.")


def _body_float(image: np.ndarray) -> np.ndarray:
    """Return a fresh float32 grayscale body crop for phase correlation.

    Args: image is an already validated grayscale/BGR crop in ticket-local pixels.
    Returns: Owned float32 grayscale array, protecting input pixels from windowing.
    Assumes uint8 image contract; OpenCV conversion errors propagate. No geometric
    preprocessing or coordinate change is performed.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return gray.astype(np.float32)


def _validate_threshold(name: str, value: float, lower: float,
                        upper: float | None, strict: bool) -> None:
    """Validate one finite real working threshold without accepting booleans.

    Args: name labels errors; value is the proposed threshold; lower/upper bound
        its range (None means no upper limit); strict makes the lower bound exclusive.
    Returns: None. Raises TypeError for bool/non-real values and ValueError for
        non-finite/out-of-range values. Units come from the caller (pixels, ratio,
        or CPR fraction); no normalization, clipping, or coordinate conversion.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, not a boolean.")
    if (not math.isfinite(value) or (value <= lower if strict else value < lower)
            or (upper is not None and value > upper)):
        raise ValueError(f"{name} must be finite and in its documented range.")
