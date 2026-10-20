"""Phase 5.2b ticket-body pixel measurements, without semantic classification.

The crop core accepts two already-extracted ticket_scroll_body images; the frame
wrapper below guarantees this existing ROI through UIConfig/layout mapping. Never
compare the whole frame for production similarity. MAD means pixel change, NOT
scroll: loading, replacement, overlays, dimming and live values can all increase
it. Meaning requires surrounding Phase 4/materialization/stable-view context.

Always retain raw grayscale MAD and CPR (existing strict >10 intensity semantics
by default). Optional phase correlation uses fresh float32 grayscale arrays and a
Hann window, matching ticket_scroll's principle. dx/dy describe A-to-B motion;
only inverse VERTICAL translation -dy is applied to B. dx is diagnostic and a
vertical-model validity check. No affine/features/flow/repair is implemented.

Provisional guards come from Phase 5.2a's 26 adjacent pairs: response .660-.960,
|dx|<=.026px, |dy|<=145px/566=.257, overlap>=.744. Defaults response>=.60,
|dx|<=2px, shift/height<=.30 and overlap>=.70 leave margins while rejecting the
observed .118-.220-response, .385-.488-height aliases. The 2px lateral allowance
is deliberately conservative relative to observed subpixel noise, not a measured
maximum valid horizontal shift. These configurable guards are not probabilities,
semantic thresholds or proof of registration. They need validation on other
scales/layouts/footage. Response can slightly exceed one and is never clipped.

Exact integer dy uses corresponding row slices, with no interpolation. Fractional
dy uses OpenCV INTER_LINEAR and an identically warped float support image; only
support==1 pixels enter residuals. Overlap is valid area / full crop area, not
shared semantic content. Raw and aligned domains differ. Unnecessary alignment
can worsen stable comparisons; it is supplemental and never replaces raw values.
Callers can explicitly disable it. Rejected alignment keeps finite measured
motion/response but returns None overlap/residuals and a technical reason. The
first failed guard is reported deterministically; no semantic labels exist.

The pair batch compares ONLY caller-specified indexes, preserving order and
duplicates. It reuses the Phase 5.1 operation-local bounded frame cache, validates
physical references, and stores no images, timestamps, quality features or scores
in comparisons. Unresolved frame references cannot supply pixel metrics and raise
ValueError rather than dropping pairs or inventing raw values. No all-pairs
engine, candidate removal, ranking, representative selection, ticket grouping,
OCR, VLM, embeddings or Phase 5.3 behavior is implemented.
"""

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from numbers import Integral, Real
from pathlib import Path

import cv2
import numpy as np

from events_diary.change_metrics import compute_changed_pixel_ratio, compute_mad
from events_diary.config import ROI, UIConfig
from events_diary.frame_materialization import (
    MaterializedEvidenceCandidate,
    VideoFrameResolver,
)
from events_diary.layout import map_roi_to_frame


@dataclass(frozen=True)
class TicketViewMeasurements:
    """Raw body metrics and optional guarded vertical-alignment diagnostics.

    Motion is A-to-B in crop pixels. Finite measured components survive rejection;
    unavailable/nonfinite components are None. overlap/aligned_* exist only for
    accepted alignment. alignment_rejection_reason is technical, not semantic:
    not_requested, crop_too_small, phase_correlation_failed, non_finite_motion,
    low_response, excessive_dx, excessive_dy, warp_failed, no_overlap, or low_overlap.
    """

    raw_mad: float
    raw_cpr: float
    dx_pixels: float | None = None
    dy_pixels: float | None = None
    alignment_response: float | None = None
    overlap_fraction: float | None = None
    aligned_mad: float | None = None
    aligned_cpr: float | None = None
    alignment_rejection_reason: str | None = None

    @property
    def alignment_valid(self) -> bool:
        """Derive availability without redundant stored state or semantic meaning."""
        return self.aligned_mad is not None


@dataclass(frozen=True)
class TicketViewComparison:
    """Lightweight caller-specified candidate references plus body measurements.

    Indexes refer to the caller's materialized_candidates sequence, whose context
    supplies video identity and Investigation ownership. No full candidate/image,
    duplicated timestamp, quality score or semantic class is copied here.
    """

    candidate_a_index: int
    candidate_b_index: int
    raw_mad: float
    raw_cpr: float
    dx_pixels: float | None
    dy_pixels: float | None
    alignment_response: float | None
    overlap_fraction: float | None
    aligned_mad: float | None
    aligned_cpr: float | None
    alignment_rejection_reason: str | None

    @property
    def alignment_valid(self) -> bool:
        """Return whether supplemental aligned metrics are available."""
        return self.aligned_mad is not None


def _parameters(
    attempt_alignment,
    changed_pixel_threshold,
    min_alignment_response,
    max_abs_dx_pixels,
    max_vertical_shift_fraction,
    min_overlap_fraction,
):
    """Validate configurable geometric guards and existing CPR parameter semantics."""
    if not isinstance(attempt_alignment, bool):
        raise TypeError("attempt_alignment must be bool.")
    if isinstance(changed_pixel_threshold, bool) or not isinstance(
        changed_pixel_threshold, int
    ):
        raise TypeError("changed_pixel_threshold must be an integer, not bool.")
    if not 0 <= changed_pixel_threshold <= 255:
        raise ValueError("changed_pixel_threshold must be in [0,255].")
    for name, value in (
        ("min_alignment_response", min_alignment_response),
        ("max_abs_dx_pixels", max_abs_dx_pixels),
        ("max_vertical_shift_fraction", max_vertical_shift_fraction),
        ("min_overlap_fraction", min_overlap_fraction),
    ):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be real, not bool.")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    if max_vertical_shift_fraction > 1 or not 0 < min_overlap_fraction <= 1:
        raise ValueError("Shift fraction must be in [0,1]; overlap fraction in (0,1].")


def _overlap(a, b, dy):
    """Extract integer correspondences or fractional full-support warped pixels."""
    height, width = a.shape
    if float(dy).is_integer():
        shift = int(dy)
        if abs(shift) >= height:
            return None, None, 0.0
        if shift >= 0:
            pixels_a, pixels_b = a[: height - shift], b[shift:]
        else:
            pixels_a, pixels_b = a[-shift:], b[: height + shift]
        return pixels_a, pixels_b, float((height - abs(shift)) / height)
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, -dy]])
    warped = cv2.warpAffine(
        b,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    support = cv2.warpAffine(
        np.ones(a.shape, dtype=np.float32),
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    mask = support == 1.0
    if not mask.any():
        return None, None, 0.0
    return a[mask].reshape(-1, 1), warped[mask].reshape(-1, 1), float(mask.mean())


def compare_ticket_body_crops(
    crop_a: np.ndarray,
    crop_b: np.ndarray,
    *,
    changed_pixel_threshold: int = 10,
    attempt_alignment: bool = True,
    min_alignment_response: float = 0.60,
    max_abs_dx_pixels: float = 2.0,
    max_vertical_shift_fraction: float = 0.30,
    min_overlap_fraction: float = 0.70,
) -> TicketViewMeasurements:
    """Compare same-sized uint8 grayscale/BGR crops of the existing ticket body.

    Inputs are never mutated or resized. Existing MAD/CPR APIs validate nonempty
    arrays, dtype, channel count and equal shape. Wrong types raise TypeError;
    malformed crops/guard ranges raise ValueError. Tiny (<2px) valid crops keep raw
    metrics but cannot align. attempt_alignment=False skips phase correlation.
    Guards use inclusive boundaries in response, dx, dy/height and actual support.
    No epsilon changes their calibrated limits. Phase/OpenCV alignment failures
    preserve raw metrics; invalid alignment never substitutes raw for aligned.
    """
    _parameters(
        attempt_alignment,
        changed_pixel_threshold,
        min_alignment_response,
        max_abs_dx_pixels,
        max_vertical_shift_fraction,
        min_overlap_fraction,
    )
    result = TicketViewMeasurements(
        compute_mad(crop_a, crop_b),
        compute_changed_pixel_ratio(crop_a, crop_b, changed_pixel_threshold),
    )
    if not attempt_alignment:
        return replace(result, alignment_rejection_reason="not_requested")
    a, b = (
        cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        for crop in (crop_a, crop_b)
    )
    if min(a.shape) < 2:
        return replace(result, alignment_rejection_reason="crop_too_small")
    try:
        window = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
        (dx, dy), response = cv2.phaseCorrelate(
            a.astype(np.float32), b.astype(np.float32), window
        )
    except cv2.error:
        return replace(result, alignment_rejection_reason="phase_correlation_failed")
    result = replace(
        result,
        dx_pixels=float(dx) if math.isfinite(dx) else None,
        dy_pixels=float(dy) if math.isfinite(dy) else None,
        alignment_response=float(response) if math.isfinite(response) else None,
    )
    if any(
        value is None
        for value in (result.dx_pixels, result.dy_pixels, result.alignment_response)
    ):
        return replace(result, alignment_rejection_reason="non_finite_motion")
    for failed, reason in (
        (response < min_alignment_response, "low_response"),
        (abs(dx) > max_abs_dx_pixels, "excessive_dx"),
        (abs(dy) / a.shape[0] > max_vertical_shift_fraction, "excessive_dy"),
    ):
        if failed:
            return replace(result, alignment_rejection_reason=reason)
    try:
        pixels_a, pixels_b, fraction = _overlap(a, b, dy)
    except cv2.error:
        return replace(result, alignment_rejection_reason="warp_failed")
    if pixels_a is None:
        return replace(result, alignment_rejection_reason="no_overlap")
    if fraction < min_overlap_fraction:
        return replace(result, alignment_rejection_reason="low_overlap")
    return replace(
        result,
        overlap_fraction=fraction,
        aligned_mad=compute_mad(pixels_a, pixels_b),
        aligned_cpr=compute_changed_pixel_ratio(
            pixels_a, pixels_b, changed_pixel_threshold
        ),
    )


def _body(frame, config):
    """Map/crop only ticket_scroll_body, rejecting unsupported layouts or geometry."""
    height, width = frame.shape[:2]
    roi = map_roi_to_frame(config.ticket_scroll_body, config, width, height)
    return roi.crop(frame)


def _compare_pairs(candidates, pairs, config, resolver, parameters):
    """Load verified physical references and compare only the supplied pair sequence."""
    results = []
    for a_index, b_index in pairs:
        crops = []
        for index in (a_index, b_index):
            materialized = candidates[index]
            frame = resolver.resolve(materialized.requested_timestamp_seconds)
            if frame.image is None:
                raise ValueError(f"Unable to resolve compared candidate {index}.")
            if (
                frame.frame_index != materialized.frame_index
                or frame.actual_timestamp_seconds
                != materialized.actual_timestamp_seconds
                or frame.resolution_status != materialized.resolution_status
            ):
                raise ValueError(
                    "Materialized reference disagrees with this video/resolver."
                )
            crops.append(_body(frame.image, config))
        measured = compare_ticket_body_crops(*crops, **parameters)
        results.append(TicketViewComparison(a_index, b_index, **asdict(measured)))
    return tuple(results)


def compare_ticket_view_pairs(
    *,
    video_path: Path,
    materialized_candidates: Sequence[MaterializedEvidenceCandidate],
    pairs: Sequence[tuple[int, int]],
    ui_config: UIConfig,
    resolver: VideoFrameResolver | None = None,
    max_frames: int = 16,
    changed_pixel_threshold: int = 10,
    attempt_alignment: bool = True,
    min_alignment_response: float = 0.60,
    max_abs_dx_pixels: float = 2.0,
    max_vertical_shift_fraction: float = 0.30,
    min_overlap_fraction: float = 0.70,
) -> tuple[TicketViewComparison, ...]:
    """Compare caller-provided candidate indexes in order, retaining repeated pairs.

    No pairs are generated/deduplicated and no candidate is modified. Supplied
    resolver is caller-owned, active, and must match video_path. Otherwise use a
    local bounded resolver and release it on every exit. Metadata-only candidates
    must already be resolved; unresolved/invalid references raise ValueError before
    any pair is silently omitted. Index/cache/type errors raise TypeError/ValueError.
    Empty pairs return (). Quality fields do not affect comparison. API assumes the
    surrounding operation supplies correct single-video identity, verified against
    each resolved physical index/time/status on loading.
    """
    parameters = {
        "changed_pixel_threshold": changed_pixel_threshold,
        "attempt_alignment": attempt_alignment,
        "min_alignment_response": min_alignment_response,
        "max_abs_dx_pixels": max_abs_dx_pixels,
        "max_vertical_shift_fraction": max_vertical_shift_fraction,
        "min_overlap_fraction": min_overlap_fraction,
    }
    _parameters(**parameters)
    if not isinstance(ui_config, UIConfig) or not isinstance(
        ui_config.ticket_scroll_body, ROI
    ):
        raise TypeError("ui_config must provide ticket_scroll_body ROI.")
    if isinstance(max_frames, bool) or not isinstance(max_frames, Integral):
        raise TypeError("max_frames must be an integer, not bool.")
    if max_frames <= 0:
        raise ValueError("max_frames must be positive.")
    if not isinstance(materialized_candidates, Sequence) or not isinstance(
        pairs, Sequence
    ):
        raise TypeError("materialized_candidates and pairs must be sequences.")
    candidates, requested = tuple(materialized_candidates), tuple(pairs)
    for pair in requested:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise TypeError("Each pair must contain two indexes.")
        for index in pair:
            if isinstance(index, bool) or not isinstance(index, Integral):
                raise TypeError("Candidate indexes must be integers, not bool.")
            if not 0 <= index < len(candidates):
                raise ValueError("Candidate index is outside the supplied sequence.")
            item = candidates[index]
            if not isinstance(item, MaterializedEvidenceCandidate):
                raise TypeError(
                    "Compared candidates must be MaterializedEvidenceCandidate records."
                )
            if (
                item.resolution_status not in ("resolved", "video_end_adjusted")
                or item.frame_index is None
                or item.actual_timestamp_seconds is None
            ):
                raise ValueError(
                    "Compared candidates must have resolved physical metadata."
                )
    if not requested:
        return ()
    path = Path(video_path).resolve()
    if resolver is not None:
        if not isinstance(resolver, VideoFrameResolver):
            raise TypeError("resolver must be VideoFrameResolver.")
        if resolver.video_path != path:
            raise ValueError("Resolver must belong to video_path.")
        return _compare_pairs(candidates, requested, ui_config, resolver, parameters)
    with VideoFrameResolver(path, max_frames=max_frames) as owned:
        return _compare_pairs(candidates, requested, ui_config, owned, parameters)
