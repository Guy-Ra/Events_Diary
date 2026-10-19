"""Phase 4.1 fixed-context temporal Investigation window construction only.

Expand an existing Phase 3 InvestigationInterval with provisional requested
context of 0.5 seconds before and 0.5 seconds after, clipping to supplied video
bounds. Preserve the original Investigation core, its index, and requested context
separately from the effective clipped processing window. Requested context never
changes because of clipping; actual context is derivable from the boundary gaps.

Zero-duration Investigations and zero context are allowed. Overlapping windows
remain separate and preserve input order; they are not merged or deduplicated.
This module reads no video, decodes/stores no frames, and inspects no visual signals.
It introduces no activity levels, adaptive sampling, scroll detection, evidence
selection, OCR, VLM, or Phase 4.2 behavior. The fixed context values are provisional
MVP parameters, not calibrated extraction policy. Inputs are never mutated.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

from events_diary.investigation_lifecycle import InvestigationInterval


@dataclass(frozen=True)
class InvestigationWindow:
    """Immutable core and clipped processing bounds plus requested temporal context.

    Investigation index/start/end are copied from the original interval. Window
    bounds are effective after clipping to video bounds. context_before_seconds
    and context_after_seconds are REQUESTED amounts; actual context may be smaller
    and is derivable without extra fields. All quantities are seconds, not frames.
    """

    investigation_index: int
    investigation_start_time_seconds: float
    investigation_end_time_seconds: float
    window_start_time_seconds: float
    window_end_time_seconds: float
    context_before_seconds: float
    context_after_seconds: float


def _validate_time(name: str, value: float) -> None:
    """Validate finite real seconds, rejecting bool/non-real types with TypeError."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, not a boolean.")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")


def _validate_parameters(start, end, before, after):
    """Validate video range and nonnegative requested context, including empty batches."""
    for name, value in (("video_start_time_seconds", start), ("video_end_time_seconds", end),
                        ("context_before_seconds", before), ("context_after_seconds", after)):
        _validate_time(name, value)
    if start < 0:
        raise ValueError("video_start_time_seconds must be non-negative.")
    if end < start:
        raise ValueError("video_end_time_seconds must be >= video_start_time_seconds.")
    if before < 0 or after < 0:
        raise ValueError("requested context must be non-negative.")


def build_investigation_window(
    investigation: InvestigationInterval, *, video_start_time_seconds: float,
    video_end_time_seconds: float, context_before_seconds: float = 0.5,
    context_after_seconds: float = 0.5,
) -> InvestigationWindow:
    """Expand a valid Investigation core with fixed context, clipped to video bounds.

    Preserve index and original boundaries. Effective start is max(video start,
    core start minus requested pre-context); end is min(video end, core end plus
    requested post-context). Returned context fields retain the requested values.
    No frame/video access, signal interpretation, or lifecycle modification occurs.

    TypeError rejects wrong interval types or bool/non-real timestamp/context
    values. ValueError rejects non-finite values, negative video start/context,
    reversed video/core bounds, or a core outside the supplied video. Zero-duration
    cores and zero context are valid. Inputs are unchanged; defaults are provisional.
    """
    _validate_parameters(video_start_time_seconds, video_end_time_seconds,
                         context_before_seconds, context_after_seconds)
    if not isinstance(investigation, InvestigationInterval):
        raise TypeError("investigation must be InvestigationInterval.")
    start, end = investigation.start_time_seconds, investigation.end_time_seconds
    _validate_time("investigation.start_time_seconds", start)
    _validate_time("investigation.end_time_seconds", end)
    if start > end:
        raise ValueError("Investigation start must not exceed its end.")
    if start < video_start_time_seconds or end > video_end_time_seconds:
        raise ValueError("Investigation must lie inside the supplied video range.")
    return InvestigationWindow(
        investigation.investigation_index, start, end,
        max(video_start_time_seconds, start - context_before_seconds),
        min(video_end_time_seconds, end + context_after_seconds),
        context_before_seconds, context_after_seconds)


def build_investigation_windows(
    investigations: Sequence[InvestigationInterval], *, video_start_time_seconds: float,
    video_end_time_seconds: float, context_before_seconds: float = 0.5,
    context_after_seconds: float = 0.5,
) -> list[InvestigationWindow]:
    """Build one window per interval in supplied order, preserving overlaps.

    Delegate every interval to build_investigation_window without modifying or
    sorting input. Empty input returns [] after validating common parameters.
    TypeError rejects non-Sequence input; other validation errors propagate from
    the same validators used by the single-window API. No merging occurs.
    """
    if not isinstance(investigations, Sequence):
        raise TypeError("investigations must be a Sequence of InvestigationInterval.")
    if not investigations:
        _validate_parameters(video_start_time_seconds, video_end_time_seconds,
                             context_before_seconds, context_after_seconds)
        return []
    return [build_investigation_window(
        investigation, video_start_time_seconds=video_start_time_seconds,
        video_end_time_seconds=video_end_time_seconds,
        context_before_seconds=context_before_seconds, context_after_seconds=context_after_seconds)
        for investigation in investigations]
