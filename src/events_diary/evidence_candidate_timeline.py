"""Phase 4.4 pure composition of existing evidence requests and annotations.

Every AdaptiveSamplingPlan request becomes exactly one EvidenceCandidate, retaining
its original timestamp and order. No sparse/dense request is ranked, filtered,
deduplicated or added. Sampling classification, semantic anchors and stable-view
sources coexist; dense reasons remain the upstream interval's unchanged tuple.
Stable spans annotate existing requests, including times before confirmation.
No raw signals, frame indexes, images, decoding, detectors, OCR or selection occur.

All boundary/equality comparisons use math.isclose(rel_tol=0, abs_tol=1e-9 seconds).
This is representation tolerance only: distinct requested values are never merged,
even when closer than that tolerance. Ordering and duplicate validation are exact.
Dense/stable membership is inclusive at both ends. Dense intervals must already
be merged and disjoint; stable views must be disjoint and in chronological order.
Multiple tolerant matches fail instead of inventing precedence. Dense sampling
is the single sampling label whenever a dense interval matches.

A right-censored stable end=None extends only to this Investigation's core end,
as specified by the Phase 4.4 contract; it never extends through trailing context.
No new timestamp is invented for a stable start/confirmation/end. Missing stable
coverage is an error. Current Phase 4.2 plans always contain their mandatory window
and core anchors, even for zero-duration windows, so an empty request tuple cannot
be a valid plan. Inputs are validated without repairing them or recomputing plans.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real

from events_diary.adaptive_sampling import AdaptiveSamplingPlan, DenseSamplingInterval
from events_diary.investigation_window import InvestigationWindow
from events_diary.ticket_scroll import TicketScrollState
from events_diary.ticket_stable_views import TicketStableView

TIMESTAMP_ABSOLUTE_TOLERANCE_SECONDS = 1e-9
SOURCE_ORDER = ("sparse_sampling", "dense_sampling", "window_start", "investigation_start",
                "investigation_end", "window_end", "stable_ticket_view")
_WINDOW_TIMES = ("window_start_time_seconds", "investigation_start_time_seconds",
                 "investigation_end_time_seconds", "window_end_time_seconds",
                 "context_before_seconds", "context_after_seconds")
_ANCHORS = tuple(zip(SOURCE_ORDER[2:6], _WINDOW_TIMES[:4]))


@dataclass(frozen=True)
class EvidenceCandidate:
    """One original requested timestamp enriched with ordered explainable sources.

    stable_view_index refers to the supplied stable_views sequence, not a new view
    identity. Sparse candidates have empty dense_reasons. Candidates outside stable
    spans have stable_view_index=None and stable_view_follows_scroll=False. This
    record carries no pixel data, frame index, score or selection decision.
    """

    investigation_index: int
    timestamp_seconds: float
    sources: tuple[str, ...]
    dense_reasons: tuple[str, ...]
    stable_view_index: int | None
    stable_view_follows_scroll: bool


def _close(a, b):
    """Compare time equality using only the documented absolute tolerance."""
    return math.isclose(a, b, rel_tol=0, abs_tol=TIMESTAMP_ABSOLUTE_TOLERANCE_SECONDS)


def _inside(time, start, end):
    """Test inclusive interval membership with tiny boundary representation tolerance."""
    return (time >= start or _close(time, start)) and (time <= end or _close(time, end))


def _number(name, value, *, minimum=None, positive=False):
    """Validate real finite nonboolean metadata without normalizing its values."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be real, not bool.")
    if not math.isfinite(value) or (minimum is not None and value < minimum) or (positive and value <= 0):
        raise ValueError(f"{name} must be finite and in its documented range.")


def _index(name, value):
    """Validate a nonnegative nonboolean Investigation identifier."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, not bool.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")


def _window(window):
    """Validate window metadata because its frozen record has no constructor checks."""
    if not isinstance(window, InvestigationWindow):
        raise TypeError("window must be InvestigationWindow.")
    _index("investigation_index", window.investigation_index)
    for name in _WINDOW_TIMES:
        _number(name, getattr(window, name), minimum=0)
    if not (window.window_start_time_seconds <= window.investigation_start_time_seconds
            <= window.investigation_end_time_seconds <= window.window_end_time_seconds):
        raise ValueError("window must contain an ordered Investigation core.")


def _plan(window, plan):
    """Validate cross-model plan correspondence and existing interval/request order."""
    if not isinstance(plan, AdaptiveSamplingPlan):
        raise TypeError("sampling_plan must be AdaptiveSamplingPlan.")
    _window(plan.window)
    if (plan.window.investigation_index != window.investigation_index
            or any(not _close(getattr(window, f), getattr(plan.window, f)) for f in _WINDOW_TIMES)):
        raise ValueError("sampling_plan.window must match supplied window semantically.")
    for name in ("source_fps", "sparse_sampling_fps", "dense_sampling_fps"):
        _number(name, getattr(plan, name), positive=True)
    for name in ("pre_padding_seconds", "post_padding_seconds"):
        _number(name, getattr(plan, name), minimum=0)
    if plan.sparse_sampling_fps > plan.source_fps or plan.dense_sampling_fps != plan.source_fps:
        raise ValueError("plan rates must preserve source-rate dense sampling and bounded sparse rate.")
    if not isinstance(plan.dense_intervals, tuple) or not isinstance(plan.requested_timestamps_seconds, tuple):
        raise TypeError("plan intervals and requested timestamps must be tuples.")
    lower, upper = window.window_start_time_seconds, window.window_end_time_seconds
    previous = None
    for interval in plan.dense_intervals:
        if not isinstance(interval, DenseSamplingInterval):
            raise TypeError("dense_intervals must contain DenseSamplingInterval.")
        # Numeric/reason validation is already guaranteed by this model's constructor.
        if not (_inside(interval.start_time_seconds, lower, upper) and _inside(interval.end_time_seconds, lower, upper)):
            raise ValueError("dense interval outside window.")
        if previous is not None and interval.start_time_seconds <= previous:
            raise ValueError("dense intervals must be ordered, non-overlapping and already merged.")
        previous = interval.end_time_seconds
    previous = None
    for time in plan.requested_timestamps_seconds:
        _number("requested timestamp", time)
        if not _inside(time, lower, upper):
            raise ValueError("requested timestamp outside window.")
        if previous is not None and time <= previous:
            raise ValueError("requested timestamps must be chronological and unique.")
        previous = time
    for _, field in _ANCHORS:
        if not any(_close(time, getattr(window, field)) for time in plan.requested_timestamps_seconds):
            raise ValueError("sampling plan is missing a mandatory semantic anchor request.")


def _views(window, views):
    """Validate stable spans/core ownership without recomputing stability/provenance."""
    if not isinstance(views, Sequence) or isinstance(views, (str, bytes)):
        raise TypeError("stable_views must be a sequence of TicketStableView.")
    lower, upper = window.investigation_start_time_seconds, window.investigation_end_time_seconds
    spans = []
    for view in views:
        if not isinstance(view, TicketStableView):
            raise TypeError("stable_views must contain TicketStableView.")
        _index("stable investigation_index", view.investigation_index)
        if view.investigation_index != window.investigation_index:
            raise ValueError("stable view Investigation index must match window.")
        start, confirmed = view.stable_start_time_seconds, view.stable_confirmed_time_seconds
        end = upper if view.stable_end_time_seconds is None else view.stable_end_time_seconds
        for name, value in (("stable start", start), ("stable confirmation", confirmed), ("stable end", end)):
            _number(name, value)
        if not (_inside(start, lower, upper) and _inside(end, lower, upper)
                and start <= end and _inside(confirmed, start, end)):
            raise ValueError("stable span/confirmation must be ordered inside Investigation core.")
        if spans and start <= spans[-1][1]:
            raise ValueError("stable views must be chronological and non-overlapping (inclusive bounds).")
        if not isinstance(view.follows_scroll, bool):
            raise TypeError("follows_scroll must be bool.")
        if view.prior_scroll_end_time_seconds is not None:
            _number("prior scroll end", view.prior_scroll_end_time_seconds)
        direction = view.prior_scroll_direction
        if direction is not None and not isinstance(direction, TicketScrollState):
            raise TypeError("prior scroll direction must be TicketScrollState or None.")
        if view.follows_scroll:
            if (view.prior_scroll_end_time_seconds is None
                    or not _inside(view.prior_scroll_end_time_seconds, lower, start)
                    or direction is TicketScrollState.QUIET):
                raise ValueError("post-scroll provenance must remain consistent with the supplied span.")
        elif view.prior_scroll_end_time_seconds is not None or direction is not None:
            raise ValueError("ordinary stable view must not have prior scroll provenance.")
        spans.append((start, end))
    return spans


def build_evidence_candidate_timeline(
    *, window: InvestigationWindow, sampling_plan: AdaptiveSamplingPlan,
    stable_views: Sequence[TicketStableView],
) -> tuple[EvidenceCandidate, ...]:
    """Enrich every request exactly once; never run sampling/detectors or repair input.

    All source labels follow SOURCE_ORDER. Sampling is dense when a dense interval
    matches, otherwise sparse; anchors never replace this label. Stable bounds are
    inclusive, with None ends capped at the core end. Raise TypeError for malformed
    types, ValueError for mismatches, invalid ordering/bounds, ambiguous membership,
    missing mandatory anchor requests or any stable view with no request coverage.
    Distinct timestamps are preserved exactly, even within equality tolerance.
    """
    _window(window)
    _plan(window, sampling_plan)
    spans = _views(window, stable_views)
    candidates, covered = [], set()
    for time in sampling_plan.requested_timestamps_seconds:
        dense = [d for d in sampling_plan.dense_intervals if _inside(time, d.start_time_seconds, d.end_time_seconds)]
        matching = [i for i, (start, end) in enumerate(spans) if _inside(time, start, end)]
        if len(dense) > 1 or len(matching) > 1:
            raise ValueError("timestamp matches multiple dense intervals or stable views within tolerance.")
        labels = {"dense_sampling" if dense else "sparse_sampling"}
        labels.update(label for label, field in _ANCHORS if _close(time, getattr(window, field)))
        stable_index = matching[0] if matching else None
        follows_scroll = False
        if stable_index is not None:
            covered.add(stable_index)
            labels.add("stable_ticket_view")
            follows_scroll = stable_views[stable_index].follows_scroll
        candidates.append(EvidenceCandidate(window.investigation_index, time,
            tuple(label for label in SOURCE_ORDER if label in labels), dense[0].reasons if dense else (),
            stable_index, follows_scroll))
    if len(covered) != len(stable_views):
        raise ValueError("stable view has no requested timestamp coverage.")
    return tuple(candidates)
