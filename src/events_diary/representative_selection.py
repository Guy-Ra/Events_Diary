"""Phase 5.3b pure role-first selection of existing materialized references.

Ownership precedes role coverage, stability and local quality. Inputs are one
Investigation's complete Phase 4 timeline and its existing window/plan/views;
composition is reused to validate exact index correspondence, not to recompute
signals or change requests. Outputs contain indexes into those supplied sequences
and deterministic reason strings only. No images or decoding are required by the
selector.

Every stable view keeps exactly one primary ticket reference. Both requested and
physical times must lie in its stable span, and physical time must belong to the
half-open Investigation core [start,end). This allocates a shared SWITCH endpoint
to the incoming core, while a nominal video-end request can still represent the
last physical frame before end. Existing 1e-9-second equality tolerance treats
representation-equivalent endpoints identically on both sides of that boundary.
No image-based ownership or ticket identity is inferred. A resolved frame at/after
confirmation is always preferred; choose the
nearest physical time. Only distances tied within the existing 1e-9-second
representation tolerance compete on sharpness, then contrast, then request/index
order. Brightness is never ranked. Missing quality sorts last within that tie.
If no post-confirmation frame is available, choose the nearest eligible in-span
frame and mark fallback. No eligible frame for a declared view is a structural
error; no views is a valid empty primary set. Similarity never removes views,
including post-scroll views or returns to earlier positions.

Map/route evidence defaults on: one physical-time midpoint per contiguous run of
relevant candidate dense reasons. Runs split at a nonmatching candidate, an
outside-core request, or a request gap greater than one source-frame period plus
the existing representation tolerance. No arbitrary interaction timeout is added.
Unresolved members do not invent a break; a run with no resolved in-core member
cannot supply context and is skipped. Midpoint ties favor earlier physical time,
then request/index order, never quality. A selected member carries the union of
map/route reasons of its run. There is no fixed global quota or map similarity.

Optional scroll provenance uses the same one-midpoint rule for ticket_scroll
dense reasons. These labels are PADDED/MERGED Phase 4 activity context, not the
original positive-pair bursts or a reconstruction of human interactions. A
midpoint can be a quiet reading frame. It is context, never a substitute for a
stable primary slot. Optional boundary context adds only existing resolved core
start/end anchors, including transitional/incoming frames deliberately labeled
investigation_boundary. Neither option defaults on; window anchors alone never
select anything. Primary and activity roles require physical core ownership;
boundary context intentionally describes the exact transition instead.

One index can satisfy several roles and is returned once with reasons in the
documented canonical order, chronologically by original request/index. Different
indexes resolving to the same physical frame are not silently merged: upstream
references/provenance remain distinct. No global quality ranking, OCR, VLM,
semantic loading/section/duplicate classifier, Enum, Phase 4 modification, or
Phase 5.4 closure is implemented.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real

from events_diary.adaptive_sampling import AdaptiveSamplingPlan
from events_diary.evidence_candidate_timeline import (
    TIMESTAMP_ABSOLUTE_TOLERANCE_SECONDS,
    EvidenceCandidate,
    build_evidence_candidate_timeline,
)
from events_diary.frame_materialization import MaterializedEvidenceCandidate
from events_diary.investigation_window import InvestigationWindow
from events_diary.ticket_stable_views import TicketStableView

_EPS = TIMESTAMP_ABSOLUTE_TOLERANCE_SECONDS
_REASONS = ("stable_ticket", "post_scroll_stable", "stable_confirmation_fallback",
            "map_activity", "route_activity", "scroll_provenance", "investigation_boundary")
_MAP = ("route_visual_change", "map_translation", "map_scale_change")


@dataclass(frozen=True)
class RepresentativeSelection:
    """One index into caller-supplied candidate/materialization sequences and reasons."""

    candidate_index: int
    selection_reasons: tuple[str, ...]


@dataclass(frozen=True)
class InvestigationSelection:
    """Chronological reference-only selections for one existing Investigation index."""

    investigation_index: int
    selections: tuple[RepresentativeSelection, ...]


def _inside(value, start, end):
    """Use Phase 4's representation tolerance for inclusive stable/request bounds."""
    return start-_EPS <= value <= end+_EPS


def _real(name, value):
    """Reject malformed scalar metadata without normalizing or repairing it."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be real, not bool.")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative.")


def _inputs(window, sampling_plan, candidates, materialized, views):
    """Reuse Phase 4 consistency checks and validate physical scalar references."""
    for name, seq in (("candidates", candidates), ("materialized_candidates", materialized), ("stable_views", views)):
        if not isinstance(seq, Sequence) or isinstance(seq, (str, bytes)):
            raise TypeError(f"{name} must be a sequence.")
    candidates, materialized, views = tuple(candidates), tuple(materialized), tuple(views)
    expected = build_evidence_candidate_timeline(window=window, sampling_plan=sampling_plan, stable_views=views)
    if not all(isinstance(c, EvidenceCandidate) for c in candidates):
        raise TypeError("candidates must contain EvidenceCandidate records.")
    for c in candidates:
        if isinstance(c.investigation_index, bool) or not isinstance(c.investigation_index, Integral):
            raise TypeError("Candidate Investigation index must be an integer, not bool.")
        _real("candidate timestamp", c.timestamp_seconds)
        if c.stable_view_index is not None and (
            isinstance(c.stable_view_index, bool) or not isinstance(c.stable_view_index, Integral)
        ):
            raise TypeError("Candidate stable-view index must be an integer, not bool.")
        if not isinstance(c.stable_view_follows_scroll, bool):
            raise TypeError("Candidate follows-scroll annotation must be bool.")
    if candidates != expected or len(candidates) != len(materialized):
        raise ValueError("Candidate/plan/view/materialization index correspondence is inconsistent.")
    for c, m in zip(candidates, materialized, strict=True):
        if not isinstance(m, MaterializedEvidenceCandidate):
            raise TypeError("materialized_candidates must contain MaterializedEvidenceCandidate records.")
        if m.evidence_candidate != c:
            raise ValueError("Materialization must reference the candidate at the same index.")
        quality = (m.ticket_sharpness_laplacian_variance, m.ticket_brightness_mean, m.ticket_contrast_std)
        if m.resolution_status == "unresolved":
            if any(v is not None for v in (m.frame_index, m.actual_timestamp_seconds, *quality)):
                raise ValueError("Unresolved materialization must have no physical/quality values.")
            continue
        if m.resolution_status not in ("resolved", "video_end_adjusted"):
            raise ValueError("Unknown materialization resolution status.")
        if isinstance(m.frame_index, bool) or not isinstance(m.frame_index, Integral):
            raise TypeError("Resolved frame_index must be an integer, not bool.")
        if m.frame_index < 0:
            raise ValueError("Resolved frame_index must be nonnegative.")
        _real("actual_timestamp_seconds", m.actual_timestamp_seconds)
        if not math.isclose(m.actual_timestamp_seconds, m.frame_index/sampling_plan.source_fps, rel_tol=0, abs_tol=_EPS):
            raise ValueError("Physical frame index/time must match the plan's source FPS.")
        for name, value in zip(("sharpness", "brightness", "contrast"), quality, strict=True):
            if value is not None:
                _real(name, value)
    return candidates, materialized, views


def _owned(window, item):
    """Allocate representationally equal shared endpoints to the incoming core."""
    time = item.actual_timestamp_seconds
    start, end = window.investigation_start_time_seconds, window.investigation_end_time_seconds
    return (item.resolution_status != "unresolved" and time is not None
            and (time >= start or math.isclose(time, start, rel_tol=0, abs_tol=_EPS))
            and time < end and not math.isclose(time, end, rel_tol=0, abs_tol=_EPS))


def _activity_bursts(candidates, window, sampling_plan, relevant):
    """Group adjacent core requests with relevant dense labels at source cadence.

    Return local candidate-index tuples, retaining unresolved members as temporal
    context. These are padded metadata runs, not direct-positive signal episodes.
    """
    groups, current = [], []
    gap = 1/sampling_plan.source_fps
    for i, c in enumerate(candidates):
        matches = (_inside(c.timestamp_seconds, window.investigation_start_time_seconds,
                           window.investigation_end_time_seconds)
                   and any(reason in c.dense_reasons for reason in relevant))
        if not matches or (current and c.timestamp_seconds-candidates[current[-1]].timestamp_seconds > gap+_EPS):
            if current:
                groups.append(tuple(current))
            current = []
        if matches:
            current.append(i)
    if current:
        groups.append(tuple(current))
    return tuple(groups)


def _midpoint(group, candidates, materialized, window):
    """Choose a resolved in-core physical midpoint, with no quality ranking."""
    valid = [i for i in group if _owned(window, materialized[i])]
    if not valid:
        return None
    center = (materialized[valid[0]].actual_timestamp_seconds+materialized[valid[-1]].actual_timestamp_seconds)/2
    distance = min(abs(materialized[i].actual_timestamp_seconds-center) for i in valid)
    tied = [i for i in valid if abs(abs(materialized[i].actual_timestamp_seconds-center)-distance) <= _EPS]
    return min(tied, key=lambda i: (materialized[i].actual_timestamp_seconds, candidates[i].timestamp_seconds, i))


def _stable_choice(group, view, candidates, materialized):
    """Prefer post-confirmation physical/request times, then local temporal ties."""
    confirmed = view.stable_confirmed_time_seconds
    post = [i for i in group if candidates[i].timestamp_seconds >= confirmed-_EPS
            and materialized[i].actual_timestamp_seconds >= confirmed-_EPS]
    pool = post or group
    distance = min(abs(materialized[i].actual_timestamp_seconds-confirmed) for i in pool)
    tied = [i for i in pool if abs(abs(materialized[i].actual_timestamp_seconds-confirmed)-distance) <= _EPS]

    def key(i):
        """Rank diagnostics only inside this same-view temporal tie."""
        item = materialized[i]
        sharp = item.ticket_sharpness_laplacian_variance
        contrast = item.ticket_contrast_std
        return (-(sharp if sharp is not None else -math.inf),
                -(contrast if contrast is not None else -math.inf), candidates[i].timestamp_seconds, i)

    return min(tied, key=key), not post


def select_investigation_representatives(
    *, window: InvestigationWindow, sampling_plan: AdaptiveSamplingPlan,
    candidates: Sequence[EvidenceCandidate],
    materialized_candidates: Sequence[MaterializedEvidenceCandidate],
    stable_views: Sequence[TicketStableView],
    include_scroll_provenance: bool = False,
    include_boundary_context: bool = False,
    include_map_route_evidence: bool = True,
) -> InvestigationSelection:
    """Select references without decoding, mutating inputs or collapsing stable views.

    Supply one complete composed timeline, matching materializations and the
    exact window/plan/views used to compose it. Indexes are local to these supplied
    sequences, not globally renumbered. Reuse composition only to validate these
    associations. TypeError rejects malformed types; ValueError rejects mismatched
    metadata or declared stable views with no eligible physical frame. Unresolved
    candidates are never selected. Missing quality is allowed, ranked last only
    in a temporal tie; invalid/nonfinite quality is rejected.

    One stable representative is mandatory per view, with explicit pre-confirmation
    fallback only when no owned resolved post-confirmation frame exists. No views
    returns a valid possibly-empty result; optional contexts remain independent.
    Activity runs use source-period request adjacency, NOT an arbitrary timeout or
    original positive-pair evidence. Boundary context is the only role allowed to
    deliberately select an exact outgoing end frame depicting incoming content.
    Canonical reason order is stable_ticket, post_scroll_stable,
    stable_confirmation_fallback, map_activity, route_activity, scroll_provenance,
    investigation_boundary. Sort final references by original request then index.
    """
    for name, value in (("include_scroll_provenance", include_scroll_provenance),
                        ("include_boundary_context", include_boundary_context),
                        ("include_map_route_evidence", include_map_route_evidence)):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be bool.")
    candidates, materialized, views = _inputs(window, sampling_plan, candidates, materialized_candidates, stable_views)
    reasons = {}

    def add(index, labels):
        """Merge role satisfaction by candidate index without copying upstream data."""
        reasons.setdefault(index, set()).update(labels)

    for vi, view in enumerate(views):
        end = view.stable_end_time_seconds
        if end is None:
            end = window.investigation_end_time_seconds
        group = [i for i, c in enumerate(candidates) if c.stable_view_index == vi
                 and _owned(window, materialized[i])
                 and _inside(materialized[i].actual_timestamp_seconds, view.stable_start_time_seconds, end)]
        if not group:
            raise ValueError(f"Stable view {vi} has no eligible owned materialized candidate.")
        index, fallback = _stable_choice(group, view, candidates, materialized)
        labels = ["stable_ticket"]
        if view.follows_scroll:
            labels.append("post_scroll_stable")
        if fallback:
            labels.append("stable_confirmation_fallback")
        add(index, labels)
    for enabled, relevant, scroll in ((include_map_route_evidence, _MAP, False),
                                      (include_scroll_provenance, ("ticket_scroll",), True)):
        if not enabled:
            continue
        for group in _activity_bursts(candidates, window, sampling_plan, relevant):
            index = _midpoint(group, candidates, materialized, window)
            if index is None:
                continue
            if scroll:
                labels = ["scroll_provenance"]
            else:
                union = {r for i in group for r in candidates[i].dense_reasons}
                labels = []
                if "map_translation" in union or "map_scale_change" in union:
                    labels.append("map_activity")
                if "route_visual_change" in union:
                    labels.append("route_activity")
            add(index, labels)
    if include_boundary_context:
        for i, c in enumerate(candidates):
            if materialized[i].resolution_status != "unresolved" and any(
                source in c.sources for source in ("investigation_start", "investigation_end")
            ):
                add(i, ("investigation_boundary",))
    selections = tuple(RepresentativeSelection(i, tuple(r for r in _REASONS if r in reasons[i]))
                       for i in sorted(reasons, key=lambda i: (candidates[i].timestamp_seconds, i)))
    return InvestigationSelection(window.investigation_index, selections)
