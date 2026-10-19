"""Pure composition tests using existing model records, no frames/raw signals."""

from dataclasses import FrozenInstanceError, fields, replace

import pytest

from events_diary.adaptive_sampling import AdaptiveSamplingPlan, DenseSamplingInterval
from events_diary.evidence_candidate_timeline import (
    SOURCE_ORDER,
    EvidenceCandidate,
    build_evidence_candidate_timeline,
)
from events_diary.investigation_window import InvestigationWindow
from events_diary.ticket_scroll import TicketScrollState
from events_diary.ticket_stable_views import TicketStableView


def _window():
    """Return a core with distinct before/after context and four semantic anchors."""
    return InvestigationWindow(2, 1, 3, 0, 4, 1, 1)


def _plan(**changes):
    """Construct a frozen plan directly, without running upstream algorithms."""
    return replace(AdaptiveSamplingPlan(_window(), 10, 1, 10, 0.5, 0.5,
        (DenseSamplingInterval(1.5, 2.5, ("ticket_scroll", "ticket_body_change")),),
        (0, 1, 1.5, 2, 2.5, 3, 4)), **changes)


def _view(start=1.5, end=2.5, **changes):
    """Return a supplied stable annotation with no recomputed stability."""
    return replace(TicketStableView(2, start, start + 0.5, end, False, None, None), **changes)


def _build(plan=None, views=(), window=None):
    """Call composition using only the three approved model inputs."""
    return build_evidence_candidate_timeline(window=window or _window(),
        sampling_plan=plan if plan is not None else _plan(), stable_views=views)


def test_exact_frozen_candidate_fields_and_every_request_preserved():
    plan = _plan()
    candidates = _build(plan)
    assert isinstance(candidates, tuple)
    assert len(candidates) == len(plan.requested_timestamps_seconds)
    assert tuple(c.timestamp_seconds for c in candidates) == plan.requested_timestamps_seconds
    assert [f.name for f in fields(EvidenceCandidate)] == ["investigation_index", "timestamp_seconds",
        "sources", "dense_reasons", "stable_view_index", "stable_view_follows_scroll"]
    with pytest.raises(FrozenInstanceError):
        candidates[0].sources = ()


def test_sparse_dense_precedence_reasons_and_empty_stable_metadata():
    candidates = _build()
    assert candidates[0].sources == ("sparse_sampling", "window_start")
    assert candidates[0].dense_reasons == ()
    assert candidates[3].sources == ("dense_sampling",)
    assert candidates[3].dense_reasons == ("ticket_scroll", "ticket_body_change")
    assert all(not {"dense_sampling", "sparse_sampling"}.issubset(c.sources) for c in candidates)
    assert all(c.stable_view_index is None and not c.stable_view_follows_scroll for c in candidates)


def test_all_anchors_coexist_with_sampling_and_stable_sources():
    window = InvestigationWindow(2, 0, 4, 0, 4, 0, 0)
    plan = _plan(window=window, dense_intervals=(DenseSamplingInterval(0, 4, ("map_translation",)),))
    candidates = _build(plan, [_view(0, 4)], window)
    assert candidates[0].sources == ("dense_sampling", "window_start", "investigation_start", "stable_ticket_view")
    assert candidates[-1].sources == ("dense_sampling", "investigation_end", "window_end", "stable_ticket_view")
    assert candidates[0].dense_reasons == ("map_translation",)
    assert SOURCE_ORDER == ("sparse_sampling", "dense_sampling", "window_start", "investigation_start",
                            "investigation_end", "window_end", "stable_ticket_view")


def test_four_distinct_anchors_and_sparse_anchor_label():
    candidates = {c.timestamp_seconds: c for c in _build()}
    for time, label in ((0, "window_start"), (1, "investigation_start"),
                        (3, "investigation_end"), (4, "window_end")):
        assert candidates[time].sources == ("sparse_sampling", label)


def test_stable_association_inclusive_bounds_and_post_scroll_propagation():
    view = _view(follows_scroll=True, prior_scroll_end_time_seconds=1.4,
                 prior_scroll_direction=TicketScrollState.SCROLLING_UP)
    candidates = _build(views=[view])
    stable = [c for c in candidates if c.stable_view_index is not None]
    assert [c.timestamp_seconds for c in stable] == [1.5, 2, 2.5]
    assert all(c.stable_view_index == 0 and c.stable_view_follows_scroll for c in stable)
    assert all(c.sources == ("dense_sampling", "stable_ticket_view") for c in stable)
    assert view.prior_scroll_direction is TicketScrollState.SCROLLING_UP


def test_multiple_stable_views_and_right_censoring_stops_at_core_end():
    views = [_view(1, 1.5), _view(2, None)]
    candidates = _build(views=views)
    assert [c.stable_view_index for c in candidates] == [None, 0, 0, 1, 1, 1, None]
    assert all(not c.stable_view_follows_scroll for c in candidates)


def test_stability_annotation_starts_before_confirmation_without_new_requests():
    view = _view(1, 2.5, stable_confirmed_time_seconds=1.7)
    candidates = _build(views=[view])
    assert candidates[1].stable_view_index == 0
    assert 1.7 not in [c.timestamp_seconds for c in candidates]


def test_tiny_matching_noise_and_distinct_requests_never_merged():
    times = (0, 1 + 2e-10, 1.5, 2, 2 + 1e-12, 2.5, 3, 4)
    candidates = _build(_plan(requested_timestamps_seconds=times),
                        [_view(1.5 + 2e-10, 2.5 - 2e-10)])
    assert tuple(c.timestamp_seconds for c in candidates) == times
    assert "investigation_start" in candidates[1].sources
    assert candidates[2].stable_view_index == candidates[5].stable_view_index == 0
    assert candidates[3].timestamp_seconds != candidates[4].timestamp_seconds


def test_normal_distinct_source_times_remain_separate():
    times = (0, 1, 2, 3, 3.2, 3.3, 4)
    assert tuple(c.timestamp_seconds for c in _build(_plan(requested_timestamps_seconds=times))) == times


def test_window_semantic_equality_accepts_tiny_noise():
    window = replace(_window(), context_before_seconds=1 + 1e-10)
    assert len(_build(window=window)) == 7


@pytest.mark.parametrize("views", [
    [_view(1, 2), _view(1.5, 2.5)], [_view(2, 2.5), _view(1, 1.5)],
    [_view(1, 2), _view(2, 2.5)], [_view(1, None), _view(2, 2.5)],
])
def test_overlap_touching_and_nonchronological_stable_views_rejected(views):
    with pytest.raises(ValueError, match="chronological|overlapping"):
        _build(views=views)


def test_stable_view_without_coverage_raises_no_synthetic_candidate():
    plan = _plan(requested_timestamps_seconds=(0, 1, 3, 4))
    with pytest.raises(ValueError, match="coverage"):
        _build(plan, [_view(1.2, 2.7)])


def test_tolerance_cannot_choose_between_nearby_stable_spans():
    views = [_view(1, 2), _view(2 + 2e-10, 3)]
    with pytest.raises(ValueError, match="multiple"):
        _build(views=views)


@pytest.mark.parametrize("intervals", [
    (DenseSamplingInterval(1, 2, ("ticket_scroll",)), DenseSamplingInterval(1.5, 3, ("ticket_scroll",))),
    (DenseSamplingInterval(1, 2, ("ticket_scroll",)), DenseSamplingInterval(2, 3, ("ticket_scroll",))),
    (DenseSamplingInterval(2, 3, ("ticket_scroll",)), DenseSamplingInterval(1, 1.5, ("ticket_scroll",))),
    (DenseSamplingInterval(-0.0, 5, ("ticket_scroll",)),),
])
def test_bad_dense_intervals_rejected(intervals):
    with pytest.raises(ValueError):
        _build(_plan(dense_intervals=intervals))


def test_tolerance_cannot_choose_between_nearby_dense_intervals():
    intervals = (DenseSamplingInterval(1, 2, ("ticket_scroll",)),
                 DenseSamplingInterval(2 + 2e-10, 3, ("map_translation",)))
    with pytest.raises(ValueError, match="multiple"):
        _build(_plan(dense_intervals=intervals))


@pytest.mark.parametrize("times,error", [
    ((0, 1, 1, 3, 4), ValueError), ((0, 2, 1, 3, 4), ValueError),
    ((-0.1, 0, 1, 3, 4), ValueError), ((0, 1, 3, 4, 5), ValueError),
    ((0, 1, float("nan"), 3, 4), ValueError), ((0, 1, float("inf"), 4), ValueError),
    ((False, 1, 3, 4), TypeError), ((0, "1", 3, 4), TypeError),
    ((), ValueError), ((0, 4), ValueError),
])
def test_malformed_requests_and_missing_anchors_rejected(times, error):
    with pytest.raises(error):
        _build(_plan(requested_timestamps_seconds=times))


def test_zero_duration_window_still_has_one_mandatory_candidate():
    window = InvestigationWindow(2, 1, 1, 1, 1, 0, 0)
    plan = _plan(window=window, dense_intervals=(), requested_timestamps_seconds=(1,))
    result = _build(plan, window=window)
    assert len(result) == 1
    assert result[0].sources == ("sparse_sampling", "window_start", "investigation_start", "investigation_end", "window_end")


@pytest.mark.parametrize("view,error", [
    (_view(investigation_index=3), ValueError), (_view(investigation_index=True), TypeError),
    (_view(stable_start_time_seconds=float("nan")), ValueError),
    (_view(stable_end_time_seconds=float("inf")), ValueError),
    (_view(stable_start_time_seconds=True), TypeError),
    (_view(stable_confirmed_time_seconds=0), ValueError),
    (_view(stable_end_time_seconds=1), ValueError),
    (_view(stable_end_time_seconds=3.1), ValueError),
    (_view(follows_scroll=1), TypeError), (_view(follows_scroll=True), ValueError),
    (_view(prior_scroll_end_time_seconds=1), ValueError),
    (_view(prior_scroll_direction="mixed"), TypeError),
])
def test_malformed_stable_views_rejected(view, error):
    with pytest.raises(error):
        _build(views=[view])


def test_plan_window_mismatch_rejected():
    with pytest.raises(ValueError, match="match"):
        _build(_plan(window=replace(_window(), investigation_index=3)))
    with pytest.raises(ValueError, match="match"):
        _build(_plan(window=replace(_window(), investigation_start_time_seconds=1.1)))


@pytest.mark.parametrize("field,value,error", [
    ("source_fps", True, TypeError), ("dense_sampling_fps", float("inf"), ValueError),
    ("pre_padding_seconds", float("nan"), ValueError), ("source_fps", 0, ValueError),
    ("sparse_sampling_fps", 20, ValueError), ("dense_sampling_fps", 5, ValueError),
    ("dense_intervals", [], TypeError), ("requested_timestamps_seconds", [], TypeError),
    ("dense_intervals", (None,), TypeError),
])
def test_constructible_invalid_plan_metadata_rejected(field, value, error):
    with pytest.raises(error):
        _build(_plan(**{field: value}))


@pytest.mark.parametrize("views", [None, "bad", [None]])
def test_wrong_view_sequence_types(views):
    with pytest.raises(TypeError):
        _build(views=views)


def test_inputs_not_mutated_and_reason_tuple_preserved():
    window, plan, views = _window(), _plan(), [_view()]
    snapshot = (replace(window), replace(plan), [replace(v) for v in views])
    result = _build(plan, views, window)
    assert (window, plan, views) == snapshot
    assert result[3].dense_reasons is plan.dense_intervals[0].reasons
