"""Synthetic ownership, role coverage, temporal ties and reference-only selection."""

import math
from dataclasses import FrozenInstanceError, asdict, fields, replace

import pytest

from events_diary.adaptive_sampling import AdaptiveSamplingPlan, DenseSamplingInterval
from events_diary.evidence_candidate_timeline import build_evidence_candidate_timeline
from events_diary.frame_materialization import MaterializedEvidenceCandidate
from events_diary.investigation_window import InvestigationWindow
from events_diary.representative_selection import (
    InvestigationSelection,
    RepresentativeSelection,
    _activity_bursts,
    select_investigation_representatives,
)
from events_diary.ticket_stable_views import TicketStableView


def view(start=0., confirmed=.5, end=1.5, *, post=False, inv=0):
    """Provide existing stable metadata, including valid optional scroll provenance."""
    return TicketStableView(inv, start, confirmed, end, post, start if post else None, None)


def data(times=(.4, .5, .6, 1.), views=None, dense=(), window=None):
    """Compose real model fixtures without image arrays or decoder access."""
    window = window or InvestigationWindow(0, 0., 3., 0., 3., 0., 0.)
    views = (view(),) if views is None else tuple(views)
    anchors = (window.window_start_time_seconds, window.investigation_start_time_seconds,
               window.investigation_end_time_seconds, window.window_end_time_seconds)
    times = tuple(sorted({*times, *anchors}))
    plan = AdaptiveSamplingPlan(window, 10., 1., 10., .5, .5,
        tuple(DenseSamplingInterval(*d) for d in dense), times)
    candidates = build_evidence_candidate_timeline(window=window, sampling_plan=plan, stable_views=views)
    materialized = []
    for c in candidates:
        endpoint = c.timestamp_seconds == window.window_end_time_seconds
        frame = math.floor(c.timestamp_seconds*10+.5)
        if endpoint:
            frame = max(0, frame-1)
        materialized.append(MaterializedEvidenceCandidate(c, "video_end_adjusted" if endpoint else "resolved",
            frame, frame/10, 100., 200., 50.))
    return {"window": window, "sampling_plan": plan, "candidates": candidates,
            "materialized_candidates": tuple(materialized), "stable_views": views}


def select(d, **options):
    """Call the production API with independent coverage options."""
    return select_investigation_representatives(**d, **options)


def at(d, time):
    """Find an original fixture request by its exact timestamp."""
    return next(i for i, c in enumerate(d["candidates"]) if c.timestamp_seconds == time)


def quality(d, time, sharp=100., contrast=50., brightness=200.):
    """Change scalar diagnostics without changing role/time eligibility."""
    items = list(d["materialized_candidates"])
    i = at(d, time)
    items[i] = replace(items[i], ticket_sharpness_laplacian_variance=sharp,
                       ticket_contrast_std=contrast, ticket_brightness_mean=brightness)
    return {**d, "materialized_candidates": tuple(items)}


def unresolved(d, time):
    """Retain an explicitly unresolved upstream record at the same index."""
    items = list(d["materialized_candidates"])
    i = at(d, time)
    items[i] = replace(items[i], resolution_status="unresolved", frame_index=None,
        actual_timestamp_seconds=None, ticket_sharpness_laplacian_variance=None,
        ticket_brightness_mean=None, ticket_contrast_std=None)
    return {**d, "materialized_candidates": tuple(items)}


def test_frozen_small_models():
    r = RepresentativeSelection(1, ("stable_ticket",))
    result = InvestigationSelection(0, (r,))
    for record, name, value in ((r, "candidate_index", 3), (result, "investigation_index", 9)):
        with pytest.raises(FrozenInstanceError):
            setattr(record, name, value)
    assert [f.name for f in fields(r)] == ["candidate_index", "selection_reasons"]
    assert [f.name for f in fields(result)] == ["investigation_index", "selections"]


def test_nearest_post_confirmation_beats_global_quality_and_brightness():
    d = quality(data(), .4, sharp=100000, brightness=255)
    d = quality(d, 1., sharp=200000)
    d = quality(d, .5, sharp=1, brightness=0)
    assert select(d).selections == (RepresentativeSelection(at(d, .5), ("stable_ticket",)),)


@pytest.mark.parametrize("second_sharp,second_contrast,expected", [(101, 1, .5000000001),
    (100, 51, .5000000001), (100, 50, .5), (99, 1000, .5)])
def test_temporal_ties_use_only_local_quality(second_sharp, second_contrast, expected):
    d = data(times=(.5, .5000000001))
    d = quality(d, .5000000001, sharp=second_sharp, contrast=second_contrast, brightness=255)
    assert select(d).selections[0].candidate_index == at(d, expected)


def test_missing_quality_sorts_last_only_in_temporal_tie():
    d = data(times=(.5, .5000000001))
    d = quality(d, .5, sharp=None)
    assert select(d).selections[0].candidate_index == at(d, .5000000001)
    d = quality(data(), .5, sharp=None, contrast=None)
    assert select(d).selections[0].candidate_index == at(d, .5)


def test_physical_time_preference_and_actual_ownership():
    d = data(times=(.49, .51, .61), views=(view(confirmed=.51),))
    # .51 resolves to .5, still physically pre-confirmation; .61 resolves to .6.
    assert select(d).selections[0].candidate_index == at(d, .61)


def test_multiple_post_scroll_views_all_retained_without_fixed_quota():
    vs = tuple(view(i, i+.5, i+.8, post=i>0) for i in range(7))
    window = InvestigationWindow(0, 0., 8., 0., 8., 0., 0.)
    d = data(times=tuple(i+.5 for i in range(7)), views=vs, window=window)
    d = quality(d, 6.5, sharp=0)
    selections = select(d).selections
    assert len(selections) == 7
    assert all(s.selection_reasons == ("stable_ticket", "post_scroll_stable") for s in selections[1:])
    assert select(quality(d, .5, sharp=99999)).selections == selections


def test_no_post_candidate_fallback_is_explicit():
    d = data(times=(.2, .4), views=(view(.2, .7, .9),))
    r = select(d).selections[0]
    assert r.candidate_index == at(d, .4)
    assert r.selection_reasons == ("stable_ticket", "stable_confirmation_fallback")


def test_unresolved_post_candidate_uses_valid_pre_fallback():
    d = unresolved(data(times=(.4, .5)), .5)
    r = select(d).selections[0]
    assert r.candidate_index == at(d, .4)
    assert "stable_confirmation_fallback" in r.selection_reasons


def test_zero_associated_or_zero_resolved_stable_candidates_raise():
    with pytest.raises(ValueError, match="coverage"):
        data(times=(), views=(view(.2, .5, .9),))
    d = unresolved(data(times=(.5,), views=(view(.2, .5, .9),)), .5)
    with pytest.raises(ValueError, match="no eligible"):
        select(d)


def test_empty_ticket_set_with_no_views_is_valid():
    d = data(times=(), views=())
    assert select(d) == InvestigationSelection(0, ())


def test_context_and_shared_end_are_not_primary_and_boundaries_are_optional():
    window = InvestigationWindow(0, 1., 3., 0., 4., 1., 1.)
    d = data(times=(.9, 1.5, 2.9), views=(view(1., 1.5, 3.),), window=window)
    d = quality(d, .9, sharp=999999)
    assert select(d).selections == (RepresentativeSelection(at(d, 1.5), ("stable_ticket",)),)
    full = select(d, include_boundary_context=True)
    assert [s.candidate_index for s in full.selections] == [at(d, 1.), at(d, 1.5), at(d, 3.)]
    assert full.selections[-1].selection_reasons == ("investigation_boundary",)
    assert at(d, 0.) not in [s.candidate_index for s in full.selections]
    assert at(d, 4.) not in [s.candidate_index for s in full.selections]


def test_end_only_stable_association_cannot_claim_incoming_physical_frame():
    window = InvestigationWindow(0, 1., 3., 0., 4., 1., 1.)
    d = data(times=(), views=(view(2.7, 2.9, 3.),), window=window)
    with pytest.raises(ValueError, match="no eligible"):
        select(d)


def test_nominal_end_adjusted_frame_can_represent_ongoing_view():
    d = data(times=(), views=(view(2.7, 2.9, None),))
    r = select(d).selections[0]
    assert r.candidate_index == at(d, 3.) and r.selection_reasons == ("stable_ticket",)


def test_rounding_equivalent_core_start_includes_incoming_frame():
    start = .1+.2
    window = InvestigationWindow(0, start, 1., 0., 1., start, 0.)
    d = data(times=(), views=(view(start, start, .6),), window=window)
    assert select(d).selections[0].candidate_index == at(d, start)


def test_rounding_equivalent_core_end_excludes_incoming_frame():
    end = .1+.2
    window = InvestigationWindow(0, 0., end, 0., 1., 0., 1.-end)
    d = data(times=(), views=(view(.2, .25, end),), window=window)
    with pytest.raises(ValueError, match="no eligible"):
        select(d)


@pytest.mark.parametrize("reason,label", [("map_translation", "map_activity"),
    ("map_scale_change", "map_activity"), ("route_visual_change", "route_activity")])
def test_one_midpoint_per_map_route_burst_and_option(reason, label):
    d = data(times=(.1, .2, .3, .4, .5), views=(), dense=((.1, .5, (reason,)),))
    assert select(d).selections == (RepresentativeSelection(at(d, .3), (label,)),)
    assert select(d, include_map_route_evidence=False).selections == ()


def test_mixed_burst_reason_union_and_same_candidate_role_merge():
    d = data(times=(.3, .4, .5), views=(view(.3, .4, .5, post=True),),
        dense=((.3, .5, ("ticket_scroll", "route_visual_change", "map_translation")),))
    r = select(d, include_scroll_provenance=True).selections
    assert r == (RepresentativeSelection(at(d, .4),
        ("stable_ticket", "post_scroll_stable", "map_activity", "route_activity", "scroll_provenance")),)


def test_reason_union_across_different_neighbor_trigger_kinds():
    d = data(times=(.1, .2, .3, .4), views=(), dense=(
        (.1, .2, ("map_translation",)), (.3, .4, ("route_visual_change",))))
    assert select(d).selections == (RepresentativeSelection(at(d, .2), ("map_activity", "route_activity")),)


def test_scroll_optional_bounded_and_stable_views_unchanged():
    d = data(times=(.1, .2, .3, .5, .7, .8, .9, 1.5), views=(view(confirmed=.5, end=.6), view(1.1, 1.5, 1.8, post=True)),
        dense=((.1, .3, ("ticket_scroll",)), (.7, .9, ("ticket_scroll",))))
    base = select(d).selections
    full = select(d, include_scroll_provenance=True).selections
    assert len(base) == 2 and len(full) == 4
    assert [s for s in full if "stable_ticket" in s.selection_reasons] == list(base)
    assert [s.candidate_index for s in full if "scroll_provenance" in s.selection_reasons] == [at(d, .2), at(d, .8)]


def test_bursts_split_on_nonmatching_candidate_and_cadence_gap():
    d = data(times=(.1, .2, .4, .6, .7), views=(), dense=(
        (.1, .2, ("map_translation",)), (.4, .7, ("map_translation",))))
    groups = _activity_bursts(d["candidates"], d["window"], d["sampling_plan"], ("map_translation",))
    assert groups == ((at(d, .1), at(d, .2)), (at(d, .4),), (at(d, .6), at(d, .7)))
    assert len(select(d).selections) == 3


def test_unresolved_activity_members_do_not_invent_breaks_or_selections():
    d = data(times=(.1, .2, .3), views=(), dense=((.1, .3, ("ticket_scroll",)),))
    d = unresolved(d, .2)
    assert len(select(d, include_scroll_provenance=True).selections) == 1
    d = unresolved(unresolved(d, .1), .3)
    assert select(d, include_scroll_provenance=True).selections == ()


@pytest.mark.parametrize("map_on,scroll_on,boundary_on", [(a,b,c) for a in (False,True) for b in (False,True) for c in (False,True)])
def test_independent_options(map_on, scroll_on, boundary_on):
    d = data(times=(.2, .3, .4), views=(), dense=((.2, .4, ("ticket_scroll", "map_translation")),))
    r = select(d, include_map_route_evidence=map_on, include_scroll_provenance=scroll_on, include_boundary_context=boundary_on)
    labels = {label for s in r.selections for label in s.selection_reasons}
    assert ("map_activity" in labels) == map_on
    assert ("scroll_provenance" in labels) == scroll_on
    assert ("investigation_boundary" in labels) == boundary_on


def test_chronological_deterministic_immutable_outputs_and_inputs():
    d = data(times=(.2, .3, .4, .5), dense=((.2, .4, ("ticket_scroll", "map_translation")),))
    before = {name: repr(value) for name, value in d.items()}
    r = select(d, include_scroll_provenance=True, include_boundary_context=True)
    assert r == select(d, include_scroll_provenance=True, include_boundary_context=True)
    assert [s.candidate_index for s in r.selections] == sorted({s.candidate_index for s in r.selections})
    assert before == {name: repr(value) for name, value in d.items()}
    assert set(asdict(r)) == {"investigation_index", "selections"}


@pytest.mark.parametrize("bad", ["length", "reference", "view_index", "foreign_investigation", "follows", "order", "status", "time", "quality"])
def test_structural_mismatches_rejected(bad):
    d = data()
    candidates, mats = list(d["candidates"]), list(d["materialized_candidates"])
    i = at(d, .5)
    if bad == "length":
        mats.pop()
    elif bad == "reference":
        mats[i] = replace(mats[i], evidence_candidate=candidates[0])
    elif bad in ("view_index", "foreign_investigation", "follows"):
        field, value = {"view_index": ("stable_view_index", 9), "foreign_investigation": ("investigation_index", 1),
                        "follows": ("stable_view_follows_scroll", True)}[bad]
        candidates[i] = replace(candidates[i], **{field: value})
        mats[i] = replace(mats[i], evidence_candidate=candidates[i])
    elif bad == "order":
        candidates.reverse()
        mats.reverse()
    elif bad == "status":
        mats[i] = replace(mats[i], resolution_status="invented")
    elif bad == "time":
        mats[i] = replace(mats[i], actual_timestamp_seconds=.6)
    else:
        mats[i] = replace(mats[i], ticket_sharpness_laplacian_variance=float("nan"))
    with pytest.raises(ValueError):
        select({**d, "candidates": tuple(candidates), "materialized_candidates": tuple(mats)})


@pytest.mark.parametrize("name", ["include_scroll_provenance", "include_map_route_evidence", "include_boundary_context"])
def test_options_reject_non_bool(name):
    with pytest.raises(TypeError):
        select(data(), **{name: 1})


@pytest.mark.parametrize("field,value", [("investigation_index", False), ("timestamp_seconds", False),
                                        ("stable_view_index", False), ("stable_view_follows_scroll", 0)])
def test_boolean_numeric_metadata_is_not_silently_equal(field, value):
    d = data()
    candidates = list(d["candidates"])
    items = list(d["materialized_candidates"])
    candidates[0] = replace(candidates[0], **{field: value})
    items[0] = replace(items[0], evidence_candidate=candidates[0])
    with pytest.raises(TypeError):
        select({**d, "candidates": tuple(candidates), "materialized_candidates": tuple(items)})
