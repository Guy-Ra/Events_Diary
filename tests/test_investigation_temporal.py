"""Offline temporal confirmation tests using synthetic semantic entries only."""

from dataclasses import FrozenInstanceError, fields, replace
from itertools import product

import pytest

from events_diary.investigation_detection import (
    InvestigationCandidate,
    InvestigationCandidateKind,
)
from events_diary.investigation_temporal import (
    CONFIRMATION_WINDOW_SECONDS,
    CandidateConfirmationResult,
    CandidateTimelineEntry,
    ConfirmationStatus,
    ConfirmedInvestigationEvent,
    confirm_investigation_candidates,
)

KINDS = (InvestigationCandidateKind.OPEN, InvestigationCandidateKind.SWITCH,
         InvestigationCandidateKind.CLOSE)
NONE = InvestigationCandidateKind.NONE


def _timeline(times, kinds, initial=False):
    """Build contiguous irregular-time pairs, preserving shared-frame presence."""
    entries, before_time, present = [], 0, initial
    for index, (time, kind) in enumerate(zip(times, kinds)):
        after = present if kind is NONE else kind is not InvestigationCandidateKind.CLOSE
        entries.append(CandidateTimelineEntry(index, before_time, time, present, after,
                                             InvestigationCandidate(kind, "synthetic")))
        before_time, present = time, after
    return entries


def test_exact_statuses_models_frozen_and_fields():
    assert CONFIRMATION_WINDOW_SECONDS == 0.7
    assert {status.name: status.value for status in ConfirmationStatus} == {
        "CONFIRMED": "confirmed", "SUPERSEDED": "superseded",
        "INSUFFICIENT_FUTURE_CONTEXT": "insufficient_future_context"}
    entry = _timeline([0.1], [KINDS[0]])[0]
    result = CandidateConfirmationResult(0, KINDS[0], 0.1,
                                         ConfirmationStatus.CONFIRMED, 0.8, None)
    event = ConfirmedInvestigationEvent(KINDS[0], 0, 0.1, 0.8)
    assert [f.name for f in fields(entry)] == ["pair_index", "time_before_seconds",
        "time_after_seconds", "present_before", "present_after", "candidate"]
    assert [f.name for f in fields(result)] == ["candidate_pair_index", "candidate_kind",
        "candidate_time_seconds", "status", "confirmed_at_seconds", "superseded_by_pair_index"]
    assert [f.name for f in fields(event)] == ["kind", "candidate_pair_index",
                                            "candidate_time_seconds", "confirmed_at_seconds"]
    for record, attribute in ((entry, "pair_index"), (result, "candidate_pair_index"),
                              (event, "candidate_pair_index")):
        with pytest.raises(FrozenInstanceError):
            setattr(record, attribute, 100)
    with pytest.raises(ValueError, match="NONE"):
        ConfirmedInvestigationEvent(NONE, 0, 0, 1)


def test_empty_and_none_only_timelines():
    assert confirm_investigation_candidates([]) == ([], [])
    assert confirm_investigation_candidates(_timeline([0.2, 1.1], [NONE, NONE])) == ([], [])


@pytest.mark.parametrize("kind", KINDS)
def test_stable_candidate_followed_by_none_confirms_at_original_time(kind):
    entries = _timeline([0.1, 0.25, 0.61, 0.8], [kind, NONE, NONE, NONE], initial=True)
    snapshot = [replace(e) for e in entries]
    events, results = confirm_investigation_candidates(entries)
    assert events == [ConfirmedInvestigationEvent(kind, 0, 0.1, 0.1 + 0.7)]
    assert results == [CandidateConfirmationResult(0, kind, 0.1,
                         ConfirmationStatus.CONFIRMED, 0.1 + 0.7, None)]
    assert entries == snapshot


@pytest.mark.parametrize("first,second", list(product(KINDS, repeat=2)))
def test_any_later_candidate_supersedes_even_same_kind(first, second):
    entries = _timeline([0.1, 0.4, 1.2], [first, second, NONE], initial=True)
    events, results = confirm_investigation_candidates(entries)
    assert results[0] == CandidateConfirmationResult(0, first, 0.1,
                             ConfirmationStatus.SUPERSEDED, None, 1)
    assert len(events) == 1
    assert events[0].kind is second
    assert events[0].candidate_time_seconds == 0.4
    assert events[0].confirmed_at_seconds == 0.4 + 0.7


def test_supersession_at_inclusive_boundary_and_first_competitor_wins():
    entries = _timeline([0.1, 0.8, 0.85, 1.6], [KINDS[0], KINDS[1], KINDS[2], NONE])
    _, results = confirm_investigation_candidates(entries)
    assert results[0].status is ConfirmationStatus.SUPERSEDED
    assert results[0].superseded_by_pair_index == 1
    assert results[1].superseded_by_pair_index == 2
    assert results[2].status is ConfirmationStatus.CONFIRMED


def test_later_candidate_outside_window_does_not_supersede():
    entries = _timeline([0.1, 0.8, 0.81, 1.6], [KINDS[0], NONE, KINDS[2], NONE])
    events, results = confirm_investigation_candidates(entries)
    assert [r.status for r in results] == [ConfirmationStatus.CONFIRMED] * 2
    assert [e.candidate_pair_index for e in events] == [0, 2]


def test_short_ending_never_confirms_or_shortens_window():
    entries = _timeline([0.1, 0.79], [KINDS[1], NONE], initial=True)
    events, results = confirm_investigation_candidates(entries)
    assert events == []
    assert results == [CandidateConfirmationResult(0, KINDS[1], 0.1,
                        ConfirmationStatus.INSUFFICIENT_FUTURE_CONTEXT, None, None)]


def test_superseding_before_early_end_is_still_decisive():
    entries = _timeline([0.1, 0.4], [KINDS[0], KINDS[2]])
    events, results = confirm_investigation_candidates(entries)
    assert events == []
    assert results[0].status is ConfirmationStatus.SUPERSEDED
    assert results[1].status is ConfirmationStatus.INSUFFICIENT_FUTURE_CONTEXT


def test_seconds_based_irregular_sampling_and_custom_window():
    entries = _timeline([0.15, 0.19, 1.3], [KINDS[0], NONE, NONE])
    events, _ = confirm_investigation_candidates(entries, confirmation_window_seconds=0.9)
    assert events[0].confirmed_at_seconds == 0.15 + 0.9
    events, results = confirm_investigation_candidates(entries, confirmation_window_seconds=2)
    assert not events
    assert results[0].status is ConfirmationStatus.INSUFFICIENT_FUTURE_CONTEXT


def test_close_reopen_accidental_open_superseded_by_switch():
    entries = _timeline([3.8, 4.5, 7.0, 7.1, 7.6, 8.3],
                        [KINDS[2], NONE, NONE, KINDS[0], KINDS[1], NONE], initial=True)
    events, results = confirm_investigation_candidates(entries)
    assert [r.status for r in results] == [ConfirmationStatus.CONFIRMED,
                                          ConfirmationStatus.SUPERSEDED, ConfirmationStatus.CONFIRMED]
    assert results[1].superseded_by_pair_index == 4
    assert [e.candidate_time_seconds for e in events] == [3.8, 7.6]
    assert [e.confirmed_at_seconds for e in events] == [4.5, 7.6 + 0.7]


def test_unassociated_presence_reversal_with_full_coverage_reports_ambiguity():
    entries = _timeline([0.1, 0.3, 0.8], [KINDS[0], NONE, NONE])
    entries[1] = replace(entries[1], present_after=False)
    entries[2] = replace(entries[2], present_before=False, present_after=False)
    with pytest.raises(ValueError, match="Unassociated presence reversal"):
        confirm_investigation_candidates(entries)


@pytest.mark.parametrize("value,error", [(True, TypeError), ("0.7", TypeError),
    (None, TypeError), (1j, TypeError), (float("nan"), ValueError), (float("inf"), ValueError),
    (-float("inf"), ValueError), (0, ValueError), (-0.1, ValueError)])
def test_invalid_window_even_with_empty_input(value, error):
    with pytest.raises(error):
        confirm_investigation_candidates([], confirmation_window_seconds=value)


@pytest.mark.parametrize("field,value,error", [
    ("pair_index", -1, ValueError), ("pair_index", True, TypeError), ("pair_index", 0.5, TypeError),
    ("time_before_seconds", float("nan"), ValueError),
    ("time_after_seconds", float("inf"), ValueError),
    ("time_after_seconds", -1, ValueError), ("time_after_seconds", True, TypeError),
    ("time_before_seconds", "0", TypeError), ("present_after", 1, TypeError),
    ("candidate", None, TypeError), ("present_after", False, ValueError),
])
def test_invalid_entry_fields(field, value, error):
    entry = _timeline([0.1], [KINDS[0]])[0]
    with pytest.raises(error):
        confirm_investigation_candidates([replace(entry, **{field: value})])


@pytest.mark.parametrize("changes", [
    {"pair_index": 0}, {"time_before_seconds": 0.09}, {"time_before_seconds": 0.11},
    {"time_before_seconds": 0, "time_after_seconds": 0.05}, {"present_before": False},
])
def test_nonchronological_or_nonconsecutive_timeline_rejected(changes):
    entries = _timeline([0.1, 0.8], [KINDS[0], NONE])
    entries[1] = replace(entries[1], **changes)
    with pytest.raises(ValueError):
        confirm_investigation_candidates(entries)


def test_invalid_sequence_or_entry_type():
    for bad in (None, iter([])):
        with pytest.raises(TypeError, match="Sequence"):
            confirm_investigation_candidates(bad)
    with pytest.raises(TypeError, match="CandidateTimelineEntry"):
        confirm_investigation_candidates([None])
