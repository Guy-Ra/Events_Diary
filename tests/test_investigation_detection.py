"""Candidate rule tests using existing evidence records, without real videos."""

from dataclasses import FrozenInstanceError, fields, replace
from itertools import product
from unittest.mock import Mock

import pytest

from events_diary import investigation_detection as detection
from events_diary.investigation_detection import (
    InvestigationCandidate,
    InvestigationCandidateKind,
    detect_investigation_candidate,
)
from events_diary.investigation_signals import (
    ChangeSignal,
    InvestigationPresenceEvidence,
    InvestigationSignals,
    SignalStrength,
)
from events_diary.map_motion import MapMotionEstimate
from events_diary.route_selection import RouteSelectionChangeEvidence
from events_diary.ticket_scroll import TicketScrollEvidence


def _presence(present):
    """Return valid synthetic evidence representing a present/absent ticket."""
    return InvestigationPresenceEvidence(0.9 if present else 0, 0.1 if present else 0, 0)


def _signals(strengths, route=False, scroll=False):
    """Build existing immutable records with deliberately strong map context."""
    changes = [ChangeSignal(strength, 100, 0.9) for strength in strengths]
    return InvestigationSignals(
        *changes, ChangeSignal(SignalStrength.HIGH, 100, 1),
        RouteSelectionChangeEvidence(True, None, 0.1, 0.1, 0.1, 0.1, 0.2), route,
        MapMotionEstimate(valid=True, dx=100, dy=-100, scale=1.5, rotation_degrees=2),
        TicketScrollEvidence(0, 0, 0, 0, 100, 1, 0, 0, 0, -20, 1), scroll)


def test_exact_candidate_enum_and_frozen_model():
    assert {member.name: member.value for member in InvestigationCandidateKind} == {
        "NONE": "none", "OPEN": "open", "SWITCH": "switch", "CLOSE": "close"}
    candidate = InvestigationCandidate(InvestigationCandidateKind.NONE, "reason")
    assert [field.name for field in fields(candidate)] == ["kind", "reason"]
    with pytest.raises(FrozenInstanceError):
        candidate.reason = "changed"


@pytest.mark.parametrize("before,after", list(product([False, True], repeat=2)))
@pytest.mark.parametrize("route,scroll", list(product([False, True], repeat=2)))
@pytest.mark.parametrize("strengths", [
    (SignalStrength.LOW, SignalStrength.NOT_AVAILABLE, SignalStrength.LOW),
    (SignalStrength.MEDIUM, SignalStrength.LOW, SignalStrength.NOT_AVAILABLE),
    (SignalStrength.MEDIUM, SignalStrength.HIGH, SignalStrength.NOT_AVAILABLE),
    (SignalStrength.MEDIUM, SignalStrength.MEDIUM, SignalStrength.MEDIUM),
])
def test_presence_transition_rules_and_context(before, after, route, scroll, strengths):
    a, b = _presence(before), _presence(after)
    signals = _signals(strengths, route, scroll)
    snapshot = (replace(a), replace(b), replace(signals))
    result = detect_investigation_candidate(a, b, signals)
    count = strengths.count(SignalStrength.MEDIUM) + strengths.count(SignalStrength.HIGH)
    kind = InvestigationCandidateKind
    if not before and not after:
        expected = InvestigationCandidate(kind.NONE, "no ticket present before or after")
    elif before and not after:
        expected = InvestigationCandidate(kind.CLOSE, "ticket presence disappeared")
    elif not before:
        expected = (InvestigationCandidate(kind.OPEN,
                    f"ticket appeared with {count} meaningful ticket-change signals") if count >= 2
                    else InvestigationCandidate(kind.NONE,
                    "ticket appeared without enough supporting ticket-change signals"))
    else:
        expected = (InvestigationCandidate(kind.SWITCH,
                    f"route-selection changed with {count} meaningful ticket-change signals")
                    if route and count >= 1 else InvestigationCandidate(kind.NONE,
                    "ticket remained present without sufficient switch evidence"))
    assert result == expected
    assert (a, b, signals) == snapshot


@pytest.mark.parametrize("position", [0, 1, 2])
@pytest.mark.parametrize("strength", [SignalStrength.MEDIUM, SignalStrength.HIGH])
def test_each_ticket_signal_can_support_switch(position, strength):
    strengths = [SignalStrength.LOW] * 3
    strengths[position] = strength
    result = detect_investigation_candidate(_presence(True), _presence(True),
                                            _signals(strengths, route=True))
    assert result.kind is InvestigationCandidateKind.SWITCH


def test_existing_presence_classifier_is_called_internally(monkeypatch):
    a, b = _presence(True), _presence(True)
    classifier = Mock(side_effect=[False, True])
    monkeypatch.setattr(detection, "is_investigation_present", classifier)
    signals = _signals([SignalStrength.MEDIUM] * 3)
    assert detect_investigation_candidate(a, b, signals).kind is InvestigationCandidateKind.OPEN
    assert classifier.call_args_list[0].args == (a,)
    assert classifier.call_args_list[1].args == (b,)
    assert classifier.call_count == 2


def test_unavailable_presence_uses_existing_false_behavior():
    unavailable = InvestigationPresenceEvidence(None, float("nan"), None)
    result = detect_investigation_candidate(unavailable, unavailable,
                                            _signals([SignalStrength.HIGH] * 3, route=True))
    assert result.kind is InvestigationCandidateKind.NONE


def test_wrong_input_types_propagate():
    with pytest.raises(TypeError, match="InvestigationSignals"):
        detect_investigation_candidate(_presence(True), _presence(True), None)
    with pytest.raises(TypeError, match="InvestigationPresenceEvidence"):
        detect_investigation_candidate(None, _presence(True), _signals([SignalStrength.LOW] * 3))
