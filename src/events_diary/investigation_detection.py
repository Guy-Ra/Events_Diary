"""Phase 3.2 pair-level Investigation candidate hypotheses.

OPEN means a ticket appeared with at least two meaningful ticket changes. SWITCH
means a present ticket remains present with route replacement and at least one
meaningful ticket change, or two during meaningful map motion. CLOSE means presence
disappeared. NONE means the pair
does not support these rules. MEDIUM/HIGH changes vote; LOW/NOT_AVAILABLE do not.
Raw presence uses the existing single-frame classifier; callers can explicitly
supply effective presence from bounded offline temporal preprocessing.

Candidates are not confirmed events or aircraft identities. Phase 3.3 temporal
confirmation and Phase 3.4 lifecycle handling come later. In particular, a video
starting with a ticket already open has no opening transition here. A transient
presence disappearance can produce CLOSE without duration or debounce logic.
Map residual/motion are not positive votes. Scroll is context, never a global
veto. No measurements, threshold tuning, scores, timestamps, or state are added.
"""

import math
from dataclasses import dataclass
from enum import Enum

from events_diary.investigation_signals import (
    ChangeSignal,
    InvestigationPresenceEvidence,
    InvestigationSignals,
    SignalStrength,
    is_investigation_present,
)

# Provisional Phase 4.2a calibrated motion bounds, reused here without importing
# the higher-level Phase 4 planner into Phase 3. These do not tune geometric fits.
MAP_TRANSLATION_THRESHOLD_PIXELS = 1.0
MAP_SCALE_CHANGE_THRESHOLD = 0.01


def _has_meaningful_map_motion(signals: InvestigationSignals) -> bool:
    """Require a valid finite translation >1px or scale delta >0.01.

    Compare scale against 1 +/- the bound so literal 1.01/0.99 stay exactly on
    the non-motion boundary despite subtraction rounding. Missing/non-finite
    components cannot vote; invalid fits always use the normal SWITCH rule.
    """
    motion = signals.map_motion
    if not motion.valid:
        return False
    translation = (motion.dx is not None and motion.dy is not None
                   and math.isfinite(motion.dx) and math.isfinite(motion.dy)
                   and math.hypot(motion.dx, motion.dy) > MAP_TRANSLATION_THRESHOLD_PIXELS)
    scale = (motion.scale is not None and math.isfinite(motion.scale)
             and (motion.scale > 1 + MAP_SCALE_CHANGE_THRESHOLD
                  or motion.scale < 1 - MAP_SCALE_CHANGE_THRESHOLD))
    return bool(translation or scale)


class InvestigationCandidateKind(Enum):
    """Pair-level hypothesis labels, without confirmation or lifecycle state."""

    NONE = "none"
    OPEN = "open"
    SWITCH = "switch"
    CLOSE = "close"


@dataclass(frozen=True)
class InvestigationCandidate:
    """Immutable candidate kind and deterministic debugging reason only.

    No confidence, score, identity, timestamp, frame index, or lifecycle state is
    stored. Construction does not measure or confirm anything.
    """

    kind: InvestigationCandidateKind
    reason: str


def _is_meaningful_change(signal: ChangeSignal) -> bool:
    """Return True only for explicit MEDIUM/HIGH generic change labels."""
    return signal.strength in (SignalStrength.MEDIUM, SignalStrength.HIGH)


def detect_investigation_candidate(
    presence_before: InvestigationPresenceEvidence,
    presence_after: InvestigationPresenceEvidence,
    signals: InvestigationSignals,
    *,
    effective_presence_before: bool | None = None,
    effective_presence_after: bool | None = None,
) -> InvestigationCandidate:
    """Classify a pair hypothesis using existing presence and signal records.

    Presence evidence refers to frames A/B; signals must refer to that same pair.
    The existing presence API supplies both decisions, including its unavailable
    evidence behavior. Wrong public record types raise TypeError; errors are not
    suppressed. Inputs are not mutated and no low-level algorithms are repeated.

    False->False gives NONE; False->True needs >=2 meaningful ticket signals for
    OPEN; True->False gives CLOSE unconditionally; True->True needs route change
    and >=1 meaningful ticket signal for SWITCH, increased to >=2 during valid
    meaningful map translation/scale. Motion never vetoes a strong replacement.
    Optional effective-presence booleans override raw decisions after measurement
    validation; the original evidence is retained and never altered. Callers must
    obtain these overrides from temporal preprocessing, not change raw thresholds.
    Scroll does not vote or veto. This does not confirm events or manage
    Investigation lifecycle, startup-active initialization, or temporal state.
    """
    if not isinstance(signals, InvestigationSignals):
        raise TypeError("signals must be InvestigationSignals.")
    before = is_investigation_present(presence_before)
    after = is_investigation_present(presence_after)
    for value in (effective_presence_before, effective_presence_after):
        if value is not None and not isinstance(value, bool):
            raise TypeError("effective presence must be bool or None.")
    if effective_presence_before is not None:
        before = effective_presence_before
    if effective_presence_after is not None:
        after = effective_presence_after
    kind = InvestigationCandidateKind
    if not before and not after:
        return InvestigationCandidate(kind.NONE, "no ticket present before or after")
    if before and not after:
        return InvestigationCandidate(kind.CLOSE, "ticket presence disappeared")
    count = sum(_is_meaningful_change(signal) for signal in (
        signals.header_change, signals.aircraft_image_change, signals.ticket_body_change))
    if not before and after:
        if count >= 2:
            return InvestigationCandidate(
                kind.OPEN, f"ticket appeared with {count} meaningful ticket-change signals")
        return InvestigationCandidate(
            kind.NONE, "ticket appeared without enough supporting ticket-change signals")
    required = 2 if _has_meaningful_map_motion(signals) else 1
    if signals.route_selection_change and count >= required:
        return InvestigationCandidate(
            kind.SWITCH, f"route-selection changed with {count} meaningful ticket-change signals")
    return InvestigationCandidate(
        kind.NONE, "ticket remained present without sufficient switch evidence")
