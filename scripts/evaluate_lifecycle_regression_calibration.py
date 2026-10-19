"""Phase 3.6a manual lifecycle regression calibration, with no production changes.

Decode consecutive canonical frames using the existing validation decoder. Measure
presence and pair signals with production APIs; run the unmodified candidate,
confirmation and lifecycle stages as the baseline. Retain semantic measurements,
not decoded frames. All rules and presence alternatives below are OFFLINE proposals
only: neither thresholds nor APIs nor Phase 4 sampling are modified.

Compare five SWITCH hypotheses over every measured pair, including newly introduced
candidates, and four absence policies. Re-run production confirmation/lifecycle on
each counterfactual timeline. Effective presence booleans are adapted to synthetic
presence records only for counterfactual candidate classification; raw measurements
remain untouched. Unsupported temporal cases are reported as errors, never silently
converted into confirmed/rejected events. Counterfactuals are evaluated separately,
then jointly, to expose interactions and missed replacement evidence.

Known event annotations come from previous manual validation; they label evaluation
results only, never classifier inputs. Motion means a valid finite fit with translation
strictly >1px OR abs(scale-1) strictly >0.01. No global pan veto exists. The frame-based
and causal time-based absence proposals have explicit latency and evidence-loss risks.
Time/FPS assumes constant frame rate; absence run durations are sampled estimates,
with terminal runs right-censored. These few manual videos are not a benchmark.

Write one report and one SWITCH comparison CSV under lifecycle_regression_calibration.
Matching artifacts are replaced on rerun; unrelated work is preserved. No new
dependencies, production tests, frame evidence selection, OCR, identity extraction,
documentation updates, or Phase 4.3 behavior are introduced.
"""

import csv
import math
from dataclasses import asdict
from pathlib import Path

from validate_phase2 import _sequential_frames

from events_diary.config import UIConfig
from events_diary.investigation_detection import (
    InvestigationCandidate,
    InvestigationCandidateKind,
    detect_investigation_candidate,
)
from events_diary.investigation_lifecycle import build_investigation_lifecycle
from events_diary.investigation_signals import (
    InvestigationPresenceEvidence,
    SignalStrength,
    analyze_investigation_pair,
    estimate_investigation_presence,
    is_investigation_present,
)
from events_diary.investigation_temporal import (
    CandidateTimelineEntry,
    confirm_investigation_candidates,
)
from events_diary.video import read_video_metadata

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/debug/map_motion/lifecycle_regression_calibration"
SOURCES = ("investigation_switch", "investigation_started_active",
           "investigation_close_reopen", "combined_interaction", "map_pan", "map_zoom")
KNOWN = {"investigation_switch": (48,), "investigation_started_active": (23,),
         "investigation_close_reopen": (75,)}
KIND = InvestigationCandidateKind
RULES = (
    "R0: route AND >=1 MEDIUM/HIGH (current)",
    "R1: route AND >=1 without motion; route AND >=2 during motion",
    "R2: route AND >=2 regardless of motion",
    "R3: current without motion; route AND >=2 AND >=1 HIGH during motion",
    "R4: current without motion; >=2 AND >=1 HIGH during motion, route not required",
)
POLICIES = ("A raw presence", "B bridge one absent frame between true samples, <=0.2s",
            "C causal absence requires 3 consecutive frames",
            "D causal absence requires >=0.2s elapsed from first absent sample")


def _measure(metadata):
    """Measure each source frame/pair once using unchanged production APIs."""
    config = UIConfig()
    presences, signals, candidates = [], [], []
    previous = None
    for index, frame in _sequential_frames(metadata.path, metadata, config):
        presence = estimate_investigation_presence(frame, config)
        if previous is not None:
            signal = analyze_investigation_pair(previous, frame, config)
            signals.append(signal)
            candidates.append(detect_investigation_candidate(presences[-1], presence, signal))
        presences.append(presence)
        previous = frame
        if index and index % 25 == 0:
            print(f"  {metadata.path.name}: {index}/{metadata.frame_count - 1} pairs", flush=True)
    if not signals:
        raise ValueError("At least one consecutive pair is required.")
    return presences, signals, candidates


def _motion(signal):
    """Return valid finite map magnitude, scale delta and calibrated motion vote."""
    motion = signal.map_motion
    magnitude = (math.hypot(motion.dx, motion.dy) if motion.dx is not None
                 and motion.dy is not None and math.isfinite(motion.dx)
                 and math.isfinite(motion.dy) else None)
    delta = abs(motion.scale - 1) if motion.scale is not None and math.isfinite(motion.scale) else None
    meaningful = motion.valid and ((magnitude is not None and magnitude > 1)
                                  or (delta is not None and delta > 0.01))
    return magnitude, delta, meaningful


def _ticket_counts(signal):
    """Count existing MEDIUM/HIGH and HIGH ticket labels without new thresholds."""
    strengths = (signal.header_change.strength, signal.aircraft_image_change.strength,
                 signal.ticket_body_change.strength)
    return (sum(s in (SignalStrength.MEDIUM, SignalStrength.HIGH) for s in strengths),
            sum(s is SignalStrength.HIGH for s in strengths))


def _counter_candidates(signals, states, rule):
    """Apply an explicitly offline SWITCH proposal while reusing OPEN/CLOSE rules."""
    evidence = {True: InvestigationPresenceEvidence(1, 1, 1),
                False: InvestigationPresenceEvidence(0, 0, 0)}
    candidates = []
    for index, signal in enumerate(signals):
        candidate = detect_investigation_candidate(evidence[states[index]], evidence[states[index + 1]], signal)
        if states[index] and states[index + 1]:
            count, high = _ticket_counts(signal)
            moving = _motion(signal)[2]
            route = signal.route_selection_change
            votes = (route and count >= 1,
                     route and count >= (2 if moving else 1),
                     route and count >= 2,
                     route and count >= 1 and (not moving or (count >= 2 and high >= 1)),
                     (count >= 2 and high >= 1) if moving else (route and count >= 1))
            candidate = InvestigationCandidate(KIND.SWITCH if votes[rule] else KIND.NONE,
                                               f"OFFLINE {RULES[rule]}")
        candidates.append(candidate)
    return candidates


def _presence_states(raw, fps, policy):
    """Simulate explicit offline bridging or causal absence persistence policies.

    B uses one future sample and has no backdating of long absences. C/D delay a
    close transition until persistence is observed, without changing raw signals.
    Initial presence is raw; initial absent startup is never treated as active.
    """
    states = list(raw)
    if policy == 1:
        for index in range(1, len(raw) - 1):
            if raw[index - 1] and not raw[index] and raw[index + 1] and 1 / fps <= 0.2:
                states[index] = True
    elif policy in (2, 3):
        active, absence_start = raw[0], None
        for index, present in enumerate(raw):
            if present:
                active, absence_start = True, None
            elif active:
                if absence_start is None:
                    absence_start = index
                persistent = (index - absence_start + 1 >= 3 if policy == 2
                              else (index - absence_start) / fps >= 0.2)
                if persistent:
                    active = False
            states[index] = active
    return states


def _interpret(candidates, states, metadata):
    """Run unchanged production temporal/lifecycle APIs on a semantic timeline.

    Report unsupported counterfactual temporal input explicitly; it has no invented
    outcome or usable lifecycle. Baseline errors are raised by the caller.
    """
    entries = [CandidateTimelineEntry(i, i / metadata.fps, (i + 1) / metadata.fps,
               states[i], states[i + 1], candidate) for i, candidate in enumerate(candidates)]
    try:
        events, results = confirm_investigation_candidates(entries)
        intervals, anomalies = build_investigation_lifecycle(initial_present=states[0],
            video_start_time_seconds=0, video_end_time_seconds=metadata.duration_seconds,
            confirmed_events=events)
    except ValueError as error:
        return [], [], [], [], str(error)
    return events, results, intervals, anomalies, None


def _timeline(outcome):
    """Format original event times, confirmation times and lifecycle boundaries."""
    events, _, intervals, anomalies, error = outcome
    if error:
        return f"UNSUPPORTED temporal/lifecycle case: {error}"
    return (f"events={[(e.kind.value, e.candidate_pair_index, e.candidate_time_seconds, e.confirmed_at_seconds) for e in events]}; "
            f"intervals={[(i.investigation_index, i.start_time_seconds, i.end_time_seconds, i.start_reason.value, i.end_reason.value) for i in intervals]}; "
            f"anomalies={anomalies}")


def _row(name, index, signal, states, metadata, candidate, result):
    """Capture full SWITCH forensics and manual annotations in a compact CSV row."""
    route, motion = signal.route_selection_evidence, signal.map_motion
    magnitude, delta, meaningful = _motion(signal)
    row = {"video_name": metadata.path.name, "pair_index": index, "time_before": index / metadata.fps,
           "time_after": (index + 1) / metadata.fps, "presence_before": states[index],
           "presence_after": states[index + 1], "is_known_true_switch": index in KNOWN.get(name, ()),
           "is_false_pan_switch": name == "combined_interaction" and index == 97,
           "ticket_signal_count_medium_or_high": _ticket_counts(signal)[0]}
    for label, change in (("header", signal.header_change), ("image", signal.aircraft_image_change),
                          ("body", signal.ticket_body_change)):
        row.update({f"{label}_strength": change.strength.value, f"{label}_cpr": change.cpr})
    row.update(route_change=signal.route_selection_change, route_valid=route.valid, route_reason=route.reason,
               route_added_fraction=route.added_fraction, route_removed_fraction=route.removed_fraction,
               route_changed_fraction=route.changed_fraction, map_motion_valid=motion.valid,
               map_reason=motion.reason, dx=motion.dx, dy=motion.dy, translation_magnitude=magnitude,
               scale=motion.scale, scale_delta=delta, rotation_degrees=motion.rotation_degrees,
               meaningful_map_motion=meaningful, ticket_scroll_detected=signal.ticket_scroll_detected,
               current_candidate_kind=candidate.kind.value, candidate_reason=candidate.reason,
               current_confirmation_status=result.status.value if result else None,
               confirmation_time=result.confirmed_at_seconds if result else None,
               superseded_by_pair_index=result.superseded_by_pair_index if result else None)
    return row


def _evaluate(name):
    """Evaluate baseline, every SWITCH rule and every presence option for one video."""
    path = ROOT / "data/videos" / f"{name}_test.mp4"
    print(f"Input: {path}", flush=True)
    metadata = read_video_metadata(path)
    presences, signals, candidates = _measure(metadata)
    raw = [is_investigation_present(p) for p in presences]
    baseline = _interpret(candidates, raw, metadata)
    if baseline[-1]:
        raise ValueError(baseline[-1])
    by_index = {r.candidate_pair_index: r for r in baseline[1]}
    lines = [(f"\n{path.name}: {metadata.duration_seconds}s, {metadata.fps} FPS, "
              f"{metadata.frame_count} frames, {metadata.width}x{metadata.height}; initial_present={raw[0]}"),
             "CURRENT: " + _timeline(baseline), "Current non-NONE candidate timeline:"]
    rows = []
    for index, candidate in enumerate(candidates):
        if candidate.kind is not KIND.NONE:
            result = by_index[index]
            superseder = (candidates[result.superseded_by_pair_index].kind.value
                          if result.superseded_by_pair_index is not None else None)
            lines.append(f"  pair {index}, {index / metadata.fps:g}-{(index + 1) / metadata.fps:g}s: "
                         f"{raw[index]}->{raw[index + 1]} {candidate.kind.value}: {candidate.reason}; "
                         f"{result.status.value}; confirmed_at={result.confirmed_at_seconds}; "
                         f"superseded_by_pair={result.superseded_by_pair_index}/{superseder}")
        if candidate.kind is KIND.SWITCH:
            row = _row(name, index, signals[index], raw, metadata, candidate, by_index[index])
            rows.append(row)
            lines.append("    SWITCH measurements: " + str(row))
    lines.append("Absence runs (sample interval estimate; final run right-censored):")
    index = 0
    while index < len(raw):
        if raw[index]:
            index += 1
            continue
        start = index
        while index < len(raw) and not raw[index]:
            index += 1
        end = index / metadata.fps if index < len(raw) else (index - 1) / metadata.fps
        lines.append(f"  {start / metadata.fps:g}-{end:g}s; absent frames={index - start}; "
                     f"estimated duration={end - start / metadata.fps:g}s; censored={index == len(raw)}")
    for rule, description in enumerate(RULES):
        revised = _counter_candidates(signals, raw, rule)
        outcome = _interpret(revised, raw, metadata)
        known = KNOWN.get(name, ())
        retained = [i for i in known if revised[i].kind is KIND.SWITCH]
        confirmed = [e.candidate_pair_index for e in outcome[0] if e.kind is KIND.SWITCH]
        changes = [(i, candidates[i].kind.value, c.kind.value) for i, c in enumerate(revised)
                   if c.kind is not candidates[i].kind]
        lines.append(f"{description}: true candidates preserved={retained}; missed={[i for i in known if i not in retained]}; "
                     f"true confirmed preserved={[i for i in known if i in confirmed]}; "
                     f"changed candidates={changes}; " + _timeline(outcome))
    for policy, description in enumerate(POLICIES):
        effective = _presence_states(raw, metadata.fps, policy)
        changed = [i / metadata.fps for i in range(len(raw)) if raw[i] != effective[i]]
        revised = _counter_candidates(signals, effective, 0)
        lines.append(f"Presence {description}: changed frame times={changed}; "
                     + _timeline(_interpret(revised, effective, metadata)))
        if policy:
            joint = _counter_candidates(signals, effective, 1)
            lines.append("  Joint presence option + R1: " + _timeline(_interpret(joint, effective, metadata)))
    if name == "combined_interaction":
        lines.append("Presence transition neighborhood (signals are incoming A-to-B pair context; map transform is B-to-A):")
        for index in range(round(14.2 * metadata.fps), round(15.1 * metadata.fps) + 1):
            signal = signals[index - 1]
            lines.append(f"  frame={index}, timestamp={index / metadata.fps:g}s, present={raw[index]}, "
                         f"presence={asdict(presences[index])}; "
                         f"H/I/B={signal.header_change.strength.value}/{signal.aircraft_image_change.strength.value}/"
                         f"{signal.ticket_body_change.strength.value}; route={signal.route_selection_change}; "
                         f"map={asdict(signal.map_motion)}")
    return lines, rows


def main():
    """Write reproducible calibration diagnostics without applying proposed fixes."""
    lines = ["Phase 3.6a Lifecycle Regression Calibration (offline proposals only)",
             "Presence: bottom dark >=0.50 AND body edges >=0.05; light diagnostic only.",
             "Ticket CPR: MEDIUM >=0.02 each; HIGH header >=0.40, image/body >=0.60.",
             "Map meaningful: valid AND (hypot(dx,dy)>1px OR abs(scale-1)>0.01).",
             "Existing temporal confirmation horizon=0.7s; no production defaults changed.",
             "Known true SWITCH annotations: switch pair48, started-active pair23, close/reopen pair75.",
             "Combined final replacement at pair146 is a known true replacement, currently OPEN, not a baseline SWITCH.",
             "Annotations label events only. No identity inferred. CSV contains every current SWITCH candidate.",
             "Rules 1/3/4 retain current behavior when motion is absent; R4's motion branch does not require route.",
             "C uses N=3; D uses 0.2s elapsed; both causal and delay the candidate, never backdate it."]
    rows = []
    for name in SOURCES:
        section, measurements = _evaluate(name)
        lines.extend(section)
        rows.extend(measurements)
    lines.append("\nCompact comparison: video | pair | known true / false pan | H/I/B (CPR) | count | route changed | translation | scale delta | scroll")
    for row in rows:
        lines.append(f"{row['video_name']} | {row['pair_index']} | {row['is_known_true_switch']}/{row['is_false_pan_switch']} | "
                     + "/".join(f"{row[label + '_strength']}({row[label + '_cpr']})" for label in ("header", "image", "body"))
                     + f" | {row['ticket_signal_count_medium_or_high']} | {row['route_changed_fraction']} | "
                     f"{row['translation_magnitude']} | {row['scale_delta']} | {row['ticket_scroll_detected']}")
    lines.append("Limits: no true switch under substantial simultaneous map motion is established in these recordings. "
                 "Preserving measured true examples cannot prove general safety; no proposal is implemented. "
                 "A real replacement routed through raw False->True remains OPEN under SWITCH-only proposals. "
                 "Causal absence delay can lose pair-level replacement evidence; offline look-ahead bridging is distinct.")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = OUTPUT / "lifecycle_regression_calibration_report.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (OUTPUT / "switch_candidate_comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("\n".join(lines))
    print(f"Report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
