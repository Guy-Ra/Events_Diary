"""Phase 4.3a offline stable-ticket-view calibration; no production detector.

Reuse the shared consecutive-frame Phase 3 evaluator, including raw/effective
presence, motion-aware candidates, confirmation and lifecycle, then existing
InvestigationWindow and adaptive sampling APIs with unchanged defaults. Retain
original pair signals, phase-correlation diagnostics and production temporal scroll
states. No scroll/change measurements, classifiers, thresholds or dense sampling
behavior are replaced. Dense during-scroll evidence remains available.

Report exploratory burst groupings at 0/0.1/0.2-second gaps. Direction changes
inside a grouping remain one interaction envelope with existing up/down labels;
no new enum or direction-based split is introduced. Stable definitions A and C
are identical scroll-only decisions (C exposes body diagnostics without a veto).
B additionally requires the existing body LOW label and finite CPR below its
existing MEDIUM boundary. This body condition is an offline hypothesis, not a
new production threshold. All definitions require effective ticket presence and
stay within a single lifecycle core; replacements/absence cannot join views.

For each maximal eligible consecutive-pair run, create at most one diagnostic
candidate per dwell duration: 0.2/0.3/0.5/0.7/1.0 seconds. Stable start is the first
pair's before-time; confirmation is the first observed after-time meeting dwell.
No timestamps or frames are selected for retention. Initial quiet regions are
reported separately from post-scroll pauses. The isolated scroll recording uses
the same explicitly evaluation-only full-video window as Phase 4.2; other sources
use actual production lifecycle windows. Overlapping windows remain independent.

Coverage checks inspect actual planner timestamp requests, not hypothetical new
dense sampling. Confirmation need not itself be requested: report requests inside
the observed stable run, before/after confirmation and after dense coverage ends.
Requests may later resolve to duplicate frames. Source-index/FPS timing assumes
constant FPS; last-frame quiet runs are right-censored, not proven final stops.
Body activity conflicts and short-gap envelopes are diagnostic proxies, not
independent visual ground truth or semantic reading/identity recognition.

Write only a report and full pair-timeline CSV in stable_ticket_view_calibration.
Do not change production code/tests, remove moving evidence, extract ticket fields,
deduplicate content, select best frames, or implement Phase 4.3b/4.4/Phase 5.
Matching local outputs are replaced on rerun; unrelated work is preserved.
"""

import csv
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import median

# Allow direct execution and explicit imports across script directories.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from events_diary.adaptive_sampling import (
    SamplingSignalEntry,
    build_adaptive_sampling_plan,
)
from events_diary.config import UIConfig
from events_diary.investigation_lifecycle import build_investigation_lifecycle
from events_diary.investigation_signals import (
    TICKET_BODY_MEDIUM_CPR,
    SignalStrength,
    is_investigation_present,
)
from events_diary.investigation_temporal import confirm_investigation_candidates
from events_diary.investigation_window import (
    InvestigationWindow,
    build_investigation_windows,
)
from events_diary.ticket_scroll import TicketScrollState, analyze_ticket_scroll_sequence
from events_diary.video import read_video_metadata
from scripts.evaluation.evaluate_investigation_candidates import _measure_timeline

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/debug/map_motion/stable_ticket_view_calibration"
SOURCES = ("ticket_scroll", "combined_interaction", "investigation_switch",
           "investigation_started_active", "investigation_close_reopen")
DWELLS = (0.2, 0.3, 0.5, 0.7, 1.0)
GAPS = (0.0, 0.1, 0.2)
EPSILON = 1e-9  # Display/observation-boundary arithmetic only, not detector tuning.


def _runs(indices):
    """Return inclusive index bounds of contiguous selected pair positions."""
    runs = []
    for index in indices:
        if runs and index == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], index)
        else:
            runs.append((index, index))
    return runs


def _body_quiet(signal):
    """Evaluate the offline B hypothesis using the unchanged existing body label."""
    body = signal.ticket_body_change
    return (body.strength is SignalStrength.LOW and body.cpr is not None
            and math.isfinite(body.cpr) and body.cpr < TICKET_BODY_MEDIUM_CPR)


def _distribution(values):
    """Summarize finite original measurements without replacing missing evidence."""
    finite = sorted(v for v in values if v is not None and math.isfinite(v))
    if not finite:
        return "unavailable"
    return (f"n={len(finite)}, min={finite[0]:.6g}, median={median(finite):.6g}, "
            f"max={finite[-1]:.6g}, unavailable={len(values) - len(finite)}")


def _bursts(indices, signals, entries, gap):
    """Describe positive scroll envelopes joined only by a specified time gap.

    This exploratory grouping does not change production scroll states. Direction
    reversals inside an envelope are retained together, regardless of sign.
    """
    bursts = []
    for index in indices:
        if not signals[index].ticket_scroll_detected:
            continue
        if bursts and entries[index].time_before_seconds - entries[bursts[-1][-1]].time_after_seconds <= gap + EPSILON:
            bursts[-1].append(index)
        else:
            bursts.append([index])
    return bursts


def _direction(signal):
    """Report existing state labels for positive A-to-B content motion only."""
    if not signal.ticket_scroll_detected:
        return None
    return (TicketScrollState.SCROLLING_UP.value if signal.ticket_scroll_evidence.dy < 0
            else TicketScrollState.SCROLLING_DOWN.value)


def _coverage(start, end, confirmed, plan):
    """Inspect real requested timestamps inside a stable run and near confirmation."""
    inside = [t for t in plan.requested_timestamps_seconds if start - EPSILON <= t <= end + EPSILON]
    dense = any(d.start_time_seconds - EPSILON <= confirmed <= d.end_time_seconds + EPSILON
                for d in plan.dense_intervals)
    before = [t for t in inside if t <= confirmed + EPSILON]
    after = [t for t in inside if t >= confirmed - EPSILON]
    return (f"confirmation_in_dense={dense}; stable_run_requests={len(inside)}; "
            f"requests_by_confirmation={len(before)}; nearest_before={max(before) if before else None}; "
            f"nearest_after={min(after) if after else None}; represented={bool(inside)}; "
            f"request_at_confirmation={any(abs(t - confirmed) <= EPSILON for t in inside)}")


def _context(start, end, bursts, entries):
    """Classify a quiet period by observed earlier/later scroll, without human intent."""
    previous = [b for b in bursts if entries[b[-1]].time_after_seconds <= start + EPSILON]
    following = [b for b in bursts if entries[b[0]].time_before_seconds >= end - EPSILON]
    if not previous:
        return "ordinary/pre-scroll quiet", None, following[0] if following else None
    return ("intermediate pause followed by scroll" if following else "post-scroll final observed pause",
            previous[-1], following[0] if following else None)


def _analyze_window(window, signals, entries, metadata):
    """Compare offline definitions inside one core against its unchanged sampling plan."""
    selected = [i for i, e in enumerate(entries)
                if window.window_start_time_seconds <= e.time_before_seconds
                and e.time_after_seconds <= window.window_end_time_seconds]
    plan = build_adaptive_sampling_plan(window=window, source_fps=metadata.fps,
        signal_entries=[SamplingSignalEntry(i, entries[i].time_before_seconds,
                        entries[i].time_after_seconds, signals[i]) for i in selected])
    core = [i for i in selected if window.investigation_start_time_seconds <= entries[i].time_before_seconds
            and entries[i].time_after_seconds <= window.investigation_end_time_seconds
            and entries[i].present_before and entries[i].present_after]
    lines = [(f"  Core {window.investigation_index}: {window.investigation_start_time_seconds:g}-"
              f"{window.investigation_end_time_seconds:g}s; window={window.window_start_time_seconds:g}-"
              f"{window.window_end_time_seconds:g}s; eligible pairs={len(core)}"),
             "    Phase 4.2 dense=" + str([(d.start_time_seconds, d.end_time_seconds, d.reasons)
                                           for d in plan.dense_intervals])]
    if not core:
        return lines
    grouped = {}
    for gap in GAPS:
        bursts = _bursts(core, signals, entries, gap)
        grouped[gap] = bursts
        lines.append(f"    Exploratory burst gap<={gap:g}s: {len(bursts)} bursts")
        for number, burst in enumerate(bursts):
            start, end = entries[burst[0]].time_before_seconds, entries[burst[-1]].time_after_seconds
            before = (entries[bursts[number - 1][-1]].time_after_seconds if number
                      else entries[core[0]].time_before_seconds)
            after = (entries[bursts[number + 1][0]].time_before_seconds if number + 1 < len(bursts)
                     else entries[core[-1]].time_after_seconds)
            directions = list(dict.fromkeys(_direction(signals[i]) for i in burst))
            lines.append(f"      burst {number}: {start:g}-{end:g}s; duration={end-start:g}s; "
                         f"directions={directions}; positives={len(burst)}; "
                         f"quiet before/after={start-before:g}/{after-end:g}s; "
                         f"internal nonpositive gaps={burst[-1]-burst[0]+1-len(burst)}")
    bursts = grouped[0.0]
    quiet_indices = [i for i in core if not signals[i].ticket_scroll_detected]
    for definition in ("A/C scroll-only", "B scroll + LOW body"):
        runs = _runs(i for i in quiet_indices if definition.startswith("A") or _body_quiet(signals[i]))
        lines.append(f"    {definition}: {len(runs)} maximal quiet runs; A and C decisions identical.")
        for first, last in runs:
            start, end = entries[first].time_before_seconds, entries[last].time_after_seconds
            context, previous, following = _context(start, end, bursts, entries)
            values = signals[first:last + 1]
            body_conflicts = sum(not _body_quiet(s) for s in values)
            header = dict(Counter(s.header_change.strength.value for s in values))
            image = dict(Counter(s.aircraft_image_change.strength.value for s in values))
            lines.append(f"      quiet {start:g}-{end:g}s; duration={end-start:g}s; {context}; "
                         f"previous_scroll_end={entries[previous[-1]].time_after_seconds if previous else None}; "
                         f"next_scroll_start={entries[following[0]].time_before_seconds if following else None}; "
                         f"body_nonquiet_pairs={body_conflicts}; header={header}; image={image}; "
                         f"censored_at_last_frame={last == len(entries)-1}")
            lines.append("        body CPR: " + _distribution([s.ticket_body_change.cpr for s in values])
                         + "; MAD: " + _distribution([s.ticket_body_change.mad for s in values]))
        for dwell in DWELLS:
            count = Counter()
            details = []
            for first, last in runs:
                start, end = entries[first].time_before_seconds, entries[last].time_after_seconds
                confirmation = next((i for i in range(first, last + 1)
                    if entries[i].time_after_seconds - start >= dwell - EPSILON), None)
                if confirmation is None:
                    continue
                confirmed = entries[confirmation].time_after_seconds
                context, previous, _ = _context(start, end, bursts, entries)
                count[context] += 1
                conflicts = sum(not _body_quiet(signals[i]) for i in range(first, confirmation + 1))
                if conflicts:
                    count["body conflict during dwell"] += 1
                if any(entries[b[0]].time_before_seconds < confirmed < entries[b[-1]].time_after_seconds
                       for b in grouped[0.2]):
                    count["inside exploratory ongoing-scroll envelope"] += 1
                # Show post-scroll and control candidates alike, so absence of
                # scroll cannot silently masquerade as confirmed ticket stability.
                details.append(f"        candidate start={start:g}s; confirmed={confirmed:g}s; "
                    f"run_end={end:g}s; {context}; body_conflicts_during_dwell={conflicts}; "
                    f"delay_from_quiet_start={confirmed-start:g}s; "
                    f"delay_from_last_scroll={confirmed-entries[previous[-1]].time_after_seconds if previous else None}; "
                    + _coverage(start, end, confirmed, plan))
            candidates = sum(count[k] for k in ("ordinary/pre-scroll quiet",
                             "intermediate pause followed by scroll", "post-scroll final observed pause"))
            lines.append(f"      SUMMARY {definition}; X={dwell:g}s; candidates={candidates}; {dict(count)}")
            lines.extend(details)
    return lines


def _evaluate(name):
    """Measure one source through production APIs and return report/CSV diagnostics."""
    metadata = read_video_metadata(ROOT / "data/videos" / f"{name}_test.mp4")
    print(f"Input: {metadata.path}", flush=True)
    presences, signals, entries = _measure_timeline(metadata.path, metadata, UIConfig())
    states = analyze_ticket_scroll_sequence([s.ticket_scroll_evidence for s in signals])
    events, results = confirm_investigation_candidates(entries)
    intervals, anomalies = build_investigation_lifecycle(initial_present=entries[0].present_before,
        video_start_time_seconds=0, video_end_time_seconds=metadata.duration_seconds, confirmed_events=events)
    isolated = name == "ticket_scroll"
    windows = ((InvestigationWindow(0, 0, metadata.duration_seconds, 0, metadata.duration_seconds, 0, 0),)
               if isolated else build_investigation_windows(intervals,
               video_start_time_seconds=0, video_end_time_seconds=metadata.duration_seconds))
    lines = [(f"\n{name}: {metadata.duration_seconds:g}s; {metadata.fps:g} FPS; {metadata.frame_count} frames; "
              f"{metadata.width}x{metadata.height}; pairs={len(signals)}"),
             f"  lifecycle={[(i.start_time_seconds, i.end_time_seconds, i.start_reason.value, i.end_reason.value) for i in intervals]}; anomalies={anomalies}",
             f"  confirmation={[(r.candidate_pair_index, r.status.value) for r in results]}",
             "  Window provenance: " + ("evaluation-only isolated full-video" if isolated else "actual lifecycle"),
             f"  Pair positives={sum(s.ticket_scroll_detected for s in signals)}; temporal states={dict(Counter(s.value for s in states))}"]
    rows = []
    for i, (signal, entry, state) in enumerate(zip(signals, entries, states)):
        scroll, body = signal.ticket_scroll_evidence, signal.ticket_body_change
        rows.append({"video_name": metadata.path.name, "pair_index": i,
            "time_before_seconds": entry.time_before_seconds, "time_after_seconds": entry.time_after_seconds,
            "ticket_scroll_detected": signal.ticket_scroll_detected, "scroll_direction": _direction(signal),
            "ticket_scroll_state": state.value, "scroll_dx": scroll.dx, "scroll_dy": scroll.dy,
            "motion_response": scroll.motion_response, "body_strength": body.strength.value,
            "body_mad": body.mad, "body_cpr": body.cpr, "header_strength": signal.header_change.strength.value,
            "image_strength": signal.aircraft_image_change.strength.value,
            "raw_present_before": is_investigation_present(presences[i]),
            "raw_present_after": is_investigation_present(presences[i + 1]),
            "effective_present_before": entry.present_before, "effective_present_after": entry.present_after,
            "candidate_kind": entry.candidate.kind.value})
    for window in windows:
        lines.extend(_analyze_window(window, signals, entries, metadata))
    return lines, rows


def main():
    """Write five-video calibration diagnostics without changing retained evidence."""
    lines = ["Phase 4.3a Stable Ticket View Calibration (offline analysis only)",
             "A/C: no pair-level scroll; C records body diagnostics but makes identical decisions.",
             f"B: additionally body strength LOW, finite CPR < existing {TICKET_BODY_MEDIUM_CPR:g} boundary.",
             "All definitions require effective presence, contiguous observations and one lifecycle core.",
             "One diagnostic candidate per maximal run and dwell; no continuous duplicate candidates.",
             "Motion response is diagnostic, not calibrated confidence. Directions describe content A-to-B, not user gesture.",
             "Candidate delay is measured from signal-defined quiet start, not independent visible-stop ground truth.",
             "A post-scroll final pause is final only within the observed core, not proof the user finished."]
    rows = []
    for name in SOURCES:
        section, timeline = _evaluate(name)
        lines.extend(section)
        rows.extend(timeline)
    lines += [("\nProposed model discussion (no production model created): stable_start_time_seconds and "
              "stable_confirmed_time_seconds are essential; include investigation_index and optionally stable_end_time_seconds "
              "for offline extent/provenance. Prior scroll start/end are optional context, not mandatory identity of a view. "
              "Direction is optional context; omit it or use Optional[TicketScrollState] for ambiguous reversals. "
              "Keep during-scroll evidence and every qualifying intermediate pause available for later selection."),
              ("Limits: pairwise LOW does not prove no cumulative slow drift; QUIET can mean insufficient evidence; "
              "generic ROI changes do not identify distinct sections. No new footage or independent stop labels are assumed. "
              "Phase 4.2 requests are not unique decoded frames. No stability duration or body rule is promoted to production.")]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = OUTPUT / "stable_ticket_view_calibration_report.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (OUTPUT / "stable_ticket_view_timeline.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("\n".join(lines))
    print(f"Report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
