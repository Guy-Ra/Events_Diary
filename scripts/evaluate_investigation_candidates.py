"""Phase 3.2 real-video evaluation of pair-level Investigation candidates.

Evaluate every consecutive pair of the switch, startup-active, and close/reopen videos
through existing single-frame presence, pair aggregation, and candidate APIs. No
candidate rules, thresholds, measurements, or production modules are modified.
Candidates are hypotheses, not confirmed events or lifecycle transitions. Startup
with a ticket already active intentionally has no synthetic OPEN pair. This tool
does not perform temporal confirmation, merge candidates, reject clicks, identify
aircraft, or initialize lifecycle state.

Reuse the existing sequential decoder and canonical layout checks. CSV timelines
retain presence fractions, ticket strengths/CPR, route fractions, and contextual
scroll/map labels for auditing unexpected candidates. Report all non-NONE pairs,
counts, compressed timeline spans, and manually known switch neighborhoods. These
spans are display summaries only, never merged events. Landmark times organize
reporting only; they do not affect candidate classification. Index/FPS times assume
constant source FPS. Presence runs and candidate spacing are retrospective display
statistics only. "Settled" below means the first subsequent all-LOW ticket pair
with unchanged presence, not a duration requirement or confirmation algorithm.
Results concern three manual recordings, not a benchmark.

Paths resolve from the repository. Outputs under investigation_candidates replace
matching CSV/report files on rerun, preserving unrelated artifacts. Decode/layout,
API, and I/O failures propagate. No new dependencies or debug-image dumps are used.
"""

import csv
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from validate_phase2 import _sequential_frames

from events_diary.config import UIConfig
from events_diary.investigation_detection import detect_investigation_candidate
from events_diary.investigation_signals import (
    analyze_investigation_pair,
    estimate_investigation_presence,
    is_investigation_present,
)
from events_diary.video import read_video_metadata

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/debug/map_motion/investigation_candidates"
SOURCES = ("investigation_switch", "investigation_started_active", "investigation_close_reopen")


def _evaluate(path, metadata, config):
    """Return rows from production APIs on every pair, without caching decisions.

    Presence is measured separately for A and B as specified. Only the previous
    full frame is retained; nominal timestamps are based on metadata FPS.
    """
    rows, previous = [], None
    for index, frame in _sequential_frames(path, metadata, config):
        if previous is not None:
            before = estimate_investigation_presence(previous, config)
            after = estimate_investigation_presence(frame, config)
            signals = analyze_investigation_pair(previous, frame, config)
            candidate = detect_investigation_candidate(before, after, signals)
            row = {"frame_a": index - 1, "frame_b": index,
                   "time_a": (index - 1) / metadata.fps, "time_b": index / metadata.fps,
                   "present_before": is_investigation_present(before),
                   "present_after": is_investigation_present(after),
                   "candidate_kind": candidate.kind.value, "candidate_reason": candidate.reason}
            for label, signal in (("header", signals.header_change),
                                  ("aircraft_image", signals.aircraft_image_change),
                                  ("ticket_body", signals.ticket_body_change)):
                row[f"{label}_strength"] = signal.strength.value
                row[f"{label}_cpr"] = signal.cpr
            route = signals.route_selection_evidence
            row.update(route_selection_change=signals.route_selection_change,
                       route_evidence_valid=route.valid,
                       route_added_fraction=route.added_fraction,
                       route_removed_fraction=route.removed_fraction,
                       route_changed_fraction=route.changed_fraction,
                       ticket_scroll_detected=signals.ticket_scroll_detected,
                       map_residual_strength=signals.map_residual_change.strength.value,
                       map_motion_valid=signals.map_motion.valid)
            for label, evidence in (("before", before), ("after", after)):
                row.update({f"{label}_{key}": value for key, value in asdict(evidence).items()})
            rows.append(row)
        previous = frame
    if not rows:
        raise ValueError(f"No consecutive pairs decoded: {path}")
    return rows


def _describe(row):
    """Format one hypothesis with raw supporting evidence for manual inspection."""
    return (f"frames {row['frame_a']}-{row['frame_b']}, "
            f"{row['time_a']:.1f}-{row['time_b']:.1f}s: " + "; ".join(
                f"{key}={value}" for key, value in row.items()
                if key not in ("frame_a", "frame_b", "time_a", "time_b")))


def _presence_timing(rows, metadata):
    """Describe sampled presence runs and timing without confirming any candidate.

    Reconstruct frame-zero presence plus each pair's after value. Run boundaries
    use the first changed sample; the final run ends at the last sampled timestamp
    and is right-censored. Candidate spacing is measured between frame_b endpoints.
    A reversal means a later opposite presence sample, not opposite identity.
    Settling is only the first subsequent all-LOW ticket pair with equal presence;
    no minimum duration, state inference, rejection, or merging occurs.
    """
    states = [rows[0]["present_before"]] + [r["present_after"] for r in rows]
    lines = ["Presence intervals (sampled boundaries; final interval is right-censored):"]
    start = 0
    for index in range(1, len(states) + 1):
        if index == len(states) or states[index] != states[start]:
            final = index == len(states)
            end = index - 1 if final else index
            lines.append(f"  {start / metadata.fps:.1f}-{end / metadata.fps:.1f}s: "
                         f"present={states[start]}; frames {start}-{index - 1}"
                         + (" (through last sampled frame)" if final else ""))
            start = index
    candidates = [r for r in rows if r["candidate_kind"] != "none"]
    lines.append("Candidate spacing (from candidate frame_b; observations only):")
    for index, row in enumerate(candidates):
        endpoint = row["frame_b"]
        lines.append(f"  {row['candidate_kind']} frames {row['frame_a']}-{endpoint}:")
        run_start = endpoint
        while run_start > 0 and states[run_start - 1] == states[endpoint]:
            run_start -= 1
        lines.append(f"    current constant-presence run begins at frame {run_start} / "
                     f"{run_start / metadata.fps:.1f}s; presence is already constant at the "
                     "candidate endpoint (0 additional frames to start, not temporal confirmation)")
        if index + 1 < len(candidates):
            delta = candidates[index + 1]["frame_b"] - endpoint
            lines.append(f"    next candidate: {delta} frames / {delta / metadata.fps:.1f}s")
        else:
            lines.append("    no next candidate observed before end of recording")
        reversal = next((i for i in range(endpoint + 1, len(states))
                         if states[i] != states[endpoint]), None)
        if reversal is not None:
            delta = reversal - endpoint
            lines.append(f"    presence reverses at frame {reversal} / {reversal / metadata.fps:.1f}s: "
                         f"{delta} frames / {delta / metadata.fps:.1f}s later")
        else:
            lines.append("    no later presence reversal observed (not proof of permanence)")
        settled = next((r for r in rows if r["frame_a"] >= endpoint
                        and r["present_before"] == r["present_after"] == states[endpoint]
                        and all(r[key] == "low" for key in (
                            "header_strength", "aircraft_image_strength", "ticket_body_strength"))), None)
        if settled is not None:
            delta = settled["frame_a"] - endpoint
            lines.append(f"    first subsequent unchanged-presence/all-LOW ticket pair: "
                         f"{settled['time_a']:.1f}-{settled['time_b']:.1f}s; "
                         f"{delta} frames / {delta / metadata.fps:.1f}s to its first frame")
        else:
            lines.append("    no subsequent unchanged-presence/all-LOW ticket pair observed")
    return lines


def _summary(name, metadata, rows):
    """Report timeline spans and known neighborhoods without imposing detections."""
    lines = [f"\n{name}: {metadata.path}",
             (f"{metadata.width}x{metadata.height}; fps={metadata.fps}; "
              f"frames={metadata.frame_count}; duration={metadata.duration_seconds}s; pairs={len(rows)}"),
             f"Candidate counts: {dict(Counter(row['candidate_kind'] for row in rows))}",
             (f"First pair starts with presence={rows[0]['present_before']}; "
              "there is no synthetic startup pair."), "Timeline (display spans, not merged events):"]
    start = 0
    for index in range(1, len(rows) + 1):
        if index == len(rows) or rows[index]["candidate_kind"] != rows[start]["candidate_kind"]:
            lines.append(f"  {rows[start]['time_a']:.1f}-{rows[index - 1]['time_b']:.1f}s: "
                         f"{rows[start]['candidate_kind']} ({index - start} pairs)")
            start = index
    lines.append("All non-NONE pairs:")
    lines.extend(_describe(row) for row in rows if row["candidate_kind"] != "none")
    lines.extend(_presence_timing(rows, metadata))
    # Manual landmarks only: expose missing-switch evidence without changing rules.
    windows = {
        "investigation_switch": ((1.4, 1.7), (4.7, 5.0), (6.9, 7.0)),
        "investigation_started_active": ((2.1, 2.6),),
        "investigation_close_reopen": ((3.7, 4.1), (6.7, 7.9)),
    }[name]
    lines.append("Known opening/switch/traffic neighborhoods (annotations only):")
    for start, end in windows:
        lines.extend(_describe(row) for row in rows
                     if row["time_a"] >= start and row["time_b"] <= end)
    if name == "investigation_close_reopen":
        lines += [("Visual annotations from consecutive ticket crops: frames 0-38 show the original "
                   "ticket, with partial closing at frame 38 (3.8s), already classified absent; "
                   "frame 39 (3.9s) exposes the map fully. A short-lived different ticket "
                   "slides in around frames 70-72, is dimmed at 74-75, and is replaced by the "
                   "final ticket at frame 76 (7.6s), which remains through frame 119. "
                   "These are visual annotations, not automatic identity or lifecycle inference.")]
        absent = [r for r in rows if not r["present_before"] and not r["present_after"]]
        false_candidates = [r for r in absent if r["candidate_kind"] != "none"]
        lines.append(f"Both-absent pairs: {len(absent)}; non-NONE among them: {len(false_candidates)}")
        for label, start, end in (("Real close of A", 3.7, 4.1),
                                  ("Short accidental ticket opening", 6.7, 7.3),
                                  ("Replacement by stable final ticket B", 7.5, 7.8)):
            matches = [r for r in rows if start <= r["time_a"] and r["time_b"] <= end
                       and r["candidate_kind"] != "none"]
            lines.append(f"{label}: " + (", ".join(
                f"{r['candidate_kind']} frames {r['frame_a']}-{r['frame_b']} "
                f"({r['time_a']:.1f}-{r['time_b']:.1f}s)" for r in matches)
                or "no candidate"))
        lines.append("Interpret the final ticket's actual candidate kind as emitted: a continuous "
                     "True->True panel transition can be SWITCH rather than a second OPEN. "
                     "Candidate spacing and all-LOW pairs must not be mistaken for confirmation. "
                     "Presence tracks the panel, not which ticket remains open; future temporal "
                     "design needs to account for subsequent replacement while presence stays True.")
    return lines


def main():
    """Write three candidate timelines and observational timing without tuning."""
    config = UIConfig()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lines = ["Phase 3.2 pair-level candidate evaluation",
             "Candidates are hypotheses; no temporal confirmation or lifecycle is implemented.",
             "Scroll is context, not a veto. Map residual/motion are not positive votes."]
    for name in SOURCES:
        path = ROOT / "data/videos" / f"{name}_test.mp4"
        print(f"Input: {path}", flush=True)
        metadata = read_video_metadata(path)
        rows = _evaluate(path, metadata, config)
        csv_path = OUTPUT / f"{name}_candidates.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        lines.extend(_summary(name, metadata, rows))
        print(f"CSV: {csv_path}", flush=True)
    lines += [("\nLimitations: three manually recorded canonical-layout videos, provisional upstream "
              "presence/change/route thresholds, no held-out validation. Missing/non-finite presence "
              "is classified absent by the existing API, so transient disappearance may create CLOSE. "
              "Route replacement may miss switches without sufficient added and removed route pixels. "
              "No rules or thresholds were loosened. Phase 3.3 confirms candidates; Phase 3.4 manages "
              "lifecycle, including startup-active initialization.")]
    report = "\n".join(lines) + "\n"
    report_path = OUTPUT / "investigation_candidate_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
