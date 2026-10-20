"""Phase 4 closure validation of the actual seven-video semantic pipeline.

Reuse Phase 4.3's shared preparation: sequential validated video -> raw/effective
presence -> candidates/confirmation/lifecycle -> windows -> adaptive plans -> stable
views. Compose candidates through the production Phase 4.4 API. No CV, sampling,
stability, composition or lifecycle algorithm is reimplemented. Assertions compare
existing outputs at stage boundaries; scenario landmarks are validation expectations
only, never inputs used to generate results. No threshold/rate is retuned.

All seven recordings use actual lifecycle intervals, including isolated motion
controls where real intervals exist. There is no synthetic fallback. Every request
must survive exactly; all stable views need coverage; dense moving scroll evidence
and semantic anchors must remain. Raw/effective presence and candidate-time/video-end
boundaries remain distinct. Censored stable annotations associate through core end,
without claiming that the final requested endpoint resolves to a new source frame.

Report source-FPS nominal counts plus a fair production full-rate request reference
with the same anchors. Totals across overlapping windows count their context more
than once and are not unique physical-frame totals. No frame resolution, ranking,
OCR, VLM or Phase 5 selection occurs. Write one report/summary CSV only. Failure
records FAIL and stops further videos; no production result is silently repaired.
"""

import csv
import math
import sys
from collections import Counter
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path

# Allow direct execution and explicit imports across script directories.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from events_diary.adaptive_sampling import (
    DENSE_REASON_ORDER,
    SamplingSignalEntry,
    build_adaptive_sampling_plan,
)
from events_diary.evidence_candidate_timeline import (
    SOURCE_ORDER,
    TIMESTAMP_ABSOLUTE_TOLERANCE_SECONDS,
    build_evidence_candidate_timeline,
)
from events_diary.investigation_lifecycle import build_investigation_lifecycle
from events_diary.investigation_signals import SignalStrength, is_investigation_present
from events_diary.investigation_temporal import bridge_transient_absence
from scripts.evaluation.evaluate_adaptive_sampling_plan import _pair_reasons
from scripts.evaluation.evaluate_ticket_stable_views import _build_contexts

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/debug/map_motion/phase4_closure"
SOURCES = ("investigation_switch", "investigation_started_active", "investigation_close_reopen",
           "ticket_scroll", "map_pan", "map_zoom", "combined_interaction")
GAPS = (
    "True aircraft selection during substantial simultaneous map pan/zoom remains unvalidated.",
    "Genuine very short CLOSE/reopen may be bridged by the provisional 0.15s absence bound.",
    "Body LOW does not prove semantic completeness, finished loading or reading intent.",
    "Slow cumulative ticket drift can escape pairwise LOW body-change labels.",
    "Direction reversal without a stable pause has synthetic tests but no strong real-video validation.",
    "Footage is predominantly canonical 1920x1080 at 10 FPS in one FR24 environment.",
    "Higher FPS and other layouts/themes need real-video validation; automatic layout recovery is absent.",
    "Prolonged real activity/route cross-talk may produce high dense candidate volume.",
    "Requests are timestamps, not unique decoded frames; endpoint resolution and final selection remain future work.",
)


def _close(a, b):
    """Use the production composer's documented representation tolerance."""
    return math.isclose(a, b, rel_tol=0, abs_tol=TIMESTAMP_ABSOLUTE_TOLERANCE_SECONDS)


def _inside(time, start, end):
    """Check expected inclusive membership for validation, never generate candidates."""
    return (time >= start or _close(time, start)) and (time <= end or _close(time, end))


def _validate_video(name):
    """Assert stage contracts on real outputs and return status/report/summary rows."""
    metadata, anomalies, contexts, phase3 = _build_contexts(name, include_phase3=True)
    timeline, signals, intervals = phase3["timeline"], phase3["signals"], phase3["intervals"]
    checks = []

    def check(label, condition):
        """Record an explicit check result without correcting an observed output."""
        checks.append((label, bool(condition)))

    lines = [(f"\n{name}: {metadata.width}x{metadata.height}, {metadata.fps:g} FPS, "
              f"{metadata.frame_count} frames, {metadata.duration_seconds:g}s; actual lifecycle; no synthetic context")]
    raw = tuple(is_investigation_present(p) for p in phase3["raw_presence"])
    effective = (timeline[0].present_before,) + tuple(e.present_after for e in timeline)
    bridged = bridge_transient_absence([(i/metadata.fps, p) for i, p in enumerate(raw)])
    check("raw presence retained separately; effective timeline equals production bridge", effective == bridged)
    check("every interval has exactly one Phase4 context", len(contexts) == len(intervals))
    check("no lifecycle anomalies", not anomalies)
    check("source-density consecutive coverage", len(timeline) == metadata.frame_count - 1)
    snapshot = tuple(asdict(i) for i in intervals)
    reference, reference_anomalies = build_investigation_lifecycle(initial_present=effective[0],
        video_start_time_seconds=0, video_end_time_seconds=metadata.duration_seconds,
        confirmed_events=phase3["events"])
    check("Phase4 preserves Phase3 intervals exactly", tuple(reference) == intervals and reference_anomalies == anomalies)
    lines.append("  Lifecycle=" + str([(i.investigation_index, i.start_time_seconds, i.end_time_seconds,
                                       i.start_reason.value, i.end_reason.value) for i in intervals]))
    lines.append("  Confirmed=" + str([(e.kind.value, e.candidate_time_seconds, e.confirmed_at_seconds)
                                       for e in phase3["events"]]))
    changed = [i for i in range(len(raw)) if raw[i] != effective[i]]
    lines.append(f"  Raw/effective presence differing frame indexes={changed}")
    rows, candidates_by_core = [], {}
    for interval, (window, plan, views, observed_end) in zip(intervals, contexts):
        prefix = f"I{interval.investigation_index} "
        model_snapshot = (asdict(window), asdict(plan), tuple(asdict(v) for v in views))
        candidates = build_evidence_candidate_timeline(window=window, sampling_plan=plan, stable_views=views)
        candidates_by_core[interval.investigation_index] = candidates
        times = tuple(c.timestamp_seconds for c in candidates)
        check(prefix + "window core exactly preserves interval", window.investigation_index == interval.investigation_index
              and window.investigation_start_time_seconds == interval.start_time_seconds
              and window.investigation_end_time_seconds == interval.end_time_seconds)
        check(prefix + "window clipped to video", 0 <= window.window_start_time_seconds <= interval.start_time_seconds
              <= interval.end_time_seconds <= window.window_end_time_seconds <= metadata.duration_seconds)
        check(prefix + "plan.window matches", plan.window == window)
        check(prefix + "requests exactly one-to-one with chronological unique candidates",
              times == plan.requested_timestamps_seconds and len(times) == len(set(times))
              and all(a < b for a, b in pairwise(times)))
        check(prefix + "requests inside window", all(_inside(t, window.window_start_time_seconds,
                                                               window.window_end_time_seconds) for t in times))
        dense = plan.dense_intervals
        check(prefix + "dense intervals ordered, disjoint, bounded", all(
              _inside(d.start_time_seconds, window.window_start_time_seconds, window.window_end_time_seconds)
              and _inside(d.end_time_seconds, window.window_start_time_seconds, window.window_end_time_seconds)
              for d in dense) and all(a.end_time_seconds < b.start_time_seconds for a, b in pairwise(dense)))
        check(prefix + "exactly one sampling label; ordered sources", all(
              sum(label in c.sources for label in ("sparse_sampling", "dense_sampling")) == 1
              and c.sources == tuple(label for label in SOURCE_ORDER if label in c.sources) for c in candidates))
        check(prefix + "dense reasons explain existing intervals; sparse reasons empty", all(
              (any(_inside(c.timestamp_seconds, d.start_time_seconds, d.end_time_seconds)
                   and c.dense_reasons == d.reasons for d in dense)
               and c.dense_reasons == tuple(r for r in DENSE_REASON_ORDER if r in c.dense_reasons)
               and bool(c.dense_reasons)) if "dense_sampling" in c.sources else
              (c.dense_reasons == () and not any(_inside(c.timestamp_seconds, d.start_time_seconds,
                                                        d.end_time_seconds) for d in dense))
              for c in candidates))
        for label, time in (("window_start", window.window_start_time_seconds),
                            ("investigation_start", interval.start_time_seconds),
                            ("investigation_end", interval.end_time_seconds),
                            ("window_end", window.window_end_time_seconds)):
            matching = [c for c in candidates if _close(c.timestamp_seconds, time)]
            check(prefix + label + " anchor preserved with sampling classification",
                  bool(matching) and all(label in c.sources for c in matching))
            check(prefix + label + " labels match only its timestamp; coincidences share candidates",
                  all((label in c.sources) == _close(c.timestamp_seconds, time) for c in candidates))
        ends = [v.stable_end_time_seconds if v.stable_end_time_seconds is not None
                else interval.end_time_seconds for v in views]
        check(prefix + "stable views ordered, disjoint and inside core", all(
              interval.start_time_seconds <= v.stable_start_time_seconds <= end <= interval.end_time_seconds
              for v, end in zip(views, ends)) and all(ends[i] < views[i+1].stable_start_time_seconds
                                                    for i in range(len(views)-1)))
        for index, (view, end) in enumerate(zip(views, ends)):
            associated = [c for c in candidates if c.stable_view_index == index]
            check(prefix + f"view{index} covered with unchanged post-scroll provenance",
                  bool(associated) and all(c.stable_view_follows_scroll == view.follows_scroll for c in associated))
            check(prefix + f"view{index} calibrated 0.5s dwell",
                  _close(view.stable_confirmed_time_seconds - view.stable_start_time_seconds, 0.5))
            support = [e for e in timeline if view.stable_start_time_seconds <= e.time_before_seconds
                       and e.time_after_seconds <= min(end, observed_end)]
            check(prefix + f"view{index} supported by effective presence, no scroll and body LOW",
                  bool(support) and support[0].time_before_seconds == view.stable_start_time_seconds
                  and all(e.present_before and e.present_after and not signals[e.pair_index].ticket_scroll_detected
                          and signals[e.pair_index].ticket_body_change.strength is SignalStrength.LOW for e in support))
            lines.append(f"  {prefix}view{index}: {view.stable_start_time_seconds:g}-{end:g}s; "
                         f"censored={view.stable_end_time_seconds is None}; candidates={len(associated)}; "
                         f"follows_scroll={view.follows_scroll}; prior_end={view.prior_scroll_end_time_seconds}; "
                         f"direction={view.prior_scroll_direction}; observed_input_end={observed_end}")
        check(prefix + "stable membership unambiguous and censored views capped at core end", all(
              [i for i, (v, end) in enumerate(zip(views, ends))
               if _inside(c.timestamp_seconds, v.stable_start_time_seconds, end)]
              == ([] if c.stable_view_index is None else [c.stable_view_index]) for c in candidates))
        positives = [e for e in timeline if window.window_start_time_seconds <= e.time_before_seconds
                     and e.time_after_seconds <= window.window_end_time_seconds
                     and signals[e.pair_index].ticket_scroll_detected]
        check(prefix + "dense during-scroll requests retained alongside stable annotations", all(any(
              _inside(c.timestamp_seconds, e.time_before_seconds, e.time_after_seconds)
              and "dense_sampling" in c.sources and "ticket_scroll" in c.dense_reasons for c in candidates)
              for e in positives))
        check(prefix + "all upstream models unchanged by composition", model_snapshot ==
              (asdict(window), asdict(plan), tuple(asdict(v) for v in views)))
        full = build_adaptive_sampling_plan(window=window, signal_entries=[], source_fps=metadata.fps,
                                             sparse_sampling_fps=metadata.fps)
        duration = window.window_end_time_seconds - window.window_start_time_seconds
        dense_duration = sum(d.end_time_seconds - d.start_time_seconds for d in dense)
        labels = Counter(label for c in candidates for label in c.sources)
        row = {"video_name": metadata.path.name, "investigation_index": interval.investigation_index,
            "core_start": interval.start_time_seconds, "core_end": interval.end_time_seconds,
            "window_start": window.window_start_time_seconds, "window_end": window.window_end_time_seconds,
            "window_duration": duration, "dense_interval_count": len(dense), "dense_duration": dense_duration,
            "sparse_duration": duration-dense_duration, "requested_timestamp_count": len(times),
            "candidate_count": len(candidates), "sparse_candidate_count": labels["sparse_sampling"],
            "dense_candidate_count": labels["dense_sampling"], "stable_view_count": len(views),
            "stable_candidate_count": labels["stable_ticket_view"],
            "post_scroll_stable_candidate_count": sum(c.stable_view_follows_scroll for c in candidates),
            "anchor_candidate_count": sum(any(a in c.sources for a in
                ("window_start", "investigation_start", "investigation_end", "window_end")) for c in candidates),
            "approximate_full_source_count": max(1, math.ceil(duration*metadata.fps-1e-9)),
            "full_rate_reference_count": len(full.requested_timestamps_seconds),
            "reduction_percent": 100*(1-len(times)/len(full.requested_timestamps_seconds))}
        rows.append(row)
        lines.append("  SUMMARY " + str(row))
        lines.append("  Dense intervals=" + str([(d.start_time_seconds, d.end_time_seconds, d.reasons) for d in dense]))
        direct_reasons = Counter(reason for e in timeline
            if window.window_start_time_seconds <= e.time_before_seconds
            and e.time_after_seconds <= window.window_end_time_seconds
            for reason in _pair_reasons(SamplingSignalEntry(e.pair_index, e.time_before_seconds,
                e.time_after_seconds, signals[e.pair_index]), metadata.fps))
        lines.append("  Direct trigger pair counts (before padding/merging)=" + str(dict(direct_reasons)))
    check("Phase3 interval records still unchanged after all Phase4 work", snapshot == tuple(asdict(i) for i in intervals))
    if name == "combined_interaction":
        actual = [(i.start_time_seconds, i.end_time_seconds, i.start_reason.value, i.end_reason.value) for i in intervals]
        check("combined exact corrected lifecycle", actual == [(0, 14.7, "video_started_active", "switch"),
                                                             (14.7, 18.6, "switch", "video_end")])
        check("combined false pan SWITCH absent; actual replacement SWITCH retained",
              [(e.kind.value, e.candidate_time_seconds) for e in phase3["events"]] == [("switch", 14.7)])
        check("combined only real replacement is a non-NONE candidate", [(e.pair_index, e.candidate.kind.value)
              for e in timeline if e.candidate.kind.value != "none"] == [(146, "switch")])
        check("combined loading frame raw/effective remains distinct", changed == [146] and not raw[146] and effective[146])
        for label, core, time, sampling, stable, reason in (
            ("initial stable", 0, 0, "sparse_sampling", True, None),
            ("first moving scroll", 0, 2.9, "dense_sampling", False, "ticket_scroll"),
            ("first reading pause", 0, 3.7, "dense_sampling", True, "ticket_scroll"),
            ("second moving scroll", 0, 6.3, "dense_sampling", False, "ticket_scroll"),
            ("second stable region", 0, 7.2, "dense_sampling", True, "ticket_scroll"),
            ("pan", 0, 9.7, "dense_sampling", True, "map_translation"),
            ("post-pan sparse stability", 0, 12, "sparse_sampling", True, None),
            ("actual switch outgoing anchor", 0, 14.7, "dense_sampling", False, "route_visual_change"),
            ("actual switch incoming anchor", 1, 14.7, "dense_sampling", True, "route_visual_change"),
            ("final stable ticket", 1, 18.6, "sparse_sampling", True, None)):
            found = [c for c in candidates_by_core.get(core, ()) if _close(c.timestamp_seconds, time)]
            check("combined " + label, len(found) == 1 and sampling in found[0].sources
                  and ("stable_ticket_view" in found[0].sources) == stable
                  and (reason is None or reason in found[0].dense_reasons))
            lines.append(f"  Sequence {label}: {found}")
    elif name == "investigation_close_reopen":
        check("real CLOSE boundary remains 3.8s", any(i.end_time_seconds == 3.8 and i.end_reason.value == "close" for i in intervals))
        check("accidental OPEN remains superseded", any(r.candidate_pair_index == 70 and r.status.value == "superseded"
                                                        for r in phase3["confirmations"]))
    elif name == "ticket_scroll":
        positives = [s.ticket_scroll_detected for s in signals]
        bursts = sum(p and (i == 0 or not positives[i-1]) for i, p in enumerate(positives))
        views = [v for _, _, vs, _ in contexts for v in vs]
        check("multiple scroll bursts and intermediate reading pauses retained", bursts > 1
              and sum(v.follows_scroll and v.stable_end_time_seconds is not None for v in views) >= 2)
        check("final stable stop retained", any(v.follows_scroll and v.stable_end_time_seconds is None for v in views))
        lines.append(f"  Scroll-positive pairs={sum(positives)}; contiguous bursts={bursts}; stable views={len(views)}")
    elif name in ("map_pan", "map_zoom"):
        reason_counts = Counter(r for cs in candidates_by_core.values() for c in cs for r in c.dense_reasons)
        expected = "map_translation" if name == "map_pan" else "map_scale_change"
        check(name + " activity explained by expected map reason", reason_counts[expected] > 0)
        lines.append(f"  Dense candidate reasons (overlap allowed)={dict(reason_counts)}; no sparsity target imposed.")
    lines.extend(f"  {'PASS' if ok else 'FAIL'}: {label}" for label, ok in checks)
    passed = all(ok for _, ok in checks)
    lines.append(f"  RESULT: {'PASS' if passed else 'FAIL'} ({len(checks)} checks)")
    print(f"{name}: {'PASS' if passed else 'FAIL'}", flush=True)
    return passed, lines, rows


def main():
    """Write closure evidence, stopping on failure and never declaring partial success."""
    lines = ["Phase 4 end-to-end closure validation", "All contexts use actual Phase3 lifecycle intervals.",
             "Defaults unchanged: context/dense padding .5s, sparse1FPS, dense sourceFPS, map>1px/>.01, stable dwell.5s, bridge.15s.",
             "Reference reduction uses production full-rate requests with identical semantic anchors."]
    rows, passed = [], True
    for name in SOURCES:
        try:
            ok, section, measured = _validate_video(name)
        except (ValueError, TypeError, OSError, AssertionError, RuntimeError) as error:
            lines.append(f"FAIL: {name}: {type(error).__name__}: {error}; remaining videos not evaluated.")
            passed = False
            break
        lines.extend(section)
        rows.extend(measured)
        if not ok:
            passed = False
            break
    if rows:
        numeric = [key for key in rows[0] if key not in ("video_name", "investigation_index", "core_start", "core_end",
                    "window_start", "window_end", "reduction_percent")]
        totals = {key: sum(row[key] for row in rows) for key in numeric}
        totals["reduction_percent"] = 100*(1-totals["candidate_count"]/totals["full_rate_reference_count"])
        lines.append(f"\nTOTALS across {len(rows)} windows (overlapping context counted separately): {totals}")
    lines.append("\nKnown validation gaps (manual representative evidence, not an accuracy benchmark):")
    lines.extend("- " + gap for gap in GAPS)
    lines.append(f"\nOVERALL: {'PASS' if passed else 'FAIL'}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = OUTPUT / "phase4_validation_report.txt"
    report.write_text("\n".join(lines)+"\n", encoding="utf-8")
    if rows:
        with (OUTPUT / "phase4_investigation_summary.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print("\n".join(lines))
    print(f"Report: {report}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
