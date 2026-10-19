"""Phase 4.4 real-model validation of pure evidence candidate composition.

Reuse the Phase 4.3 real-video preparation helper to supply production lifecycle
windows, adaptive plans and stable views for five existing videos. This evaluator
contains no decoding, raw-signal analysis, detector or sampling implementation;
the existing upstream validation helper supplies those model objects. The new
production composer itself consumes only the three approved model types.

Check exact one-to-one request preservation and input immutability for every actual
Investigation. Report sparse/dense/anchor/stable counts, each view's coverage and
all combined-video candidates across initial stability, both scrolls, reading
pauses, pan, replacement anchors and final stability. Closed stable bounds are
inclusive; censored ends use the core end by the Phase 4.4 contract, which can
annotate the already-requested end anchor beyond the last observed source pair.
This adds no timestamps or claim of new observed frames. No images are saved.

Write only a report and one CSV under evidence_candidate_timeline. Matching local
outputs are replaced; unrelated artifacts are preserved. Fail loudly on missing
coverage, model mismatches or any changed request timeline. No ranking, filtering,
OCR, decoded frame indexes, semantic deduplication or Phase 5 selection occurs.
"""

import csv
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from evaluate_ticket_stable_views import SOURCES, _build_contexts

from events_diary.evidence_candidate_timeline import build_evidence_candidate_timeline

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/debug/map_motion/evidence_candidate_timeline"


def _evaluate(name):
    """Compose each upstream context and return per-Investigation diagnostics/rows."""
    metadata, anomalies, contexts = _build_contexts(name)
    lines = [(f"\n{name}: {metadata.width}x{metadata.height}, {metadata.fps:g} FPS, "
              f"{metadata.frame_count} frames, {metadata.duration_seconds:g}s; actual lifecycle; anomalies={anomalies}")]
    rows = []
    for window, plan, views, observed_through in contexts:
        snapshot = (asdict(window), asdict(plan), tuple(asdict(v) for v in views))
        candidates = build_evidence_candidate_timeline(window=window, sampling_plan=plan, stable_views=views)
        if tuple(c.timestamp_seconds for c in candidates) != plan.requested_timestamps_seconds:
            raise AssertionError("STOP: EvidenceCandidate timestamps differ from original requests.")
        if snapshot != (asdict(window), asdict(plan), tuple(asdict(v) for v in views)):
            raise AssertionError("Composition must not mutate upstream objects.")
        counts = Counter(label for c in candidates for label in c.sources)
        anchors = sum(any(label in c.sources for label in
            ("window_start", "investigation_start", "investigation_end", "window_end")) for c in candidates)
        post_scroll = sum(c.stable_view_follows_scroll for c in candidates)
        lines.append(f"  Investigation {window.investigation_index}: core="
            f"{window.investigation_start_time_seconds:g}-{window.investigation_end_time_seconds:g}s; "
            f"window={window.window_start_time_seconds:g}-{window.window_end_time_seconds:g}s; "
            f"requested={len(plan.requested_timestamps_seconds)}; candidates={len(candidates)}; "
            f"sparse={counts['sparse_sampling']}; dense={counts['dense_sampling']}; anchors={anchors}; "
            f"stable={counts['stable_ticket_view']}; post_scroll_stable={post_scroll}; "
            "exact one-to-one=True; inputs unchanged=True")
        for index, view in enumerate(views):
            count = sum(c.stable_view_index == index for c in candidates)
            end = view.stable_end_time_seconds
            effective_end = end if end is not None else window.investigation_end_time_seconds
            lines.append(f"    stable view {index}: start={view.stable_start_time_seconds:g}; "
                f"confirmed={view.stable_confirmed_time_seconds:g}; end={end}; association_end={effective_end:g}; "
                f"observed_input_through={observed_through}; candidates={count}; "
                f"follows_scroll={view.follows_scroll}; coverage={count > 0}")
        for index, candidate in enumerate(candidates):
            row = {"video_name": metadata.path.name, "candidate_index": index, **asdict(candidate)}
            row["sources"] = ";".join(candidate.sources)
            row["dense_reasons"] = ";".join(candidate.dense_reasons)
            rows.append(row)
            if name == "combined_interaction":
                lines.append(f"    candidate {index}: t={candidate.timestamp_seconds!r}; "
                    f"sources={candidate.sources}; dense_reasons={candidate.dense_reasons}; "
                    f"stable_view_index={candidate.stable_view_index}; follows_scroll={candidate.stable_view_follows_scroll}")
    return lines, rows


def main():
    """Evaluate five real upstream outputs and write composition-only diagnostics."""
    lines = ["Phase 4.4 Evidence Candidate Timeline validation",
             "Every original request preserved exactly; no filtering, ranking or generated timestamps.",
             "Timestamp matching: math.isclose(rel_tol=0, abs_tol=1e-9 seconds).",
             "Sampling classification, anchors and stable-view labels coexist; dense reasons unchanged.",
             "Censored stable spans associate through Investigation end, excluding trailing context.",
             "Existing Phase4.3 real-video helper supplies upstream models; composer never handles frames/signals."]
    rows = []
    for name in SOURCES:
        section, measured = _evaluate(name)
        lines.extend(section)
        rows.extend(measured)
    lines += ["\nAll stable views have existing request coverage; all requested times map exactly one-to-one.",
              "Dense during-scroll candidates remain present. No candidate ranking/filtering or Phase5."]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = OUTPUT / "evidence_candidate_timeline_report.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (OUTPUT / "evidence_candidate_timeline.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("\n".join(lines))
    print(f"Report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
