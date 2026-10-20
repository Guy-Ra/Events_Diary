"""Bounded real-video validation of Phase 5.2b production body measurements.

Rebuild unchanged Phase 3/4 contexts, materialize their actual candidates through
Phase 5.1, and compare only explicitly assembled neighboring/control pairs using
the production API. Within each stable span use temporal neighbors plus one
first/last probe, never exhaustive combinations. Other relationships reproduce
the important Phase 5.2a scroll, pan, replacement and loading controls. Categories
are evaluation context, not semantic output from the comparison layer.

Only ticket_scroll_body enters production measurements. Raw MAD/CPR always remain
available; optional vertical residuals cover fully supported overlap and can be
rejected. Print counts, first failed geometric guards and metric distributions,
including cases where unnecessary alignment worsens a stable comparison. Loading
and replacement still require context: no low residual establishes identity.
Scalar report/CSV artifacts contain no candidate selection, ranking, grouping,
OCR or Phase 5.3 decisions. No crops are saved and no production settings tuned.
"""

import csv
import math
import sys
from collections import Counter
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from events_diary.config import UIConfig
from events_diary.evidence_candidate_timeline import build_evidence_candidate_timeline
from events_diary.frame_materialization import (
    VideoFrameResolver,
    materialize_evidence_candidates,
)
from events_diary.ticket_view_similarity import compare_ticket_view_pairs
from scripts.evaluation.evaluate_ticket_stable_views import _build_contexts

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/debug/map_motion/ticket_view_similarity"
SOURCES = ("ticket_scroll", "combined_interaction", "investigation_switch")
METRICS = (
    "raw_mad",
    "raw_cpr",
    "dx_pixels",
    "dy_pixels",
    "alignment_response",
    "overlap_fraction",
    "aligned_mad",
    "aligned_cpr",
)


def _pairs(name, contexts, phase3, materialized):
    """Build bounded diagnostic references without inventing candidate timestamps."""
    candidates = [m.evidence_candidate for m in materialized]
    pairs, groups = [], {}
    for window, _, views, _ in contexts:
        investigation = window.investigation_index
        for view_index, view in enumerate(views):
            group = [
                i
                for i, c in enumerate(candidates)
                if c.investigation_index == investigation
                and c.stable_view_index == view_index
            ]
            groups[(investigation, view_index)] = (view, group)
            pairs.extend(
                ("same_stable_span", "neighbors", a, b) for a, b in pairwise(group)
            )
            if len(group) > 2:
                pairs.append(("same_stable_span", "first_last", group[0], group[-1]))
        for left, right in pairwise(range(len(views))):
            if not views[right].follows_scroll:
                continue
            _, ga = groups[(investigation, left)]
            _, gb = groups[(investigation, right)]
            if ga and gb:
                pairs.append(
                    ("post_scroll_stable_views", "boundary_edges", ga[-1], gb[0])
                )
                a = min(
                    ga,
                    key=lambda i: abs(
                        candidates[i].timestamp_seconds
                        - views[left].stable_confirmed_time_seconds
                    ),
                )
                b = min(
                    gb,
                    key=lambda i: abs(
                        candidates[i].timestamp_seconds
                        - views[right].stable_confirmed_time_seconds
                    ),
                )
                pairs.append(
                    ("post_scroll_stable_views", "confirmation_neighborhood", a, b)
                )
        owned = [
            i
            for i, c in enumerate(candidates)
            if c.investigation_index == investigation
        ]
        for a, b in pairwise(owned):
            ia, ib = materialized[a].frame_index, materialized[b].frame_index
            if (
                ia is not None
                and ib == ia + 1
                and all("dense_sampling" in candidates[i].sources for i in (a, b))
                and phase3["signals"][ia].ticket_scroll_detected
            ):
                pairs.append(("adjacent_during_scroll", "existing_positive_pair", a, b))
    for event in phase3["events"]:
        if event.kind.value != "switch":
            continue
        time = event.candidate_time_seconds
        before = [
            i
            for i, c in enumerate(candidates)
            if c.timestamp_seconds < time and c.stable_view_index is not None
        ]
        after = [
            i
            for i, c in enumerate(candidates)
            if c.timestamp_seconds >= time and c.stable_view_index is not None
        ]
        if before and after:
            a = max(before, key=lambda i: candidates[i].timestamp_seconds)
            b = min(after, key=lambda i: candidates[i].timestamp_seconds)
            pairs.append(("true_ticket_replacement", "boundary", a, b))
            view, group = groups[
                (candidates[b].investigation_index, candidates[b].stable_view_index)
            ]
            settled = min(
                group,
                key=lambda i: abs(
                    candidates[i].timestamp_seconds - view.stable_confirmed_time_seconds
                ),
            )
            pairs.append(
                ("true_ticket_replacement", "confirmation_neighborhood", a, settled)
            )

    def landmark(time, investigation):
        """Resolve one actual request within floating timestamp tolerance."""
        found = [
            i
            for i, c in enumerate(candidates)
            if c.investigation_index == investigation
            and math.isclose(c.timestamp_seconds, time, rel_tol=0, abs_tol=1e-9)
        ]
        if len(found) != 1:
            raise ValueError(
                f"Missing/ambiguous actual candidate: {name} I{investigation} t{time}"
            )
        return found[0]

    if name == "combined_interaction":
        for a, b in ((8.0, 9.7), (9.7, 12.0), (8.0, 12.0)):
            pairs.append(
                (
                    "same_ticket_across_map_pan",
                    "manual_pan_control",
                    landmark(a, 0),
                    landmark(b, 0),
                )
            )
        controls = (
            (14.5, 0, 14.6, 0),
            (14.6, 0, 14.7, 1),
            (14.5, 0, 14.7, 1),
            (14.7, 1, 15.2, 1),
        )
    elif name == "investigation_switch":
        controls = (
            (1.5, 0, 1.6, 0),
            (1.6, 0, 1.7, 0),
            (1.7, 0, 2.2, 0),
            (4.7, 0, 4.8, 0),
            (4.8, 0, 4.9, 1),
        )
    else:
        controls = ()
    for a, ai, b, bi in controls:
        pairs.append(
            ("loading_transition", "manual_landmark", landmark(a, ai), landmark(b, bi))
        )
    return pairs


def _evaluate(name):
    """Use real production candidates, materialization and bounded cached comparisons."""
    metadata, anomalies, contexts, phase3 = _build_contexts(name, include_phase3=True)
    if anomalies:
        raise RuntimeError(f"Unexpected lifecycle anomalies: {anomalies}")
    config = UIConfig()
    candidates = tuple(
        c
        for window, plan, views, _ in contexts
        for c in build_evidence_candidate_timeline(
            window=window, sampling_plan=plan, stable_views=views
        )
    )
    snapshot = tuple(asdict(c) for c in candidates)
    with VideoFrameResolver(
        metadata.path, metadata=metadata, max_frames=16
    ) as resolver:
        materialized = materialize_evidence_candidates(
            video_path=metadata.path,
            candidates=candidates,
            ui_config=config,
            resolver=resolver,
        )
        pairs = _pairs(name, contexts, phase3, materialized)
        comparisons = compare_ticket_view_pairs(
            video_path=metadata.path,
            materialized_candidates=materialized,
            pairs=[(a, b) for _, _, a, b in pairs],
            ui_config=config,
            resolver=resolver,
        )
        rows = []
        for (relationship, variant, a, b), comparison in zip(
            pairs, comparisons, strict=True
        ):
            rows.append(
                {
                    "video_name": metadata.path.name,
                    "investigation_index": candidates[a].investigation_index,
                    "investigation_index_b": candidates[b].investigation_index,
                    "relationship": relationship,
                    "pair_variant": variant,
                    "timestamp_a": candidates[a].timestamp_seconds,
                    "timestamp_b": candidates[b].timestamp_seconds,
                    **asdict(comparison),
                    "alignment_valid": comparison.alignment_valid,
                }
            )
        cache = f"cache_hits={resolver.cache_hits}; decodes={resolver.decode_attempts}; bounded_cache={resolver.cache_size}/16"
    if tuple(asdict(c) for c in candidates) != snapshot:
        raise AssertionError("Semantic candidates changed")
    summary = (
        f"{name}: {metadata.width}x{metadata.height}, {metadata.fps:g} FPS, "
        f"frames={metadata.frame_count}, duration={metadata.duration_seconds:g}s; "
        f"candidates={len(candidates)} pairs={len(rows)}; {cache}"
    )
    print(summary, flush=True)
    return rows, summary


def _stats(values):
    """Summarize finite measurements without fitting semantic thresholds."""
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if not finite:
        return "unavailable"
    return f"n={len(finite)} min={min(finite):.6g} median={np.median(finite):.6g} max={max(finite):.6g}"


def _summary(rows, lines):
    """Expose acceptance, first-failure reasons and raw versus supplemental residuals."""
    accepted = sum(r["alignment_valid"] for r in rows)
    lines.append(
        f"pairs={len(rows)} attempted={len(rows)} accepted={accepted} rejected={len(rows) - accepted}"
    )
    counts = Counter(
        r["alignment_rejection_reason"] for r in rows if not r["alignment_valid"]
    )
    lines.append(f"First failed guards: {dict(counts)}")
    for reason in ("low_response", "excessive_dx", "excessive_dy", "low_overlap"):
        lines.append(f"  {reason}={counts[reason]}")
    for relationship in sorted({r["relationship"] for r in rows}):
        group = [r for r in rows if r["relationship"] == relationship]
        supported = [r for r in group if r["alignment_valid"]]
        lines.append(
            f"\n{relationship}: n={len(group)} accepted={len(supported)} rejected={len(group) - len(supported)}"
        )
        for key in METRICS:
            lines.append(f"  {key}: {_stats([r[key] for r in group])}")
        lines.append(
            f"  aligned MAD improves={sum(r['aligned_mad'] < r['raw_mad'] for r in supported)} "
            f"worsens={sum(r['aligned_mad'] > r['raw_mad'] for r in supported)}"
        )
        lines.append(
            f"  rejection reasons={dict(Counter(r['alignment_rejection_reason'] for r in group if not r['alignment_valid']))}"
        )
    lines.append(
        "\nNon-within-span diagnostic details (context categories, not classifier labels):"
    )
    lines.extend(str(r) for r in rows if r["relationship"] != "same_stable_span")


def main():
    """Write bounded validation artifacts, failing visibly without adjusting defaults."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = OUTPUT / "ticket_view_similarity_report.txt"
    lines = [
        "Phase 5.2b production ticket-view similarity validation",
        "Production Phase3/4 requests + Phase5.1 materialization + Phase5.2b body comparisons; no selection.",
        "Bounded policy: stable temporal neighbors/first-last, adjacent scroll, consecutive stable views and known controls.",
        "ROI=ticket_scroll_body; CPR strict >10; dx/dy A-to-B, inverse vertical-only B-to-A alignment.",
        "Defaults: response>=.60, |dx|<=2px, |dy|/height<=.30, actual supported overlap>=.70.",
        "Calibration: adjacent n=26 response>=.660, |dx|<=.026px, |dy|/height<=.257, overlap>=.744.",
        "Large observed aliases: response=.118-.220, shift fraction=.385-.488. Margins preserve adjacent cases.",
        "The 2px lateral allowance is provisional, not an empirically established maximum. Other resolutions need validation.",
        "Integer shifts use exact rows; fractional INTER_LINEAR support==1 only. Raw and aligned domains differ.",
        "Rejected alignment keeps raw and finite diagnostics but no overlap/residual. First failure, not every failing guard, is counted.",
    ]
    rows = []
    try:
        for name in SOURCES:
            measured, info = _evaluate(name)
            rows.extend(measured)
            lines.append(info)
        _summary(rows, lines)
        adjacent = [r for r in rows if r["relationship"] == "adjacent_during_scroll"]
        if not adjacent or any(
            not r["alignment_valid"] or r["aligned_mad"] >= r["raw_mad"]
            for r in adjacent
        ):
            raise RuntimeError(
                "Adjacent-scroll alignment unreliable; stop and inspect, do not tune silently"
            )
    except (ValueError, TypeError, OSError, RuntimeError, AssertionError) as error:
        lines.append(f"STOP: {type(error).__name__}: {error}")
        report.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(lines[-1], flush=True)
        return 1
    lines.extend(
        [
            "\nInterpretation limits:",
            "Stable raw differences stay available even when unnecessary alignment worsens them; alignment acceptance is not identity.",
            "Adjacent scroll should have strongly reduced residual; large post-scroll aliases should be rejected, not repaired.",
            "Pan affects other frame areas; body metrics remain isolated. Loading may align successfully while pixels differ substantially.",
            "Replacement rejected by these guards has unavailable aligned residual; do not impute an aligned value from raw.",
            "A low accepted residual can reflect common templates/cropped regions, not shared ticket identity or section coverage.",
            "Representative/manual small canonical-layout validation, not a benchmark; other FPS/themes/layouts and alias cases remain open.",
            "No semantic classifier, quality weighting, candidate removal/ranking/selection, Phase4 changes or Phase5.3 implementation.",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (OUTPUT / "ticket_view_similarity.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    accepted = sum(row["alignment_valid"] for row in rows)
    print(
        f"TOTAL pairs={len(rows)} attempted={len(rows)} accepted={accepted} rejected={len(rows) - accepted}",
        flush=True,
    )
    print(f"Report: {report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
