"""Phase 5 closure checks on seven real lifecycle-derived video pipelines.

Run unchanged lifecycle, windows, adaptive sampling, stable-view detection,
candidate composition, physical materialization and production representative
selection. Never substitute synthetic Investigations. Validate A (ticket only),
B (default map/route), and C (full optional context) against actual metadata and
the explicitly recorded Phase 5.3b regression references. References belong to
this validator, never to production policy; unexpected differences fail visibly.

Audit every original request in the CSV, including unselected and adjusted end
anchors. Independently check primary physical ownership, confirmation eligibility,
stable/post-scroll coverage, chronological uniqueness, determinism and additive
configuration reasons. Reuse production metadata-run grouping only to audit its
bounded context coverage; do not invent another event detector. These are padded
activity runs, not direct-positive motion episodes or semantic map states.

Similarity is supplemental validation only: at most two existing diagnostic pairs
per relevant relationship per recording, no exhaustive comparisons. Raw MAD <=5
is a local unchanged-body regression bound, and residual >10 is a local changed-
body check, not a production threshold or semantic classifier. Rejected alignment
must retain raw measurements and omit aligned values. No selection uses these
measurements. Reuse existing representative contact sheets rather than generating
another image collection. Write scalar reports/CSVs even on a failed check, return
nonzero on failure, and never tune algorithms, defaults or regression references.
"""

import csv
import math
import sys
from collections import Counter
from dataclasses import asdict, fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from events_diary.config import UIConfig
from events_diary.evidence_candidate_timeline import build_evidence_candidate_timeline
from events_diary.frame_materialization import (
    VideoFrameResolver,
    materialize_evidence_candidates,
)
from events_diary.representative_selection import (
    _activity_bursts,
    select_investigation_representatives,
)
from events_diary.ticket_view_similarity import compare_ticket_view_pairs
from scripts.evaluation.evaluate_representative_selection import (
    CONFIGS,
    MAP_REASONS,
    SOURCES,
    _controls,
)
from scripts.evaluation.evaluate_ticket_stable_views import _build_contexts
from scripts.evaluation.evaluate_ticket_view_similarity import _pairs

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/debug/map_motion/phase5_validation"
EPS = 1e-9
# Historical observations, intentionally separate from actual detected metadata.
# Tuple order: candidate count, stable views, post-scroll views, A/B/C selections.
REFERENCE = {
    ("combined_interaction", 0): (80, 3, 2, 3, 5, 9),
    ("combined_interaction", 1): (15, 1, 0, 1, 2, 4),
    ("ticket_scroll", 0): (66, 5, 4, 5, 5, 9),
    ("investigation_switch", 0): (27, 1, 0, 1, 2, 4),
    ("investigation_switch", 1): (14, 1, 0, 1, 2, 4),
    ("investigation_started_active", 0): (16, 1, 0, 1, 2, 4),
    ("investigation_started_active", 1): (13, 1, 0, 1, 2, 4),
    ("investigation_close_reopen", 0): (16, 1, 0, 1, 1, 3),
    ("investigation_close_reopen", 1): (29, 1, 0, 1, 2, 4),
    ("map_pan", 0): (76, 1, 0, 1, 3, 5),
    ("map_zoom", 0): (82, 1, 0, 1, 2, 4),
}
ENDPOINTS = {
    "combined_interaction": (18.6, 18.5, 185),
    "ticket_scroll": (8.4, 8.3, 83),
    "investigation_switch": (8.1, 8.0, 80),
    "map_pan": (12.5, 12.4, 124),
    "map_zoom": (8.9, 8.8, 88),
}
REASONS = (
    "stable_ticket",
    "post_scroll_stable",
    "stable_confirmation_fallback",
    "map_activity",
    "route_activity",
    "scroll_provenance",
    "investigation_boundary",
)


def _close(a, b):
    """Compare timestamp representation using the established absolute tolerance."""
    return math.isclose(a, b, rel_tol=0, abs_tol=EPS)


def _owned(window, item):
    """Independently audit physical core ownership, excluding outgoing endpoints."""
    time = item.actual_timestamp_seconds
    return (
        time is not None
        and item.resolution_status != "unresolved"
        and time >= window.investigation_start_time_seconds - EPS
        and time < window.investigation_end_time_seconds
        and not _close(time, window.investigation_end_time_seconds)
    )


def _eligible(window, view, candidate, item, vi):
    """Audit both semantic association and resolved in-span physical membership."""
    end = view.stable_end_time_seconds
    end = window.investigation_end_time_seconds if end is None else end
    return (
        candidate.stable_view_index == vi
        and _owned(window, item)
        and view.stable_start_time_seconds - EPS
        <= candidate.timestamp_seconds
        <= end + EPS
        and view.stable_start_time_seconds - EPS
        <= item.actual_timestamp_seconds
        <= end + EPS
    )


def _similarity(name, metadata, contexts, phase3, materialized, resolver, check):
    """Compare a small fixed bound of existing diagnostic pairs through production."""
    chosen, counts = [], Counter()
    relevant = {
        "same_stable_span",
        "adjacent_during_scroll",
        "same_ticket_across_map_pan",
        "true_ticket_replacement",
    }
    for relationship, variant, a, b in _pairs(name, contexts, phase3, materialized):
        if relationship in relevant and counts[relationship] < 2:
            chosen.append((relationship, variant, a, b))
            counts[relationship] += 1
    measured = compare_ticket_view_pairs(
        video_path=metadata.path,
        materialized_candidates=materialized,
        pairs=[(a, b) for _, _, a, b in chosen],
        ui_config=UIConfig(),
        resolver=resolver,
    )
    rows = []
    for (relationship, variant, a, b), result in zip(chosen, measured, strict=True):
        label = f"similarity {relationship} {a}->{b}"
        check(
            label + " finite raw measurements",
            math.isfinite(result.raw_mad) and 0 <= result.raw_cpr <= 1,
        )
        if result.alignment_valid:
            check(
                label + " supported alignment",
                result.alignment_rejection_reason is None
                and result.overlap_fraction >= 0.70
                and math.isfinite(result.aligned_mad)
                and 0 <= result.aligned_cpr <= 1,
            )
        else:
            check(
                label + " rejected fit retains raw only",
                result.alignment_rejection_reason is not None
                and result.aligned_mad is None
                and result.aligned_cpr is None
                and result.overlap_fraction is None,
            )
        if relationship in ("same_stable_span", "same_ticket_across_map_pan"):
            check(
                label + " unchanged-body raw MAD <=5 (regression bound only)",
                result.raw_mad <= 5,
            )
        elif relationship == "adjacent_during_scroll":
            check(
                label + " alignment reduces residual",
                result.alignment_valid and result.aligned_mad < result.raw_mad,
            )
        else:
            check(
                label + " rejected or different residual",
                not result.alignment_valid or result.aligned_mad > 10,
            )
        rows.append(
            {
                "video_name": metadata.path.name,
                "relationship": relationship,
                "variant": variant,
                "requested_a": materialized[a].requested_timestamp_seconds,
                "requested_b": materialized[b].requested_timestamp_seconds,
                **asdict(result),
            }
        )
    return rows


def _video(name, rows, summaries, similarities, checks, lines, totals, unique):
    """Run and audit every actual Investigation without changing upstream records."""
    metadata, anomalies, contexts, phase3 = _build_contexts(name, include_phase3=True)

    def check(label, condition):
        """Retain every invariant result, including failure details for review."""
        checks.append((name, label, bool(condition)))

    lines.append(
        f"\n{name}: {metadata.width}x{metadata.height}, {metadata.fps:g} FPS, "
        f"{metadata.frame_count} frames, {metadata.duration_seconds:g}s; input={metadata.path}"
    )
    check("real lifecycle has no anomalies", not anomalies)
    check("one real context per interval", len(contexts) == len(phase3["intervals"]))
    check(
        "reference Investigation indexes",
        tuple(w.investigation_index for w, _, _, _ in contexts)
        == tuple(i for n, i in REFERENCE if n == name),
    )
    groups = tuple(
        build_evidence_candidate_timeline(window=w, sampling_plan=p, stable_views=v)
        for w, p, v, _ in contexts
    )
    candidates = tuple(c for group in groups for c in group)
    upstream = tuple(asdict(record) for record in phase3["intervals"])
    context_snapshot = tuple(
        (asdict(w), asdict(p), tuple(asdict(v) for v in views))
        for w, p, views, _ in contexts
    )
    snapshot = tuple(asdict(c) for c in candidates)
    with VideoFrameResolver(
        metadata.path, metadata=metadata, max_frames=16
    ) as resolver:
        materialized = materialize_evidence_candidates(
            video_path=metadata.path,
            candidates=candidates,
            ui_config=UIConfig(),
            resolver=resolver,
        )
        check(
            "candidate preservation through materialization",
            len(materialized) == len(candidates),
        )
        check(
            "original semantic candidate identity/order",
            all(
                m.evidence_candidate is c
                for m, c in zip(materialized, candidates, strict=True)
            ),
        )
        check(
            "lightweight materialized fields",
            tuple(f.name for f in fields(materialized[0]))
            == (
                "evidence_candidate",
                "resolution_status",
                "frame_index",
                "actual_timestamp_seconds",
                "ticket_sharpness_laplacian_variance",
                "ticket_brightness_mean",
                "ticket_contrast_std",
            ),
        )
        for c, m in zip(candidates, materialized, strict=True):
            check(
                f"request {c.timestamp_seconds:g} physical resolution",
                m.resolution_status != "unresolved"
                and _close(m.actual_timestamp_seconds, m.frame_index / metadata.fps),
            )
            if m.resolution_status == "resolved":
                check(
                    f"request {c.timestamp_seconds:g} nearest-frame error",
                    m.timestamp_error_seconds <= 0.5 / metadata.fps + EPS,
                )
            check(
                f"request {c.timestamp_seconds:g} raw diagnostics",
                all(
                    v is not None and math.isfinite(v) and v >= 0
                    for v in (
                        m.ticket_sharpness_laplacian_variance,
                        m.ticket_brightness_mean,
                        m.ticket_contrast_std,
                    )
                ),
            )
            totals[m.resolution_status] += 1
            if m.frame_index is not None:
                unique.add((name, m.frame_index))
        endpoints = [
            m
            for m in materialized
            if m.requested_timestamp_seconds == metadata.duration_seconds
        ]
        check(
            "all duration anchors explicitly adjusted",
            bool(endpoints)
            and all(
                m.resolution_status == "video_end_adjusted"
                and m.frame_index == metadata.frame_count - 1
                for m in endpoints
            ),
        )
        if name in ENDPOINTS:
            requested, actual, index = ENDPOINTS[name]
            check(
                "reference endpoint resolution",
                all(
                    _close(m.requested_timestamp_seconds, requested)
                    and _close(m.actual_timestamp_seconds, actual)
                    and m.frame_index == index
                    for m in endpoints
                ),
            )
        lines.append(
            "  Endpoint anchors="
            + str(
                [
                    (
                        m.requested_timestamp_seconds,
                        m.actual_timestamp_seconds,
                        m.frame_index,
                        m.resolution_status,
                    )
                    for m in endpoints
                ]
            )
        )
        materialized_snapshot = tuple(asdict(m) for m in materialized)
        offset = 0
        for (window, plan, views, _), group in zip(contexts, groups, strict=True):
            inv = window.investigation_index
            local = materialized[offset : offset + len(group)]
            expected = REFERENCE.get((name, inv))
            post_count = sum(v.follows_scroll for v in views)
            actual = (len(group), len(views), post_count)
            check(
                f"I{inv} upstream regression reference actual={actual}, expected={expected}",
                expected is not None and actual == expected[:3],
            )
            map_bursts = _activity_bursts(group, window, plan, MAP_REASONS)
            scroll_bursts = _activity_bursts(group, window, plan, ("ticket_scroll",))
            boundaries = {
                i
                for i, c in enumerate(group)
                if any(
                    r in c.sources for r in ("investigation_start", "investigation_end")
                )
                and local[i].resolution_status != "unresolved"
            }
            previous = {}
            for ci, (config, (map_on, scroll_on, boundary_on)) in enumerate(
                CONFIGS.items()
            ):
                kwargs = {
                    "window": window,
                    "sampling_plan": plan,
                    "candidates": group,
                    "materialized_candidates": local,
                    "stable_views": views,
                    "include_map_route_evidence": map_on,
                    "include_scroll_provenance": scroll_on,
                    "include_boundary_context": boundary_on,
                }
                result = select_investigation_representatives(**kwargs)
                selections = result.selections
                chosen = {s.candidate_index: s.selection_reasons for s in selections}
                label = f"I{inv} {config}"
                check(
                    label + " deterministic",
                    result == select_investigation_representatives(**kwargs),
                )
                check(
                    label + " index identity and uniqueness",
                    result.investigation_index == inv
                    and len(chosen) == len(selections)
                    and all(0 <= i < len(group) for i in chosen),
                )
                check(
                    label + " chronological",
                    tuple(chosen)
                    == tuple(
                        sorted(chosen, key=lambda i: (group[i].timestamp_seconds, i))
                    ),
                )
                check(
                    label + " merged canonical reasons",
                    all(
                        rs and rs == tuple(r for r in REASONS if r in rs)
                        for rs in chosen.values()
                    ),
                )
                check(
                    label + " additive reference/reason coverage",
                    all(
                        i in chosen and set(rs) <= set(chosen[i])
                        for i, rs in previous.items()
                    ),
                )
                previous = chosen
                check(
                    label
                    + f" count regression actual={len(chosen)} expected={expected[3 + ci] if expected else None}",
                    expected is not None and len(chosen) == expected[3 + ci],
                )
                primary = {i for i, rs in chosen.items() if "stable_ticket" in rs}
                pre, fallback = 0, 0
                for vi, view in enumerate(views):
                    primary_view = [
                        i for i in primary if group[i].stable_view_index == vi
                    ]
                    check(label + f" V{vi} exactly one primary", len(primary_view) == 1)
                    if len(primary_view) != 1:
                        continue
                    i = primary_view[0]
                    eligible = [
                        j
                        for j, c in enumerate(group)
                        if _eligible(window, view, c, local[j], vi)
                    ]
                    post = [
                        j
                        for j in eligible
                        if group[j].timestamp_seconds
                        >= view.stable_confirmed_time_seconds - EPS
                        and local[j].actual_timestamp_seconds
                        >= view.stable_confirmed_time_seconds - EPS
                    ]
                    is_pre = (
                        group[i].timestamp_seconds
                        < view.stable_confirmed_time_seconds - EPS
                        or local[i].actual_timestamp_seconds
                        < view.stable_confirmed_time_seconds - EPS
                    )
                    pre += is_pre
                    fallback += "stable_confirmation_fallback" in chosen[i]
                    check(label + f" V{vi} association/core ownership", i in eligible)
                    check(
                        label + f" V{vi} confirmation preference",
                        not post or not is_pre,
                    )
                    check(
                        label + f" V{vi} fallback reason",
                        ("stable_confirmation_fallback" in chosen[i]) == (not post),
                    )
                    check(
                        label + f" V{vi} post-scroll retained",
                        ("post_scroll_stable" in chosen[i]) == view.follows_scroll,
                    )
                    lines.append(
                        f"  {label} V{vi}: confirmed={view.stable_confirmed_time_seconds:g}; "
                        f"selected C{i} request={group[i].timestamp_seconds:g}, actual={local[i].actual_timestamp_seconds:g}; "
                        f"eligible_post={len(post)}; reasons={chosen[i]}"
                    )
                check(label + " no extra primary slots", len(primary) == len(views))
                covered = []
                for role, bursts, enabled, reasons in (
                    ("map", map_bursts, map_on, ("map_activity", "route_activity")),
                    ("scroll", scroll_bursts, scroll_on, ("scroll_provenance",)),
                ):
                    representatives = {
                        i for i, rs in chosen.items() if any(r in rs for r in reasons)
                    }
                    representable = [
                        b for b in bursts if any(_owned(window, local[i]) for i in b)
                    ]
                    coverage = sum(
                        len(set(b) & representatives) == 1 for b in representable
                    )
                    check(
                        label + f" bounded {role} coverage",
                        coverage == (len(representable) if enabled else 0)
                        and len(representatives)
                        == (len(representable) if enabled else 0)
                        and all(_owned(window, local[i]) for i in representatives),
                    )
                    covered.extend((len(bursts), len(representable), coverage))
                boundary_selected = {
                    i for i, rs in chosen.items() if "investigation_boundary" in rs
                }
                check(
                    label + " optional existing core anchors only",
                    boundary_selected == (boundaries if boundary_on else set()),
                )
                check(
                    label + " every selected frame resolves",
                    all(local[i].resolution_status != "unresolved" for i in chosen),
                )
                physical = Counter(local[i].frame_index for i in chosen)
                duplicates = sum(n - 1 for n in physical.values())
                check(
                    label + " current reference has no duplicate physical selections",
                    duplicates == 0,
                )
                for i, rs in chosen.items():
                    controls = _controls(name, local[i])
                    check(
                        label + f" C{i} transition/loading not primary",
                        "stable_ticket" not in rs
                        or all(
                            control == "settled_incoming_ticket" for control in controls
                        ),
                    )
                    if name == "combined_interaction" and local[i].frame_index in (
                        146,
                        147,
                    ):
                        check(
                            label + f" C{i} loading/mixed context guard",
                            local[i].frame_index == 147
                            and boundary_on
                            and rs == ("investigation_boundary",),
                        )
                    if name == "investigation_switch" and (
                        local[i].frame_index == 16
                        or (inv == 0 and local[i].frame_index == 49)
                    ):
                        check(
                            label + f" C{i} switch boundary context only",
                            boundary_on and rs == ("investigation_boundary",),
                        )
                    if name == "investigation_started_active" and inv == 1:
                        check(
                            label + f" C{i} previous-ticket context excluded",
                            local[i].frame_index not in (19, 20),
                        )
                    if name == "ticket_scroll" and not scroll_on:
                        check(
                            label + f" C{i} no active-scroll reference",
                            not (
                                local[i].frame_index
                                and phase3["signals"][
                                    local[i].frame_index - 1
                                ].ticket_scroll_detected
                            ),
                        )
                if name == "combined_interaction" and inv == 1:
                    check(
                        label + " settled 15.2s primary",
                        any(local[i].frame_index == 152 for i in primary),
                    )
                if name == "investigation_started_active" and inv == 1:
                    check(
                        label + " settled 2.9s primary",
                        any(local[i].frame_index == 29 for i in primary),
                    )
                summary = {
                    "video_name": metadata.path.name,
                    "investigation_index": inv,
                    "configuration": config,
                    "candidate_count": len(group),
                    "stable_views": len(views),
                    "post_scroll_views": post_count,
                    "selected_count": len(chosen),
                    "reduction_percent": 100 * (1 - len(chosen) / len(group)),
                    "stable_covered": len(primary),
                    "post_scroll_covered": sum(
                        "post_scroll_stable" in rs for rs in chosen.values()
                    ),
                    "map_bursts": covered[0],
                    "map_representable": covered[1],
                    "map_covered": covered[2],
                    "scroll_bursts": covered[3],
                    "scroll_representable": covered[4],
                    "scroll_covered": covered[5],
                    "boundary_anchors": len(boundaries),
                    "boundary_selected": len(boundary_selected),
                    "pre_confirmation_primary": pre,
                    "confirmation_fallbacks": fallback,
                    "duplicate_candidate_selections": len(selections) - len(chosen),
                    "duplicate_physical_frames": duplicates,
                    "unresolved_selected": sum(
                        local[i].resolution_status == "unresolved" for i in chosen
                    ),
                }
                summary.update(
                    {r: sum(r in rs for rs in chosen.values()) for r in REASONS}
                )
                summaries.append(summary)
                lines.append("  SUMMARY " + str(summary))
                for i, (c, m) in enumerate(zip(group, local, strict=True)):
                    view = (
                        views[c.stable_view_index]
                        if c.stable_view_index is not None
                        else None
                    )
                    rows.append(
                        {
                            "video_name": metadata.path.name,
                            "investigation_index": inv,
                            "configuration": config,
                            "candidate_index": i,
                            "video_candidate_index": offset + i,
                            "requested_timestamp": c.timestamp_seconds,
                            "actual_timestamp": m.actual_timestamp_seconds,
                            "frame_index": m.frame_index,
                            "resolution_status": m.resolution_status,
                            "sharpness": m.ticket_sharpness_laplacian_variance,
                            "brightness": m.ticket_brightness_mean,
                            "contrast": m.ticket_contrast_std,
                            "stable_view_index": c.stable_view_index,
                            "stable_confirmed_time": view.stable_confirmed_time_seconds
                            if view
                            else None,
                            "selected": i in chosen,
                            "selection_reasons": ";".join(chosen.get(i, ())),
                            "manual_controls": ";".join(_controls(name, m)),
                        }
                    )
            offset += len(group)
        similarities.extend(
            _similarity(name, metadata, contexts, phase3, materialized, resolver, check)
        )
        check(
            "bounded shared image cache",
            resolver.cache_size <= resolver.max_frames == 16,
        )
        check(
            "materialized records unchanged",
            materialized_snapshot == tuple(asdict(m) for m in materialized),
        )
    check(
        "Phase 3 intervals unchanged",
        upstream == tuple(asdict(record) for record in phase3["intervals"]),
    )
    check(
        "Phase 4 metadata unchanged",
        context_snapshot
        == tuple(
            (asdict(w), asdict(p), tuple(asdict(v) for v in views))
            for w, p, views, _ in contexts
        ),
    )
    check(
        "original candidates unchanged",
        snapshot == tuple(asdict(c) for c in candidates),
    )
    totals.update(
        recordings=1,
        investigations=len(phase3["intervals"]),
        windows=len(contexts),
        candidates=len(candidates),
        materialized=len(materialized),
    )
    print(
        f"{name}: {'PASS' if all(ok for source, _, ok in checks if source == name) else 'FAIL'}; "
        f"Investigations={len(contexts)} candidates={len(candidates)}",
        flush=True,
    )


def _csv(name, rows):
    """Write scalar tables when rows exist, including partial evidence on failure."""
    if rows:
        with (OUTPUT / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    """Run closure integration, preserve diagnostics on failure, and never retune."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows, summaries, similarities, checks, unique = [], [], [], [], set()
    totals = Counter()
    lines = [
        "Phase 5 end-to-end closure validation",
        "Seven real lifecycle pipelines; no synthetic fallback Investigations.",
        "A=ticket minimal; B=default production; C=full context. Coverage configurations, not quality levels.",
        "Production modules/defaults unchanged; candidate metadata runs remain padded context, not direct-positive episodes.",
        "Regression references: 434 candidates, 17 stable views, 6 post-scroll; selected A/B/C=17/28/54.",
        "Similarity: <=2 pairs/relationship/video; unchanged raw MAD<=5 and changed residual>10 are diagnostic regression bounds only.",
    ]
    try:
        for name in SOURCES:
            _video(name, rows, summaries, similarities, checks, lines, totals, unique)
        baseline = [r for r in summaries if r["configuration"] == "A_ticket_minimal"]
        checks.append(
            (
                "ALL",
                "reference totals",
                (
                    totals["candidates"],
                    sum(r["stable_views"] for r in baseline),
                    sum(r["post_scroll_views"] for r in baseline),
                )
                == (434, 17, 6),
            )
        )
        checks.append(
            (
                "ALL",
                "four required similarity relationships exercised",
                {r["relationship"] for r in similarities}
                == {
                    "same_stable_span",
                    "adjacent_during_scroll",
                    "same_ticket_across_map_pan",
                    "true_ticket_replacement",
                },
            )
        )
        totals["unique_physical_frames"] = len(unique)
        lines.append("\nTOTAL materialization=" + str(dict(totals)))
        for config in CONFIGS:
            group = [r for r in summaries if r["configuration"] == config]
            sums = {
                key: sum(r[key] for r in group)
                for key in group[0]
                if key
                not in (
                    "video_name",
                    "investigation_index",
                    "configuration",
                    "reduction_percent",
                )
            }
            sums["reduction_percent"] = 100 * (
                1 - sums["selected_count"] / sums["candidate_count"]
            )
            lines.append(f"TOTAL {config}: {sums}")
        lines.extend(
            [
                "\nBounded similarity results:",
                *(str(r) for r in similarities),
                "\nExisting contact sheets: data/debug/map_motion/representative_selection/ (no new images needed).",
                "Limits: mostly canonical 1920x1080 10FPS; CFR/index-FPS contract; VFR/backend diversity unvalidated.",
                "Stable spans do not certify loading completeness; live values and return positions are not semantically collapsed.",
                "Alignment can alias on large/repetitive shifts; activity midpoints need not show final/extreme map state.",
                "No OCR, VLM, embeddings, semantic classifier, map-image similarity or Phase 6 work. Representative/manual validation, not a labeled accuracy benchmark.",
            ]
        )
    except (ValueError, TypeError, OSError, RuntimeError, AssertionError) as error:
        checks.append(("ALL", f"STOP: {type(error).__name__}: {error}", False))
    lines.extend(
        f"{'PASS' if ok else 'FAIL'}: {source}: {label}" for source, label, ok in checks
    )
    passed = bool(checks) and all(ok for _, _, ok in checks)
    lines.append(f"CHECKS: {sum(ok for _, _, ok in checks)}/{len(checks)}")
    lines.append(f"OVERALL: {'PASS' if passed else 'FAIL'}")
    _csv("phase5_validation.csv", rows)
    _csv("phase5_selection_summary.csv", summaries)
    _csv("phase5_similarity_checks.csv", similarities)
    report = OUTPUT / "phase5_validation_report.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        "\n".join(
            line
            for line in lines
            if line.startswith(("TOTAL", "FAIL", "CHECKS", "OVERALL"))
        ),
        flush=True,
    )
    print(f"Report: {report}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
