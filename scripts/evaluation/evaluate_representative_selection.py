"""Seven-video validation of production Phase 5.3b reference selection.

Use the existing lifecycle/window/plan/stable pipeline, Phase 4 composition and
Phase 5.1 materialization, then call the pure production selector for A (ticket
minimal), B (default map/route context), C (full optional context). Indexes in the
production result are local to each supplied Investigation sequence; CSV rows
also give the original video-wide candidate index for debugging. Every original
candidate remains unchanged. No policy is reimplemented or tuned here.

Coverage counts distinguish primary stable reasons from other references that
happen to share a stable-view annotation. Map/scroll burst counts refer to padded
candidate-metadata runs at source cadence, exactly as production specifies, not
the prior calibration's direct-positive pairs or human interactions. Report both
all runs and runs with owned resolved members, explicit selected indexes/times,
fallbacks, physical duplicates, optional boundaries and known manual controls.
The previous pair's actual positive scroll evidence is additional debug context,
never a selector input or a second production burst classifier.

Known controls are manual landmarks, not a loading classifier. Assert that all
stable views remain covered and owned, no available post-confirmation choice is
replaced by a pre-confirmation frame, and the current settled/control examples
behave correctly. Full context deliberately retains transition boundaries with
context reasons; they must never masquerade as primary stable evidence. A small
union of selected references and manual controls is rendered once per important
Investigation with full-frame thumbnails and native ticket-body crops. No image
similarity, OCR, VLM, semantic section labels or production settings are added.
"""

import csv
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from textwrap import wrap

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np

from events_diary.config import UIConfig
from events_diary.evidence_candidate_timeline import build_evidence_candidate_timeline
from events_diary.frame_materialization import (
    VideoFrameResolver,
    materialize_evidence_candidates,
)
from events_diary.layout import map_roi_to_frame
from events_diary.representative_selection import (
    _activity_bursts,
    _owned,
    select_investigation_representatives,
)
from scripts.evaluation.evaluate_ticket_stable_views import _build_contexts

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/debug/map_motion/representative_selection"
SOURCES = ("combined_interaction", "ticket_scroll", "investigation_switch",
           "investigation_started_active", "investigation_close_reopen", "map_pan", "map_zoom")
FOCUS = {"combined_interaction", "ticket_scroll", "investigation_switch", "map_pan", "map_zoom"}
CONFIGS = {"A_ticket_minimal": (False, False, False), "B_default": (True, False, False), "C_full_context": (True, True, True)}
MAP_REASONS = ("route_visual_change", "map_translation", "map_scale_change")


def _controls(name, item):
    """Name explicit manual landmarks without deriving image semantics."""
    index = item.frame_index
    inv = item.evidence_candidate.investigation_index
    labels = []
    if name == "combined_interaction":
        labels = {146: ["spinner_dimming"], 147: ["mixed_incoming_ticket"], 152: ["settled_incoming_ticket"]}.get(index, [])
    elif name == "investigation_switch":
        if index == 16:
            labels.append("sliding_open")
        if index == 49 and inv == 0:
            labels.append("outgoing_boundary_shows_incoming")
    elif name == "investigation_started_active" and inv == 1 and index in (19, 20):
        labels.append("previous_ticket_pre_context")
    return tuple(labels)


def _caption(tile, text, y):
    """Draw readable bounded debug labels without changing measured source pixels."""
    for line in wrap(text, width=57):
        cv2.putText(tile, line, (4, y), cv2.FONT_HERSHEY_SIMPLEX, .35, (15, 15, 15), 1, cv2.LINE_AA)
        y += 14
    return y


def _sheets(name, inv, labels, candidates, materialized, resolver, config):
    """Render each selected reference once across configurations, with control probes."""
    indexes = sorted(labels)
    paths = []
    for page in range(0, len(indexes), 8):
        tiles = []
        for i in indexes[page:page+8]:
            c, m = candidates[i], materialized[i]
            handle = resolver.resolve(c.timestamp_seconds)
            if handle.image is None or handle.frame_index != m.frame_index:
                raise RuntimeError("Contact-sheet physical reference mismatch")
            frame = handle.image
            roi = map_roi_to_frame(config.ticket_scroll_body, config, frame.shape[1], frame.shape[0])
            tile = np.full((862, 384, 3), 245, np.uint8)
            y = _caption(tile, f"{name} I{inv} C{i} t={c.timestamp_seconds:.2f}s V{c.stable_view_index}", 14)
            _caption(tile, " | ".join(labels[i]), y)
            tile[80:296] = cv2.resize(frame, (384, 216))
            tile[296:862, 24:360] = roi.crop(frame)
            tiles.append(tile)
        sheet = np.full((862*((len(tiles)+3)//4), 1536, 3), 245, np.uint8)
        for j, tile in enumerate(tiles):
            sheet[(j//4)*862:(j//4+1)*862, (j%4)*384:(j%4+1)*384] = tile
        path = OUTPUT / f"{name}_i{inv}_{page//8+1:02d}.png"
        if not cv2.imwrite(str(path), sheet):
            raise OSError(f"Cannot save {path}")
        paths.append(path.name)
    return paths


def _evaluate(name):
    """Validate the actual production selector for three independent configurations."""
    metadata, anomalies, contexts, phase3 = _build_contexts(name, include_phase3=True)
    if anomalies:
        raise RuntimeError(f"Unexpected lifecycle anomalies: {anomalies}")
    config = UIConfig()
    groups = [build_evidence_candidate_timeline(window=w, sampling_plan=p, stable_views=v) for w, p, v, _ in contexts]
    all_candidates = tuple(c for group in groups for c in group)
    snapshot = tuple(asdict(c) for c in all_candidates)
    rows, summaries, lines, images = [], [], [], []
    offset = 0
    with VideoFrameResolver(metadata.path, metadata=metadata) as resolver:
        all_materialized = materialize_evidence_candidates(video_path=metadata.path, candidates=all_candidates,
                                                          ui_config=config, resolver=resolver)
        materialized_snapshot = tuple(asdict(m) for m in all_materialized)
        for (window, plan, views, _), candidates in zip(contexts, groups, strict=True):
            inv = window.investigation_index
            materialized = all_materialized[offset:offset+len(candidates)]
            map_bursts = _activity_bursts(candidates, window, plan, MAP_REASONS)
            scroll_bursts = _activity_bursts(candidates, window, plan, ("ticket_scroll",))
            eligible_map = [b for b in map_bursts if any(_owned(window, materialized[i]) for i in b)]
            eligible_scroll = [b for b in scroll_bursts if any(_owned(window, materialized[i]) for i in b)]
            labels = defaultdict(list)
            for configuration, (map_on, scroll_on, boundary_on) in CONFIGS.items():
                result = select_investigation_representatives(window=window, sampling_plan=plan, candidates=candidates,
                    materialized_candidates=materialized, stable_views=views, include_map_route_evidence=map_on,
                    include_scroll_provenance=scroll_on, include_boundary_context=boundary_on)
                assert result == select_investigation_representatives(window=window, sampling_plan=plan, candidates=candidates,
                    materialized_candidates=materialized, stable_views=views, include_map_route_evidence=map_on,
                    include_scroll_provenance=scroll_on, include_boundary_context=boundary_on)
                selected = {s.candidate_index: s.selection_reasons for s in result.selections}
                primary = [i for i in selected if "stable_ticket" in selected[i]]
                pre = [i for i in primary if materialized[i].actual_timestamp_seconds < views[candidates[i].stable_view_index].stable_confirmed_time_seconds-1e-9
                       or candidates[i].timestamp_seconds < views[candidates[i].stable_view_index].stable_confirmed_time_seconds-1e-9]
                assert len(primary) == len(views)
                assert len({candidates[i].stable_view_index for i in primary}) == len(views)
                assert all(_owned(window, materialized[i]) for i in primary)
                assert not pre, "Current recordings unexpectedly required confirmation fallback"
                assert all(materialized[i].resolution_status != "unresolved" for i in selected)
                assert not any("investigation_boundary" in rs for rs in selected.values()) or boundary_on
                assert not any("scroll_provenance" in rs for rs in selected.values()) or scroll_on
                map_selected = [i for i in selected if "map_activity" in selected[i] or "route_activity" in selected[i]]
                scroll_selected = [i for i in selected if "scroll_provenance" in selected[i]]
                boundary_selected = [i for i in selected if "investigation_boundary" in selected[i]]
                map_coverage = sum(any(i in selected and ("map_activity" in selected[i] or "route_activity" in selected[i]) for i in b) for b in eligible_map)
                scroll_coverage = sum(any(i in scroll_selected for i in b) for b in eligible_scroll)
                boundary_available = [i for i, c in enumerate(candidates) if materialized[i].resolution_status != "unresolved"
                                      and any(a in c.sources for a in ("investigation_start", "investigation_end"))]
                assert map_coverage == (len(eligible_map) if map_on else 0)
                assert scroll_coverage == (len(eligible_scroll) if scroll_on else 0)
                assert len(map_selected) == (len(eligible_map) if map_on else 0)
                assert len(scroll_selected) == (len(eligible_scroll) if scroll_on else 0)
                assert len(boundary_selected) == (len(boundary_available) if boundary_on else 0)
                for i, reasons in selected.items():
                    c, m = candidates[i], materialized[i]
                    controls = _controls(name, m)
                    if "stable_ticket" in reasons:
                        assert not any(x != "settled_incoming_ticket" for x in controls), (name, inv, i, controls)
                    frame = m.frame_index
                    active = bool(frame and phase3["signals"][frame-1].ticket_scroll_detected)
                    if name == "ticket_scroll" and configuration != "C_full_context":
                        assert not active
                    row = {"video_name": metadata.path.name, "investigation_index": inv, "configuration": configuration,
                        "candidate_index": i, "video_candidate_index": offset+i, "timestamp": c.timestamp_seconds,
                        "actual_timestamp": m.actual_timestamp_seconds, "frame_index": frame,
                        "selection_reasons": ";".join(reasons), "stable_view_index": c.stable_view_index,
                        "primary_stable": "stable_ticket" in reasons, "post_scroll_stable": "post_scroll_stable" in reasons,
                        "resolution_status": m.resolution_status, "preceding_pair_scroll_positive": active,
                        "known_manual_controls": ";".join(controls)}
                    rows.append(row)
                    labels[i].append(configuration[0]+":"+"/".join(reasons))
                if name == "combined_interaction" and inv == 1:
                    assert any(materialized[i].frame_index == 152 for i in primary), "Settled 15.2s frame not primary"
                physical = Counter(materialized[i].frame_index for i in selected)
                summary = {"video_name": metadata.path.name, "investigation_index": inv, "configuration": configuration,
                    "candidate_count": len(candidates), "stable_views": len(views), "selected_count": len(selected),
                    "reduction_percent": 100*(1-len(selected)/len(candidates)),
                    "selected_indexes": tuple(selected), "selected_times": tuple(candidates[i].timestamp_seconds for i in selected),
                    "stable_views_covered": len(primary), "post_scroll_views": sum(v.follows_scroll for v in views),
                    "post_scroll_covered": sum("post_scroll_stable" in selected[i] for i in primary),
                    "map_metadata_bursts": len(map_bursts), "map_representable_bursts": len(eligible_map), "map_covered": map_coverage,
                    "scroll_metadata_bursts": len(scroll_bursts), "scroll_representable_bursts": len(eligible_scroll), "scroll_covered": scroll_coverage,
                    "boundary_available": len(boundary_available), "boundary_selected": len(boundary_selected),
                    "duplicate_physical_frames_selected": sum(n-1 for n in physical.values()), "unresolved_selected": 0,
                    "pre_confirmation_primary": len(pre), "primary_fallbacks": sum("stable_confirmation_fallback" in selected[i] for i in primary),
                    "manual_controls_selected": tuple((i, _controls(name, materialized[i]), selected[i]) for i in selected if _controls(name, materialized[i]))}
                summaries.append(summary)
                lines.append(str(summary))
            for i, item in enumerate(materialized):
                if _controls(name, item):
                    labels[i].append("manual audit: "+"/".join(_controls(name, item)))
            if name in FOCUS:
                images.extend(_sheets(name, inv, labels, candidates, materialized, resolver, config))
            offset += len(candidates)
        assert snapshot == tuple(asdict(c) for c in all_candidates)
        assert materialized_snapshot == tuple(asdict(m) for m in all_materialized)
    print(f"{name}: PASS; {len(all_candidates)} candidates, {len(contexts)} Investigations", flush=True)
    return rows, summaries, lines, images


def main():
    """Write real validation evidence and stop on an invariant failure without tuning."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = OUTPUT / "representative_selection_report.txt"
    lines = ["Phase 5.3b production representative selection",
        "A=ticket minimal; B=default map/route context; C=map + optional scroll/boundary context. These are coverage configurations.",
        "Selector is pure; existing Phase3/4 and Phase5.1 outputs are reused unchanged. No image-similarity decision, classifier or tuning.",
        "Indexes are local to each Investigation input; video_candidate_index gives the original flattened index.",
        "Primary physical ownership=[core start,core end), with stable association and inclusive stable span; video-end adjustment can remain inside core.",
        "Both physical/request times prefer post-confirmation; nearest physical time, then quality only within 1e-9s temporal ties.",
        "Activity bursts use padded/merged candidate dense reasons, split by nonmatching requests or gaps >1/source FPS +1e-9s.",
        "One owned resolved physical midpoint per burst. This is metadata activity context, NOT direct-positive-pair or human-interaction reconstruction.",
        "Map/route context is independent of stable ticket evidence; a context-only frame is not implicitly a primary OCR representative.",
        "Different candidate indexes resolving to the same physical frame are retained as separate references and counted explicitly."]
    rows, summaries, images = [], [], []
    try:
        for name in SOURCES:
            measured, summary, section, sheets = _evaluate(name)
            rows.extend(measured)
            summaries.extend(summary)
            lines.extend(section)
            images.extend(sheets)
        for configuration in CONFIGS:
            group = [r for r in summaries if r["configuration"] == configuration]
            original = sum(r["candidate_count"] for r in group)
            selected = sum(r["selected_count"] for r in group)
            lines.append(f"TOTAL {configuration}: selected={selected}/{original}, reduction={100*(1-selected/original):.6g}%; "
                f"stable={sum(r['stable_views_covered'] for r in group)}/{sum(r['stable_views'] for r in group)}; "
                f"post_scroll={sum(r['post_scroll_covered'] for r in group)}/{sum(r['post_scroll_views'] for r in group)}; "
                f"map_bursts={sum(r['map_covered'] for r in group)}/{sum(r['map_representable_bursts'] for r in group)}; "
                f"scroll_bursts={sum(r['scroll_covered'] for r in group)}/{sum(r['scroll_representable_bursts'] for r in group)}; "
                f"boundaries={sum(r['boundary_selected'] for r in group)}/{sum(r['boundary_available'] for r in group)}; "
                f"physical_duplicates={sum(r['duplicate_physical_frames_selected'] for r in group)}; unresolved=0; pre_confirmation=0")
        lines.extend(["Known-control interpretation:",
            "Combined incoming primary is settled 15.2s; dimmed 14.6s and mixed 14.7s are not primary. Full boundary context may intentionally include 14.7s.",
            "Switch sliding/incoming boundary frames are not outgoing primary ticket representatives; started-active pre-context never becomes incoming primary.",
            "Every stable/post-scroll view remains; optional activity midpoint references do not replace those slots.",
            "Midpoint map evidence bounds dense volume but does not guarantee final/extreme scale state or semantic route ownership. No map-image similarity is implemented.",
            "Padded reason runs can bridge direct motion pauses or include quiet frames. Their coverage counts must not be compared as if they were Phase5.3a direct-positive burst counts.",
            "No current fallback/unresolved selection or ownership inconsistency occurred. Stable spans still do not certify loading completeness in general.",
            "Representative manual footage, not a selection-accuracy benchmark; other layouts/FPS, missing stable views and timing edge cases need wider real validation.",
            "No production Phase4/5.1/5.2 behavior changed; no semantic section/loading/duplicate classifier, OCR/VLM, Enum or Phase5.4 closure.",
            "Contact sheets: "+str(images), "OVERALL: PASS"])
        for name, table in (("representative_selection.csv", rows), ("representative_selection_summary.csv", summaries)):
            with (OUTPUT / name).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(table[0]))
                writer.writeheader()
                writer.writerows(table)
    except (ValueError, TypeError, OSError, RuntimeError, AssertionError) as error:
        lines.append(f"STOP: {type(error).__name__}: {error}; no production repair or threshold tuning")
        report.write_text("\n".join(lines)+"\n", encoding="utf-8")
        print(lines[-1], flush=True)
        return 1
    report.write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("\n".join(line for line in lines if line.startswith(("TOTAL", "OVERALL"))), flush=True)
    print(f"Report: {report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
