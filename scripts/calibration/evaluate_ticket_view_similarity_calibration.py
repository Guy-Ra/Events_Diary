"""Phase 5.2a ticket-view similarity and section-diversity calibration only.

Rebuild real Phase 3/4 timelines through shared evaluation preparation, then call
the production Phase 5.1 materializer and its bounded frame resolver. Original
candidate objects/timestamps and every materialized result remain intact. This
tool creates diagnostic comparison pairs, never representative-frame selections.
Use only UIConfig.ticket_scroll_body mapped through the existing layout utility.

Compare every pair WITHIN each stable span to characterize its variation (not all
candidates against all candidates), consecutive view boundaries/confirmation-time
probes, adjacent dense source frames with existing positive scroll evidence, and
explicit real-candidate landmarks for pan, confirmed replacement and loading.
Landmarks organize manual controls and never generate new candidate timestamps.
Stable identities include Investigation ownership; inclusive boundaries can also
carry active-scroll context. Pair categories overlap and are not ground-truth
semantic section labels. Duplicated physical frames/candidates are not removed.

Raw grayscale MAD/CPR reuse production APIs, with their existing CPR threshold 10.
Phase correlation uses the same fresh float32 crops and Hann-window concept as
ticket_scroll.py. dx/dy are A-to-B estimates, response an uncalibrated diagnostic.
Retain every finite estimate without a response/motion gate. Report primary
vertical-only B-to-A alignment by -dy and diagnostic XY alignment by (-dx,-dy).
Warp a float support image identically with INTER_LINEAR, constant-zero borders;
only support==1 pixels enter residuals, as in map_motion.py. Report overlap relative
to the entire body crop. This is geometric support, not measured shared text or
semantic coverage. Non-finite/no-overlap fits have unavailable aligned metrics.

Phase correlation can alias large shifts and latch onto fixed/repeated template
elements; high response or low residual does not prove correct content matching.
No more complex registration, optical flow, embeddings or extra dependency is
introduced. No structural metric is added: native phase/pixel diagnostics suffice
to test this limited hypothesis, without implying semantic completeness.

Keep scalar rows and grayscale body crops, not a full frame per candidate. Retain
only the few full frames needed for pan-control MAD and at most eight diagnostic
tiles. Write one report, pair CSV and two small contact sheets. No production
similarity model, thresholds, enum, grouping, ranking, filtering, deduplication,
OCR, VLM, normalization, final frame selection, Phase 5.2b or Phase 5.3 is implemented.
"""

import csv
import math
import sys
from collections import Counter
from dataclasses import asdict
from itertools import combinations, pairwise
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np

from events_diary.change_metrics import compute_changed_pixel_ratio, compute_mad
from events_diary.config import UIConfig
from events_diary.evidence_candidate_timeline import build_evidence_candidate_timeline
from events_diary.frame_materialization import (
    VideoFrameResolver,
    materialize_evidence_candidates,
)
from events_diary.layout import map_roi_to_frame
from scripts.evaluation.evaluate_ticket_stable_views import _build_contexts

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/debug/map_motion/ticket_view_similarity_calibration"
SOURCES = ("ticket_scroll", "combined_interaction", "investigation_switch")
METRICS = ("raw_mad", "raw_cpr", "dx", "dy", "phase_response", "aligned_mad",
           "aligned_cpr", "overlap_fraction", "xy_aligned_mad", "xy_aligned_cpr",
           "xy_overlap_fraction", "full_frame_mad")


def _stats(values):
    """Report raw min/quantiles/max without fitting a similarity threshold."""
    values = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    if not len(values):
        return "n=0"
    p10, median, p90 = np.percentile(values, (10, 50, 90))
    return (f"n={len(values)} min={values.min():.6g} P10={p10:.6g} median={median:.6g} "
            f"P90={p90:.6g} max={values.max():.6g}")


def _aligned(a, b, dx, dy):
    """Inverse-warp B and measure only pixels with full linear source support."""
    size = (a.shape[1], a.shape[0])
    matrix = np.array([[1., 0., -dx], [0., 1., -dy]])
    warped = cv2.warpAffine(b, matrix, size, flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    support = cv2.warpAffine(np.ones(b.shape, dtype=np.float32), matrix, size,
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = support == 1
    overlap = float(mask.mean())
    if not mask.any():
        return None, None, overlap, warped, mask
    pixels_a, pixels_b = a[mask].reshape(-1, 1), warped[mask].reshape(-1, 1)
    return compute_mad(pixels_a, pixels_b), compute_changed_pixel_ratio(pixels_a, pixels_b), overlap, warped, mask


def _comparison(a, b):
    """Return raw/phase/aligned diagnostics without accepting or rejecting a fit."""
    window = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(a.astype(np.float32), b.astype(np.float32), window)
    row = {"raw_mad": compute_mad(a, b), "raw_cpr": compute_changed_pixel_ratio(a, b),
           "dx": float(dx), "dy": float(dy), "phase_response": float(response),
           "aligned_mad": None, "aligned_cpr": None, "overlap_fraction": None,
           "xy_aligned_mad": None, "xy_aligned_cpr": None, "xy_overlap_fraction": None}
    warped, mask = None, None
    if math.isfinite(dx) and math.isfinite(dy):
        mad, cpr, overlap, warped, mask = _aligned(a, b, 0, dy)
        row.update(aligned_mad=mad, aligned_cpr=cpr, overlap_fraction=overlap)
        mad, cpr, overlap, _, _ = _aligned(a, b, dx, dy)
        row.update(xy_aligned_mad=mad, xy_aligned_cpr=cpr, xy_overlap_fraction=overlap)
    return row, warped, mask


def _prepare(name):
    """Materialize unchanged real requests and retain body crops/scalars for analysis."""
    metadata, anomalies, contexts, phase3 = _build_contexts(name, include_phase3=True)
    if anomalies:
        raise RuntimeError(f"Unexpected lifecycle anomalies: {anomalies}")
    config = UIConfig()
    candidates = tuple(c for window, plan, views, _ in contexts
                       for c in build_evidence_candidate_timeline(window=window, sampling_plan=plan, stable_views=views))
    snapshot = tuple(asdict(c) for c in candidates)
    full_times = (8., 9.7, 12.) if name == "combined_interaction" else ()
    crops, full, records = {}, {}, []
    with VideoFrameResolver(metadata.path, metadata=metadata) as resolver:
        results = materialize_evidence_candidates(video_path=metadata.path, candidates=candidates,
                                                   ui_config=config, resolver=resolver)
        if len(results) != len(candidates):
            raise AssertionError("Materializer lost candidates")
        roi = map_roi_to_frame(config.ticket_scroll_body, config, metadata.width, metadata.height)
        for index, result in enumerate(results):
            if result.frame_index is None:
                raise RuntimeError(f"Unresolved candidate {name} {index}")
            physical = result.frame_index
            need_full = any(math.isclose(result.requested_timestamp_seconds, t, rel_tol=0, abs_tol=1e-9) for t in full_times)
            if physical not in crops or (need_full and physical not in full):
                handle = resolver.resolve(result.requested_timestamp_seconds)
                if handle.image is None:
                    raise RuntimeError("Failed independent image access")
                crops[physical] = cv2.cvtColor(roi.crop(handle.image), cv2.COLOR_BGR2GRAY)
                if need_full:
                    full[physical] = handle.image.copy()
            records.append({"index": index, "result": result, "candidate": result.evidence_candidate})
    if snapshot != tuple(asdict(c) for c in candidates):
        raise AssertionError("Semantic candidates mutated")
    return metadata, contexts, phase3, records, crops, full


def _make_pairs(name, contexts, phase3, records):
    """Construct reproducible diagnostic probes, never selected evidence representatives."""
    pairs, stable_groups = [], {}
    for window, _, views, _ in contexts:
        for view_index, view in enumerate(views):
            group = [r for r in records if r["candidate"].investigation_index == window.investigation_index
                     and r["candidate"].stable_view_index == view_index]
            stable_groups[(window.investigation_index, view_index)] = (view, group)
            pairs.extend(("same_stable_span", "all_within_span", a, b) for a, b in combinations(group, 2))
        for left, right in pairwise(range(len(views))):
            if not views[right].follows_scroll:
                continue
            _, ga = stable_groups[(window.investigation_index, left)]
            _, gb = stable_groups[(window.investigation_index, right)]
            if ga and gb:
                pairs.append(("post_scroll_stable_views", "boundary_edges", ga[-1], gb[0]))
                a = min(ga, key=lambda r: abs(r["candidate"].timestamp_seconds-views[left].stable_confirmed_time_seconds))
                b = min(gb, key=lambda r: abs(r["candidate"].timestamp_seconds-views[right].stable_confirmed_time_seconds))
                pairs.append(("post_scroll_stable_views", "confirmation_neighborhood", a, b))
        owned = [r for r in records if r["candidate"].investigation_index == window.investigation_index]
        for a, b in pairwise(owned):
            ia, ib = a["result"].frame_index, b["result"].frame_index
            if (ib == ia+1 and all("dense_sampling" in r["candidate"].sources for r in (a, b))
                    and phase3["signals"][ia].ticket_scroll_detected):
                pairs.append(("adjacent_during_scroll", "existing_positive_pair", a, b))
    for event in phase3["events"]:
        if event.kind.value != "switch":
            continue
        time = event.candidate_time_seconds
        before = [r for r in records if r["candidate"].timestamp_seconds < time and r["candidate"].stable_view_index is not None]
        after = [r for r in records if r["candidate"].timestamp_seconds >= time and r["candidate"].stable_view_index is not None]
        if before and after:
            a = max(before, key=lambda r: r["candidate"].timestamp_seconds)
            b = min(after, key=lambda r: r["candidate"].timestamp_seconds)
            pairs.append(("true_ticket_replacement", "boundary", a, b))
            incoming_view, incoming = stable_groups[(b["candidate"].investigation_index, b["candidate"].stable_view_index)]
            settled = min(incoming, key=lambda r: abs(r["candidate"].timestamp_seconds-incoming_view.stable_confirmed_time_seconds))
            pairs.append(("true_ticket_replacement", "confirmation_neighborhood", a, settled))

    def landmark(time, investigation):
        """Find an existing exact-tolerance request; never synthesize a timestamp."""
        found = [r for r in records if r["candidate"].investigation_index == investigation and
                 math.isclose(r["candidate"].timestamp_seconds, time, rel_tol=0, abs_tol=1e-9)]
        if len(found) != 1:
            raise ValueError(f"Missing/ambiguous actual candidate {name} I{investigation} t{time}")
        return found[0]

    if name == "combined_interaction":
        for a, b in ((8., 9.7), (9.7, 12.), (8., 12.)):
            pairs.append(("same_ticket_across_map_pan", "manual_pan_control", landmark(a, 0), landmark(b, 0)))
        for a, ai, b, bi in ((14.5, 0, 14.6, 0), (14.6, 0, 14.7, 1),
                              (14.5, 0, 14.7, 1), (14.7, 1, 15.2, 1)):
            pairs.append(("loading_transition", "manual_combined_landmark", landmark(a, ai), landmark(b, bi)))
    elif name == "investigation_switch":
        for a, ai, b, bi in ((1.5, 0, 1.6, 0), (1.6, 0, 1.7, 0), (1.7, 0, 2.2, 0),
                              (4.7, 0, 4.8, 0), (4.8, 0, 4.9, 1)):
            pairs.append(("loading_transition", "manual_switch_landmark", landmark(a, ai), landmark(b, bi)))
    return pairs, stable_groups


def _tile(row, a, b, warped, mask):
    """Render one bounded diagnostic example; display scaling does not enter metrics."""
    tile = np.full((350, 704, 3), 245, dtype=np.uint8)
    diff = np.zeros_like(a) if warped is None else cv2.absdiff(a, warped)
    if mask is not None:
        diff[~mask] = 0
    arrays = (a, b, b if warped is None else warped, diff)
    for i, (label, image) in enumerate(zip(("A", "B", "B aligned vertical", "supported abs diff"), arrays)):
        tile[66:349, i*176:i*176+168] = cv2.cvtColor(cv2.resize(image, (168, 283)), cv2.COLOR_GRAY2BGR)
        cv2.putText(tile, label, (i*176+2, 60), cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 0, 0), 1)
    captions = [f"{row['video_name']} {row['category']} {row['timestamp_a']:.2f}->{row['timestamp_b']:.2f}s",
                f"MAD raw/aligned {row['raw_mad']:.2f}/{row['aligned_mad']}; dy {row['dy']:.2f}; response {row['phase_response']:.3f}",
                f"overlap {row['overlap_fraction']}; full MAD {row['full_frame_mad']}"]
    for y, text in zip((13, 29, 44), captions):
        cv2.putText(tile, text, (2, y), cv2.FONT_HERSHEY_SIMPLEX, .36, (0, 0, 0), 1, cv2.LINE_AA)
    return tile


def _evaluate(name):
    """Measure all diagnostic pairs, reporting category and per-span variation."""
    metadata, contexts, phase3, records, crops, full = _prepare(name)
    pairs, stable_groups = _make_pairs(name, contexts, phase3, records)
    rows, tiles, examples = [], [], set()
    for category, variant, ra, rb in pairs:
        ca, cb = ra["candidate"], rb["candidate"]
        ma, mb = ra["result"], rb["result"]
        a, b = crops[ma.frame_index], crops[mb.frame_index]
        metrics, warped, mask = _comparison(a, b)
        row = {"video_name": metadata.path.name, "investigation_index": ca.investigation_index,
            "investigation_index_b": cb.investigation_index, "candidate_index_a": ra["index"], "candidate_index_b": rb["index"],
            "category": category, "pair_variant": variant, "timestamp_a": ca.timestamp_seconds, "timestamp_b": cb.timestamp_seconds,
            "actual_timestamp_a": ma.actual_timestamp_seconds, "actual_timestamp_b": mb.actual_timestamp_seconds,
            "frame_index_a": ma.frame_index, "frame_index_b": mb.frame_index,
            "stable_view_index_a": ca.stable_view_index, "stable_view_index_b": cb.stable_view_index,
            "same_stable_view": ca.investigation_index == cb.investigation_index and ca.stable_view_index is not None
                                and ca.stable_view_index == cb.stable_view_index,
            "follows_scroll_a": ca.stable_view_follows_scroll, "follows_scroll_b": cb.stable_view_follows_scroll,
            "scroll_active_a": bool(ma.frame_index and phase3["signals"][ma.frame_index-1].ticket_scroll_detected),
            "scroll_active_b": bool(mb.frame_index and phase3["signals"][mb.frame_index-1].ticket_scroll_detected),
            "replacement_boundary": any(e.kind.value == "switch" and ca.timestamp_seconds < e.candidate_time_seconds
                                         <= cb.timestamp_seconds for e in phase3["events"]),
            "sharpness_a": ma.ticket_sharpness_laplacian_variance, "sharpness_b": mb.ticket_sharpness_laplacian_variance,
            "brightness_a": ma.ticket_brightness_mean, "brightness_b": mb.ticket_brightness_mean,
            "contrast_a": ma.ticket_contrast_std, "contrast_b": mb.ticket_contrast_std,
            **metrics, "full_frame_mad": None, "body_height": a.shape[0]}
        if category == "same_ticket_across_map_pan":
            row["full_frame_mad"] = compute_mad(full[ma.frame_index], full[mb.frame_index])
        rows.append(row)
        # These are named diagnostic examples, not evidence selected for downstream roles.
        example_key = category
        if category == "post_scroll_stable_views":
            example_key += f"_{ca.stable_view_index}"
        allowed = ((name == "ticket_scroll" and (
            (category == "same_stable_span" and ca.stable_view_index == 2)
            or (category == "post_scroll_stable_views" and ca.stable_view_index in (0, 2) and variant == "boundary_edges")
            or category == "adjacent_during_scroll"))
            or (name == "combined_interaction" and (
                category in ("same_ticket_across_map_pan", "loading_transition")
                or (category == "true_ticket_replacement" and variant == "confirmation_neighborhood")
                or (category == "post_scroll_stable_views" and ca.stable_view_index == 1 and variant == "boundary_edges"))))
        if allowed and example_key not in examples:
            examples.add(example_key)
            if len(tiles) < 4:
                tiles.append(_tile(row, a, b, warped, mask))
    lines = [(f"\n{name}: {metadata.width}x{metadata.height}, {metadata.fps:g} FPS; "
              f"candidates={len(records)} unique_frames={len(crops)}; pairs={len(rows)}"),
             f"  Category counts={dict(Counter(r['category'] for r in rows))}"]
    _group_report(rows, lines)
    for (investigation, view_index), (view, group) in stable_groups.items():
        measured = [r for r in rows if r["category"] == "same_stable_span" and r["investigation_index"] == investigation
                    and r["stable_view_index_a"] == view_index]
        lines.append(f"  Stable I{investigation}/V{view_index}: start={view.stable_start_time_seconds}; "
                     f"confirmed={view.stable_confirmed_time_seconds}; end={view.stable_end_time_seconds}; "
                     f"follows_scroll={view.follows_scroll}; candidates={len(group)}; all within-span pairs={len(measured)}")
        for key in ("raw_mad", "raw_cpr", "dy", "phase_response", "aligned_mad", "overlap_fraction"):
            lines.append(f"    {key}: {_stats([r[key] for r in measured])}")
    lines.append("  Non-within-span pair details:")
    lines.extend("    " + str(r) for r in rows if r["category"] != "same_stable_span")
    n = len(records)
    lines.append(f"  Complexity: all N(N-1)/2={n*(n-1)//2}; diagnostic pairs={len(rows)} (categories can repeat physical pairs).")
    print(f"{name}: {len(records)} materialized candidates, {len(rows)} comparison pairs", flush=True)
    return rows, lines, tiles, len(records)


def _group_report(rows, lines):
    """Summarize categories, including worsening alignments without rejecting them."""
    for category in sorted({r["category"] for r in rows}):
        group = [r for r in rows if r["category"] == category]
        lines.append(f"  Category {category}: n={len(group)}")
        for key in METRICS:
            lines.append(f"    {key}: {_stats([r[key] for r in group])}")
        supported = [r for r in group if r["aligned_mad"] is not None]
        lines.append(f"    vertical alignment: improves={sum(r['aligned_mad'] < r['raw_mad'] for r in supported)} "
                     f"worsens={sum(r['aligned_mad'] > r['raw_mad'] for r in supported)} "
                     f"unavailable={len(group)-len(supported)}; no response/overlap threshold fitted.")


def _findings(rows, lines):
    """State empirical comparisons without fitting a classifier or hiding failed fits."""
    groups = {category: [r for r in rows if r["category"] == category]
              for category in {r["category"] for r in rows}}
    lines.append("\nEmpirical category comparisons:")
    for left, right in (("same_stable_span", "post_scroll_stable_views"),
                        ("same_stable_span", "true_ticket_replacement"),
                        ("post_scroll_stable_views", "true_ticket_replacement"),
                        ("loading_transition", "true_ticket_replacement")):
        for key in ("raw_mad", "raw_cpr", "aligned_mad", "aligned_cpr", "phase_response", "overlap_fraction"):
            a = [r[key] for r in groups[left] if r[key] is not None and math.isfinite(r[key])]
            b = [r[key] for r in groups[right] if r[key] is not None and math.isfinite(r[key])]
            if a and b:
                intersects = max(min(a), min(b)) <= min(max(a), max(b))
                lines.append(f"  {left} vs {right}, {key}: observed ranges overlap={intersects}; "
                             f"ranges={min(a):.6g}-{max(a):.6g} / {min(b):.6g}-{max(b):.6g}. "
                             "A range gap in this small sample is not a calibrated threshold.")
    for row in groups["same_ticket_across_map_pan"]:
        lines.append(f"  Pan control {row['timestamp_a']:g}->{row['timestamp_b']:g}s: "
                     f"full-frame MAD={row['full_frame_mad']:.6g}; body MAD={row['raw_mad']:.6g}.")
    for row in groups["post_scroll_stable_views"]:
        lines.append(f"  Post-scroll I{row['investigation_index']} V{row['stable_view_index_a']}->V{row['stable_view_index_b']} "
                     f"{row['pair_variant']} ({row['timestamp_a']:g}->{row['timestamp_b']:g}s): "
                     f"dy={row['dy']:.6g}; response={row['phase_response']:.6g}; geometric overlap={row['overlap_fraction']}; "
                     f"MAD raw/aligned={row['raw_mad']:.6g}/{row['aligned_mad']}. "
                     "Validate matched rows visually before claiming shared/new section content.")


def main():
    """Write calibration evidence and stop on decode/API failures, never tune production."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lines = ["Phase 5.2a ticket-view similarity / section diversity calibration",
             "Real production Phase3/4/5.1 outputs; all candidate records retained; no representative chosen.",
             "Raw CPR threshold=10 (existing default). dx/dy A-to-B; primary inverse warp vertical-only, diagnostic XY warp also reported.",
             "Only fully supported overlap enters aligned metrics. Geometric overlap does NOT establish shared semantic content.",
             "Raw and aligned metrics use different pixel domains; lower aligned MAD can partly reflect cropping and is not proof of correct registration.",
             "Phase response is not calibrated confidence. No additional structural metric/dependency introduced."]
    rows, sheets, counts = [], [], []
    try:
        for name in SOURCES:
            measured, section, tiles, count = _evaluate(name)
            rows.extend(measured)
            lines.extend(section)
            counts.append(count)
            sheets.extend(tiles)
    except (ValueError, TypeError, OSError, RuntimeError, AssertionError) as error:
        lines.append(f"STOP: {type(error).__name__}: {error}; no production changes.")
        (OUTPUT / "ticket_view_similarity_calibration_report.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
        print(lines[-1], flush=True)
        return 1
    lines.append(f"\nTOTAL candidates={sum(counts)}; pairs={len(rows)}; category counts={dict(Counter(r['category'] for r in rows))}")
    _group_report(rows, lines)
    _findings(rows, lines)
    lines += ["\nInterpretation limits and proposed Phase5.2b boundary:",
        "Inspect actual pair distributions and contact sheets before interpreting shift/overlap as shared content. Finite phase fits are not accepted production alignments.",
        "Prefer a measurement-only comparison record initially: pair references; dx/dy; phase response; geometric overlap; raw MAD/CPR; aligned MAD/CPR; explicit availability/convention.",
        "dx remains useful to expose nonvertical/wrong fits. Candidate indexes and original references supply ownership/time/quality; avoid redundant image storage.",
        "No clean semantic near_duplicate/overlapping_section/distinct_section classifier or threshold is approved by this exploratory set. Loading/replacement/template aliases require context.",
        "An overlapping-section hypothesis needs visibly correct alignment and newly exposed-content evidence, not just geometric overlap. Do not force an enum or final grouping.",
        "Production should not compare all candidates pairwise: 202 candidates imply 20301 pairs; 360 imply 64620; 1000 imply 499500.",
        "Prefer temporal neighbors, stable-span variation checks and neighboring post-scroll spans; later compare to retained coverage context only after representative policy is separately approved.",
        "Exhaustive within-span comparisons here characterize variation only; they are not a proposed production quadratic algorithm or a chosen representative.",
        "Remaining gaps: small canonical 10FPS environment, ground-truth row overlap/section labels, large/opposing scrolls, fixed/repetitive templates, loading variants, other layouts/themes/FPS/codecs.",
        "No OCR, embeddings, VLM, ranking, filtering, semantic deduplication, representative selection, production similarity API, Phase5.2b or Phase5.3."]
    report = OUTPUT / "ticket_view_similarity_calibration_report.txt"
    report.write_text("\n".join(lines)+"\n", encoding="utf-8")
    with (OUTPUT / "ticket_view_similarity_pairs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # At most eight diagnostic examples across at most two small sheets.
    chosen = sheets[:4] + sheets[4:8]
    for index in range(0, len(chosen), 4):
        if not cv2.imwrite(str(OUTPUT / f"comparison_examples_{index//4+1}.png"), np.vstack(chosen[index:index+4])):
            raise OSError("Cannot save diagnostic sheet")
    print(lines[-1], flush=True)
    print(f"Report: {report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
