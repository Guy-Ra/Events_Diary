"""Phase 5.1b real-video validation of production physical-frame materialization.

Rebuild actual Phase 4 timelines using shared preparation and composition, then
call only the new production resolver/materializer for frame resolution/quality.
Preserve every candidate, including overlaps and nominal-duration anchors. Use
one bounded, context-managed resolver per video across all Investigation windows.

Sequential reference hashes/time metadata and scalar calibration measurements
independently audit physical identity and raw ticket-body metric parity. The
optional earlier calibration CSV is checked against identical physical indexes;
it is evidence, never an input to production resolution. Video-end anchors must
now resolve to the last frame with explicit adjustment status. Unresolved results
are recorded, never removed; validation fails if current real candidates remain
unresolved or any identity/endpoint/quality invariant fails.

Write a report and one per-candidate CSV only; save no images. Distributions are
candidate-weighted with duplicates retained and overlapping semantic groups.
Moving-scroll tags use the preceding pair's existing evidence; stable association
includes inclusive boundaries and does not imply semantic usefulness. Raw metrics
remain diagnostics; no normalization, thresholds, ranking, filtering, duplicate
removal, representative selection, OCR, similarity or Phase 5.2 occurs.
"""

import csv
import math
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

# Allow direct execution and explicit imports across script directories.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from events_diary.config import UIConfig
from events_diary.evidence_candidate_timeline import build_evidence_candidate_timeline
from events_diary.frame_materialization import (
    VideoFrameResolver,
    materialize_evidence_candidates,
)
from scripts.calibration.evaluate_frame_materialization_calibration import (
    _hash,
    _reference,
)
from scripts.evaluation.evaluate_ticket_stable_views import _build_contexts

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/debug/map_motion/frame_materialization"
CALIBRATION = ROOT / "data/debug/map_motion/frame_materialization_calibration/frame_materialization_calibration.csv"
SOURCES = ("combined_interaction", "ticket_scroll", "investigation_switch", "map_pan", "map_zoom")


def _distribution(values):
    """Describe raw values without normalizing or fitting thresholds."""
    values = np.asarray(values, dtype=float)
    if not len(values):
        return "n=0"
    p10, median, p90 = np.percentile(values, (10, 50, 90))
    return (f"n={len(values)} min={values.min():.9g} P10={p10:.9g} median={median:.9g} "
            f"P90={p90:.9g} max={values.max():.9g} mean={values.mean():.9g}")


def _quality_groups(rows, lines):
    """Report overlapping upstream conditions without inventing quality labels."""
    for tag in ("all", "stable_ticket", "moving_scroll", "loading_dimming", "map_pan_context", "map_zoom_context"):
        group = [r for r in rows if r["resolution_status"] != "unresolved"
                 and (tag == "all" or tag in r["condition_tags"].split(";"))]
        lines.append(f"  Quality {tag}: records={len(group)}")
        for key in ("ticket_sharpness", "ticket_brightness", "ticket_contrast"):
            lines.append(f"    {key}: {_distribution([r[key] for r in group])}")


def _evaluate(name, calibration):
    """Validate all real candidates of one video and retain scalar audit evidence."""
    config = UIConfig()
    metadata, anomalies, contexts, phase3 = _build_contexts(name, include_phase3=True)
    if anomalies:
        raise AssertionError(f"Unexpected lifecycle anomalies: {anomalies}")
    _, reference = _reference(metadata)
    candidates = tuple(c for window, plan, views, _ in contexts
                       for c in build_evidence_candidate_timeline(window=window, sampling_plan=plan, stable_views=views))
    snapshot = tuple(asdict(c) for c in candidates)
    rows, endpoints, checks = [], [], []
    with VideoFrameResolver(metadata.path, metadata=metadata) as resolver:
        results = materialize_evidence_candidates(video_path=metadata.path, candidates=candidates,
                                                   ui_config=config, resolver=resolver)
        checks.append(("exact result count and original candidate identity/order",
                       len(results) == len(candidates) and all(r.evidence_candidate is c for r, c in zip(results, candidates))))
        # Capture materialization cache counters before separate image-access auditing.
        stats = {"max_frames": resolver.max_frames, "cache_hits": resolver.cache_hits,
                 "cache_misses": resolver.cache_misses, "decode_attempts": resolver.decode_attempts,
                 "retained_frames": resolver.cache_size}
        local_indexes = Counter()
        for result in results:
            candidate = result.evidence_candidate
            investigation = candidate.investigation_index
            candidate_index = local_indexes[investigation]
            local_indexes[investigation] += 1
            row = {"video_name": metadata.path.name, "investigation_index": investigation,
                "candidate_index": candidate_index, "requested_timestamp": result.requested_timestamp_seconds,
                "resolution_status": result.resolution_status, "frame_index": result.frame_index,
                "actual_timestamp": result.actual_timestamp_seconds, "timestamp_error": result.timestamp_error_seconds,
                "ticket_sharpness": result.ticket_sharpness_laplacian_variance,
                "ticket_brightness": result.ticket_brightness_mean, "ticket_contrast": result.ticket_contrast_std,
                "condition_tags": ""}
            tags = []
            if candidate.stable_view_index is not None:
                tags.append("stable_ticket")
            if result.frame_index is not None:
                index = result.frame_index
                handle = resolver.resolve(candidate.timestamp_seconds)
                checks.append((f"I{investigation} C{candidate_index} decoded pixels/index/time match sequential reference",
                               handle.frame_index == index and _hash(handle.image) == reference[index][1]
                               and math.isclose(result.actual_timestamp_seconds, reference[index][0], rel_tol=0, abs_tol=1e-8)))
                if index and phase3["signals"][index-1].ticket_scroll_detected and "dense_sampling" in candidate.sources:
                    tags.append("moving_scroll")
                if name == "combined_interaction" and index == 146:
                    tags.append("loading_dimming")
                if "map_translation" in candidate.dense_reasons:
                    tags.append("map_pan_context")
                if "map_scale_change" in candidate.dense_reasons:
                    tags.append("map_zoom_context")
                previous = calibration.get((metadata.path.name, investigation, candidate_index))
                if previous and previous["frame_index"]:
                    checks.append((f"I{investigation} C{candidate_index} raw body metrics match calibration",
                        int(previous["frame_index"]) == index and all(math.isclose(row[current], float(previous[old]), rel_tol=1e-12, abs_tol=1e-9)
                        for current, old in (("ticket_sharpness", "ticket_body_sharpness"),
                                             ("ticket_brightness", "ticket_body_brightness"),
                                             ("ticket_contrast", "ticket_body_contrast")))))
            if result.requested_timestamp_seconds == metadata.duration_seconds:
                endpoints.append((result.requested_timestamp_seconds, result.actual_timestamp_seconds,
                                  result.frame_index, result.resolution_status))
                checks.append(("nominal-duration anchor now resolves to last physical frame",
                    result.resolution_status == "video_end_adjusted" and result.frame_index == metadata.frame_count-1
                    and result.actual_timestamp_seconds == (metadata.frame_count-1)/metadata.fps))
            row["condition_tags"] = ";".join(tags)
            rows.append(row)
    checks.append(("input semantic records unchanged", snapshot == tuple(asdict(c) for c in candidates)))
    checks.append(("all current candidates resolve", all(r["resolution_status"] != "unresolved" for r in rows)))
    checks.append(("one video-end adjustment; cache bounded", len(endpoints) == 1 and stats["retained_frames"] <= stats["max_frames"]))
    counts = Counter(r["resolution_status"] for r in rows)
    physical = Counter(r["frame_index"] for r in rows if r["frame_index"] is not None)
    lines = [f"\n{name}: {metadata.width}x{metadata.height}, {metadata.fps:g} FPS, {metadata.frame_count} frames; actual lifecycle",
             (f"  candidates={len(rows)} statuses={dict(counts)} unique_physical_frames={len(physical)} "
              f"duplicate_groups={sum(n>1 for n in physical.values())}; largest_group={max(physical.values(), default=0)}"),
             f"  Materialization cache (before independent image auditing)={stats}", f"  Endpoints={endpoints}",
             "  ALL absolute timestamp error: " + _distribution([r["timestamp_error"] for r in rows if r["frame_index"] is not None]),
             "  Normal nearest absolute timestamp error: " + _distribution([r["timestamp_error"] for r in rows if r["resolution_status"] == "resolved"])]
    _quality_groups(rows, lines)
    lines.extend(f"  {'PASS' if passed else 'FAIL'}: {label}" for label, passed in checks)
    passed = all(ok for _, ok in checks)
    print(f"{name}: {'PASS' if passed else 'FAIL'}; statuses={dict(counts)}; cache_hits={stats['cache_hits']}", flush=True)
    return passed, rows, lines, stats


def main():
    """Write reproducible production materialization evidence, stopping on any failure."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    calibration = {}
    if CALIBRATION.is_file():
        with CALIBRATION.open(encoding="utf-8", newline="") as stream:
            calibration = {(r["video_name"], int(r["investigation_index"]), int(r["candidate_index"])): r
                           for r in csv.DictReader(stream)}
    lines = ["Phase 5.1b production materialization validation",
             "Nearest CFR index; halfway -> later frame; exact duration -> last decodable frame, explicitly adjusted.",
             "Candidate records retain no images. Cache bound=16 frames; raw mapped ticket_scroll_body metrics only.",
             f"Earlier calibration comparison available={bool(calibration)}; no production thresholds tuned.",
             "Quality groups overlap; dense reasons denote expanded context, not necessarily instantaneous motion."]
    rows, stats, overall = [], [], True
    try:
        for name in SOURCES:
            ok, measured, section, cache = _evaluate(name, calibration)
            rows.extend(measured)
            lines.extend(section)
            stats.append(cache)
            if not ok:
                overall = False
                break
    except (ValueError, TypeError, RuntimeError, OSError, AssertionError) as error:
        lines.append(f"STOP: {type(error).__name__}: {error}")
        overall = False
    physical = Counter((r["video_name"], r["frame_index"]) for r in rows if r["frame_index"] is not None)
    lines += [(f"\nTOTAL candidates={len(rows)} statuses={dict(Counter(r['resolution_status'] for r in rows))} "
              f"unique_physical_frames={len(physical)} duplicate_groups={sum(n>1 for n in physical.values())} "
              f"largest_group={max(physical.values(), default=0)}"),
              (f"TOTAL cache hits={sum(s['cache_hits'] for s in stats)} misses={sum(s['cache_misses'] for s in stats)} "
               f"decode_attempts={sum(s['decode_attempts'] for s in stats)}"),
              "TOTAL error: " + _distribution([r["timestamp_error"] for r in rows if r["frame_index"] is not None])]
    _quality_groups(rows, lines)
    lines += ["Raw quality reproduces calibration diagnostics, not semantic usefulness. Moving scroll may be sharper than stable views.",
              "CFR/index-FPS convention is explicit; VFR, other backends/codecs and layouts need wider validation.",
              "No ranking/filtering/selection, duplicate removal, normalization, OCR, similarity or Phase5.2.",
              f"OVERALL: {'PASS' if overall else 'FAIL'}"]
    report = OUTPUT / "frame_materialization_report.txt"
    report.write_text("\n".join(lines)+"\n", encoding="utf-8")
    if rows:
        with (OUTPUT / "frame_materialization.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print("\n".join(line for line in lines if line.lstrip().startswith(("TOTAL", "OVERALL", "STOP"))))
    print(f"Report: {report}", flush=True)
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
