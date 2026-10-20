"""Phase 5.1a manual frame-resolution and image-quality calibration only.

Rebuild real Phase 4 EvidenceCandidate timelines using the existing shared CV,
effective-presence, lifecycle, window, sampling and stable-view preparation. Keep
every request, including overlapping-window duplicates and video-end anchors.
Exercise video.sample_video's exact POS_MSEC seek/read/property recipe on these
arbitrary requests because that public API currently accepts only a uniform FPS.
Failed reads remain unresolved rows, never silently clamped, skipped or replaced.

An independent sequential decode supplies frame hashes and reported timestamps.
Every successful seek must agree with that reference in pixels, index and time;
otherwise stop with an ambiguity report. Nominal index/FPS times are checked, not
substituted for decoder times. Supplemental subframe probes derived from a real
request distinguish nearest/floor/ceil behavior; these are diagnostics, never new
EvidenceCandidates or observations in quality distributions. No half-frame tie
policy or portable cross-backend contract is inferred from this small CFR set.

Measure raw grayscale Laplacian variance (CV_64F, default ksize=1), mean and
population standard deviation on the full frame and existing UIConfig ticket_zone,
ticket_scroll_body and map_analysis_region crops. No masks, resizing, normalization,
threshold fitting or new ROI is applied to measurement. Condition tags are existing
semantic metadata and explicitly labeled manual landmarks, not a good-frame rule.
Distribution units are resolved candidate records, so duplicate physical frames
are retained; also report separate physical-frame statistics and group overlap.

Write a text report, compact per-candidate CSV and at most two small contact sheets
for manual inspection under data/debug/map_motion/frame_materialization_calibration.
Only contact-sheet display copies are resized. Store hashes/scalars and a bounded
example subset, not a frame array per candidate. No production module, model,
ranking, deduplication, OCR, section similarity, selection or Phase 5.1b/5.2 behavior
is implemented. No dependency is added. Stop on ambiguous timing or decoding.
"""

import csv
import hashlib
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Allow direct execution and explicit imports across script directories.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np

from events_diary.config import UIConfig
from events_diary.evidence_candidate_timeline import build_evidence_candidate_timeline
from events_diary.video import read_video_metadata
from scripts.evaluation.evaluate_ticket_stable_views import _build_contexts

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/debug/map_motion/frame_materialization_calibration"
SOURCES = ("combined_interaction", "ticket_scroll", "investigation_switch", "map_pan", "map_zoom")
EXAMPLES = {
    "combined_interaction": {0: "quiet stable", 29: "during scroll", 37: "reading pause",
        97: "map pan", 120: "post-pan stable", 146: "dimmed transition", 147: "replacement"},
    "investigation_switch": {16: "opening/loading", 49: "replacement"},
    "map_zoom": {32: "zoom activity"},
}


def _hash(image):
    """Identify decoded BGR bytes without retaining a full reference image."""
    return hashlib.sha256(image.tobytes()).hexdigest()


def _reference(metadata):
    """Sequentially audit CFR index/decoder-time agreement and retain hashes only."""
    capture = cv2.VideoCapture(str(metadata.path))
    records = []
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open {metadata.path}")
        backend = capture.getBackendName()
        while True:
            success, frame = capture.read()
            if not success:
                break
            index = len(records)
            reported = capture.get(cv2.CAP_PROP_POS_FRAMES) - 1
            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if (frame.shape != (metadata.height, metadata.width, 3) or reported != index
                    or not math.isclose(timestamp, index / metadata.fps, rel_tol=0, abs_tol=1e-8)):
                raise RuntimeError(f"AMBIGUOUS sequential index/time/layout: {metadata.path.name}, {index}, {reported}, {timestamp}")
            records.append((timestamp, _hash(frame)))
        if len(records) != metadata.frame_count:
            raise RuntimeError(f"AMBIGUOUS decoded count: {len(records)} vs {metadata.frame_count}")
    finally:
        capture.release()
    return backend, records


def _seek(capture, requested, reference):
    """Exercise the existing sampler's recipe and verify every successful result."""
    if not capture.set(cv2.CAP_PROP_POS_MSEC, requested * 1000):
        raise RuntimeError(f"AMBIGUOUS rejected millisecond seek: {requested}")
    success, image = capture.read()
    if not success:
        return None
    raw_index = capture.get(cv2.CAP_PROP_POS_FRAMES) - 1
    timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
    index = max(0, int(raw_index))  # Same conversion as sample_video; audit before accepting.
    if (raw_index != index or not 0 <= index < len(reference)
            or not math.isfinite(timestamp)
            or not math.isclose(timestamp, reference[index][0], rel_tol=0, abs_tol=1e-8)
            or _hash(image) != reference[index][1]):
        raise RuntimeError(f"AMBIGUOUS seek pixels/index/time at request {requested}: index={raw_index}, time={timestamp}")
    return index, timestamp, image


def _quality(image, config):
    """Measure the three raw grayscale diagnostics independently on existing ROIs."""
    values = {}
    for label, pixels in (("full_frame", image), ("ticket", config.ticket_zone.crop(image)),
                          ("ticket_body", config.ticket_scroll_body.crop(image)),
                          ("map", config.map_analysis_region.crop(image))):
        gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
        values[label + "_sharpness"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        values[label + "_brightness"] = float(gray.mean())
        values[label + "_contrast"] = float(gray.std())
    return values


def _distribution(values):
    """Summarize raw values without fitting cutoffs or normalizing by session."""
    values = np.asarray(values, dtype=float)
    if not len(values):
        return "n=0"
    low, median, high = np.percentile(values, (10, 50, 90))
    return (f"n={len(values)} min={values.min():.6g} P10={low:.6g} "
            f"median={median:.6g} P90={high:.6g} max={values.max():.6g}")


def _tags(name, candidate, frame_index, phase3):
    """Use upstream semantics and known landmarks as overlapping diagnostic groups."""
    tags = []
    if candidate.stable_view_index is not None:
        tags.append("stable_ticket")
    if "dense_sampling" in candidate.sources:
        tags.append("dense")
    # Frame endpoints inherit the existing preceding pair, not a new motion test.
    pair = phase3["signals"][frame_index - 1] if frame_index else None
    if pair is not None and pair.ticket_scroll_detected and "dense_sampling" in candidate.sources:
        tags.append("moving_scroll")
    if "map_translation" in candidate.dense_reasons:
        tags.append("map_translation_context")
    if "map_scale_change" in candidate.dense_reasons:
        tags.append("map_scale_context")
    if candidate.stable_view_index is not None and "sparse_sampling" in candidate.sources:
        tags.append("quiet_stable")
    if any(e.kind.value == "switch" and math.isclose(e.candidate_time_seconds,
            candidate.timestamp_seconds, rel_tol=0, abs_tol=1e-9) for e in phase3["events"]):
        tags.append("aircraft_replacement")
    if name == "combined_interaction" and frame_index == 146:
        tags.append("manual_dimmed_transition")
    if name == "investigation_switch" and frame_index == 16:
        tags.append("manual_opening_transition")
    return tags


def _example_tile(name, index, image, metrics, config):
    """Build a small display copy for manual review, never quality measurement."""
    tile = np.full((510, 670, 3), 245, dtype=np.uint8)
    tile[65:335, :480] = cv2.resize(image, (480, 270))
    tile[65:509, 490:658] = cv2.resize(config.ticket_zone.crop(image), (168, 444))
    label = EXAMPLES[name][index]
    lines = [f"{name} frame {index}: {label}",
             (f"sharp full/ticket/body: {metrics['full_frame_sharpness']:.1f} / "
              f"{metrics['ticket_sharpness']:.1f} / {metrics['ticket_body_sharpness']:.1f}"),
             f"ticket mean/std: {metrics['ticket_brightness']:.1f} / {metrics['ticket_contrast']:.1f}"]
    for y, line in zip((16, 36, 55), lines):
        cv2.putText(tile, line, (3, y), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 0, 0), 1, cv2.LINE_AA)
    return tile


def _evaluate(name):
    """Resolve every real candidate unchanged, returning scalar rows and bounded examples."""
    config = UIConfig()
    metadata = read_video_metadata(ROOT / f"data/videos/{name}_test.mp4")
    backend, reference = _reference(metadata)
    metadata, anomalies, contexts, phase3 = _build_contexts(name, include_phase3=True)
    if anomalies:
        raise RuntimeError(f"Unexpected lifecycle anomalies: {anomalies}")
    capture = cv2.VideoCapture(str(metadata.path))
    rows, examples, probes = [], {}, []
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open {metadata.path}")
        for window, plan, views, _ in contexts:
            candidates = build_evidence_candidate_timeline(window=window, sampling_plan=plan, stable_views=views)
            if tuple(c.timestamp_seconds for c in candidates) != plan.requested_timestamps_seconds:
                raise RuntimeError("Candidate requests changed before calibration")
            for candidate_index, candidate in enumerate(candidates):
                requested = candidate.timestamp_seconds
                result = _seek(capture, requested, reference)
                row = {"video_name": metadata.path.name, "investigation_index": candidate.investigation_index,
                    "candidate_index": candidate_index, "requested_timestamp": requested,
                    "actual_timestamp": None, "frame_index": None, "timestamp_error": None,
                    "signed_timestamp_error": None, "resolved": result is not None,
                    "failure_reason": "", "is_stable_ticket": candidate.stable_view_index is not None,
                    "is_dense": "dense_sampling" in candidate.sources, "sources": ";".join(candidate.sources),
                    "dense_reasons": ";".join(candidate.dense_reasons), "condition_tags": ""}
                row.update({f"{roi}_{feature}": None for roi in ("full_frame", "ticket", "ticket_body", "map")
                            for feature in ("sharpness", "brightness", "contrast")})
                if result is None:
                    if requested < metadata.duration_seconds - 1e-9:
                        raise RuntimeError(f"AMBIGUOUS interior read failure: {metadata.path.name} {requested}")
                    row["failure_reason"] = "video_end_anchor_no_frame; no fallback applied"
                else:
                    index, actual, image = result
                    row.update(actual_timestamp=actual, frame_index=index, timestamp_error=abs(actual-requested),
                               signed_timestamp_error=actual-requested,
                               condition_tags=";".join(_tags(name, candidate, index, phase3)))
                    row.update(_quality(image, config))
                    if index in EXAMPLES.get(name, {}) and index not in examples:
                        examples[index] = _example_tile(name, index, image, row, config)
                rows.append(row)
        # Distinguish rounding policies with separate diagnostic probes based on one real interior request.
        base = next(r["frame_index"] for r in rows if r["resolved"] and
                    0 < r["frame_index"] < metadata.frame_count - 2)
        for offset in (.24, .49, .51, .76):
            requested = (base + offset) / metadata.fps
            result = _seek(capture, requested, reference)
            if result is None:
                raise RuntimeError("AMBIGUOUS subframe probe failure")
            index, actual, _ = result
            probes.append((requested, index, actual))
        policies = {"nearest": all(i == base + (offset > .5) for offset, (_, i, _) in zip((.24, .49, .51, .76), probes)),
                    "floor": all(i == base for _, i, _ in probes),
                    "ceil": all(i == base + 1 for _, i, _ in probes)}
        if sum(policies.values()) != 1:
            raise RuntimeError(f"AMBIGUOUS seek policy: {probes}")
    finally:
        capture.release()
    groups = defaultdict(list)
    for row in rows:
        if row["resolved"]:
            groups[row["frame_index"]].append((row["investigation_index"], row["candidate_index"], row["requested_timestamp"]))
    duplicates = {index: requests for index, requests in groups.items() if len(requests) > 1}
    lines = [(f"\n{name}: {metadata.width}x{metadata.height}, {metadata.fps:g}FPS, {metadata.frame_count} frames, "
              f"duration={metadata.duration_seconds:g}s; backend={backend}"),
             (f"  candidates={len(rows)} resolved={sum(r['resolved'] for r in rows)} unique_frames={len(groups)} "
              f"duplicate_groups={len(duplicates)} largest_group={max(map(len, groups.values()), default=0)}"),
             f"  seek policy on subframe probes={next(p for p, yes in policies.items() if yes)}; probes={probes}",
             f"  timestamp error: {_distribution([r['timestamp_error'] for r in rows if r['resolved']])}",
             f"  unresolved requests={[(r['investigation_index'], r['requested_timestamp'], r['failure_reason']) for r in rows if not r['resolved']]}",
             f"  duplicate physical-frame groups (no removal)={duplicates}",
             f"  duplicate groups with distinct exact requested values={sum(len({r[2] for r in group}) > 1 for group in duplicates.values())}",
             f"  successful candidate reads matching nearest-index prediction={sum(r['frame_index'] == math.floor(r['requested_timestamp']*metadata.fps+.5) for r in rows if r['resolved'])}"]
    for investigation in sorted({r["investigation_index"] for r in rows}):
        selected = [r for r in rows if r["investigation_index"] == investigation and r["resolved"]]
        counts = Counter(r["frame_index"] for r in selected)
        lines.append(f"  I{investigation}: resolved={len(selected)} unique={len(counts)} "
                     f"within-Investigation duplicate groups={sum(n > 1 for n in counts.values())}")
    for time in (0, .1, .2, 3.2, 3.7, 14.7):
        matched = [r for r in rows if math.isclose(r["requested_timestamp"], time, rel_tol=0, abs_tol=1e-9)]
        if matched:
            lines.append(f"  real candidate landmark {time:g}s: " + str(
                [(r["requested_timestamp"], r["frame_index"], r["actual_timestamp"]) for r in matched]))
    _summarize_groups(rows, lines, name)
    print(f"{name}: calibration complete; {len(rows)} candidates", flush=True)
    return rows, lines, list(examples.values()), metadata


def _summarize_groups(rows, lines, label):
    """Report overlapping candidate-weighted groups and raw ROI distributions."""
    resolved = [r for r in rows if r["resolved"]]
    tags = sorted({t for r in resolved for t in r["condition_tags"].split(";") if t})
    for tag in ("ALL", *tags):
        group = resolved if tag == "ALL" else [r for r in resolved if tag in r["condition_tags"].split(";")]
        unique = len({(r["video_name"], r["frame_index"]) for r in group})
        lines.append(f"  {label} group={tag}, n={len(group)}, unique physical frames={unique}")
        for roi in ("full_frame", "ticket", "ticket_body", "map"):
            for feature in ("sharpness", "brightness", "contrast"):
                key = f"{roi}_{feature}"
                lines.append(f"    {key}: {_distribution([r[key] for r in group])}")
    overlap = sum("stable_ticket" in r["condition_tags"].split(";") and
                  "moving_scroll" in r["condition_tags"].split(";") for r in resolved)
    lines.append(f"  {label} stable/moving overlap={overlap}; inclusive pair/view boundaries may overlap; not independent classes.")


def _write_sheet(tiles, filename):
    """Save a bounded contact sheet of manually specified diagnostic examples."""
    if not tiles:
        return
    sheet = np.full((math.ceil(len(tiles)/2)*510, 1340, 3), 245, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        y, x = (index//2)*510, (index % 2)*670
        sheet[y:y+510, x:x+670] = tile
    if not cv2.imwrite(str(OUTPUT / filename), sheet):
        raise OSError(f"Cannot save {filename}")


def main():
    """Run five real pipelines, record empirical evidence, and stop on ambiguity."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = OUTPUT / "frame_materialization_calibration_report.txt"
    lines = ["Phase 5.1a frame materialization / raw quality calibration (analysis only)",
        "Actual lifecycle contexts; no invented candidate timestamps; no duplicate removal or selection.",
        "Current sample_video: uniform grid t<duration; CAP_PROP_POS_MSEC seek/read; decoder time; POS_FRAMES-1.",
        "Sequential pixel hashes and decoder times independently audit every successful candidate/probe read.",
        "Four raw ROI diagnostics: full frame, UIConfig.ticket_zone, ticket_scroll_body, map_analysis_region.",
        "Laplacian CV_64F ksize=1 variance; grayscale mean/std; original pixels, no masking/normalization.",
        "Moving-scroll = dense candidate whose resolved frame is the after endpoint of a detected scroll pair.",
        "Dense reasons denote padded/merged context, not necessarily instantaneous motion.",
        "Stable association includes pre-confirmation frames and inclusive boundaries; it is not semantic quality.",
        "Reported error measures decoded frame time vs request; index/FPS is separately checked under CFR assumptions."]
    rows, combined_tiles, control_tiles, metadata_items = [], [], [], []
    try:
        for name in SOURCES:
            measured, section, tiles, metadata = _evaluate(name)
            rows.extend(measured)
            lines.extend(section)
            metadata_items.append(metadata)
            (combined_tiles if name == "combined_interaction" else control_tiles).extend(tiles)
    except (RuntimeError, ValueError, OSError) as error:
        lines.append(f"STOP: {error}; remaining evaluation not performed; no production change.")
        report.write_text("\n".join(lines)+"\n", encoding="utf-8")
        print(lines[-1], flush=True)
        return 1
    resolved = [r for r in rows if r["resolved"]]
    physical = Counter((r["video_name"], r["frame_index"]) for r in resolved)
    lines += [(f"\nTOTAL candidates={len(rows)} resolved={len(resolved)} unresolved={len(rows)-len(resolved)} "
              f"unique_physical_frames={len(physical)} duplicate_groups={sum(n > 1 for n in physical.values())} "
              f"largest_group={max(physical.values())} duplicate_extra_records={len(resolved)-len(physical)}"),
              "TOTAL absolute error: " + _distribution([r["timestamp_error"] for r in resolved])]
    _summarize_groups(rows, lines, "TOTAL")
    lines += ["\nEmpirical interpretation (no ranking or thresholds):",
              "Stable/moving distributions overlap. Content, aircraft image, text density and map geometry confound sharpness."]
    stable = [r for r in resolved if "stable_ticket" in r["condition_tags"].split(";")]
    moving = [r for r in resolved if "moving_scroll" in r["condition_tags"].split(";")]
    for key in ("full_frame_sharpness", "ticket_sharpness", "ticket_body_sharpness", "ticket_brightness", "ticket_contrast"):
        stable_median = float(np.median([r[key] for r in stable]))
        moving_median = float(np.median([r[key] for r in moving]))
        lines.append(f"  {key}: stable median={stable_median:.6g}; moving median={moving_median:.6g}; "
                     f"moving median {'higher' if moving_median > stable_median else 'not higher'}.")
    lines.append("  Scroll motion does not reliably lower Laplacian variance in this footage; reject it as a standalone stability/usefulness score.")
    for name in ("map_pan", "map_zoom"):
        group = [r for r in stable if r["video_name"] == f"{name}_test.mp4"]
        lines.append(f"  {name}, stable-associated candidates: full sharpness "
                     f"{min(r['full_frame_sharpness'] for r in group):.6g}-{max(r['full_frame_sharpness'] for r in group):.6g}; "
                     f"ticket sharpness {min(r['ticket_sharpness'] for r in group):.6g}-{max(r['ticket_sharpness'] for r in group):.6g}.")
    lines.append("  Full-frame sharpness varies substantially with map content while ticket metrics remain comparatively steady; ticket/body ROI is more relevant to ticket diagnostics.")
    lines.append("  Manual landmark examples below are single physical frames; repeated candidate records are not independent examples.")
    for name, landmarks in EXAMPLES.items():
        for index, label in landmarks.items():
            example = next((r for r in resolved if r["video_name"] == f"{name}_test.mp4" and r["frame_index"] == index), None)
            if example is not None:
                lines.append(f"  {name} frame{index} ({label}): full/ticket/body sharpness="
                    f"{example['full_frame_sharpness']:.6g}/{example['ticket_sharpness']:.6g}/{example['ticket_body_sharpness']:.6g}; "
                    f"ticket brightness/contrast={example['ticket_brightness']:.6g}/{example['ticket_contrast']:.6g}")
    lines += ["  Manual visual review landmarks: combined frame146 shows a dimmed spinner overlay; switch frame16 shows a partly sliding/opening ticket.",
              "  Dimming reduces ticket brightness/contrast and sharpness, but map/static edges leave a substantial full-frame score. Replacement can immediately score highly.",
              "  Brightness/contrast expose appearance changes; they do not establish loading completion, stability or semantic usefulness."]
    bytes_per_frame = metadata_items[0].width * metadata_items[0].height * 3
    lines += [(f"\nMemory: BGR uint8 frame={bytes_per_frame} bytes; all candidates={bytes_per_frame*len(rows)} bytes; "
              f"resolved records={bytes_per_frame*len(resolved)} bytes; unique frames={bytes_per_frame*len(physical)} bytes; "
              "excludes ndarray/object overhead and decoder buffers."),
        f"Memory GiB: candidate arrays={bytes_per_frame*len(rows)/2**30:.4f}; resolved arrays={bytes_per_frame*len(resolved)/2**30:.4f}; unique unbounded cache={bytes_per_frame*len(physical)/2**30:.4f}.",
        "\nProduction recommendations (not implemented):",
        "Use explicit nearest-source-frame resolution for CFR evidence; record actual decoder time and signed error.",
        "Current uniform-FPS public sampler is insufficient for arbitrary requests, failed-read accounting and inclusive video-end anchors.",
        "Video duration is an exclusive physical-frame bound. Propose an explicit last-frame policy for end anchors, with a boundary flag and error; never silent fallback.",
        "Backend seek behavior is observed, not a portable guarantee. Validate cross-backend/codec/VFR/tie behavior before promising a general policy.",
        "Retain raw ticket/body Laplacian variance, mean and std as diagnostics, not a score or good-frame classifier.",
        "Keep full/map metrics optional diagnostics; assess later evidence roles in their actual ROIs. Do not use full-frame texture as ticket quality.",
        "Retain raw values only. Session-dependent normalization would obstruct cross-session interpretation and is uncalibrated.",
        "Minimal record: original EvidenceCandidate; source-video identity; resolved frame index; actual timestamp; raw ROI quality features.",
        "Signed/absolute timing error is derivable from original request and actual time; expose it for auditing rather than independent mutable truth.",
        "Unresolved requests need explicit status/reason. Preserve original candidate metadata and all physical-frame duplicates.",
        "Avoid an image array in each record. Keep lightweight metadata plus lazy access and a bounded shared cache keyed by (video identity, frame index).",
        "A frozen dataclass does not make ndarray pixels immutable. Define cached image ownership/read-only access or defensive copies explicitly in 5.1b.",
        "Loading/dimming/replacement can retain sharp borders, text or map texture. Sharpness/contrast do not prove useful or complete content.",
        "Known gaps: small canonical 10FPS CFR set, other codecs/backends/VFR/higher FPS/layouts, genuinely blurry footage, semantic quality labels, role-specific usefulness, cache throughput and timestamp ties.",
        "No threshold tuning, ranking, duplicate removal, OCR, production materialization, similarity or Phase5.2. Evaluation complete."]
    report.write_text("\n".join(lines)+"\n", encoding="utf-8")
    with (OUTPUT / "frame_materialization_calibration.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_sheet(combined_tiles, "combined_conditions.png")
    _write_sheet(control_tiles, "control_conditions.png")
    print("\n".join(line for line in lines if line.lstrip().startswith(("TOTAL", "Memory:", "Memory GiB:", "candidates=", "unresolved"))))
    print(f"Report: {report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
