"""Validate Phase 6.1b production OCR on the current seven-video dataset.

Run the unchanged production Investigation/window/plan/stable-view pipeline and
selector, materialize physical candidates, then call run_ocr_for_representatives.
Full optional selection context is requested to audit that map, route, scroll
provenance and boundary-only roles never receive OCR. The ticket targets are the
same primary stable representatives as default selection, checked explicitly.

Use only the approved RapidOCR CPU/color2x/full-ticket default. Compare original
crop pixels, raw ordered text/confidence/quadrilaterals and bundled model hashes
against the existing 6.1a V2/A runs. Reuse the seven human-verified annotation sets
for a regression check, without editing truth, recalibrating or repairing OCR.
The calibration scorer is an evaluator-only import, never a production dependency.

Outputs under data/debug/ocr/ticket_ocr contain per-detection raw results, compact
per-representative summaries, selection/skip audit and regression measurements.
Only numeric/string provenance is saved; no bulk image output, semantic pairing,
field parsing, VLM, fusion, multi-view merge or Phase 6.2 logic is implemented.
"""

import csv
import hashlib
import json
import statistics
import sys
import time
from dataclasses import asdict
from importlib.metadata import distribution
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from events_diary.config import UIConfig
from events_diary.evidence_candidate_timeline import build_evidence_candidate_timeline
from events_diary.frame_materialization import (
    VideoFrameResolver,
    materialize_evidence_candidates,
)
from events_diary.representative_selection import select_investigation_representatives
from events_diary.ticket_ocr import RapidOCREngine, run_ocr_for_representatives
from scripts.calibration.evaluate_ocr_engine_calibration import _load_truth, _score
from scripts.evaluation.evaluate_representative_selection import SOURCES, _controls
from scripts.evaluation.evaluate_ticket_stable_views import _build_contexts

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT/"data/debug/ocr/ticket_ocr"
CALIBRATION = ROOT/"data/debug/ocr/ocr_engine_calibration"


def _csv(path, rows):
    """Write one explicitly scoped debug table, keeping original text in UTF-8."""
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class MeasuredEngine(RapidOCREngine):
    """Measure actual production engine inference without changing its behavior."""

    def __init__(self):
        """Construct one CPU engine for all seven videos and retain call diagnostics."""
        started = time.perf_counter()
        super().__init__()
        self.initialization_ms = 1000*(time.perf_counter()-started)
        self.calls = []

    def run(self, image):
        """Time native inference and record processed input identity, never output repair."""
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        started = time.perf_counter()
        result = super().run(image)
        self.calls.append({"latency_ms": 1000*(time.perf_counter()-started), "processed_sha256": digest})
        return result


def _baseline():
    """Read the unchanged calibration artifacts needed for fixed-default regression."""
    manifest = json.loads((CALIBRATION/"input_manifest.json").read_text(encoding="utf-8"))
    samples = {s["sample_id"]: s for s in manifest["samples"] if s["kind"] == "stable"}
    native = {}
    for line in (CALIBRATION/"ocr_boxes.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        if (row["engine"], row["preprocessing_variant"], row["roi_strategy"]) == ("RapidOCR", "V2_upscale2x", "A"):
            native[row["sample_id"]] = row["boxes"]
    versions = json.loads((CALIBRATION/"engine_environment.json").read_text(encoding="utf-8"))
    model_dir = Path(distribution("rapidocr-onnxruntime").locate_file("rapidocr_onnxruntime/models"))
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in model_dir.glob("*.onnx")}
    if hashes != versions["RapidOCR"]["models"]:
        raise RuntimeError("Production bundled models differ from calibration; no silent recalibration.")
    return samples, native, hashes


def _regression(sample_id, result, image, baseline_sample, baseline_boxes, truth):
    """Compare exact source crop and raw detections, then score verified components."""
    roi = result.crop_roi
    pixels_match = hashlib.sha256(roi.crop(image).tobytes()).hexdigest() == baseline_sample["pixel_sha256"]
    old = [(b["text"], b["confidence"], b["bbox_source_frame"]) for b in baseline_boxes]
    new = [(d.text, d.confidence, [list(point) for point in d.source_polygon]) for d in result.detections]
    text_match = [r[0] for r in old] == [r[0] for r in new]
    confidence_match = len(old) == len(new) and all(a[1] == b[1] for a,b in zip(old,new,strict=True))
    geometry_match = len(old) == len(new) and all(np.array_equal(a[2],b[2]) for a,b in zip(old,new,strict=True))
    boxes = [{"text": d.text, "confidence": d.confidence,
        "bbox_full_ticket": [[x-roi.x, y-roi.y] for x,y in d.source_polygon]} for d in result.detections]
    verified = [r for r in truth if r["sample_id"] == sample_id]
    scores = _score(boxes, verified)
    return {"sample_id": sample_id, "crop_pixels_match": pixels_match, "ordered_text_match": text_match,
        "confidence_match": confidence_match, "source_geometry_match": geometry_match,
        "verified_components": len(verified), **{k: scores[k] for k in
        ("cer", "wer", "exact_match_rate", "exact_field_match_rate", "detection_recall")}}, scores


def main():
    """Evaluate real selected frames and write a complete report, including skips."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    samples, baseline_boxes, model_hashes = _baseline()
    truth = _load_truth()
    if len(truth) != 151 or len({r["sample_id"] for r in truth}) != 7:
        raise RuntimeError("Expected the existing seven human-verified calibration crops; do not substitute draft truth.")
    engine = MeasuredEngine()
    sessions = (engine._engine.text_det.infer.session, engine._engine.text_cls.infer.session, engine._engine.text_rec.session.session)
    providers = [s.get_providers() for s in sessions]
    if any(p != ["CPUExecutionProvider"] for p in providers):
        raise RuntimeError("Unexpected non-CPU provider")
    config = UIConfig()
    detections, summaries, audits, regressions, measurements = [], [], [], [], []
    lines = ["Phase 6.1b Production Ticket OCR real-video validation", engine.engine_name,
        f"Engine initialization={engine.initialization_ms:.2f}ms; providers={providers}; bundled model hashes={model_hashes}",
        "One engine across all videos. Selector C includes optional context only to verify role skips; OCR targets equal default selector B's ticket targets.",
        "Strategy A=(16,121,336,831) at1920x1080, mapped proportionally; BGR crop->cubic2x->RapidOCR. No alternate engine/filter.",
        "Coordinates: source polygon = processed polygon /2 + mapped integer crop origin; no rounding/clipping, no semantic reading order.",
        "latency_ms is native engine wall time, repeated per detection; batch wall time includes crop/upscale/resolve/mapping. Do not sum repeated detection-row latency."]
    for name in SOURCES:
        metadata, anomalies, contexts = _build_contexts(name)
        if anomalies:
            raise RuntimeError(f"Unexpected lifecycle anomalies: {anomalies}")
        lines.append(f"{name}: {asdict(metadata)}")
        with VideoFrameResolver(metadata.path, metadata=metadata) as resolver:
            for window, plan, views, _ in contexts:
                candidates = build_evidence_candidate_timeline(window=window, sampling_plan=plan, stable_views=views)
                materialized = materialize_evidence_candidates(video_path=metadata.path,
                    candidates=candidates, ui_config=config, resolver=resolver)
                inputs_before = (tuple(asdict(c) for c in candidates), tuple(asdict(m) for m in materialized))
                default = select_investigation_representatives(window=window, sampling_plan=plan,
                    candidates=candidates, materialized_candidates=materialized, stable_views=views)
                selected = select_investigation_representatives(window=window, sampling_plan=plan,
                    candidates=candidates, materialized_candidates=materialized, stable_views=views,
                    include_scroll_provenance=True, include_boundary_context=True)
                expected = {s.candidate_index for s in default.selections
                    if {"stable_ticket", "post_scroll_stable"}.intersection(s.selection_reasons)}
                first_call = len(engine.calls)
                hits = resolver.cache_hits
                started = time.perf_counter()
                results = run_ocr_for_representatives(video_path=metadata.path,
                    investigation_index=window.investigation_index, selections=selected.selections,
                    candidates=candidates, materialized_candidates=materialized, ui_config=config,
                    resolver=resolver, engine=engine)
                elapsed = 1000*(time.perf_counter()-started)
                if {r.candidate_index for r in results} != expected or len(results) != len(expected):
                    raise AssertionError("Context selection changed ticket targets or duplicate OCR occurred")
                if inputs_before != (tuple(asdict(c) for c in candidates), tuple(asdict(m) for m in materialized)):
                    raise AssertionError("OCR mutated upstream records")
                executed = {r.candidate_index for r in results}
                reasons = {s.candidate_index: s.selection_reasons for s in selected.selections}
                for selection in selected.selections:
                    i = selection.candidate_index
                    audits.append({"video_name": metadata.path.name, "investigation_index": window.investigation_index,
                        "candidate_index": i, "frame_index": materialized[i].frame_index,
                        "actual_timestamp": materialized[i].actual_timestamp_seconds,
                        "selection_reasons": ";".join(selection.selection_reasons), "ocr_executed": i in executed,
                        "skip_reason": "" if i in executed else "No stable_ticket/post_scroll_stable role",
                        "manual_landmarks": ";".join(_controls(name, materialized[i]))})
                for result, call in zip(results, engine.calls[first_call:], strict=True):
                    i = result.candidate_index
                    sid = f"{name}_i{window.investigation_index}_c{i}"
                    if sid not in samples or samples[sid]["materialized"]["frame_index"] != result.frame_index:
                        raise AssertionError(f"Unexpected Phase5 ticket target: {sid}")
                    frame = resolver.resolve(candidates[i].timestamp_seconds)
                    regression, scores = _regression(sid, result, frame.image, samples[sid], baseline_boxes[sid], truth)
                    regressions.append(regression)
                    measurements.extend(scores["lines"])
                    confidence = [d.confidence for d in result.detections if d.confidence is not None]
                    info = {"video_name": metadata.path.name, "investigation_index": result.investigation_index,
                        "candidate_index": i, "frame_index": result.frame_index, "actual_timestamp": result.actual_timestamp_seconds,
                        "selection_reasons": ";".join(reasons[i]), "resolution_status": result.resolution_status,
                        "engine_name": result.engine_name, "preprocessing_name": result.preprocessing_name,
                        "crop_geometry": json.dumps(asdict(result.crop_roi)), "upscale_factor": result.upscale_factor,
                        "detection_count": len(result.detections), "mean_confidence": statistics.mean(confidence) if confidence else None,
                        "min_confidence": min(confidence, default=None), "max_confidence": max(confidence, default=None), **call}
                    summaries.append({**info, "ocr_text": "\n".join(d.text for d in result.detections)})
                    for j, detection in enumerate(result.detections):
                        detections.append({**info, "detection_index": j, "text": detection.text,
                            "confidence": detection.confidence, "source_polygon": json.dumps(detection.source_polygon)})
                    if not result.detections:
                        detections.append({**info, "detection_index": None, "text": "", "confidence": None, "source_polygon": "[]"})
                lines.append(f"  I{window.investigation_index}: selected={len(selected.selections)}, OCR={len(results)}, skipped={len(selected.selections)-len(results)}, batch_wall_ms={elapsed:.2f}, cache_hits_gained={resolver.cache_hits-hits}")
        print(f"{name}: production OCR complete", flush=True)
    _csv(OUTPUT/"ticket_ocr_results.csv", detections)
    _csv(OUTPUT/"ticket_ocr_summary.csv", summaries)
    _csv(OUTPUT/"ticket_ocr_selection_audit.csv", audits)
    _csv(OUTPUT/"ticket_ocr_regression.csv", regressions)
    matches = all(all(r[k] for k in ("crop_pixels_match", "ordered_text_match", "confidence_match", "source_geometry_match")) for r in regressions)
    totals = {"representatives": len(summaries), "detections": sum(r["detection_count"] for r in summaries),
        "context_selections_skipped": sum(not r["ocr_executed"] for r in audits),
        "verified_components": len(measurements),
        "cer": sum(r["character_edits"] for r in measurements)/sum(r["characters"] for r in measurements),
        "wer": sum(r["word_edits"] for r in measurements)/sum(r["words"] for r in measurements),
        "exact_match_rate": statistics.mean(r["exact_match"] for r in measurements),
        "detection_recall": statistics.mean(r["detected"] for r in measurements),
        "median_inference_ms": statistics.median(r["latency_ms"] for r in summaries),
        "min_inference_ms": min(r["latency_ms"] for r in summaries),
        "max_inference_ms": max(r["latency_ms"] for r in summaries), "all_calibration_outputs_match": matches}
    lines += ["Totals="+json.dumps(totals), "Per-representative summary:", *(str(r) for r in summaries),
        "Selection/skip audit:", *(str(r) for r in audits), "Calibration regression:", *(str(r) for r in regressions),
        "All five ticket-scroll stable views processed independently; no merge/dedup or text correction.",
        "Combined: incoming15.2s must be primary;14.6s dimmed/14.7s boundary-only references must not execute OCR.",
        "Started-active second Investigation uses its owned selected ticket reference; previous-ticket pre-context is never promoted.",
        "Map pan/zoom: ticket-role references execute, map/route-only context is skipped.",
        "No unresolved result or geometry/provenance ambiguity is accepted. Failures propagate with no Tesseract fallback.",
        "Engine confidence is not frame validity or semantic correctness. Known time-suffix/column-merge limitations from6.1a remain.",
        "Phases1-5 code unchanged. No pairing, field parsing, column assignment, VLM/fusion, view assembly or Phase6.2 implementation."]
    (OUTPUT/"ticket_ocr_report.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    if len(summaries) != 17 or not matches:
        raise AssertionError("Production/calibration regression differs; measurements saved, no silent tuning")
    combined = [r for r in summaries if r["video_name"] == "combined_interaction_test.mp4"]
    if not any(r["actual_timestamp"] == 15.2 for r in combined) or any(r["actual_timestamp"] in (14.6,14.7) for r in combined):
        raise AssertionError("Combined incoming ownership/role regression")
    if len([r for r in summaries if r["video_name"] == "ticket_scroll_test.mp4"]) != 5:
        raise AssertionError("A stable scroll position was lost")
    print(json.dumps(totals), flush=True)
    print(f"Report: {OUTPUT/'ticket_ocr_report.txt'}", flush=True)


if __name__ == "__main__":
    main()
