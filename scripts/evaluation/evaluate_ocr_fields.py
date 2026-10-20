"""Evaluate production FR24 OCR field extraction on17 saved physical representatives.

Rehydrate existing Phase6.1b OCRTicketResult records and call production recognition,
spatial and field APIs, including the batch helper. No OCR/video/model rerun, no
new calibration, no threshold tuning and no view merging occur. Production uses
no ground-truth labels/values: verified artifacts are evaluation-only references.

The seven-crop reference reuses Phase6.3a's saved geometric detection linkage and
canonical identities, cross-checking literal labels/values against unchanged human
truth. Compound secondary labels may share a detection, not independent boxes.
Value ranks are diagnostic alternatives, not selected values for abstained fields.
Recognition, physical value LOCATION, and exact literal transcription are distinct
metrics. UNKNOWN controls retain the human105/synthetic16 separation.

Write all production fields, per-representative summaries, complete46-row human
validation tables, alternatives and reference metrics under data/debug/ocr/ocr_fields.
Every ambiguous, absent-value and special/non-text case stays visible. No optional
compound x-position heuristic, final Investigation record, VLM/fusion or Phase6.4.
"""

import csv
import hashlib
import json
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

from events_diary.config import ROI
from events_diary.ocr_fields import (
    CANONICAL_LABELS,
    extract_ocr_fields_batch,
    rank_ocr_field_candidates,
    recognize_canonical_labels,
)
from events_diary.ocr_spatial import (
    SpatialOCRObservation,
    build_spatial_observations,
    build_spatial_relations,
)
from events_diary.ticket_ocr import OCRDetection
from scripts.evaluation.evaluate_ocr_spatial import _rehydrate

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT/"data/debug/ocr/ocr_fields"
CALIBRATION = ROOT/"data/debug/ocr/canonical_label_calibration"
TRUTH = ROOT/"data/debug/ocr/ocr_engine_calibration"


def _read(path):
    """Read raw UTF-8 text/provenance without normalization or correction."""
    with path.open(encoding="utf-8-sig",newline="") as stream:
        return list(csv.DictReader(stream))


def _csv(name,rows):
    """Write one debug table with scalar/provenance fields and exact raw strings."""
    with (OUTPUT/name).open("w",encoding="utf-8",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def _references():
    """Require canonical calibration references to agree with authoritative raw truth."""
    fields = {f["field_id"]:f for f in _read(TRUTH/"ocr_ground_truth_fields.csv")}
    references = _read(CALIBRATION/"example_review_table.csv")
    if len(fields)!=94 or len(references)!=46 or any(f["review_status"]!="human_verified" for f in fields.values()):
        raise ValueError("Expected existing verified94 fields and46 label references")
    for row in references:
        f = fields[row["field_id"]]
        if row["verified_label"]!=f["raw_label_text"] or row["verified_value"]!=f["raw_value_text"] or row["sample_id"]!=f["sample_id"]:
            raise ValueError("Calibration references differ from verified labels/values")
        if row["canonical"] not in CANONICAL_LABELS:
            raise ValueError("Reference canonical identity not covered by production scope")
    return references


def _flatten(summary,field):
    """Expose a production field's separate scores, local status and source polygons."""
    value = field.selected_candidate.observation if field.selected_candidate else None
    return {"video_name":summary["video_name"],**asdict(field.provenance),
        "canonical_label":field.canonical_label,"raw_label_text":field.raw_label_text,"raw_value_text":field.raw_value_text,
        "label_detection_index":field.label_detection_index,"value_detection_index":field.value_detection_index,
        "label_source_polygon":json.dumps(field.label_match.source_polygon),
        "value_source_polygon":json.dumps(value.source_polygon) if value else "",
        "ticket_crop":json.dumps(asdict(field.label_match.observation.ticket_roi)),
        "label_match_method":field.label_match_method,"label_match_score":field.label_match_score,
        "relationship_type":field.relationship_type,"relationship_score":field.relationship_score,
        "ocr_label_confidence":field.ocr_label_confidence,"ocr_value_confidence":field.ocr_value_confidence,
        "compound":field.compound_label,"shared_geometry":field.shared_geometry,
        "matched_text_span":json.dumps(field.label_match.matched_text_span),"ambiguous":field.ambiguous,
        "status":field.status,"reason":field.reason}


def _validate_reference(ref,result,fields,matches,observations,relations):
    """Compare one verified occurrence without using its text/value to drive extraction."""
    source = int(ref["source_index"]) if ref["source_index"] else None
    target = int(ref["verified_value_index"]) if ref["verified_value_index"] else None
    if source is not None and result.detections[source].text!=ref["raw_ocr_label"]:
        raise ValueError("Saved reference label points at different raw OCR text")
    found = [f for f in fields if f.label_detection_index==source and f.canonical_label==ref["canonical"]]
    if len(found)>1:
        raise ValueError("Reference maps ambiguously to multiple production occurrences")
    field = found[0] if found else None
    candidates = rank_ocr_field_candidates(field.label_match,observations,relations,label_matches=matches) if field else ()
    eligible = [c for c in candidates if not c.is_label_candidate]
    rank = next((i for i,c in enumerate(eligible,1) if c.observation.detection_index==target),None)
    special = ref["no_text_value"]=="True"
    selected = field.value_detection_index if field else None
    location_correct = selected is not None and selected==target and not special
    exact = field is not None and field.raw_value_text is not None and field.raw_value_text==ref["verified_value"]
    status = ("label_missing" if field is None else "special_non_text" if special and field.status=="special_non_text" else
        "correct_location_exact_text" if location_correct and exact else "correct_location_text_diff" if location_correct else
        "wrong_selected" if field.status=="extracted" else field.status)
    return {"sample_id":ref["sample_id"],"field_id":ref["field_id"],"column":ref["column"],
        "raw_ocr_label":ref["raw_ocr_label"],"canonical":ref["canonical"],"canonical_correct":field is not None,
        "observed_regime":ref["observed_regime"],"selected_raw_value":field.raw_value_text if field else None,
        "verified_value":ref["verified_value"],"value_rank":rank,"rank_note":"diagnostic only for unselected/compound fields",
        "selected_detection_index":selected,"verified_value_detection_index":target,
        "location_correct":location_correct,"raw_value_exact":exact,"no_text_value":special,
        "label_match_method":field.label_match_method if field else None,"label_match_score":field.label_match_score if field else None,
        "relationship_type":field.relationship_type if field else None,"relationship_score":field.relationship_score if field else None,
        "compound":field.compound_label if field else False,"ambiguous":field.ambiguous if field else False,
        "production_status":field.status if field else None,"reason":field.reason if field else None,"validation_status":status,
        "top_alternatives":json.dumps([{"index":c.observation.detection_index,"raw_text":c.observation.text,
            "score":c.relationship_score,"is_label_candidate":c.is_label_candidate} for c in candidates[:3]],ensure_ascii=False)}


def _unknowns(records):
    """Measure production UNKNOWN behavior on the existing105 human/16 synthetic controls."""
    controls = [r for r in _read(CALIBRATION/"unknown_label_controls.csv") if r["method"]=="F_hybrid" and r["threshold"]=="0.9"]
    rows = []
    for control in controls:
        if control["kind"]=="human_verified_non_label":
            _,result,observations,_ = records[control["sample_id"]]
            index = int(control["detection_index"])
            if result.detections[index].text!=control["text"]:
                raise ValueError("Unknown reference changed")
            observation = observations[index]
        else:
            detection = OCRDetection(control["text"],None,((0.,0.),(10.,0.),(10.,10.),(0.,10.)))
            observation = SpatialOCRObservation(0,detection,ROI(0,0,20,20))
        predictions = recognize_canonical_labels((observation,))
        rows.append({"sample_id":control["sample_id"],"raw_text":control["text"],"kind":control["kind"],
            "false_canonical_assignment":bool(predictions),"canonical_candidates":";".join(p.canonical_label for p in predictions)})
    _csv("unknown_controls.csv",rows)
    return rows


def _metrics(validations):
    """Keep selected-location accuracy separate from raw transcription and abstentions."""
    rows = []
    for regime in ("technical_stack","inline_time","special_non_text","overall_text"):
        group = [r for r in validations if (r["no_text_value"] if regime=="special_non_text" else
            not r["no_text_value"] and (regime=="overall_text" or (r["observed_regime"]=="inline" if regime=="inline_time" else r["observed_regime"]!="inline")))]
        selected = [r for r in group if r["production_status"]=="extracted"]
        correct = sum(r["location_correct"] for r in group)
        exact = sum(r["location_correct"] and r["raw_value_exact"] for r in group)
        rows.append({"regime":regime,"verified_occurrences":len(group),"recognized_labels":sum(r["canonical_correct"] for r in group),
            "selected":len(selected),"correct_selected_location":correct,
            "location_accuracy_all_eligible":correct/len(group) if regime!="special_non_text" else None,
            "location_accuracy_selected":correct/len(selected) if selected else None,
            "ambiguous":sum(r["production_status"]=="ambiguous" for r in group),
            "no_value_selected":sum(r["production_status"]=="no_value_selected" for r in group),
            "label_missing":sum(not r["canonical_correct"] for r in group),
            "special_non_text":sum(r["production_status"]=="special_non_text" for r in group),
            "wrong_selected":sum(r["validation_status"]=="wrong_selected" for r in group),
            "exact_raw_values_at_correct_location":exact,"exact_value_rate_selected":exact/len(selected) if selected else None,
            "exact_value_rate_all_eligible":exact/len(group) if regime!="special_non_text" else None,
            "location_correct_text_different":sum(r["location_correct"] and not r["raw_value_exact"] for r in group)})
    _csv("verified_regression_metrics.csv",rows)
    return rows


def _report(validations,metrics,summaries,unknowns):
    """Write all46 review rows plus explicit production policy, failures and closure scope."""
    def escape(value):
        """Keep raw strings legible in Markdown without normalizing their content."""
        return "—" if value is None else str(value).replace("|","\\|").replace("\n","<br>")
    table = ["| Crop / field / column | Raw OCR label | Canonical | Selected raw value | Verified value | Value rank | Match / relationship | Status |",
        "|---|---|---|---|---|---|---|---|"]
    for r in validations:
        method = f"{r['label_match_method']} / {r['relationship_type']} ({r['relationship_score']})"
        table.append("| "+" | ".join(escape(v) for v in (r["sample_id"]+" / "+r["field_id"]+" / "+r["column"],r["raw_ocr_label"],
            r["canonical"],r["selected_raw_value"],r["verified_value"],r["value_rank"],method,r["validation_status"]))+" |")
    (OUTPUT/"field_validation_table.md").write_text("\n".join(table)+"\n",encoding="utf-8")
    counts = Counter(r["production_status"] for r in validations)
    lines = ["Phase6.3b Production Canonical Labels + OCR Field Extraction",
        f"17 representatives kept separate; total production fields={sum(r['field_count'] for r in summaries)}; verified46 occurrence statuses={dict(counts)}.",
        "Canonical set="+", ".join(CANONICAL_LABELS),
        "Matching: compact case/alphanumeric view; stable month/year age context, fixed optionalMSN, parenthetical aircraft tag; complete compound grammar. Fuzzy only after failure, similarity>=.90 and unique-best margin>=.10; no memorized dates/type-code fuzzy templates, no raw correction.",
        "Stack/inline scores are unchanged Phase6.2 features. Exact/stable label targets are flagged in diagnostics and excluded from default selection; fuzzy targets remain candidates to avoid over-filtering. No expected numeric/unit/registration formats are used.",
        "Explicit provisional extraction guards: min_relationship_score=.25, min_candidate_margin=.10. No automatic tuning during evaluation. Equal scores always abstain. These are policy guards, not probabilities or a claim of calibrated precision.",
        "Compound/shared label geometry always abstains. No optional x-position heuristic was evaluated/deployed; no fake boxes. COUNTRY_OF_REGISTRATION/FIR_UIR retain special_non_text with raw label evidence, not an inferred textual flag/lock value.",
        "Statuses: extracted, no_value_selected, ambiguous, special_non_text. Reasons distinguish weak/no candidates, close margins and compound sharing. These are not final evidence/fusion statuses.",
        "Provenance: Investigation/candidate/frame/actual time/engine/preprocessing plus original source observation indexes, raw texts/confidences, source polygons, crop metadata, canonical method/score and chosen relation. Confidence types remain separate; no global confidence.",
        "Reference metrics separate known target LOCATION from exact raw text. Value-rank columns on abstained rows are diagnostic alternatives, never selected values. No selected value for compounds is reported as an extraction success.",
        "VERIFIED METRICS:",*(str(r) for r in metrics),"UNKNOWN CONTROLS:"]
    for kind in sorted({r["kind"] for r in unknowns}):
        group = [r for r in unknowns if r["kind"]==kind]
        lines.append(f"{kind}: false assignments={sum(r['false_canonical_assignment'] for r in group)}/{len(group)}")
    lines += ["REPRESENTATIVES:",*(str(r) for r in summaries),"Human-readable field validation",*table,
        "Known failure/abstention: compound SERIAL_NUMBER/AGE source strings recognize both identities, but share one polygon. Both abstain; SERIAL_NUMBER's true1100 remains a diagnostic rank2 below16years. AGE may have rank1 but shared geometry does not independently justify selection.",
        "OCR transcription examples: selected0ft versus verified0 ft; time suffixpM versusPM; aircraft-name spacing. Correct location is not corrected text or final semantic truth.",
        "All17 production outputs are evaluated, but accuracy is scored only on the seven verified crops/46 labels. Unannotated fields are not automatically successes or failures;105 negative controls do not establish a general false-positive benchmark.",
        "Phase6.3 closure readiness: scoped MVP criteria are reviewable with canonical recognition, conservative field selection, full raw provenance, explicit compounds/abstentions, both regimes, verified regression and human tables. Known OCR text errors/compound abstentions remain limitations; not a claim of complete-ticket correctness.",
        "No Phases1-5/6.1/6.2 changes, VLM/fusion, multi-view assembly, final Investigation record, airport/departure/arrival interpretation or Phase6.4 implementation. No commit/push."]
    (OUTPUT/"ocr_fields_report.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    """Evaluate fixed production extraction and check protected inputs remain unchanged."""
    OUTPUT.mkdir(parents=True,exist_ok=True)
    protected = [ROOT/p for p in ("src/events_diary/ticket_ocr.py","src/events_diary/ocr_spatial.py",
        "scripts/calibration/evaluate_canonical_label_calibration.py","scripts/evaluation/evaluate_ocr_spatial.py")]
    protected += [TRUTH/"ocr_ground_truth_fields.csv",TRUTH/"ocr_calibration_ground_truth.csv",CALIBRATION/"example_review_table.csv",
        ROOT/"data/debug/ocr/ticket_ocr/ticket_ocr_results.csv",ROOT/"data/debug/ocr/ticket_ocr/ticket_ocr_summary.csv"]
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    loaded = _rehydrate()
    started = time.perf_counter()
    batches = extract_ocr_fields_batch(tuple(result for _,result in loaded))
    elapsed = 1000*(time.perf_counter()-started)
    references = _references()
    records,flat,summaries,validations,alternatives = {},[],[],[],[]
    for (summary,result),fields in zip(loaded,batches,strict=True):
        sid = f"{summary['video_name'].removesuffix('_test.mp4')}_i{result.investigation_index}_c{result.candidate_index}"
        observations = build_spatial_observations(result)
        relations = build_spatial_relations(observations)
        matches = recognize_canonical_labels(observations)
        records[sid] = summary,result,observations,fields
        flat.extend(_flatten(summary,f) for f in fields)
        summaries.append({"sample_id":sid,"video_name":summary["video_name"],"investigation_index":result.investigation_index,
            "candidate_index":result.candidate_index,"actual_timestamp":result.actual_timestamp_seconds,"frame_index":result.frame_index,
            "field_count":len(fields),"extracted":sum(f.status=="extracted" for f in fields),
            "ambiguous":sum(f.ambiguous for f in fields),"no_value_selected":sum(f.status=="no_value_selected" for f in fields),
            "special_non_text":sum(f.status=="special_non_text" for f in fields),"compound_occurrences":sum(f.compound_label for f in fields)})
        for ref in references:
            if ref["sample_id"]==sid:
                validations.append(_validate_reference(ref,result,fields,matches,observations,relations))
        for field in fields:
            ranked = rank_ocr_field_candidates(field.label_match,observations,relations,label_matches=matches)
            alternatives.extend({"sample_id":sid,"canonical":field.canonical_label,"label_index":field.label_detection_index,
                "rank":i,"target_index":candidate.observation.detection_index,"raw_text":candidate.observation.text,
                "relationship_type":candidate.relationship_type,"score":candidate.relationship_score,
                "obvious_label_target":candidate.is_label_candidate} for i,candidate in enumerate(ranked[:3],1))
    if len(validations)!=46:
        raise ValueError("Verified occurrence coverage changed")
    _csv("ocr_fields_results.csv",flat)
    _csv("field_extraction_summary.csv",summaries)
    _csv("field_validation_table.csv",validations)
    _csv("ranked_alternatives.csv",alternatives)
    unknowns = _unknowns(records)
    metrics = _metrics(validations)
    _report(validations,metrics,summaries,unknowns)
    if hashes!={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}:
        raise AssertionError("Production evaluation modified protected inputs")
    (OUTPUT/"input_provenance.json").write_text(json.dumps(hashes,indent=2),encoding="utf-8")
    print(f"Representatives={len(summaries)}, production fields={len(flat)}, extraction batch_ms={elapsed:.2f}")
    for metric in metrics:
        print(json.dumps(metric))
    print(f"Report: {OUTPUT/'ocr_fields_report.txt'}")


if __name__=="__main__":
    main()
