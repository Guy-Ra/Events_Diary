"""Evaluate production spatial features over saved Phase6.1b OCR, without OCR reruns.

Rehydrate the original17 OCRTicketResult records from their persisted summary and
ordered detection CSVs, preserving physical provenance, raw text, confidence and
source quadrilaterals. Call only production spatial APIs for observations/scores.
All directed non-self relations are kept; no calibration-driven formula/threshold
adjustment, semantic pairing, label recognition or cross-frame merging occurs.

The seven human-verified crops are an evaluation-only reference. Reuse Phase6.2a's
saved dominant-geometric-overlap component mapping (including its missing/merged
cases) to locate the known target detection. Rank it against ALL other detections
from that representative, not a truth-filtered list. Report missing edges, best/
worst ranks under ties, unique top1, guaranteed top3 and score margins. A1e-12
absolute tie tolerance is evaluation bookkeeping only, never a production gate.

Outputs under data/debug/ocr/ocr_spatial include per-representative provenance,
all pair geometry/scores, distributions, top candidate alternatives, ambiguity
examples and reference rankings. No image/model execution/dependency additions.
"""

import csv
import hashlib
import json
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

from events_diary.config import ROI
from events_diary.ocr_spatial import build_spatial_observations, build_spatial_relations
from events_diary.ticket_ocr import OCRDetection, OCRTicketResult

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT/"data/debug/ocr/ticket_ocr"
CALIBRATION = ROOT/"data/debug/ocr/ocr_spatial_relationship_calibration"
TRUTH = ROOT/"data/debug/ocr/ocr_engine_calibration"
OUTPUT = ROOT/"data/debug/ocr/ocr_spatial"
SCORES = ("same_column_score","same_row_score","vertical_stack_score","inline_score","local_neighbor_score")
TIE_TOLERANCE = 1e-12


def _read(path):
    """Read raw UTF-8 CSV text without whitespace, case or punctuation normalization."""
    with path.open(encoding="utf-8-sig",newline="") as stream:
        return list(csv.DictReader(stream))


def _csv(name,rows):
    """Write evaluator measurements without changing saved OCR or calibration inputs."""
    if not rows:
        raise ValueError(f"No evaluation rows for {name}")
    with (OUTPUT/name).open("w",encoding="utf-8",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def _key(row):
    """Address one physical representative within the existing single-video scope."""
    return row["video_name"],row["investigation_index"],row["candidate_index"]


def _rehydrate():
    """Rebuild original typed results, checking exact order, count and text semantics."""
    groups = defaultdict(list)
    for row in _read(INPUT/"ticket_ocr_results.csv"):
        if row["detection_index"]!="":
            groups[_key(row)].append(row)
    records = []
    for summary in _read(INPUT/"ticket_ocr_summary.csv"):
        rows = groups.pop(_key(summary),[])
        if [int(r["detection_index"]) for r in rows]!=list(range(int(summary["detection_count"]))):
            raise ValueError("Saved detection indexes/count are not the original contiguous OCR order")
        for row in rows:
            if any(row[k]!=summary[k] for k in ("frame_index","actual_timestamp","crop_geometry","engine_name","preprocessing_name","resolution_status")):
                raise ValueError("Detection/representative provenance mismatch")
        detections = tuple(OCRDetection(r["text"],float(r["confidence"]) if r["confidence"] else None,
            tuple(tuple(p) for p in json.loads(r["source_polygon"]))) for r in rows)
        if "\n".join(d.text for d in detections)!=summary["ocr_text"]:
            raise ValueError("Rehydrated raw text differs from saved OCR summary")
        result = OCRTicketResult(summary["engine_name"],summary["preprocessing_name"],ROI(**json.loads(summary["crop_geometry"])),
            int(summary["upscale_factor"]),detections,int(summary["investigation_index"]),int(summary["candidate_index"]),
            int(summary["frame_index"]),float(summary["actual_timestamp"]),summary["resolution_status"])
        records.append((summary,result))
    if groups or len(records)!=17:
        raise ValueError("Expected exactly17 unchanged production OCR representatives")
    return records


def _percentiles(values):
    """Return simple linearly interpolated quantiles using only the standard library."""
    ordered = sorted(values)
    result = {}
    for name,q in (("min",0),("p05",.05),("p25",.25),("median",.5),("p75",.75),("p95",.95),("max",1)):
        index = (len(ordered)-1)*q
        low,high = int(index),min(int(index)+1,len(ordered)-1)
        result[name] = ordered[low]+(ordered[high]-ordered[low])*(index-low)
    return result


def _references():
    """Load verified relationships and accepted geometric OCR mapping, never text-match."""
    fields = _read(TRUTH/"ocr_ground_truth_fields.csv")
    parts = _read(TRUTH/"ocr_calibration_ground_truth.csv")
    if len(fields)!=94 or len(parts)!=151 or any(r["review_status"]!="human_verified" for r in fields+parts):
        raise ValueError("Expected unchanged verified94-field/151-component reference data")
    mapping = {}
    for row in _read(CALIBRATION/"ocr_component_geometry_mapping.csv"):
        aid = row["assigned_annotation"]
        if not aid:
            continue
        key = row["sample_id"],aid
        value = float(row["overlap_detection_fraction"]),int(row["detection_index"])
        if key not in mapping or value[0]>mapping[key][0]:
            mapping[key] = value
    return fields,mapping


def _reference_ranks(summary,result,relations,fields,mapping):
    """Rank known targets against every other detection; preserve missing edges/ties."""
    rows = []
    for f in fields:
        if (f["video_name"],f["investigation_index"],f["candidate_index"])!=_key(summary):
            continue
        if json.loads(f["label_box"]) is None or json.loads(f["value_box"]) is None:
            continue
        source = mapping.get((f["sample_id"],f["field_id"]+".label"))
        target = mapping.get((f["sample_id"],f["field_id"]+".value"))
        technical = f["section"] not in ("scheduled","actual_estimated")
        score = "vertical_stack_score" if technical else "inline_score"
        row = {"sample_id":f["sample_id"],"field_id":f["field_id"],"regime":"technical_stack" if technical else "inline_time",
            "score_name":score,"source_index":source[1] if source else None,"target_index":target[1] if target else None,
            "available":source is not None and target is not None,"rank_best":None,"rank_worst":None,
            "top1_unique":False,"top3_guaranteed":False,"true_score":None,"best_false_score":None,"score_margin":None}
        if row["available"]:
            if source[1]==target[1] or max(source[1],target[1])>=len(result.detections):
                raise ValueError("Saved geometric component mapping cannot define distinct valid detections")
            group = [r for r in relations if r.source_index==source[1]]
            correct = next(r for r in group if r.target_index==target[1])
            value = getattr(correct,score)
            false = [getattr(r,score) for r in group if r.target_index!=target[1]]
            rank_best = 1+sum(s>value+TIE_TOLERANCE for s in false)
            rank_worst = 1+sum(s>=value-TIE_TOLERANCE for s in false)
            best_false = max(false,default=0)
            row.update(rank_best=rank_best,rank_worst=rank_worst,top1_unique=rank_worst==1,
                top3_guaranteed=rank_worst<=3,true_score=value,best_false_score=best_false,score_margin=value-best_false)
        rows.append(row)
    return rows


def main():
    """Analyze all saved representatives with fixed production formulas and write audits."""
    OUTPUT.mkdir(parents=True,exist_ok=True)
    protected = [INPUT/"ticket_ocr_summary.csv",INPUT/"ticket_ocr_results.csv",TRUTH/"ocr_ground_truth_fields.csv",
        TRUTH/"ocr_calibration_ground_truth.csv",CALIBRATION/"ocr_component_geometry_mapping.csv",
        ROOT/"src/events_diary/ticket_ocr.py",ROOT/"scripts/calibration/evaluate_ocr_spatial_relationships.py"]
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    fields,mapping = _references()
    summaries,relation_rows,distributions,tops,ambiguities,references,observation_rows = [],[],[],[],[],[],[]
    for summary,result in _rehydrate():
        snapshot = asdict(result)
        started = time.perf_counter()
        observations = build_spatial_observations(result)
        relations = build_spatial_relations(observations)
        latency_ms = 1000*(time.perf_counter()-started)
        if asdict(result)!=snapshot or len(relations)!=len(observations)*(len(observations)-1):
            raise AssertionError("Spatial analysis changed OCR input or lost directed pairs")
        identity = {k:summary[k] for k in ("video_name","investigation_index","candidate_index","frame_index","actual_timestamp")}
        summaries.append({**identity,"detection_count":len(observations),"relation_count":len(relations),"spatial_latency_ms":latency_ms,
            "crop_geometry":summary["crop_geometry"],"engine_name":summary["engine_name"],"preprocessing_name":summary["preprocessing_name"]})
        for o in observations:
            observation_rows.append({**identity,"detection_index":o.detection_index,"text":o.text,"ocr_confidence":o.ocr_confidence,
                "source_polygon":json.dumps(o.source_polygon),"source_bounds":json.dumps(asdict(o.source_bounds)),
                "normalized_bounds":json.dumps(asdict(o.normalized_bounds)),"normalized_center_x":o.normalized_center_x,
                "normalized_center_y":o.normalized_center_y,"normalized_width":o.normalized_width,"normalized_height":o.normalized_height})
        relation_rows.extend({**identity,**asdict(r)} for r in relations)
        by_source = defaultdict(list)
        for r in relations:
            by_source[r.source_index].append(r)
        for score in SCORES:
            distributions.append({**identity,"score_name":score,**_percentiles([getattr(r,score) for r in relations])})
            for source,group in by_source.items():
                ranked = sorted(group,key=lambda r:(-getattr(r,score),r.target_index))
                best,runner = getattr(ranked[0],score),getattr(ranked[1],score)
                for rank,r in enumerate(ranked[:3],1):
                    tops.append({**identity,"source_index":source,"score_name":score,"debug_rank":rank,
                        "target_index":r.target_index,"score":getattr(r,score),"source_text":observations[source].text,
                        "target_text":observations[r.target_index].text})
                ambiguities.append({**identity,"source_index":source,"score_name":score,
                    "best_target":ranked[0].target_index,"best_score":best,"runner_up_target":ranked[1].target_index,
                    "runner_up_score":runner,"margin":best-runner,
                    "top_tie_count":sum(abs(getattr(r,score)-best)<=TIE_TOLERANCE for r in group)})
        references.extend(_reference_ranks(summary,result,relations,fields,mapping))
    _csv("ocr_spatial_summary.csv",summaries)
    _csv("ocr_spatial_relations.csv",relation_rows)
    _csv("ocr_spatial_observations.csv",observation_rows)
    _csv("ocr_spatial_score_distributions.csv",distributions)
    _csv("ocr_spatial_top_candidates.csv",tops)
    _csv("ocr_spatial_ambiguity.csv",ambiguities)
    _csv("ocr_spatial_reference_rankings.csv",references)
    totals = {"representatives":len(summaries),"detections":sum(r["detection_count"] for r in summaries),
        "relations":len(relation_rows),"median_spatial_latency_ms":statistics.median(r["spatial_latency_ms"] for r in summaries),
        "max_spatial_latency_ms":max(r["spatial_latency_ms"] for r in summaries)}
    transfer = []
    for regime in ("technical_stack","inline_time"):
        group = [r for r in references if r["regime"]==regime]
        available = [r for r in group if r["available"]]
        transfer.append({"regime":regime,"verified_pairs":len(group),"matched_pairs":len(available),"missing_pairs":len(group)-len(available),
            "top1_unique_count":sum(r["top1_unique"] for r in available),"top3_guaranteed_count":sum(r["top3_guaranteed"] for r in available),
            "top1_matched_rate":sum(r["top1_unique"] for r in available)/len(available),
            "top3_matched_rate":sum(r["top3_guaranteed"] for r in available)/len(available),
            "top1_all_verified_rate":sum(r["top1_unique"] for r in available)/len(group),
            "top3_all_verified_rate":sum(r["top3_guaranteed"] for r in available)/len(group),
            **{"margin_"+k:v for k,v in _percentiles([r["score_margin"] for r in available]).items()}})
    _csv("ocr_spatial_transfer_summary.csv",transfer)
    lines = ["Phase6.2b Production Spatial Observation Model -- fixed geometry features",json.dumps(totals),
        "17 saved Phase6.1b results, no OCR/video/model rerun. All directed non-self pairs retained. Timing measures observation/relation construction only, single pass, excludes CSV/report I/O.",
        "Canonical source polygons remain unchanged. AABBs are derived convenience, normalized relative to saved production crop origin/size, never screen size. Edge overhang is not clipped.",
        "Gap=max(starts)-min(ends): positive separation in either direction, zero contact, negative overlap/containment. Center direction/delta is separate; reversed pair preserves gaps and reverses deltas.",
        "Fixed formulas, not fitted or tuned by this evaluator:",
        "C=.5*overlap_x/minwidth+.5*exp(-abs(dx/W)/.25).",
        "R=.5*overlap_y/minheight+.5*exp(-abs(dy/H)/max(.02,meanheight/H)).",
        "Stack=C*clamp(dy/meanheight,0,1)*exp(-max(0,gap_y/H)/.05).",
        "Inline=R*clamp(dx/meanwidth,0,1)*exp(-max(0,gap_x/W)/.1).",
        "Neighbor=1/(1+hypot(dx/W,dy/H)); separate fallback proximity, never a combined pairing rule.",
        "No score is semantic/field/OCR/fusion confidence. No hard 0.02 grouping, binary column split, section identity, canonical labels or label/value pairing.",
        "All raw alternatives retained. top_candidates.csv presents three for debugging only, does not truncate the production graph. Equal scores use index only for display. Reference ranks expose best/worst tie positions, and top1 is unique; top3 is guaranteed despite ties (1e-12 evaluation-only tolerance).",
        "Transfer uses existing dominant-overlap mapping, not OCR text matching. Compare known targets to ALL other raw detections, including unannotated texts/icons. Missing/merged mapped components remain unavailable, not invented. Seven reference crops repeat content and are not a benchmark.",
        "TRANSFER:",*(str(r) for r in transfer),"REFERENCE RANKINGS:",*(str(r) for r in references),
        "REPRESENTATIVE SUMMARY:",*(str(r) for r in summaries),"SCORE DISTRIBUTIONS:",*(str(r) for r in distributions),
        "AMBIGUITY EXAMPLES (smallest positive-score top-two margins, including ties):"]
    examples = sorted((r for r in ambiguities if r["best_score"]>0),key=lambda r:r["margin"])[:12]
    lines.extend(str(r) for r in examples)
    lines += ["Manual-band perfect thresholds from6.2a were NOT promoted. Raw box widths, gaps and merged components differ; geometric rankings may prefer unrelated text, so no grouping is finalized.",
        "Known merged serial/age label examples F20/F56 remain missing as separate label observations; raw column-wide boxes cannot be split from this geometry alone.",
        "Failures/low margins reported rather than tuned. Production scoring uses no canonical labels, section metadata, verified field IDs or values; reference IDs appear only in this evaluator.",
        "No Phase1-5/6.1 production changes, no field parsing, normalization, VLM/fusion, multi-view merge or Phase6.3 implementation."]
    (OUTPUT/"ocr_spatial_report.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    if hashes!={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}:
        raise AssertionError("Spatial evaluator modified protected input/production/calibration artifacts")
    (OUTPUT/"input_provenance.json").write_text(json.dumps(hashes,indent=2),encoding="utf-8")
    print(json.dumps(totals),flush=True)
    for row in transfer:
        print(json.dumps(row),flush=True)
    print(f"Report: {OUTPUT/'ocr_spatial_report.txt'}",flush=True)


if __name__=="__main__":
    main()
