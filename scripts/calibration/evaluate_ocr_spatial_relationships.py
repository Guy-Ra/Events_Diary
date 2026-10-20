"""Phase 6.2a development-only measurement of verified FR24 spatial relationships.

Read the existing seven human-verified crops, 94 field/cell records and 151 raw
components without creating annotations or running OCR. Field/section/column IDs
are calibration truth only: no canonical labels, value interpretation or production
pairing is implemented. Geometry calculations never inspect text contents.

Annotation boxes are ticket-local (x,y,width,height) review bands, sometimes much
wider than the rendered text. Distinguish these from saved production OCR source
quadrilaterals. Source coordinates add the actual mapped Strategy-A crop origin;
normalization divides x by crop width and y by crop height (not 1920/1080).
All bounds use continuous/exclusive-edge geometry; missing boxes stay missing.

True label/value relations come only from verified field IDs. False competitors
are other fields' components within +/-0.15 ticket heights of the label center,
including opposite-column same-row values and neighboring rows. True pairs are
always retained, even outside that window; the window is a declared sampling
choice, not a learned production threshold. Sensitivity at0.08/0.22 is reported.

Peer truth uses existing same-section left/right cells, not text matching. Section
gap measurements concern annotated sections only, not all visible ticket content.
Header, upper scroll content and technical body are analyzed separately using
verified section metadata. Candidate formulas/thresholds and leave-one-crop-out
measurements remain calibration artifacts; they are not production rules.

The polygon audit reads already saved Phase6.1b detections, checks crop/provenance
correspondence, compares actual polygons with AABBs, and reports ranking changes
on geometrically matched components. It never uses recognized text for matching.
One native detection is assigned to one component by dominant box overlap >=0.3;
merged/missing detections remain explicit limitations, never split using answers.
No dependency additions, production changes, OCR invocation, label recognition,
VLM, fusion or Phase6.2b/6.3 implementation is involved.
"""

import csv
import hashlib
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np

from events_diary.config import UIConfig
from events_diary.ticket_ocr import ticket_ocr_roi

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT/"data/debug/ocr/ocr_engine_calibration"
PRODUCTION = ROOT/"data/debug/ocr/ticket_ocr"
OUTPUT = ROOT/"data/debug/ocr/ocr_spatial_relationship_calibration"
WINDOW = .15
UPPER_SECTIONS = frozenset(("airport_cells", "scheduled", "actual_estimated", "progress_text"))
QUANTILES = (0, .05, .25, .5, .75, .95, 1)
QUANTILE_NAMES = ("min", "p05", "p25", "median", "p75", "p95", "max")


def _read_csv(path):
    """Read saved UTF-8 annotations/results without modifying the source artifact."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _csv(name, rows):
    """Save heterogeneous scalar measurement records with deterministic column order."""
    if not rows:
        raise ValueError(f"No measurement rows for {name}")
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with (OUTPUT/name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _box(values):
    """Validate a literal local annotation rectangle; never create a missing box."""
    if values is None:
        return None
    x,y,w,h = map(float, values)
    if not all(math.isfinite(v) for v in (x,y,w,h)) or x < 0 or y < 0 or w <= 0 or h <= 0:
        raise ValueError("Invalid annotation geometry")
    return {"left": x, "top": y, "right": x+w, "bottom": y+h, "width": w,
        "height": h, "center_x": x+w/2, "center_y": y+h/2}


def _union(boxes):
    """Enclose only existing literal component boxes, without inferring missing text."""
    left, top = min(b["left"] for b in boxes), min(b["top"] for b in boxes)
    return _box((left, top, max(b["right"] for b in boxes)-left, max(b["bottom"] for b in boxes)-top))


def _spaces(box, roi):
    """Project one local box into local/source pixels and normalized ticket geometry."""
    result = {"local_"+k: v for k,v in box.items()}
    for k,v in box.items():
        offset = roi.x if k in ("left", "right", "center_x") else roi.y if k in ("top", "bottom", "center_y") else 0
        result["source_"+k] = v+offset
        result["normalized_"+k] = v/(roi.width if k in ("left", "right", "width", "center_x") else roi.height)
    return result


def _regime(section):
    """Partition using verified section metadata only, never recognized text/value."""
    return "header" if section == "header" else "upper_content" if section in UPPER_SECTIONS else "technical_body"


def _load():
    """Require exact verified data and reconcile fields, components and physical crops."""
    fields = _read_csv(INPUT/"ocr_ground_truth_fields.csv")
    parts = _read_csv(INPUT/"ocr_calibration_ground_truth.csv")
    if len(fields) != 94 or len(parts) != 151:
        raise ValueError("Expected exactly94 verified fields and151 components")
    if any(r["review_status"] != "human_verified" or not r["verified_by"] for r in fields+parts):
        raise ValueError("Unverified annotations are not calibration truth")
    by_field = {f["field_id"]: f for f in fields}
    if len(by_field) != 94 or len({r["annotation_id"] for r in parts}) != 151:
        raise ValueError("Duplicate annotation IDs")
    summaries = _read_csv(PRODUCTION/"ticket_ocr_summary.csv")
    production = {(r["video_name"],r["investigation_index"],r["candidate_index"]): r for r in summaries}
    manifest = json.loads((INPUT/"input_manifest.json").read_text(encoding="utf-8"))
    samples = {s["sample_id"]: s for s in manifest["samples"]}
    rois = {}
    for sid in sorted({f["sample_id"] for f in fields}):
        sample = samples[sid]
        row = production[sample["video_name"],str(sample["investigation_index"]),str(sample["candidate_index"])]
        crop = cv2.imread(str(ROOT/sample["input_path"]))
        if crop is None or hashlib.sha256(crop.tobytes()).hexdigest() != sample["pixel_sha256"]:
            raise ValueError("Original review crop changed")
        # All verified sources are canonical1920x1080; use the production derivation
        # and compare saved metadata instead of inventing a separate origin.
        roi = ticket_ocr_roi(UIConfig(),1920,1080)
        if json.loads(row["crop_geometry"]) != {"x":roi.x,"y":roi.y,"width":roi.width,"height":roi.height}:
            raise ValueError("Saved OCR crop does not match production/reference geometry")
        if (crop.shape[:2] != (roi.height,roi.width)
                or float(row["actual_timestamp"]) != float(sample["materialized"]["actual_timestamp_seconds"])
                or int(row["frame_index"]) != sample["materialized"]["frame_index"]):
            raise ValueError("Crop/provenance correspondence differs")
        rois[sid] = roi
    if len(rois) != 7:
        raise ValueError("Expected seven verified crops")
    for f in fields:
        f["label"] = _box(json.loads(f["label_box"]))
        f["value"] = _box(json.loads(f["value_box"]))
        f["regime"] = _regime(f["section"])
        f["parts"] = []
    for part in parts:
        f = by_field[part["field_id"]]
        if any(part[k] != f[k] for k in ("sample_id","section","column","video_name","candidate_index","investigation_index")):
            raise ValueError("Field/component annotation correspondence mismatch")
        part["box"] = _box([part[k] for k in ("x","y","width","height")])
        if part["component"] in ("label","value") and part["box"] != f[part["component"]]:
            raise ValueError("Literal label/value box differs from component box")
        part["regime"] = f["regime"]
        f["parts"].append(part)
        if part["box"]["right"] > rois[part["sample_id"]].width or part["box"]["bottom"] > rois[part["sample_id"]].height:
            raise ValueError("Annotation outside original ticket crop")
    for f in fields:
        if not f["parts"]:
            raise ValueError("A field has no annotated text; do not invent geometry")
        f["box"] = _union([p["box"] for p in f["parts"]])
    return fields, parts, rois, samples


def _geometry_table(fields, parts, rois):
    """Record every verified field union and component in all three coordinate spaces."""
    rows = []
    # Empirical median verified field bounds, not a hard-coded half-ticket split.
    regions = {column:(float(np.median([f["box"]["left"]/rois[f["sample_id"]].width for f in fields if f["column"]==column])),
        float(np.median([f["box"]["right"]/rois[f["sample_id"]].width for f in fields if f["column"]==column]))) for column in ("left","right")}
    for level, records in (("field",fields),("component",parts)):
        for r in records:
            roi = rois[r["sample_id"]]
            overlap = {}
            for column,(left,right) in regions.items():
                intersection = max(0,min(r["box"]["right"]/roi.width,right)-max(r["box"]["left"]/roi.width,left))
                overlap[column+"_column_region_coverage"] = intersection/(right-left)
            rows.append({"level":level,"sample_id":r["sample_id"],"field_id":r["field_id"],
                "annotation_id":r.get("annotation_id",""),"section":r["section"],"column":r["column"],"regime":r["regime"],
                "component":r.get("component","union"),"has_label":bool(r.get("label")) if level=="field" else "",
                "has_value":bool(r.get("value")) if level=="field" else "",
                "label_box":r.get("label_box",""),"value_box":r.get("value_box",""),
                "raw_text":r.get("raw_manual_transcription",""), **overlap, **_spaces(r["box"],roi)})
    _csv("field_geometry.csv",rows)


def _pair(a,b,width,height):
    """Measure relative rectangle geometry without consulting strings or field identity."""
    overlap = max(0,min(a["right"],b["right"])-max(a["left"],b["left"]))
    vertical = max(0,min(a["bottom"],b["bottom"])-max(a["top"],b["top"]))
    dx,dy = b["center_x"]-a["center_x"],b["center_y"]-a["center_y"]
    gap = b["top"]-a["bottom"]
    hmin = overlap/min(a["width"],b["width"])
    h_iou = overlap/(a["width"]+b["width"]-overlap)
    vmin = vertical/min(a["height"],b["height"])
    align = max(0,1-abs(dx)/(.5*width))
    row = .5*vmin+.5*max(0,1-abs(dy)/max(a["height"],b["height"]))
    column = .5*hmin+.5*align
    stack = float(dy>0)*(.5*hmin+.3*align+.2*math.exp(-abs(gap)/(.05*height)))
    distance = math.hypot(dx/width,dy/height)
    return {"a_center_x":a["center_x"],"b_center_x":b["center_x"],"center_dx_abs":abs(dx),
        "center_dx_normalized":abs(dx)/width,"horizontal_intersection":overlap,
        "horizontal_overlap_min":hmin,"horizontal_overlap_union":h_iou,
        "vertical_gap":gap,"vertical_gap_normalized":gap/height,"value_center_below":dy>0,
        "value_edges_below":b["top"]>=a["bottom"],"relative_width":b["width"]/a["width"],
        "relative_height":b["height"]/a["height"],"left_alignment_normalized":abs(b["left"]-a["left"])/width,
        "right_alignment_normalized":abs(b["right"]-a["right"])/width,
        "horizontal_containment":overlap>=min(a["width"],b["width"])-1e-9,
        "vertical_center_difference_normalized":abs(dy)/height,"top_alignment_normalized":abs(b["top"]-a["top"])/height,
        "bottom_alignment_normalized":abs(b["bottom"]-a["bottom"])/height,
        "row_height_similarity":min(a["height"],b["height"])/max(a["height"],b["height"]),
        "horizontal_separation_normalized":max(0,a["left"]-b["right"],b["left"]-a["right"])/width,
        "same_column_score":column,"same_row_score":row,"vertical_stack_score":stack,
        "local_neighbor_score":float(dy>=0)/(1+distance),
        "inline_score":float(b["left"]>=a["right"])*row*math.exp(-max(0,b["left"]-a["right"])/(.1*width)),
        "euclidean_normalized":distance,"manhattan_normalized":abs(dx)/width+abs(dy)/height}


def _relationships(fields, parts, rois):
    """Generate verified true stacks/inline pairs, bounded competitors and peer controls."""
    rows = []
    for field in fields:
        if not field["label"] or not field["value"]:
            continue
        label = next(p for p in field["parts"] if p["component"]=="label")
        roi = rois[field["sample_id"]]
        for candidate in parts:
            if candidate["sample_id"] != field["sample_id"] or candidate["annotation_id"] == label["annotation_id"]:
                continue
            true = candidate["field_id"]==field["field_id"] and candidate["component"]=="value"
            if not true and (candidate["field_id"]==field["field_id"] or abs(candidate["box"]["center_y"]-field["label"]["center_y"])/roi.height>WINDOW):
                continue
            opposite = {field["column"],candidate["column"]}=={"left","right"}
            kind = ("true_same_field" if true else "opposite_column_same_row_value"
                if opposite and candidate["section"]==field["section"] and candidate["component"]=="value" else
                "neighbor_row_same_column_value" if candidate["component"]=="value" and candidate["column"]==field["column"] else
                "other_nearby_value" if candidate["component"]=="value" else "nearby_"+candidate["component"])
            rows.append({"relationship":"label_value","sample_id":field["sample_id"],"regime":field["regime"],
                "anchor_id":label["annotation_id"],"target_id":candidate["annotation_id"],"anchor_field":field["field_id"],
                "target_field":candidate["field_id"],"anchor_section":field["section"],"target_section":candidate["section"],
                "anchor_column":field["column"],"target_column":candidate["column"],"target_component":candidate["component"],
                "is_true":true,"competitor_kind":kind, **_pair(label["box"],candidate["box"],roi.width,roi.height)})
    peers = []
    for left,right in itertools.combinations(fields,2):
        if left["sample_id"]!=right["sample_id"] or {left["column"],right["column"]}!={"left","right"}:
            continue
        left,right = (left,right) if left["column"]=="left" else (right,left)
        roi = rois[left["sample_id"]]
        true = left["section"]==right["section"]
        if not true and abs(left["box"]["center_y"]-right["box"]["center_y"])/roi.height>WINDOW:
            continue
        peers.append({"relationship":"row_peer","sample_id":left["sample_id"],"regime":left["regime"],
            "anchor_id":left["field_id"],"target_id":right["field_id"],"is_true":true,
            "competitor_kind":"verified_same_section_peers" if true else "neighbor_section_peer_control",
            **_pair(left["box"],right["box"],roi.width,roi.height)})
    _csv("pair_geometry.csv",rows+peers)
    return rows,peers


def _distributions(fields,parts,rois,pairs,peers):
    """Summarize quantiles with regimes/samples kept visible instead of blindly pooled."""
    rows = []
    def add(population,feature,group,values):
        if values:
            rows.append({"population":population,"feature":feature,"group":group,"n":len(values),
                **dict(zip(QUANTILE_NAMES,map(float,np.quantile(values,QUANTILES)),strict=True))})
    features = ("horizontal_overlap_min","horizontal_overlap_union","center_dx_normalized","vertical_gap_normalized",
        "relative_width","relative_height","same_column_score","same_row_score","vertical_stack_score","local_neighbor_score","inline_score")
    for regime in ("all","upper_content","technical_body"):
        for true in (True,False):
            selected = [p for p in pairs if p["is_true"]==true and (regime=="all" or p["regime"]==regime)]
            for feature in features:
                add("label_value",feature,f"{regime}/{'true' if true else 'false'}",[p[feature] for p in selected])
    for sid in sorted(rois):
        for true in (True,False):
            for feature in features[:4]:
                add("label_value",feature,f"{sid}/{'true' if true else 'false'}",[p[feature] for p in pairs if p["sample_id"]==sid and p["is_true"]==true])
    for level,records in (("field",fields),("component",parts)):
        for column in ("left","right","full","header"):
            for regime in ("all","header","upper_content","technical_body"):
                group = [r for r in records if r["column"]==column and (regime=="all" or r["regime"]==regime)]
                for feature in ("center_x","left","right","width"):
                    add(level,"normalized_"+feature,column+"/"+regime,[r["box"][feature]/rois[r["sample_id"]].width for r in group])
    for true in (True,False):
        for feature in ("vertical_center_difference_normalized","top_alignment_normalized","bottom_alignment_normalized",
                "row_height_similarity","horizontal_separation_normalized","same_row_score"):
            add("row_peer",feature,"true" if true else "false",[p[feature] for p in peers if p["is_true"]==true])
    for section in sorted({f["section"] for f in fields}):
        for column in sorted({f["column"] for f in fields if f["section"]==section}):
            group = [f for f in fields if f["section"]==section and f["column"]==column]
            for feature in ("center_x","center_y","width","height"):
                add("section_field","normalized_"+feature,section+"/"+column,
                    [f["box"][feature]/(rois[f["sample_id"]].width if feature in ("center_x","width") else rois[f["sample_id"]].height) for f in group])
    _csv("geometry_distributions.csv",rows)
    return rows


def _cell_structure(fields,rois):
    """Measure verified within-cell stacks, upper peer layout and header positions."""
    rows = []
    for f in fields:
        components = sorted(f["parts"],key=lambda p:(p["box"]["top"],p["box"]["left"]))
        roi = rois[f["sample_id"]]
        for a,b in itertools.pairwise(components):
            rows.append({"sample_id":f["sample_id"],"field_id":f["field_id"],"section":f["section"],"column":f["column"],
                "from_component":a["component"],"to_component":b["component"],
                **_pair(a["box"],b["box"],roi.width,roi.height)})
    _csv("cell_component_geometry.csv",rows)
    upper = [{"sample_id":f["sample_id"],"field_id":f["field_id"],"section":f["section"],"column":f["column"],
        **_spaces(f["box"],rois[f["sample_id"]])} for f in fields if f["regime"] in ("header","upper_content")]
    _csv("upper_ticket_layout.csv",upper)


def _section_gaps(fields,parts,rois):
    """Measure annotated section spacing and within-section literal component gaps."""
    rows = []
    for sid,roi in rois.items():
        sections = defaultdict(list)
        for p in parts:
            if p["sample_id"]==sid:
                sections[p["section"]].append(p)
        ordered = sorted(sections,key=lambda s:min(p["box"]["top"] for p in sections[s]))
        for before,after in itertools.pairwise(ordered):
            a,b = _union([p["box"] for p in sections[before]]),_union([p["box"] for p in sections[after]])
            gap = b["top"]-a["bottom"]
            rows.append({"sample_id":sid,"kind":"across_section","from":before,"to":after,
                "gap_pixels":gap,"gap_normalized":gap/roi.height})
        for section,group in sections.items():
            for column in sorted({p["column"] for p in group}):
                stack = sorted((p for p in group if p["column"]==column),key=lambda p:(p["box"]["top"],p["box"]["left"]))
                for a,b in itertools.pairwise(stack):
                    gap = b["box"]["top"]-a["box"]["bottom"]
                    rows.append({"sample_id":sid,"kind":"within_section_components","from":a["annotation_id"],"to":b["annotation_id"],
                        "section":section,"column":column,"gap_pixels":gap,"gap_normalized":gap/roi.height})
        for column in ("left","right","full","header"):
            stack = sorted((f for f in fields if f["sample_id"]==sid and f["column"]==column),key=lambda f:f["box"]["top"])
            for a,b in itertools.pairwise(stack):
                gap = b["box"]["top"]-a["box"]["bottom"]
                rows.append({"sample_id":sid,"kind":"same_column_field_rows","from":a["field_id"],"to":b["field_id"],
                    "section":a["section"],"column":column,"same_section":a["section"]==b["section"],
                    "gap_pixels":gap,"gap_normalized":gap/roi.height})
    _csv("section_gaps.csv",rows)
    return rows


def _performance(group,predicate):
    """Measure true recall and false acceptance without conflating class proportions."""
    true = [r for r in group if r["is_true"]]
    false = [r for r in group if not r["is_true"]]
    recall = sum(predicate(r) for r in true)/len(true) if true else None
    acceptance = sum(predicate(r) for r in false)/len(false) if false else None
    return {"true_count":len(true),"false_count":len(false),"true_recall":recall,"false_acceptance":acceptance,
        "balanced_accuracy":(recall+1-acceptance)/2 if recall is not None and acceptance is not None else None}


def _gate_diagnostics(group,feature,direction,threshold):
    """Expose class overlap and concrete examples near a proposed scalar cutoff."""
    true = [r[feature] for r in group if r["is_true"]]
    false = [r[feature] for r in group if not r["is_true"]]
    margin = (min(true)-max(false) if direction==">=" else min(false)-max(true)) if true and false else None
    near = sorted(group,key=lambda r:abs(r[feature]-threshold))[:6]
    return {"class_separation_margin":margin,"near_threshold":json.dumps([
        (r.get("anchor_id",r.get("id","")),r.get("target_id",""),r["is_true"],r[feature]) for r in near])}


def _thresholds(pairs,peers,fields,parts,rois):
    """Compare bounded candidate gates, signed class margins and held-out-crop stability."""
    rows, held_out = [], []
    grids = {"horizontal_overlap_min":(">=",(.1,.5,.8,.95)),"center_dx_normalized":("<=",(.02,.05,.1,.15,.25,.35,.5)),
        "vertical_gap_normalized":("<=",(0,.002,.005,.01,.02,.05,.1)),
        "same_column_score":(">=",(.5,.7,.8,.9)),"same_row_score":(">=",(.5,.7,.9,.95)),
        "vertical_stack_score":(">=",(.5,.7,.8,.9,.95)),"local_neighbor_score":(">=",(.7,.8,.9,.95,.98)),
        "inline_score":(">=",(.5,.7,.9,.95))}
    for regime in ("all","upper_content","technical_body"):
        group = [r for r in pairs if regime=="all" or r["regime"]==regime]
        for feature,(direction,thresholds) in grids.items():
            true = [r[feature] for r in group if r["is_true"]]
            false = [r[feature] for r in group if not r["is_true"]]
            margin = (min(true)-max(false) if direction==">=" else min(false)-max(true)) if true and false else None
            for threshold in thresholds:
                def accept(r, feature=feature, direction=direction, threshold=threshold):
                    return r[feature]>=threshold if direction==">=" else r[feature]<=threshold
                near = sorted(group,key=lambda r:abs(r[feature]-threshold))[:6]
                rows.append({"task":"label_value","regime":regime,"feature":feature,"direction":direction,"threshold":threshold,
                    "class_separation_margin":margin,"near_threshold":json.dumps([(r["anchor_id"],r["target_id"],r["is_true"],r[feature]) for r in near]),
                    **_performance(group,accept)})
            for sid in sorted(rois):
                training = [r for r in group if r["sample_id"]!=sid]
                testing = [r for r in group if r["sample_id"]==sid]
                if not testing or not any(r["is_true"] for r in training):
                    continue
                candidates = [(t,_performance(training,lambda r,t=t,feature=feature,direction=direction:r[feature]>=t if direction==">=" else r[feature]<=t)) for t in thresholds]
                threshold,train = max(candidates,key=lambda item: item[1]["balanced_accuracy"] if item[1]["balanced_accuracy"] is not None else -1)
                held_out.append({"sample_id":sid,"regime":regime,"feature":feature,"training_threshold":threshold,
                    "training_balanced_accuracy":train["balanced_accuracy"], **_performance(testing,lambda r,feature=feature,direction=direction,threshold=threshold:r[feature]>=threshold if direction==">=" else r[feature]<=threshold)})
        for gap in (.005,.01,.02):
            predicate = lambda r,gap=gap: r["horizontal_overlap_min"]>=.5 and r["center_dx_normalized"]<=.1 and 0<=r["vertical_gap_normalized"]<=gap
            rows.append({"task":"label_value","regime":regime,"feature":"combined_vertical_gate","direction":"overlap>=.5,dx<=.1,0<=gap<=threshold",
                "threshold":gap,"class_separation_margin":None,"near_threshold":"",**_performance(group,predicate)})
    for threshold in (.005,.01,.02,.04,.08):
        rows.append({"task":"row_peer","regime":"all","feature":"vertical_center_difference_normalized","direction":"<=","threshold":threshold,
            **_performance(peers,lambda r,threshold=threshold:r["vertical_center_difference_normalized"]<=threshold),
            **_gate_diagnostics(peers,"vertical_center_difference_normalized","<=",threshold)})
    for threshold in (.5,.7,.9,.95):
        rows.append({"task":"row_peer","regime":"all","feature":"same_row_score","direction":">=","threshold":threshold,
            **_performance(peers,lambda r,threshold=threshold:r["same_row_score"]>=threshold),
            **_gate_diagnostics(peers,"same_row_score",">=",threshold)})
    for level,records in (("component",parts),("field",fields)):
        columns = [{"id":r.get("annotation_id",r["field_id"]),"is_true":r["column"]=="right","x":r["box"]["center_x"]/rois[r["sample_id"]].width} for r in records if r["column"] in ("left","right")]
        full = [{"id":r.get("annotation_id",r["field_id"]),"is_true":r["column"]=="full","width":r["box"]["width"]/rois[r["sample_id"]].width} for r in records if r["column"] in ("left","right","full")]
        for threshold in (.4,.45,.5,.55,.6):
            rows.append({"task":"right_column_"+level,"regime":"left_vs_right_only","feature":"normalized_center_x","direction":">=","threshold":threshold,
                **_performance(columns,lambda r,threshold=threshold:r["x"]>=threshold),
                **_gate_diagnostics(columns,"x",">=",threshold)})
        for threshold in (.5,.6,.7,.8,.85):
            rows.append({"task":"full_width_"+level,"regime":"full_vs_left_right_only","feature":"normalized_width","direction":">=","threshold":threshold,
                **_performance(full,lambda r,threshold=threshold:r["width"]>=threshold),
                **_gate_diagnostics(full,"width",">=",threshold)})
    _csv("threshold_candidates.csv",rows)
    _csv("held_out_crop_thresholds.csv",held_out)
    return rows,held_out


def _polygon_audit(parts,rois,samples,pairs):
    """Compare saved raw quadrilaterals and enclosing boxes without running OCR."""
    rows, by_component, mapping, native_pairs = [], defaultdict(list), [], []
    identities = {(s["video_name"],str(s["investigation_index"]),str(s["candidate_index"])):sid for sid,s in samples.items() if sid in rois}
    native_rows = _read_csv(PRODUCTION/"ticket_ocr_results.csv")
    for raw in native_rows:
        sid = identities.get((raw["video_name"],raw["investigation_index"],raw["candidate_index"]))
        if sid is None or not raw["detection_index"]:
            continue
        roi = rois[sid]
        polygon = np.asarray(json.loads(raw["source_polygon"]),dtype=np.float32)-[roi.x,roi.y]
        polygon = polygon.astype(np.float32)
        x,y = polygon.min(axis=0)
        right,bottom = polygon.max(axis=0)
        aabb = _box((x,y,right-x,bottom-y))
        area = abs(cv2.contourArea(polygon))
        moments = cv2.moments(polygon)
        cx,cy = (moments["m10"]/moments["m00"],moments["m01"]/moments["m00"]) if moments["m00"] else (aabb["center_x"],aabb["center_y"])
        horizontal = (np.linalg.norm(polygon[1]-polygon[0])+np.linalg.norm(polygon[2]-polygon[3]))/2
        vertical = (np.linalg.norm(polygon[3]-polygon[0])+np.linalg.norm(polygon[2]-polygon[1]))/2
        oriented = _box((cx-horizontal/2,cy-vertical/2,horizontal,vertical))
        tilt = math.degrees(math.atan2(float(polygon[1,1]-polygon[0,1]),float(polygon[1,0]-polygon[0,0])))
        row = {"sample_id":sid,"detection_index":raw["detection_index"],"source_polygon":raw["source_polygon"],
            "polygon_area":area,"aabb_area":aabb["width"]*aabb["height"],"polygon_aabb_iou":area/(aabb["width"]*aabb["height"]),
            "centroid_shift_x_normalized":abs(cx-aabb["center_x"])/roi.width,"centroid_shift_y_normalized":abs(cy-aabb["center_y"])/roi.height,
            "oriented_width_aabb_ratio":float(horizontal/aabb["width"]),"oriented_height_aabb_ratio":float(vertical/aabb["height"]),
            "top_edge_tilt_degrees":tilt}
        rows.append(row)
        overlaps = []
        for p in parts:
            if p["sample_id"]!=sid:
                continue
            b = p["box"]
            intersection = max(0,min(aabb["right"],b["right"])-max(aabb["left"],b["left"]))*max(0,min(aabb["bottom"],b["bottom"])-max(aabb["top"],b["top"]))
            overlaps.append((intersection/row["aabb_area"],p["annotation_id"]))
        overlap,aid = max(overlaps,default=(0,""))
        assigned = next((p for p in parts if p["annotation_id"]==aid),None) if overlap>=.3 else None
        row.update({"assigned_annotation":aid if assigned else "", "assigned_column":assigned["column"] if assigned else "",
            "normalized_width":aabb["width"]/roi.width,"normalized_center_x":aabb["center_x"]/roi.width})
        qualifying = sum(o>=.3 for o,_ in overlaps)
        mapping.append({"sample_id":sid,"detection_index":raw["detection_index"],"assigned_annotation":aid if overlap>=.3 else "",
            "overlap_detection_fraction":overlap,"qualifying_components":qualifying})
        if overlap>=.3:
            by_component[sid,aid].append((overlap,aabb,oriented,polygon))
    chosen = {k:max(v,key=lambda item:item[0]) for k,v in by_component.items()}
    comparisons = []
    for pair in pairs:
        a,b = chosen.get((pair["sample_id"],pair["anchor_id"])),chosen.get((pair["sample_id"],pair["target_id"]))
        if a is None or b is None:
            continue
        roi = rois[pair["sample_id"]]
        axis,oriented = _pair(a[1],b[1],roi.width,roi.height),_pair(a[2],b[2],roi.width,roi.height)
        native_pairs.append({**{k:pair[k] for k in ("sample_id","regime","anchor_id","target_id","is_true","competitor_kind")}, **axis})
        area,_ = cv2.intersectConvexConvex(a[3],b[3])
        bbox_overlap = max(0,min(a[1]["right"],b[1]["right"])-max(a[1]["left"],b[1]["left"]))*max(0,min(a[1]["bottom"],b[1]["bottom"])-max(a[1]["top"],b[1]["top"]))
        comparisons.append({"sample_id":pair["sample_id"],"anchor_id":pair["anchor_id"],"target_id":pair["target_id"],"is_true":pair["is_true"],
            "actual_polygon_intersection_area":float(area),"aabb_intersection_area":bbox_overlap,
            **{"aabb_"+k:axis[k] for k in ("horizontal_overlap_min","center_dx_normalized","vertical_stack_score","same_row_score")},
            **{"oriented_"+k:oriented[k] for k in ("horizontal_overlap_min","center_dx_normalized","vertical_stack_score","same_row_score")}})
    rankings = []
    groups = defaultdict(list)
    for p in comparisons:
        groups[p["sample_id"],p["anchor_id"]].append(p)
    for (sid,aid),group in groups.items():
        for feature in ("vertical_stack_score","same_row_score"):
            axis = sorted(group,key=lambda p:(-p["aabb_"+feature],p["target_id"]))
            oriented = sorted(group,key=lambda p:(-p["oriented_"+feature],p["target_id"]))
            rankings.append({"sample_id":sid,"anchor_id":aid,"feature":feature,"candidate_count":len(group),
                "top_candidate_changed":axis[0]["target_id"]!=oriented[0]["target_id"],
                "complete_order_changed":[p["target_id"] for p in axis]!=[p["target_id"] for p in oriented]})
    _csv("polygon_geometry.csv",rows)
    _csv("ocr_component_geometry_mapping.csv",mapping)
    _csv("polygon_pair_comparison.csv",comparisons)
    _csv("polygon_ranking_comparison.csv",rankings)
    _csv("raw_ocr_pair_geometry.csv",native_pairs)
    return rows,mapping,comparisons,rankings,native_pairs


def _score_rankings(pairs,name):
    """Report top-candidate ambiguity and true-target rank for each exploratory score."""
    groups = defaultdict(list)
    for p in pairs:
        groups[p["sample_id"],p["anchor_id"]].append(p)
    rows = []
    for (sid,aid),group in groups.items():
        true = next((p for p in group if p["is_true"]),None)
        if true is None:
            continue
        for feature in ("same_column_score","same_row_score","vertical_stack_score","local_neighbor_score","inline_score"):
            best = max(p[feature] for p in group)
            tied = [p for p in group if abs(p[feature]-best)<1e-12]
            rows.append({"sample_id":sid,"anchor_id":aid,"regime":true["regime"],"score":feature,
                "true_rank":1+sum(p[feature]>true[feature]+1e-12 for p in group),
                "top_contains_true":any(p["is_true"] for p in tied),"top_tie_count":len(tied),
                "uniquely_correct_top":len(tied)==1 and tied[0]["is_true"],
                "score_margin":true[feature]-max((p[feature] for p in group if not p["is_true"]),default=0)})
    _csv(name,rows)
    return rows


def _raw_transfer(native_pairs,thresholds):
    """Apply the existing annotation threshold grid to matched raw boxes without tuning."""
    rows = []
    for candidate in thresholds:
        if candidate["task"]!="label_value":
            continue
        regime,feature,threshold,direction = (candidate[k] for k in ("regime","feature","threshold","direction"))
        group = [r for r in native_pairs if regime=="all" or r["regime"]==regime]
        if feature=="combined_vertical_gate":
            def predicate(r,threshold=threshold):
                return r["horizontal_overlap_min"]>=.5 and r["center_dx_normalized"]<=.1 and 0<=r["vertical_gap_normalized"]<=threshold
        else:
            def predicate(r,feature=feature,threshold=threshold,direction=direction):
                return r[feature]>=threshold if direction==">=" else r[feature]<=threshold
        rows.append({"regime":regime,"feature":feature,"threshold":threshold,"direction":direction,
            "annotation_true_recall":candidate["true_recall"],"annotation_false_acceptance":candidate["false_acceptance"],
            **_performance(group,predicate), **(_gate_diagnostics(group,feature,direction,threshold)
                if feature!="combined_vertical_gate" else {"class_separation_margin":None,"near_threshold":"Mixed-feature gate; inspect individual normalized margins"})})
    _csv("raw_ocr_threshold_transfer.csv",rows)
    return rows


def _write_report(fields,parts,rois,pairs,peers,distributions,gaps,thresholds,held_out,polygons,rankings_by_score,transfer):
    """Write measured conclusions and all fourteen required architectural questions."""
    true = [r for r in pairs if r["is_true"]]
    false = [r for r in pairs if not r["is_true"]]
    polygon_rows,mapping,comparisons,rankings,native_pairs = polygons
    lines = ["Phase6.2a Spatial Relationship Calibration -- measurement only",
        f"Verified crops={len(rois)}, fields/cells={len(fields)}, components={len(parts)}, true label/value pairs={len(true)}, false competitors={len(false)}.",
        f"Column populations: fields={dict(Counter(f['column'] for f in fields))}; components={dict(Counter(p['column'] for p in parts))}.",
        "Inputs: ocr_ground_truth_fields.csv, ocr_calibration_ground_truth.csv, review_fields_source.json, ground_truth_review.md, input_manifest.json; saved Phase6.1b ticket_ocr_results/summary.csv. No annotation regenerated, no OCR invocation.",
        "Coordinate convention: saved annotations local to336x831 ticket; source pixels add(16,121); normalization x/336,y/831. Every field union encloses only existing component boxes. Flags/locks have no fabricated value box.",
        "CRITICAL annotation limitation: manual boxes often describe full field/line bands, not tight glyph boxes. Technical left label/value bands share identical x/width by construction. Perfect overlap/alignment of those bands is not proof of perfect raw OCR alignment. Seven crops share repeated headers/content; these are not independent benchmark examples.",
        "Upper-content time fields are inline (label and value side by side), not vertical stacks. A universal below-label constraint would reject them. Full-width means verified layout membership, not necessarily wide visible text.",
        "Sampling: false components from other fields within+/-0.15 ticket heights of anchor center; include labels/context as competitors, not only values. True labels with no text value are excluded from pair classification, retained in geometry. Both directions of nearby rows are represented.",
        "Peer truth: opposite-column fields in the same verified section/sample; controls are opposite-column different sections in bounded neighborhood. Section IDs are accepted annotation metadata only, never canonical labels.",
        "Pair counts by regime/truth/kind="+str(dict(Counter((p["regime"],p["is_true"],p["competitor_kind"]) for p in pairs))),
        "Distribution quantiles are linear percentiles: min,p05,p25,median,p75,p95,max. All sample/regime/column detail is saved in geometry_distributions.csv.",
        "TRUE/FALSE GEOMETRY DISTRIBUTIONS:"]
    gate = next(r for r in thresholds if r["task"]=="label_value" and r["regime"]=="technical_body" and r["feature"]=="combined_vertical_gate" and r["threshold"]==.01)
    raw_gate = next(r for r in transfer if r["regime"]=="technical_body" and r["feature"]=="combined_vertical_gate" and r["threshold"]==.01)
    peer_gate = next(r for r in thresholds if r["task"]=="row_peer" and r["feature"]=="vertical_center_difference_normalized" and r["threshold"]==.02)
    summary = ["KEY MEASURED CONCLUSIONS:",
        f"Technical vertical gate (overlap>=.5,dx<=.1W,0<=gap<=.01H): manual recall={gate['true_recall']:.4f}/false acceptance={gate['false_acceptance']:.4f}; raw matched-box recall={raw_gate['true_recall']:.4f}/false acceptance={raw_gate['false_acceptance']:.4f}, true support={raw_gate['true_count']}. Missing raw edges are excluded from that conditional recall, not repaired.",
        f"Manual peer center-dy<=.02H: recall={peer_gate['true_recall']:.4f}/false acceptance={peer_gate['false_acceptance']:.4f}; {peer_gate['true_count']} verified peers versus{peer_gate['false_count']} controls. Short label-only flag cells explain failure of stricter center/height symmetry. Automatic field-union transfer is NOT validated.",
        "Absolute y/section positions move with scrolling. Stable local offsets and column structure do not justify fixed absolute-y section templates. The same nominal header/upper/body geometry must not be pooled blindly."]
    for column in ("full","left","right"):
        values = [r["normalized_width"] for r in polygon_rows if r["assigned_column"]==column]
        summary.append(f"Raw matched {column} text widths/W: min={min(values):.4f},median={float(np.median(values)):.4f},max={max(values):.4f}. Contrast full-layout annotation bands at279/336=.83036; text width is not layout membership.")
    lines[2:2] = summary
    lines.extend(str(r) for r in distributions if r["population"]=="label_value" and r["group"].startswith(("all/","upper_content/","technical_body/")) and r["feature"] in
        ("horizontal_overlap_min","center_dx_normalized","vertical_gap_normalized"))
    lines += ["COLUMN DISTRIBUTIONS:", *(str(r) for r in distributions if r["population"] in ("component","field") and r["group"] in ("left/all","right/all","full/all")),
        "PEER DISTRIBUTIONS:", *(str(r) for r in distributions if r["population"]=="row_peer")]
    left = [p["box"]["center_x"]/rois[p["sample_id"]].width for p in parts if p["column"]=="left"]
    right = [p["box"]["center_x"]/rois[p["sample_id"]].width for p in parts if p["column"]=="right"]
    lines.append(f"Two-column component center median separation={float(np.median(right)-np.median(left)):.6f}, interval separation margin={min(right)-max(left):.6f}; no left/right center overlap in verified bands. Ambiguous middle interval=({max(left):.6f},{min(right):.6f}); full/header membership is excluded from that binary comparison.")
    lines += ["UPPER CELLS/HEADER SECTION SUMMARY:",*(str(r) for r in distributions if r["population"]=="section_field"
        and r["group"].split("/")[0] in UPPER_SECTIONS|{"header"})]
    for kind in sorted({g["kind"] for g in gaps}):
        values = [g["gap_normalized"] for g in gaps if g["kind"]==kind]
        lines.append("Gaps "+kind+": n="+str(len(values))+", "+str(dict(zip(QUANTILE_NAMES,map(float,np.quantile(values,QUANTILES)),strict=True))))
    lines += ["CANDIDATE FORMULAS (exploratory, not production):",
        "same_column=.5*horizontal_overlap/minwidth +.5*max(0,1-abs(dx)/(.5W)).",
        "same_row=.5*vertical_overlap/minheight +.5*max(0,1-abs(dy)/maxheight).",
        "vertical_stack=I(value_center_below)*(.5*horizontal_overlap/minwidth+.3*xalignment+.2*exp(-abs(vertical_edge_gap)/(.05H))).",
        "local_neighbor=I(dy>=0)/(1+hypot(dx/W,dy/H)); inline=I(value_left>=label_right)*same_row*exp(-horizontal_gap/(.1W)).",
        "Thresholds inclusive. Scalar margin=min(true)-max(false) for >=, min(false)-max(true) for <=; negative means class overlap. Combined-gate scalar margin is deliberately undefined (mixed features). Near-threshold IDs/values are in threshold_candidates.csv.",
        "TOP CANDIDATE THRESHOLDS PER REGIME (balanced accuracy; descriptive training-set results, not final rules):"]
    for regime in ("all","upper_content","technical_body"):
        candidates = [r for r in thresholds if r["task"]=="label_value" and r["regime"]==regime]
        lines.extend(str({k:v for k,v in r.items() if k!="near_threshold"}) for r in sorted(candidates,key=lambda r:r["balanced_accuracy"],reverse=True)[:6])
    lines += ["COLUMN/FULL-WIDTH/PEER CANDIDATES:", *(str(r) for r in thresholds if r["task"]!="label_value"),
        "Held-out validation: for each candidate scalar feature, select best balanced-accuracy grid threshold on six crops and evaluate the seventh. Threshold grids remain fixed; no text/value inputs. Some held-out regimes have no positives and report None recall. See held_out_crop_thresholds.csv."]
    for regime in ("all","upper_content","technical_body"):
        for feature in ("horizontal_overlap_min","center_dx_normalized","vertical_gap_normalized","vertical_stack_score","inline_score"):
            group = [r for r in held_out if r["regime"]==regime and r["feature"]==feature]
            measured = [r for r in group if r["true_recall"] is not None and r["false_acceptance"] is not None]
            if measured:
                lines.append(f"Held-out {regime}/{feature}: thresholds={sorted({r['training_threshold'] for r in group})}, macro recall={np.mean([r['true_recall'] for r in measured]):.4f}, macro false acceptance={np.mean([r['false_acceptance'] for r in measured]):.4f}; n={len(measured)} crops with both classes.")
    lines += [f"Saved polygon audit: {len(polygon_rows)} native detections from seven originals; {sum(bool(r['assigned_annotation']) for r in mapping)} assigned by geometry; {sum(r['qualifying_components']>1 for r in mapping)} overlap multiple bands; {len(comparisons)} native pair comparisons; {len(rankings)} anchor/score rankings."]
    for column in ("left","right","full","header"):
        selected = [r for r in polygon_rows if r["assigned_column"]==column]
        for feature in ("normalized_width","normalized_center_x"):
            lines.append(f"Raw OCR mapped {column}/{feature}: n={len(selected)}, "+str(dict(zip(QUANTILE_NAMES,map(float,np.quantile([r[feature] for r in selected],QUANTILES)),strict=True))))
    for feature in ("polygon_aabb_iou","centroid_shift_x_normalized","centroid_shift_y_normalized","oriented_width_aabb_ratio","oriented_height_aabb_ratio","top_edge_tilt_degrees"):
        lines.append("Polygon "+feature+": "+str(dict(zip(QUANTILE_NAMES,map(float,np.quantile([r[feature] for r in polygon_rows],QUANTILES)),strict=True))))
    lines.append(f"Polygon/AABB changed top ranks={sum(r['top_candidate_changed'] for r in rankings)}/{len(rankings)}, changed complete orders={sum(r['complete_order_changed'] for r in rankings)}/{len(rankings)}.")
    for feature in ("vertical_stack_score","same_row_score"):
        selected = [r for r in rankings if r["feature"]==feature]
        lines.append(f"Polygon/AABB {feature} changed top={sum(r['top_candidate_changed'] for r in selected)}/{len(selected)}. Candidate ranking includes false alternatives and ties; this is not a count of true semantic pairing errors.")
    lines.append("The changed same-row top is the merged F55.label example: one very low-scoring false candidate under AABB becomes an all-zero tie under oriented descriptors. Do not overstate this as loss of a correctly identified true pair.")
    lines.append(f"Raw matched pairs={len(native_pairs)}, complete true pairs={sum(p['is_true'] for p in native_pairs)}/{len(true)}; raw-box transfer recall is conditional on matched geometry, not all43 truth edges.")
    lines += ["ANNOTATION-TO-RAW THRESHOLD TRANSFER (no retuning):",*(str(r) for r in transfer if
        (r["feature"]=="combined_vertical_gate" and r["threshold"]==.01) or
        (r["feature"] in ("inline_score","vertical_stack_score") and r["threshold"]==.9))]
    for dataset,rank_rows in rankings_by_score.items():
        for regime in ("upper_content","technical_body"):
            for score in ("same_column_score","same_row_score","vertical_stack_score","local_neighbor_score","inline_score"):
                selected = [r for r in rank_rows if r["regime"]==regime and r["score"]==score]
                if selected:
                    lines.append(f"Top-candidate {dataset}/{regime}/{score}: uniquely correct={sum(r['uniquely_correct_top'] for r in selected)}/{len(selected)}, top-set contains true={sum(r['top_contains_true'] for r in selected)}/{len(selected)}. Ties are not forced decisions.")
    lines.append("Horizontal/vertical projection extents are identical to AABB extents by definition, not an empirical discovery. Actual polygon-vs-AABB intersection areas and centroid/oriented-edge score/ranking comparisons are recorded separately. Oriented descriptors are measurement approximations, not new stored geometry.")
    lines += ["REQUIRED QUESTIONS / RECOMMENDATIONS:",
        "1. Horizontal overlap: useful for technical vertical stacks and rejecting opposite-column peers, not universal because upper time labels/values are inline. Manual band widths inflate apparent separation; validate on raw tight detections before production gates.",
        "2. Center-x distance: useful for columns/stacks, but inline labels/values intentionally differ in x; full-width objects and header IDs are not ordinary column members.",
        "3. Vertical gap: combine with direction and x alignment for stacks; alone accepts other columns/rows, and inline pairs have negative edge gap. Never require below-label for all fields.",
        "4. Normalized stability: recurring manual bands are stable across these crops, but repeated snapshots and fixed canonical layout limit independence. Held-out crop diagnostics are not held-out layout/flight validation.",
        "5. Left/right identification: evaluate centers conditionally on verified two-column membership; field bounds can cross the center. Full/header cases must remain distinguishable/ambiguous rather than forced into two classes.",
        "6. Fixed50% split: see measured normalized_center_x threshold=.5 recalls/false acceptance above. Even if perfect on these selected annotations, do not claim a universal ticket splitter or apply it to full/header detections; bands and raw glyph boxes differ.",
        "7. Same-row peers: center/top/bottom alignment and vertical overlap offer candidates; label-only flag cells have shorter unions, causing center/height asymmetry. Keep alternatives/ambiguity instead of forcing equal heights.",
        "8. Full-width: verified full-layout bands can be wide but their raw text can be narrow/left-aligned. There are only four full cells (including a one-line source heading). Width alone is annotation-sensitive; do not finalize a classifier.",
        "9. Section boundaries: across-section gaps include near-zero gaps and large omitted-image/content intervals. Annotated section labels are finer than physical card groups. Spacing alone is insufficient for robust section recovery; no image-line detection implemented.",
        "10. Upper ticket: distinct spatial regime needed in calibration: header identifiers, airport cell stacks, inline time pairs and progress rows differ from technical vertical stacks. Preserve generic relations; do not hard-code canonical labels/values to select regimes.",
        "11. Axis alignment: AABBs are useful for exploratory features and many boxes are axis-aligned, but the measured centroid/oriented-edge ranking changes mean they are not lossless. Preserve original production polygons; do not finalize equivalence or discard rotation/shear on this evidence. No rotated-layout generalization is established.",
        "12. Separation: combined direction+bounded gap+horizontal alignment is a better technical-stack candidate than any single feature; inline same-row/right-neighbor geometry is a separate candidate. Report candidate recalls/false acceptance, not a semantic pairing success claim.",
        "13. Phase6.2b recommendation: retain source polygons and derive ticket-normalized extents/centers, overlap ratios, signed gaps, direction, and local-neighbor alternatives. Consider same-row/vertical-stack/inline relations independently with uncertain or multiple edges. Candidate center split=.5 is supported only for known two-column objects; peer center-dy<=.02H is promising on manual unions, not validated on automatic field unions. Technical gate overlap>=.5,dx<=.1W,gap0...01H is perfect on bands but loses raw-box recall; inline score>=.9 also fails to transfer. Do not ship these as hard grouping cutoffs. Rank alternatives with ambiguity rather than impose semantic relationships; raw merged/missing detections remain unpaired.",
        "14. Ambiguous relationships: narrow full-width text versus left cell; merged opposite-column OCR labels; country flag/no-text values; inline versus stacked fields; near-touching annotated sections; incomplete/clipped rows; repeated headers. Missing annotations/boxes stay missing, not inferred.",
        "No production code, production label/value pairing, canonical label recognition, field registry, OCR behavior change, VLM/fusion or Phase6.2b/6.3 work was added. No commit/push."]
    for window in (.08,.15,.22):
        counts = Counter()
        for f in fields:
            if not f["label"] or not f["value"]:
                continue
            counts[f["regime"]] += sum(p["sample_id"]==f["sample_id"] and p["field_id"]!=f["field_id"]
                and abs(p["box"]["center_y"]-f["label"]["center_y"])/rois[f["sample_id"]].height<=window for p in parts)
        lines.append(f"Neighborhood sensitivity +/-{window} ticket height false counts={dict(counts)}; window changes candidate population, not truth.")
    examples = [{"field_id":f["field_id"],"sample_id":f["sample_id"],"section":f["section"],"column":f["column"],
        "raw_label":f["raw_label_text"],"raw_value":f["raw_value_text"],"label_box":f["label_box"],"value_box":f["value_box"]} for f in fields if f["label"]]
    lines += ["Verified example identities for inspection only (not used in geometry formulas):",*(str(r) for r in examples)]
    (OUTPUT/"spatial_relationship_report.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    """Run measurement only, preserving all existing source/annotation artifacts."""
    OUTPUT.mkdir(parents=True,exist_ok=True)
    protected = [INPUT/n for n in ("ocr_ground_truth_fields.csv","ocr_calibration_ground_truth.csv","review_fields_source.json","ground_truth_review.md","input_manifest.json")]
    protected += [ROOT/"src/events_diary/ticket_ocr.py",PRODUCTION/"ticket_ocr_results.csv",PRODUCTION/"ticket_ocr_summary.csv"]
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    fields,parts,rois,samples = _load()
    _geometry_table(fields,parts,rois)
    _cell_structure(fields,rois)
    pairs,peers = _relationships(fields,parts,rois)
    distributions = _distributions(fields,parts,rois,pairs,peers)
    gaps = _section_gaps(fields,parts,rois)
    thresholds,held_out = _thresholds(pairs,peers,fields,parts,rois)
    polygons = _polygon_audit(parts,rois,samples,pairs)
    rankings = {"annotation":_score_rankings(pairs,"candidate_score_rankings.csv"),
        "raw_ocr":_score_rankings(polygons[4],"raw_ocr_score_rankings.csv")}
    transfer = _raw_transfer(polygons[4],thresholds)
    _write_report(fields,parts,rois,pairs,peers,distributions,gaps,thresholds,held_out,polygons,rankings,transfer)
    if hashes != {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}:
        raise AssertionError("Read-only calibration modified protected input/production files")
    (OUTPUT/"input_provenance.json").write_text(json.dumps(hashes,indent=2),encoding="utf-8")
    print(f"Crops={len(rois)}, fields={len(fields)}, components={len(parts)}, true pairs={sum(p['is_true'] for p in pairs)}, false competitors={sum(not p['is_true'] for p in pairs)}, peer positives={sum(p['is_true'] for p in peers)}")
    print(f"Read-only input hashes unchanged. Report: {OUTPUT/'spatial_relationship_report.txt'}")


if __name__ == "__main__":
    main()
