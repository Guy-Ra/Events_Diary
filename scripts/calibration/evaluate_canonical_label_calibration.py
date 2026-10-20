"""Phase 6.3a calibration-only FR24 canonical label recognition and association audit.

Read unchanged human-verified fields/components and saved production OCR outputs.
Do not run OCR, regenerate truth, modify src/, build production field models or
tune Phase6.2 formulas. Matching-only text views retain raw detection text and
offsets for audit; no normalization is applied to saved OCR or candidate values.

Compare A whitespace-only exact, B casefold exact, C punctuation/space compact,
D whole-string stable-label grammar (including compound candidates), and E bounded
Levenshtein fallback. Stable grammar accepts only observed label-specific optional
qualifiers: serial's fixed MSN, age's month/year context, and parenthetical aircraft
type tags. It never strips arbitrary parentheses or repairs O/0/I/1 globally.
Unknown text can abstain. All labels/aliases here are FR24 calibration metadata,
not a production registry or generic profile framework.

Locate reference labels by geometry alone: a native detection overlapping a
verified label band by >=.3 of its own AABB area is a reference label candidate.
Multiple label bands can map to one native detection; this exposes merged labels
instead of suppressing the secondary label. Multiple detections at one label band
are inspected as possible splits, not automatically reconstructed. This linkage
uses coarse manual bands, so metrics remain conditional on that declared mapping.

Recognition metrics cover46 occurrences and distinct reference label detections.
False-positive metrics use geometrically mapped verified non-label components;
unannotated OCR is never automatically called negative. Synthetic unknown/variable
suffix probes are separately identified, never counted as human ground truth.

For each reference field, calibration-only regime metadata selects existing stack
or inline relations. All other raw detections remain value candidates; text, truth
values and known labels do not filter/rerank candidates. Compound canonical spans
are character offsets, not pixel boxes. Shared merged geometry is never split or
converted to independent label polygons. Spatial rank success and literal OCR-text
accuracy are reported separately; compound cases remain explicitly ambiguous.

Outputs include every field example, failures, threshold alternatives and five
small review images of actual original crops. No production extraction, semantic
airport roles, view merging, VLM, fusion or Phase6.3b work is implemented.
"""

import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np

from events_diary.ocr_spatial import build_spatial_observations, build_spatial_relations
from scripts.evaluation.evaluate_ocr_spatial import _percentiles, _rehydrate

ROOT = Path(__file__).resolve().parents[2]
TRUTH = ROOT/"data/debug/ocr/ocr_engine_calibration"
SPATIAL = ROOT/"data/debug/ocr/ocr_spatial_relationship_calibration"
OCR = ROOT/"data/debug/ocr/ticket_ocr"
OUTPUT = ROOT/"data/debug/ocr/canonical_label_calibration"
METHODS = ("A_exact","B_casefold","C_compact","D_stable","E_fuzzy","F_hybrid")
THRESHOLDS = (.70,.75,.80,.85,.90,.95)
WORKING_THRESHOLD = .90
WORKING_MARGIN = .10
MONTHS = "JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC"
RENAMES = {"COUNTRY OF REG.":"COUNTRY_OF_REGISTRATION","SERIAL NUMBER (MSN)":"SERIAL_NUMBER",
    "BAROMETRIC ALT.":"BAROMETRIC_ALTITUDE","ICAO 24-BIT ADDRESS":"ICAO_ADDRESS","FIR/UIR":"FIR_UIR"}


def _read(path):
    """Read UTF-8 raw artifacts without changing whitespace or punctuation."""
    with path.open(encoding="utf-8-sig",newline="") as stream:
        return list(csv.DictReader(stream))


def _csv(name,rows):
    """Write debug-only heterogeneous scalar tables, preserving raw text columns."""
    with (OUTPUT/name).open("w",encoding="utf-8",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def _ws(text):
    """Create a whitespace-only matching view; never overwrite literal source text."""
    return " ".join(text.split())


def _compact(text):
    """Create a case-insensitive alphanumeric comparison view, with no glyph repair."""
    return "".join(c.upper() for c in text if c.isalnum())


def _distance(a,b):
    """Compute standard Levenshtein edit distance without an NLP dependency."""
    row = list(range(len(b)+1))
    for i,left in enumerate(a,1):
        following = [i]
        for j,right in enumerate(b,1):
            following.append(min(following[-1]+1,row[j]+1,row[j-1]+(left!=right)))
        row = following
    return row[-1]


def _identity(label):
    """Map only human-verified visible labels to calibration canonical identities."""
    if re.fullmatch(r"AGE \([A-Za-z]{3} \d{4}\)",label):
        return "AGE"
    if re.fullmatch(r"AIRCRAFT TYPE \([A-Z0-9]+\)",label):
        return "AIRCRAFT_TYPE"
    return RENAMES.get(label,re.sub(r"[^A-Z0-9]+","_",label).strip("_"))


def _specs(fields):
    """Derive visible variants and regimes from verified fields, not OCR guesses."""
    specs = {}
    for f in fields:
        if not f["raw_label_text"]:
            continue
        canonical = _identity(f["raw_label_text"])
        f["canonical"] = canonical
        label,value = json.loads(f["label_box"]),json.loads(f["value_box"])
        regime = ("no_text_value" if value is None else "inline" if label[1]==value[1]
            else "full_width_stack" if f["column"]=="full" else "stack")
        f["observed_regime"] = regime
        spec = specs.setdefault(canonical,{"canonical":canonical,"variants":set(),"regimes":set(),"count":0})
        spec["variants"].add(f["raw_label_text"])
        spec["regimes"].add(regime)
        spec["count"]+=1
    for canonical,spec in specs.items():
        visible = min(spec["variants"])
        stable = ("AGE" if canonical=="AGE" else "AIRCRAFT TYPE" if canonical=="AIRCRAFT_TYPE"
            else "SERIAL NUMBER" if canonical=="SERIAL_NUMBER" else visible)
        spec["stable"] = _compact(stable)
        spec["stable_visible"] = stable
        spec["suffix_rule"] = ("optional month/year, not a fixed date" if canonical=="AGE" else
            "optional parenthetical alphanumeric type tag; one verified tag only" if canonical=="AIRCRAFT_TYPE" else
            "optional fixed MSN qualifier" if canonical=="SERIAL_NUMBER" else "none")
    return specs


def _stable_parse(text,specs):
    """Recognize complete known-label chains and preserve literal character spans.

    Requiring full compact-string coverage prevents substring matches inside
    arbitrary values/words. Character spans do not imply sub-detection pixel boxes.
    """
    positions = [i for i,c in enumerate(text) if c.isalnum()]
    compact = _compact(text)
    offset,hits = 0,[]
    while offset<len(compact):
        possible = []
        for canonical,spec in specs.items():
            stable = spec["stable"]
            if not compact.startswith(stable,offset):
                continue
            end = offset+len(stable)
            if canonical=="AGE":
                suffix = re.match(r"(?:"+MONTHS+r")\d{4}",compact[end:])
                end += len(suffix[0]) if suffix else 0
            elif canonical=="SERIAL_NUMBER" and compact.startswith("MSN",end):
                end+=3
            elif canonical=="AIRCRAFT_TYPE":
                original_end = positions[end-1]+1
                tag = re.match(r"\s*\(\s*([A-Za-z0-9]{2,6})\s*\)",text[original_end:])
                if tag:
                    end += len(_compact(tag[1]))
            possible.append((end,canonical))
        if not possible:
            return []
        end,canonical = max(possible)
        start_char = positions[offset]
        end_char = positions[end] if end<len(positions) else len(text)
        hits.append({"canonical":canonical,"score":1.,"span_start":start_char,"span_end":end_char,
            "raw_span":text[start_char:end_char],"method":"stable"})
        offset = end
    return hits


def _recognize(text,specs,method,threshold=WORKING_THRESHOLD,margin=WORKING_MARGIN):
    """Return abstaining/compound label candidates without altering raw OCR text."""
    if method in METHODS[:3]:
        normalize = _ws if method==METHODS[0] else lambda t:_ws(t).casefold() if method==METHODS[1] else _compact(t)
        candidates = [c for c,s in specs.items() if any(normalize(text)==normalize(v) for v in s["variants"])]
        return {"hits":[{"canonical":c,"score":1.,"span_start":0,"span_end":len(text),"raw_span":text,"method":method} for c in candidates],
            "ambiguous":len(candidates)>1,"compound":False,"margin":None,"fuzzy_scores":[]}
    hits = _stable_parse(text,specs) if method in ("D_stable","F_hybrid") else []
    if hits or method=="D_stable":
        return {"hits":hits,"ambiguous":len(hits)>1,"compound":len(hits)>1,"margin":None,"fuzzy_scores":[]}
    compact = _compact(text)
    scores = []
    for canonical,spec in specs.items():
        variants = {_compact(v) for v in spec["variants"]}|{spec["stable"]}
        score = max((1-_distance(compact,v)/max(len(compact),len(v),1) for v in variants),default=0)
        scores.append((score,canonical))
    scores.sort(reverse=True)
    best,second = scores[0][0],scores[1][0]
    eligible = [(s,c) for s,c in scores if s>=threshold]
    ambiguous = len(eligible)>1 or bool(eligible and best-second<margin)
    hits = [{"canonical":c,"score":s,"span_start":0,"span_end":len(text),"raw_span":text,"method":"bounded_fuzzy"}
        for s,c in eligible if best-s<margin] if eligible else []
    # Keep ambiguity visible; downstream dry run abstains rather than force a label.
    return {"hits":hits,"ambiguous":ambiguous,"compound":False,"margin":best-second,"fuzzy_scores":scores[:3]}


def _bounds(polygon,roi):
    """Read a saved source polygon as a local AABB for reference linkage only."""
    points = np.asarray(polygon,dtype=float)-[roi.x,roi.y]
    x,y = points.min(axis=0)
    right,bottom = points.max(axis=0)
    return float(x),float(y),float(right),float(bottom)


def _link(fields,records):
    """Locate all raw label fragments/compound occurrences by reference-band overlap."""
    links,truth_by_detection = {},defaultdict(set)
    for f in fields:
        if not f["raw_label_text"]:
            continue
        _,result,_,_ = records[f["sample_id"]]
        x,y,w,h = json.loads(f["label_box"])
        candidates = []
        for i,d in enumerate(result.detections):
            left,top,right,bottom = _bounds(d.source_polygon,result.crop_roi)
            fraction = max(0,min(right,x+w)-max(left,x))*max(0,min(bottom,y+h)-max(top,y))/((right-left)*(bottom-top))
            if fraction>=.3:
                candidates.append((fraction,i))
                truth_by_detection[f["sample_id"],i].add(f["canonical"])
        links[f["field_id"]] = sorted(candidates,reverse=True)
    return links,truth_by_detection


def _unknowns(parts,truth_by_detection,records):
    """Use only mapped human non-label text as negatives; leave unannotated text unscored."""
    component_by_id = {p["annotation_id"]:p for p in parts}
    negatives = {}
    for row in _read(SPATIAL/"ocr_component_geometry_mapping.csv"):
        part = component_by_id.get(row["assigned_annotation"])
        if part is None or part["component"]=="label":
            continue
        sid,index = row["sample_id"],int(row["detection_index"])
        if (sid,index) in truth_by_detection:
            continue
        negatives[sid,index] = {"sample_id":sid,"detection_index":index,"kind":"human_verified_non_label",
            "text":records[sid][1].detections[index].text,"annotation_id":part["annotation_id"]}
    synthetic = ["arbitrary text","16 years","1100","281°","9 kts","Israir Airlines","ISR972","TLV",
        "NEW YORK","Airbus A330-243","THE QUICK BROWN FOX","AGELESS","MACHINERY","GROUND SPEEDING","TRACKING","SCHEDULED123"]
    return list(negatives.values()),[{"sample_id":"synthetic","detection_index":i,"kind":"synthetic_unknown","text":t,"annotation_id":""} for i,t in enumerate(synthetic)]


def _metrics(specs,records,links,truth_by_detection,negatives,synthetic):
    """Compare strategies/thresholds with occurrence recall and explicit unknown rejection."""
    rows,matches,per_label,unknown_rows = [],[],[],[]
    for method in METHODS:
        thresholds = THRESHOLDS if method in ("E_fuzzy","F_hybrid") else (1.,)
        for threshold in thresholds:
            predictions = {}
            for (sid,index),expected in truth_by_detection.items():
                text = records[sid][1].detections[index].text
                prediction = _recognize(text,specs,method,threshold)
                predictions[sid,index] = prediction
                hits = {h["canonical"] for h in prediction["hits"]}
                matches.append({"sample_id":sid,"detection_index":index,"raw_text":text,"method":method,
                    "threshold":threshold,"expected_canonical":json.dumps(sorted(expected)),"canonical_label":json.dumps(sorted(hits)),
                    "match_score":max((h["score"] for h in prediction["hits"]),default=None),
                    "verified_correct":hits==expected,"ambiguity":prediction["ambiguous"],"compound":prediction["compound"],
                    "best_second_margin":prediction["margin"],"matched_spans":json.dumps(prediction["hits"],ensure_ascii=False)})
            occurrence = []
            for canonical in specs:
                associated = [(sid,index) for (sid,index),truth in truth_by_detection.items() if canonical in truth]
                recovered = sum(any(h["canonical"]==canonical for h in predictions[key]["hits"]) for key in associated)
                strict = sum(len(predictions[key]["hits"])==1 and not predictions[key]["ambiguous"]
                    and predictions[key]["hits"][0]["canonical"]==canonical for key in associated)
                occurrence.append((canonical,len(associated),recovered))
                per_label.append({"canonical":canonical,"method":method,"threshold":threshold,"mapped_occurrences":len(associated),
                    "recognized_occurrences":recovered,"recall":recovered/specs[canonical]["count"],
                    "strict_top1_occurrences":strict,"strict_top1_accuracy":strict/specs[canonical]["count"],
                    "ambiguous_occurrences":sum(predictions[key]["ambiguous"] for key in associated)})
            false_counts = Counter()
            for item in [*negatives,*synthetic]:
                prediction = _recognize(item["text"],specs,method,threshold)
                false_counts[item["kind"]] += bool(prediction["hits"])
                unknown_rows.append({**item,"method":method,"threshold":threshold,"false_canonical_assignment":bool(prediction["hits"]),
                    "prediction":json.dumps(prediction["hits"],ensure_ascii=False),"ambiguous":prediction["ambiguous"]})
            truth_count = sum(v[1] for v in occurrence)
            margins = [p["margin"] for p in predictions.values() if p["margin"] is not None]
            rows.append({"method":method,"threshold":threshold,"mapped_label_detections":len(predictions),"mapped_label_occurrences":truth_count,
                "recognized_occurrences":sum(v[2] for v in occurrence),"canonical_recall":sum(v[2] for v in occurrence)/46,
                "detection_exact_set_accuracy":sum({h["canonical"] for h in p["hits"]}==truth_by_detection[key] for key,p in predictions.items())/len(predictions),
                "strict_top1_canonical_accuracy":sum(len(p["hits"])==1 and not p["ambiguous"] and p["hits"][0]["canonical"] in truth_by_detection[key] for key,p in predictions.items())/46,
                "wrong_canonical_assignments":sum(len({h["canonical"] for h in p["hits"]}-truth_by_detection[key]) for key,p in predictions.items()),
                "abstained_label_detections":sum(not p["hits"] for p in predictions.values()),
                "ambiguous_detection_rate":sum(p["ambiguous"] for p in predictions.values())/len(predictions),
                "missed_occurrence_rate":1-sum(v[2] for v in occurrence)/46,
                "human_unknown_count":len(negatives),"human_unknown_false_matches":false_counts["human_verified_non_label"],
                "human_unknown_false_rate":false_counts["human_verified_non_label"]/len(negatives),
                "synthetic_unknown_count":len(synthetic),"synthetic_unknown_false_matches":false_counts["synthetic_unknown"],
                **({"margin_"+k:v for k,v in _percentiles(margins).items()} if margins else {})})
    _csv("canonical_label_thresholds.csv",rows)
    _csv("canonical_label_matches.csv",matches)
    _csv("canonical_label_per_label.csv",per_label)
    _csv("unknown_label_controls.csv",unknown_rows)
    return rows,per_label


def _errors(fields,records,links,truth_by_detection,specs):
    """Classify literal label OCR differences without editing their transcription."""
    rows = []
    for f in fields:
        if not f["raw_label_text"]:
            continue
        located = links[f["field_id"]]
        raws = [records[f["sample_id"]][1].detections[i].text for _,i in located]
        kinds = []
        compound = any(len(truth_by_detection[f["sample_id"],i])>1 for _,i in located)
        if not located:
            kinds.append("missing_label")
        if compound:
            kinds.append("merged_labels")
        if len(located)>1:
            kinds.append("possible_split_label")
        if raws:
            raw = raws[0]
            if compound:
                spans = _stable_parse(raw,specs)
                raw = next((h["raw_span"] for h in spans if h["canonical"]==f["canonical"]),raw)
            expected = f["raw_label_text"]
            if raw!=expected and _compact(raw)==_compact(expected):
                if "".join(raw.split()).casefold()=="".join(expected.split()).casefold():
                    kinds.append("spacing_or_case")
                if _ws(raw).casefold()!=_ws(expected).casefold():
                    kinds.append("punctuation_or_spacing")
                if raw.casefold()==expected.casefold() and raw!=expected:
                    kinds.append("capitalization")
            elif _compact(raw)!=_compact(expected):
                kinds.append("truncated_label" if _compact(expected).startswith(_compact(raw)) else "character_substitution_or_extra_text")
                if "(" in expected:
                    kinds.append("parenthetical_error_or_merge")
            if not kinds:
                kinds.append("literal_exact")
        rows.append({"field_id":f["field_id"],"sample_id":f["sample_id"],"canonical":f["canonical"],
            "verified_label":f["raw_label_text"],"raw_ocr_fragments":json.dumps(raws,ensure_ascii=False),
            "raw_detection_indexes":json.dumps([i for _,i in located]),"error_categories":";".join(kinds),
            "separate_label_geometry_available":bool(located) and not compound})
    _csv("ocr_label_error_analysis.csv",rows)
    return rows


def _dry_run(fields,records,links,specs,mapping):
    """Rank existing raw spatial neighbors after recognition, never tune/filter candidates."""
    rows = []
    for f in fields:
        if not f["raw_label_text"]:
            continue
        sid = f["sample_id"]
        _,result,_,relations = records[sid]
        located = links[f["field_id"]]
        source = located[0][1] if located else None
        raw = result.detections[source].text if source is not None else ""
        recognition = _recognize(raw,specs,"F_hybrid")
        recognized = any(h["canonical"]==f["canonical"] for h in recognition["hits"])
        no_value = json.loads(f["value_box"]) is None
        target = mapping.get((sid,f["field_id"]+".value"))
        target = target[1] if target else None
        score = "inline_score" if f["observed_regime"]=="inline" else "vertical_stack_score"
        candidates = sorted((r for r in relations if r.source_index==source),key=lambda r:(-getattr(r,score),r.target_index)) if recognized and not no_value else []
        rank,margin = None,None
        if target is not None and candidates:
            true_score = next(getattr(r,score) for r in candidates if r.target_index==target)
            rank = 1+sum(getattr(r,score)>true_score+1e-12 for r in candidates)
            margin = true_score-max(getattr(r,score) for r in candidates if r.target_index!=target)
        compound = recognition["compound"]
        top = candidates[0].target_index if candidates else None
        chosen = result.detections[top].text if top is not None else ""
        outcome = ("NON_TEXT_VALUE_NO_ASSOCIATION" if no_value else "MISSING_OR_UNRECOGNIZED_LABEL" if not recognized else
            "FUZZY_CANONICAL_AMBIGUOUS" if recognition["ambiguous"] and not compound else
            "COMPOUND_GEOMETRY_AMBIGUOUS" if compound else "CORRECT_LOCATION" if top==target else "WRONG_TOP1_TOP3" if rank is not None and rank<=3 else "WRONG_OR_UNAVAILABLE")
        rows.append({"sample_id":sid,"field_id":f["field_id"],"column":f["column"],"canonical":f["canonical"],
            "verified_label":f["raw_label_text"],"raw_ocr_label":raw,"source_index":source,"recognized":recognized,
            "recognized_canonical":json.dumps([h["canonical"] for h in recognition["hits"]]),"compound":compound,
            "observed_regime":f["observed_regime"],"score_name":score,"verified_value":f["raw_value_text"],
            "value_status":f["value_status"],"no_text_value":no_value,"verified_value_index":target,
            "top_value_candidate":chosen,"top_value_index":top,"verified_rank":rank,"score_margin":margin,
            "potential_top1":top==target and target is not None,"potential_top3":rank is not None and rank<=3,
            "literal_value_text_exact":chosen==f["raw_value_text"] if not no_value and candidates else None,
            "result":outcome,"top_candidates":json.dumps([{"index":r.target_index,"raw_text":result.detections[r.target_index].text,
                "score":getattr(r,score)} for r in candidates[:3]],ensure_ascii=False),
            "label_box":f["label_box"],"value_box":f["value_box"]})
    _csv("field_association_dry_run.csv",rows)
    _csv("example_review_table.csv",rows)
    return rows


def _previews(rows,records,samples):
    """Draw five actual-crop reviews; compound spans never become guessed sub-boxes."""
    selected = [("clean_technical","TRACK",False),("clean_inline","SCHEDULED",False),
        ("two_column","VERTICAL_SPEED",False),("merged_serial_failure","SERIAL_NUMBER",True),("compound_age_ambiguity","AGE",True)]
    paths = []
    (OUTPUT/"review").mkdir(exist_ok=True)
    for name,canonical,compound in selected:
        row = next(r for r in rows if r["canonical"]==canonical and r["compound"]==compound)
        sid = row["sample_id"]
        _,result,_,_ = records[sid]
        crop = cv2.imread(str(ROOT/samples[sid]["input_path"]))
        canvas = np.full((crop.shape[0]+145,crop.shape[1]*2,3),245,np.uint8)
        # Copy original pixels twice for uncluttered OCR versus truth/alternative panels.
        canvas[145:,0:crop.shape[1]]=crop
        canvas[145:,crop.shape[1]:]=crop
        legend = [f"{canonical}  {row['field_id']}  {sid}","LEFT: cyan=raw label polygon; green=top spatial candidate",
            "RIGHT: yellow=alternative candidate; red=verified value band",f"Outcome: {row['result']}","Compound text spans have NO separate pixel geometry."]
        for i,text in enumerate(legend):
            cv2.putText(canvas,text,(5,18+i*24),cv2.FONT_HERSHEY_SIMPLEX,.40,(20,20,20),1,cv2.LINE_AA)
        def draw_detection(index,color,panel,result=result,crop=crop,canvas=canvas):
            """Overlay the original native polygon after translation only."""
            polygon = np.asarray(result.detections[index].source_polygon)-[result.crop_roi.x,result.crop_roi.y]
            polygon += [panel*crop.shape[1],145]
            cv2.polylines(canvas,[np.rint(polygon).astype(np.int32)],True,color,2)
        if row["source_index"] is not None:
            draw_detection(row["source_index"],(240,170,0),0)
        candidates = json.loads(row["top_candidates"])
        if candidates:
            draw_detection(candidates[0]["index"],(0,180,0),0)
        for candidate in candidates[1:]:
            draw_detection(candidate["index"],(0,190,220),1)
        if row["value_box"]!="null":
            x,y,w,h = json.loads(row["value_box"])
            cv2.rectangle(canvas,(x+crop.shape[1],y+145),(x+w+crop.shape[1],y+h+145),(0,0,230),2)
        path = OUTPUT/"review"/(name+".png")
        if not cv2.imwrite(str(path),canvas):
            raise OSError(path)
        paths.append(path)
    return paths


def _report(specs,metrics,per_label,errors,rows,paths,links,truth_by_detection,probes):
    """Write explicit metric definitions, every human-review example and conservative advice."""
    summaries = []
    for regime in ("stack","inline","overall"):
        group = [r for r in rows if not r["no_text_value"] and (regime=="overall" or
            (r["observed_regime"]=="inline" if regime=="inline" else r["observed_regime"]!="inline"))]
        ranks = [r["verified_rank"] for r in group if r["verified_rank"] is not None]
        margins = [r["score_margin"] for r in group if r["score_margin"] is not None]
        summaries.append({"regime":regime,"verified_pairs":len(group),"ranked_pairs":len(ranks),
            "potential_top1":sum(r["potential_top1"] for r in group),"potential_top3":sum(r["potential_top3"] for r in group),
            "noncompound_top1":sum(r["potential_top1"] and not r["compound"] for r in group),
            "compound_abstentions":sum(r["compound"] for r in group),"mean_rank":sum(ranks)/len(ranks),
            **{"rank_"+k:v for k,v in _percentiles(ranks).items()},**{"margin_"+k:v for k,v in _percentiles(margins).items()}})
    _csv("dry_run_summary.csv",summaries)
    catalog = [{"canonical":c,"visible_variants":json.dumps(sorted(s["variants"]),ensure_ascii=False),"stable_token":s["stable_visible"],
        "suffix_rule":s["suffix_rule"],"observed_regimes":";".join(sorted(s["regimes"])),"verified_occurrences":s["count"]} for c,s in sorted(specs.items())]
    _csv("canonical_label_catalog.csv",catalog)
    split = sum(len(v)>1 for v in links.values())
    merged = sum(len(v)>1 for v in truth_by_detection.values())
    lines = ["Phase6.3a Canonical Label Recognition Calibration -- NOT production extraction",
        f"46 verified labels; {len(specs)} canonical identities; {len(truth_by_detection)} distinct geometrically linked raw label detections; merged detections={merged}; possible split occurrences={split}; missing literal label occurrences={sum(not v for v in links.values())}.",
        "All verified labels/values/boxes unchanged; source hashes checked. Match normalization affects comparison views only, never raw OCR or values.",
        "Label reference linkage is geometric overlap/native-area>=.3 against verified bands; it is not recognized-text matching. Compound native boxes can support two reference labels. Separate missing geometry from missing literal text.",
        "Canonical variants/regimes:",*(str(r) for r in catalog),
        "Strategies: A=whitespace-only literal; B=casefold literal; C=casefold alphanumeric compact; D=complete stable-label grammar with supported suffixes and compound chains; E=pure whole-detection Levenshtein1-distance/maxlen; F=D first, otherwise E. No O/0/I/1 repair. Pure fuzzy and hybrid results are separate so exact matches cannot masquerade as fuzzy gains.",
        "Stable suffix scope: AGE optional month/year (not Mar2010 constant); SERIAL_NUMBER optional fixedMSN; AIRCRAFT_TYPE parenthetical alphanumeric tag (only one verified instance). Other parentheses are never globally dropped. Synthetic alternative dates/tags are probes, not verified accuracy examples.",
        "Canonical SCHEDULED is shared by left/right occurrences; no departure/arrival semantics assigned. Data-source heading/airport cells without verified label boxes are NOT added as canonical labels.",
        "Recognition metrics: occurrence recall denominator46. Detection exact-set accuracy requires all and only expected labels in the native box. Strict top1 accuracy requires one unambiguous correct identity; compound boxes abstain instead of forcing one label. Ambiguity includes compound candidates. Fuzzy margin is best-minus-second canonical score, undefined for exact/stable recognition and compounds; not OCR/field confidence.",
        "Unknown metrics score only geometrically mapped human non-label detections; unannotated detections are unscored. Synthetic controls are separate. This is not a whole-ticket false-positive benchmark.",
        "Matching results:",*(str(r) for r in metrics),"Per-canonical performance at selected calibration strategy:",
        *(str(r) for r in per_label if r["method"]=="F_hybrid" and r["threshold"]==WORKING_THRESHOLD),
        "Literal OCR errors:",*(str(r) for r in errors),"Synthetic suffix/substitution probes:",*(str(r) for r in probes),
        "Association dry-run: fixed Phase6.2 production relations; all other detections are candidates. No candidate text correction/filter, no new column boxes, no formula tuning. Calibration regime is derived from verified field geometry, not a production registry.",
        "Potential rank metrics are spatial target-location accuracy, NOT verified value-string accuracy. No-text/flag/lock fields never receive a value candidate. Compound geometry produces diagnostic ranks but always a compound abstention.",
        "Dry-run summaries:",*(str(r) for r in summaries),"", "Human Review Examples",""]
    table = ["| Crop / field / column | Raw OCR label | Canonical / regime | Top candidate | Verified value | Rank | Result |",
        "|---|---|---|---|---|---|---|"]
    def escape(value):
        """Escape literal source strings for a readable Markdown review table."""
        return str(value).replace("|","\\|").replace("\n","<br>")
    for r in rows:
        table.append("| "+" | ".join(escape(v) for v in (r["sample_id"]+" / "+r["field_id"]+" / "+r["column"],r["raw_ocr_label"],r["canonical"]+" / "+r["observed_regime"],r["top_value_candidate"],r["verified_value"],r["verified_rank"],r["result"]))+" |")
        lines += [f"{r['field_id']} {r['canonical']} ({r['column']}, {r['sample_id']}):",f"Raw OCR label: {r['raw_ocr_label']!r}",
            f"Recognized canonical candidates: {r['recognized_canonical']}; regime={r['observed_regime']}; compound={r['compound']}",
            "Candidate values: "+r["top_candidates"],f"Verified value: {r['verified_value']!r}; rank={r['verified_rank']}; result={r['result']}; literal OCR value exact={r['literal_value_text_exact']}"]
    (OUTPUT/"example_review_table.md").write_text("\n".join(table)+"\n",encoding="utf-8")
    lines += ["",*table,"", "Failure interpretation:",
        "Fuzzy-only matching adds no recovery over compact whole-label matching on these actual labels: the four lost occurrences belong to two compound detections. Stable compound grammar recovers the canonical identities, not independent geometry.",
        "At fuzzy thresholds.70/.75, synthetic GROUND SPEEDING and SCHEDULED123 are wrongly proposed as GROUND_SPEED/SCHEDULED (scores.7857/.75). At.90 these probes abstain. All105 human non-label controls abstain across the evaluated grid; this limited result does not establish general precision.",
        "Synthetic1CAO 24-BIT ADDRESS is accepted by fuzzy fallback as ICAO_ADDRESS at.9375, illustrating indirect substitution tolerance. Raw1CAO text is not corrected. This is a probe, not a verified OCR label example.",
        "Merged SERIAL_NUMBER/AGE strings can contain both canonical names at exact character spans. Both candidates share one original polygon; character offsets do NOT locate separate labels in pixels. Serial's true1100 can rank second behind16years. Age's true value may rank first, but this does not resolve compound ownership. All compound examples remain ambiguous.",
        "Missing-label examples: distinguish the two age labels missing as separate detections from text that is entirely absent. If all46 literal labels are present in this dataset, do not fabricate a naturally missing example; UNKNOWN controls demonstrate abstention separately.",
        "Split-label reconstruction is not implemented. Multiple geometric fragments are reported as potential splits for inspection; no absence/presence is guessed from joined OCR strings.",
        "Non-text special fields COUNTRY_OF_REGISTRATION/FIR_UIR preserve flag/lock status without forced text-value selection. Header/airline/identifier text remains UNKNOWN unless it actually matches a known label.",
        "Recommendation for6.3b: exact compact label recognition plus label-specific variable-suffix handling; retain compound candidates and raw text/provenance. Fuzzy>=.90 with margin>=.10 is a conservative candidate-only fallback, not a proven production operating point. Prefer UNKNOWN on weak/tied matches. The small in-sample label set and coarse geometry cannot establish general recognition precision.",
        "Do not infer departure/arrival from repeated SCHEDULED; do not split merged polygons using character widths or finalize their value associations. Keep technical and inline ranking separate; preserve OCR text errors even when spatial target is correct. Additional diverse labels/layouts and true missing/split/near-label unknowns remain unvalidated.",
        "Review images:",*(str(p) for p in paths),
        "No production registry/field model/extraction, generic UI framework, dependencies, VLM/fusion, Phase6.3b, commit or push."]
    (OUTPUT/"canonical_label_report.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return summaries


def main():
    """Measure saved evidence only and preserve all production/verified input bytes."""
    OUTPUT.mkdir(parents=True,exist_ok=True)
    protected = [TRUTH/n for n in ("ocr_ground_truth_fields.csv","ocr_calibration_ground_truth.csv","input_manifest.json","ground_truth_review.md","review_fields_source.json")]
    protected += [OCR/"ticket_ocr_results.csv",OCR/"ticket_ocr_summary.csv",SPATIAL/"ocr_component_geometry_mapping.csv"]
    protected += list((ROOT/"src/events_diary").glob("*.py"))
    protected += [ROOT/p for p in ("tests/test_ticket_ocr.py","tests/test_ocr_spatial.py",
        "scripts/calibration/evaluate_ocr_engine_calibration.py","scripts/calibration/evaluate_ocr_spatial_relationships.py",
        "scripts/evaluation/evaluate_ticket_ocr.py","scripts/evaluation/evaluate_ocr_spatial.py")]
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    fields,parts = _read(TRUTH/"ocr_ground_truth_fields.csv"),_read(TRUTH/"ocr_calibration_ground_truth.csv")
    if len(fields)!=94 or len(parts)!=151 or any(r["review_status"]!="human_verified" for r in fields+parts):
        raise ValueError("Expected unchanged human-verified94 fields and151 components")
    specs = _specs(fields)
    manifest = json.loads((TRUTH/"input_manifest.json").read_text(encoding="utf-8"))
    samples = {s["sample_id"]:s for s in manifest["samples"]}
    records = {}
    for summary,result in _rehydrate():
        name = summary["video_name"].removesuffix("_test.mp4")
        sid = f"{name}_i{result.investigation_index}_c{result.candidate_index}"
        observations = build_spatial_observations(result)
        records[sid] = summary,result,observations,build_spatial_relations(observations)
    links,truth_by_detection = _link(fields,records)
    negatives,synthetic = _unknowns(parts,truth_by_detection,records)
    metrics,per_label = _metrics(specs,records,links,truth_by_detection,negatives,synthetic)
    all_candidates = []
    negative_ids = {(r["sample_id"],r["detection_index"]) for r in negatives}
    for sid,(_,result,_,_) in records.items():
        for i,detection in enumerate(result.detections):
            prediction = _recognize(detection.text,specs,"F_hybrid")
            status = "verified_label" if (sid,i) in truth_by_detection else "verified_non_label" if (sid,i) in negative_ids else "unannotated_unscored"
            all_candidates.append({"sample_id":sid,"detection_index":i,"raw_text":detection.text,
                "reference_status":status,"canonical_candidates":json.dumps(prediction["hits"],ensure_ascii=False),
                "compound":prediction["compound"],"ambiguous":prediction["ambiguous"]})
    _csv("all_ocr_label_candidates.csv",all_candidates)
    errors = _errors(fields,records,links,truth_by_detection,specs)
    mapping = {}
    for r in _read(SPATIAL/"ocr_component_geometry_mapping.csv"):
        if r["assigned_annotation"]:
            key = r["sample_id"],r["assigned_annotation"]
            value = float(r["overlap_detection_fraction"]),int(r["detection_index"])
            if key not in mapping or value[0]>mapping[key][0]:
                mapping[key] = value
    rows = _dry_run(fields,records,links,specs,mapping)
    probes = [{"kind":"synthetic_probe_not_human_truth","raw_text":text,"prediction":json.dumps(_recognize(text,specs,"F_hybrid"),ensure_ascii=False)}
        for text in ("AGE (Jul 2024)","AGE (Feb 1999)","AGE (unknown)","AIRCRAFT TYPE (B738)","SERIAL NUMBER (XYZ)","1CAO 24-BIT ADDRESS")]
    _csv("synthetic_suffix_probes.csv",probes)
    paths = _previews(rows,records,samples)
    summaries = _report(specs,metrics,per_label,errors,rows,paths,links,truth_by_detection,probes)
    if hashes!={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}:
        raise AssertionError("Canonical calibration modified protected production or verified artifacts")
    (OUTPUT/"input_provenance.json").write_text(json.dumps(hashes,indent=2),encoding="utf-8")
    print(f"Verified labels=46; canonical identities={len(specs)}; raw label detections={len(truth_by_detection)}; human unknown controls={len(negatives)}")
    for r in metrics:
        print(f"{r['method']} threshold={r['threshold']}: recall={r['canonical_recall']:.4f}, strict_top1={r['strict_top1_canonical_accuracy']:.4f}, human_unknown_FP={r['human_unknown_false_matches']}, synthetic_FP={r['synthetic_unknown_false_matches']}")
    for r in summaries:
        print(json.dumps(r))
    print(f"Report: {OUTPUT/'canonical_label_report.txt'}")


if __name__=="__main__":
    main()
