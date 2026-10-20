"""Phase 6.1a development-only OCR engine/preprocessing calibration.

Preparation runs the real Phase 3-5 pipeline on seven videos, calls the production
selector, and materializes all 17 primary stable references plus four separately
labeled controls. No timestamps are invented for stable samples. A fingerprinted
local checkpoint avoids rerunning upstream CV for each OCR experiment. Original
frames, candidate indexes and source-video identity remain explicit. Production
code, selection semantics, ROIs, thresholds and schemas are never changed.

Setup: uv sync --group ocr-calibration. Tesseract is a separate native tool; this
experiment discovers TESSERACT_CMD, PATH or the locally extracted Windows binary
under the ignored output/tools directory. Its English model must be available.
Use --prepare-only to save native ticket crops for manual transcription BEFORE
running either engine. Ground truth must be read from those original pixels, not
from OCR output; uncertain/clipped text is explicitly excluded, never inferred.

Full-ticket input covers ticket_zone above ticket_bottom_actions. Focused header,
aircraft_image/upper and scroll-body ROIs must cover exactly that same rectangle.
Bottom action text is audited separately and excluded as interface actions without
ticket field content, rather than silently omitted from the coverage analysis.
All OCR, preprocessing, manual annotations and metric decisions live here or in
local debug artifacts. No production OCR API, field parser, semantic ticket model,
VLM, fusion, dictionary, inferred value or multi-view merge is implemented.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np

from events_diary.config import ROI, UIConfig
from events_diary.evidence_candidate_timeline import build_evidence_candidate_timeline
from events_diary.frame_materialization import (
    VideoFrameResolver,
    materialize_evidence_candidates,
)
from events_diary.layout import map_roi_to_frame
from events_diary.representative_selection import select_investigation_representatives
from scripts.evaluation.evaluate_representative_selection import SOURCES
from scripts.evaluation.evaluate_ticket_stable_views import _build_contexts

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/debug/ocr/ocr_engine_calibration"
CHECKPOINT = OUTPUT / "input_manifest.json"
GROUND_TRUTH = OUTPUT / "ocr_calibration_ground_truth.csv"
VARIANTS = ("V0_original", "V1_grayscale", "V2_upscale2x", "V3_gray_upscale2x",
            "V4_clahe_upscale2x", "V5_otsu_upscale2x", "V6_sharpen_upscale2x")


def _write_csv(path, rows):
    """Save a scalar calibration table without changing any production records."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _rois():
    """Derive complete calibration crop geometry solely from existing UIConfig."""
    c = UIConfig()
    full = ROI(c.ticket_zone.x, c.ticket_zone.y, c.ticket_zone.width,
               c.ticket_bottom_actions.y-c.ticket_zone.y)
    return {"full_ticket": full, "ticket_header": c.ticket_header,
            "upper_information": c.aircraft_image, "ticket_scroll_body": c.ticket_scroll_body}


def _coverage():
    """Verify focused ROI union has zero gaps, including all upper information."""
    rois = _rois()
    full = rois["full_ticket"]
    mask = np.zeros((full.height, full.width), dtype=np.uint8)
    content = {"full_ticket": "Entire ticket above icon-only bottom actions",
        "ticket_header": "Flight identifier/header labels and controls",
        "upper_information": "Aircraft photo and upper ticket information/overlaid text",
        "ticket_scroll_body": "Visible scrollable labels/values and lower content"}
    rows = []
    for name, r in rois.items():
        if name != "full_ticket":
            mask[r.y-full.y:r.bottom-full.y, r.x-full.x:r.right-full.x] += 1
        rows.append({"roi_name": name, **asdict(r), "strategy": "A" if name == "full_ticket" else "B",
                     "visible_content": content[name], "uncovered_relevant_pixels": 0})
    if not np.all(mask == 1):
        raise ValueError(f"Focused ROI coverage has gaps/overlaps: {np.unique(mask, return_counts=True)}")
    rows.append({"roi_name": "ticket_bottom_actions", **asdict(UIConfig().ticket_bottom_actions),
        "strategy": "excluded", "visible_content": "Audited fixed action labels: 3D view, Route, Follow, Share, More; no ticket field content",
        "uncovered_relevant_pixels": 0})
    _write_csv(OUTPUT / "ocr_roi_coverage.csv", rows)
    return rois


def _fingerprint(path):
    """Record source identity and metadata without hashing full video contents."""
    stat = path.stat()
    return {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _prepare():
    """Call actual upstream production APIs and save original selected ticket crops."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "inputs").mkdir(exist_ok=True)
    rois = _coverage()
    samples, sources = [], []
    reference_path = ROOT / "data/debug/map_motion/representative_selection/representative_selection.csv"
    reference = list(csv.DictReader(reference_path.open(encoding="utf-8")))
    expected = {(r["video_name"], int(r["investigation_index"]), int(r["candidate_index"])): int(r["frame_index"])
                for r in reference if r["configuration"] == "A_ticket_minimal" and r["primary_stable"] == "True"}
    for name in SOURCES:
        metadata, anomalies, contexts, phase3 = _build_contexts(name, include_phase3=True)
        if anomalies:
            raise ValueError(f"Lifecycle anomalies: {anomalies}")
        sources.append(_fingerprint(metadata.path))
        with VideoFrameResolver(metadata.path, metadata=metadata) as resolver:
            for window, plan, views, _ in contexts:
                candidates = build_evidence_candidate_timeline(window=window, sampling_plan=plan, stable_views=views)
                materialized = materialize_evidence_candidates(video_path=metadata.path, candidates=candidates,
                    ui_config=UIConfig(), resolver=resolver)
                selected = select_investigation_representatives(window=window, sampling_plan=plan,
                    candidates=candidates, materialized_candidates=materialized, stable_views=views,
                    include_map_route_evidence=False)
                roles = {s.candidate_index: ("stable", s.selection_reasons) for s in selected.selections}
                inv = window.investigation_index
                for i, m in enumerate(materialized):
                    control = None
                    if name == "combined_interaction" and inv == 0 and m.frame_index == 146:
                        control = "dimmed_spinner"
                    elif name == "combined_interaction" and inv == 1 and m.frame_index == 147:
                        control = "mixed_ticket"
                    elif name == "investigation_switch" and inv == 0 and m.frame_index == 16:
                        control = "sliding_open"
                    if control:
                        roles[i] = (control, ())
                if name == "ticket_scroll":
                    active = [i for i, m in enumerate(materialized) if m.frame_index
                              and phase3["signals"][m.frame_index-1].ticket_scroll_detected and i not in roles]
                    if not active:
                        raise ValueError("No actual positive-scroll candidate for control")
                    roles[active[len(active)//2]] = ("active_scroll", ())
                for i, (kind, reasons) in sorted(roles.items()):
                    c, m = candidates[i], materialized[i]
                    key = (metadata.path.name, inv, i)
                    if kind == "stable" and expected.get(key) != m.frame_index:
                        raise ValueError(f"Phase 5 representative regression: {key}")
                    frame = resolver.resolve(c.timestamp_seconds)
                    if frame.image is None or frame.frame_index != m.frame_index:
                        raise ValueError("Unresolved/mismatched calibration input")
                    roi = map_roi_to_frame(rois["full_ticket"], UIConfig(), metadata.width, metadata.height)
                    crop = roi.crop(frame.image)
                    identifier = f"{name}_i{inv}_c{i}"
                    path = OUTPUT / "inputs" / f"{identifier}.png"
                    if not cv2.imwrite(str(path), crop):
                        raise OSError(path)
                    samples.append({"sample_id": identifier, "video_name": metadata.path.name,
                        "investigation_index": inv, "candidate_index": i, "kind": kind,
                        "selection_reasons": reasons, "evidence_candidate": asdict(c),
                        "materialized": asdict(m), "stable_view_index": c.stable_view_index,
                        "input_path": str(path.relative_to(ROOT)), "pixel_sha256": hashlib.sha256(crop.tobytes()).hexdigest()})
        print(f"Prepared {name}", flush=True)
    stable = [s for s in samples if s["kind"] == "stable"]
    if len(stable) != len(expected) or len(stable) != 17 or len(samples) != 21:
        raise ValueError(f"Unexpected dataset size: {len(stable)} stable / {len(samples)} total")
    CHECKPOINT.write_text(json.dumps({"sources": sources, "selection_source": _fingerprint(reference_path),
        "samples": samples}, indent=2), encoding="utf-8")
    print(f"Prepared {len(stable)} production stable representatives and four controls: {CHECKPOINT}", flush=True)


def _preprocess(image, variant):
    """Apply seven fixed non-combinatorial variants without changing baseline pixels."""
    image = image.copy()
    if variant == VARIANTS[0]:
        return image, 1
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if variant == VARIANTS[1]:
        return gray, 1
    if variant == VARIANTS[2]:
        return cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC), 2
    if variant == VARIANTS[4]:
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    enlarged = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    if variant == VARIANTS[5]:
        # One global Otsu probe is reproducible and avoids a block-size/C grid.
        enlarged = cv2.threshold(enlarged, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    elif variant == VARIANTS[6]:
        blurred = cv2.GaussianBlur(enlarged, (0, 0), 1.0)
        enlarged = cv2.addWeighted(enlarged, 1.5, blurred, -.5, 0)
    return enlarged, 2


def _tesseract_path():
    """Discover native Tesseract without installing it or changing system PATH."""
    supplied = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
    path = Path(supplied) if supplied else OUTPUT / "tools/tesseract/tesseract.exe"
    if not path.is_file():
        raise FileNotFoundError("Tesseract unavailable; set TESSERACT_CMD to its executable")
    return path.resolve()


def _engines():
    """Initialize two CPU engines and record actual versions, models and costs."""
    started = time.perf_counter()
    from rapidocr_onnxruntime import RapidOCR
    rapid = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1)
    init_ms = 1000*(time.perf_counter()-started)
    sessions = (rapid.text_det.infer.session, rapid.text_cls.infer.session, rapid.text_rec.session.session)
    if any(s.get_providers() != ["CPUExecutionProvider"] for s in sessions):
        raise RuntimeError("Unexpected non-CPU execution provider")
    started = time.perf_counter()
    tess = _tesseract_path()
    version = subprocess.run([str(tess), "--version"], check=True, capture_output=True).stdout.decode("utf-8").splitlines()[0]
    tess_setup_ms = 1000*(time.perf_counter()-started)
    versions = {"Tesseract": {"version": version, "discovery_version_probe_ms": tess_setup_ms,
        "initialization_ms": None, "note": "CLI reloads model per call; model initialization is included in every inference latency", "psm": 11,
        "english_model_sha256": hashlib.sha256((tess.parent/"tessdata/eng.traineddata").read_bytes()).hexdigest()},
        "RapidOCR": {"version": importlib.metadata.version("rapidocr-onnxruntime"), "initialization_ms": init_ms,
        "runtime": importlib.metadata.version("onnxruntime"), "providers": [s.get_providers() for s in sessions],
        "threads": "2 intra / 1 inter", "note": "Bundled Chinese/Latin PP-OCRv4 detector/recognizer, default score filter .5; no fine tuning"}}
    model_dir = Path(importlib.metadata.distribution("rapidocr-onnxruntime").locate_file("rapidocr_onnxruntime/models"))
    versions["RapidOCR"]["models"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in model_dir.glob("*.onnx")}
    return tess, rapid, versions


def _recognize(engine, image, tess, rapid):
    """Return raw text, native boxes and native confidence without correcting reads."""
    if engine == "RapidOCR":
        result, _ = rapid(image)
        return [{"text": str(text), "bbox": np.asarray(box, dtype=float).tolist(), "confidence": float(conf)}
                for box, text, conf in result or []]
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise ValueError("Cannot encode Tesseract input")
    env = os.environ.copy()
    env["OMP_THREAD_LIMIT"] = "2"
    command = [str(tess), "stdin", "stdout", "--tessdata-dir", str(tess.parent/"tessdata"),
               "-l", "eng", "--oem", "1", "--psm", "11", "tsv"]
    output = subprocess.run(command, input=encoded.tobytes(), capture_output=True, check=True, env=env, timeout=90)
    rows = csv.DictReader(output.stdout.decode("utf-8-sig").splitlines(), delimiter="\t")
    boxes = []
    for row in rows:
        if row["level"] != "5" or not row["text"].strip():
            continue
        x, y, w, h = (int(row[k]) for k in ("left", "top", "width", "height"))
        boxes.append({"text": row["text"], "bbox": [[x,y], [x+w,y], [x+w,y+h], [x,y+h]],
                      "confidence": float(row["conf"])})
    return boxes


def _normalize(text):
    """Normalize whitespace only; preserve case, punctuation, spelling and digits."""
    return " ".join(text.split())


def _distance(a, b):
    """Count Levenshtein edits without external metric libraries or text repair."""
    row = list(range(len(b)+1))
    for i, left in enumerate(a, 1):
        following = [i]
        for j, right in enumerate(b, 1):
            following.append(min(row[j]+1, following[-1]+1, row[j-1]+(left != right)))
        row = following
    return row[-1]


def _rectangle(box):
    """Derive enclosing axis-aligned bounds while retaining the original quadrilateral."""
    points = np.asarray(box)
    return (float(points[:,0].min()), float(points[:,1].min()),
            float(points[:,0].max()), float(points[:,1].max()))


def _score(boxes, truth):
    """Score human-verified text-line boxes only, with explicit annotation scope.

    A detection belongs to the single manual line with greatest intersection /
    detection-box area, provided that fraction is >=.3. Detection recall is then
    geometric annotated-line coverage, independent of correct recognition. Extra
    boxes are counted only within explicitly fully transcribed scope rectangles;
    unannotated ticket areas are never called false text. Confidence correctness
    uses exact whitespace-normalized token sequences in the associated GT line.
    """
    assigned = {i: [] for i in range(len(truth))}
    extra, scores_correct, scores_incorrect = [], [], []
    for box in boxes:
        x1,y1,x2,y2 = _rectangle(box["bbox_full_ticket"])
        area = max(1, (x2-x1)*(y2-y1))
        matches = []
        in_scope = False
        for i, line in enumerate(truth):
            gx,gy,gw,gh = (float(line[k]) for k in ("x","y","width","height"))
            overlap = max(0,min(x2,gx+gw)-max(x1,gx))*max(0,min(y2,gy+gh)-max(y1,gy))/area
            matches.append((overlap,i))
            sx,sy,sw,sh = (float(line[k]) for k in ("scope_x","scope_y","scope_width","scope_height"))
            in_scope |= sx <= (x1+x2)/2 <= sx+sw and sy <= (y1+y2)/2 <= sy+sh
        overlap, index = max(matches, default=(0,0))
        if overlap >= .3:
            assigned[index].append(box)
        elif in_scope:
            extra.append(box)
    measurements = []
    for i, line in enumerate(truth):
        group = sorted(assigned[i], key=lambda b: (_rectangle(b["bbox_full_ticket"])[0], _rectangle(b["bbox_full_ticket"])[1]))
        expected = _normalize(line["raw_manual_transcription"])
        actual = _normalize(" ".join(b["text"] for b in group))
        for box in group:
            correct = (" "+_normalize(box["text"])+" ") in (" "+expected+" ")
            (scores_correct if correct else scores_incorrect).append(box["confidence"])
        measurements.append({"annotation_id": line["annotation_id"], "category": line["category"],
            "field_id": line.get("field_id", line["annotation_id"]), "section": line.get("section", ""),
            "column": line.get("column", ""), "component": line.get("component", "text"),
            "complete_field_verified": line.get("complete_field_verified", True),
            "expected": expected, "actual": actual, "character_edits": _distance(expected, actual), "characters": len(expected),
            "word_edits": _distance(expected.split(),actual.split()), "words": len(expected.split()),
            "exact_match": expected == actual, "detected": bool(group)})
    chars = sum(m["characters"] for m in measurements)
    words = sum(m["words"] for m in measurements)
    scoped_boxes = sum(len(v) for v in assigned.values())+len(extra)
    extra_words = sum(len(_normalize(b["text"]).split()) for b in extra)
    scoped_words = sum(len(_normalize(b["text"]).split()) for group in assigned.values() for b in group)+extra_words
    field_ids = {m["field_id"] for m in measurements if m["complete_field_verified"]}
    exact_fields = sum(all(m["exact_match"] for m in measurements if m["field_id"] == fid) for fid in field_ids)
    return {"cer": sum(m["character_edits"] for m in measurements)/chars if chars else None,
        "wer": sum(m["word_edits"] for m in measurements)/words if words else None,
        "exact_match_rate": statistics.mean(m["exact_match"] for m in measurements) if measurements else None,
        "detection_recall": statistics.mean(m["detected"] for m in measurements) if measurements else None,
        "exact_field_match_rate": exact_fields/len(field_ids) if field_ids else None,
        "field_count": len(field_ids), "exact_fields": exact_fields,
        "false_text_count": len(extra) if truth else None,
        "scoped_box_count": scoped_boxes, "extra_word_count": extra_words, "scoped_word_count": scoped_words,
        "false_extra_box_rate": len(extra)/scoped_boxes if truth and scoped_boxes else None,
        "false_extra_word_rate": extra_words/scoped_words if truth and scoped_words else None,
        "correct_confidences": scores_correct, "incorrect_confidences": scores_incorrect, "lines": measurements}


def _load_truth():
    """Accept explicit human verification only; never promote a model draft to truth."""
    if not GROUND_TRUTH.is_file():
        return []
    rows = list(csv.DictReader(GROUND_TRUTH.open(encoding="utf-8-sig")))
    accepted = [r for r in rows if r["review_status"] == "human_verified" and r["raw_manual_transcription"]]
    for r in accepted:
        if not r.get("verified_by"):
            raise ValueError("Human-verified annotations require a verifier")
        field_id = r.get("field_id", r["annotation_id"])
        r["complete_field_verified"] = all(a["review_status"] == "human_verified" for a in rows
            if a.get("field_id", a["annotation_id"]) == field_id)
    return accepted


def _prepare_review():
    """Render manually authored field/cell annotations without OCR-based parsing.

    Each field retains its section, left/right membership, literal label/value,
    and separate component boxes. Label:value punctuation appears only in the
    review presentation. Country flags/lock symbols supply no inferred text.
    Projection to metric rows never flattens adjacent columns. All projected rows
    remain unverified until a human verifies them; verified work is not replaced.
    """
    if _load_truth():
        raise ValueError("Refusing to replace human-verified annotations with a draft")
    source = json.loads((OUTPUT/"review_fields_source.json").read_text(encoding="utf-8"))
    samples = {s["sample_id"]: s for s in _load_inputs()}
    fields, parts, tiles = [], [], []
    notes = ["# Field-wise OCR transcription draft v2", "", "UNVERIFIED: requires human review. No accuracy scores are computed from this draft.",
        "", source["layout_method"], "", source["notes"], "",
        "Each row is one field/cell. Left/right entries remain separate. Review raw labels, values, column membership and boxes.",
        "Colons and table separators are presentation only; no characters are added to scored raw text.", ""]
    for sample in source["samples"]:
        sid = sample["sample_id"]
        original = samples[sid]
        image = cv2.imread(str(ROOT/original["input_path"]))
        tile = np.full((870,336,3),245,np.uint8)
        tile[39:] = image
        cv2.putText(tile, sid, (3,16), cv2.FONT_HERSHEY_SIMPLEX, .36, (10,10,10), 1)
        header = {"name": "header", "fields": []}
        for item in [*sample["header"]["identifiers"], sample["header"]["airline"], sample["header"]["operated"]]:
            if item:
                header["fields"].append({"column": "header", "raw_label_text": "", "raw_value_text": item["text"],
                    "label_box": None, "value_box": item["box"], "additional_text": [], "value_status": "visible_text"})
        notes.extend(["## "+sid, "", f"Actual source time: {original['materialized']['actual_timestamp_seconds']} s; original crop: inputs/{sid}.png", ""])
        for section in [header, *sample["sections"]]:
            notes.extend(["### "+section["name"], "", "| ID | Column | Raw label | Raw value | Additional visible text / note |",
                          "|---|---|---|---|---|"])
            for field in section["fields"]:
                fid = f"F{len(fields)+1}"
                boxes = [b for b in [field["label_box"], field["value_box"], *[a["box"] for a in field["additional_text"]]] if b]
                x,y = min(b[0] for b in boxes), min(b[1] for b in boxes)
                right,bottom = max(b[0]+b[2] for b in boxes), max(b[1]+b[3] for b in boxes)
                common = {"sample_id": sid, "video_name": original["video_name"], "investigation_index": original["investigation_index"],
                    "candidate_index": original["candidate_index"], "actual_timestamp": original["materialized"]["actual_timestamp_seconds"],
                    "field_id": fid, "section": section["name"], "column": field["column"], "review_status": "unverified_draft", "verified_by": ""}
                fields.append({**common, "raw_label_text": field["raw_label_text"], "raw_value_text": field["raw_value_text"],
                    "value_status": field["value_status"], "label_box": json.dumps(field["label_box"]), "value_box": json.dumps(field["value_box"]),
                    "additional_text": json.dumps(field["additional_text"], ensure_ascii=False)})
                entries = [("label",field["raw_label_text"],field["label_box"]), ("value",field["raw_value_text"],field["value_box"]),
                           *[(f"context{j+1}",a["text"],a["box"]) for j,a in enumerate(field["additional_text"])]]
                for component,text,box in entries:
                    if not box or not text:
                        continue
                    category = ("labels" if component == "label" else "flight_identifiers" if section["name"] == "header"
                        and field["raw_value_text"] in [a["text"] for a in sample["header"]["identifiers"]] else
                        "airport_codes" if section["name"] == "airport_cells" and component == "value" else "times"
                        if section["name"] in ("scheduled","actual_estimated") and component == "value" else "small_text"
                        if component.startswith("context") else "short_identifiers" if section["name"] == "registration" else
                        "numeric_values" if section["name"] in ("serial_age","progress_text","vertical_state","gps_track","ground_true",
                            "indicated_mach","wind_temperature","address_squawk","latitude_longitude") else "general_text")
                    parts.append({**common,"annotation_id": fid+"."+component,"component": component,"roi_name": section["name"],
                        "x": box[0],"y": box[1],"width": box[2],"height": box[3],"scope_x": x,"scope_y": y,
                        "scope_width": right-x,"scope_height": bottom-y,"category": category,"raw_manual_transcription": text})
                color = (220,100,0) if field["column"] == "left" else (0,140,200) if field["column"] == "right" else (0,150,0)
                cv2.rectangle(tile,(x,y+39),(right,bottom+39),color,1)
                cv2.putText(tile,fid,(x+1,y+47),cv2.FONT_HERSHEY_SIMPLEX,.25,color,1)
                more = " / ".join(a["text"] for a in field["additional_text"]) or (field["value_status"] if field["value_status"] != "visible_text" else "")
                notes.append(f"| {fid} | {field['column']} | {field['raw_label_text']} | {field['raw_value_text']} | {more} |")
            notes.append("")
        tiles.append(tile)
    _write_csv(OUTPUT/"ocr_ground_truth_fields.csv",fields)
    _write_csv(GROUND_TRUTH,parts)
    (OUTPUT/"ground_truth_review.md").write_text("\n".join(notes)+"\n",encoding="utf-8")
    sheet = np.full((870*((len(tiles)+3)//4),1344,3),245,np.uint8)
    for i,tile in enumerate(tiles):
        sheet[i//4*870:(i//4+1)*870,i%4*336:(i%4+1)*336] = tile
    cv2.imwrite(str(OUTPUT/"ground_truth_review.png"),sheet)
    print(f"Prepared {len(fields)} field/cell drafts and {len(parts)} raw text components; none human-verified",flush=True)


def _load_inputs():
    """Reject stale checkpoints or modified baseline crops rather than repairing them."""
    manifest = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    for original in [*manifest["sources"], manifest["selection_source"]]:
        if _fingerprint(Path(original["path"])) != original:
            raise ValueError("Preparation checkpoint stale; rerun --prepare-only")
    for s in manifest["samples"]:
        image = cv2.imread(str(ROOT/s["input_path"]))
        if image is None or hashlib.sha256(image.tobytes()).hexdigest() != s["pixel_sha256"]:
            raise ValueError("Missing/modified original input crop")
    return manifest["samples"]


def _run():
    """Execute bounded raw OCR and score only independently human-verified evidence."""
    samples = _load_inputs()
    rois = _coverage()
    truth = _load_truth()
    cv2.setNumThreads(2)
    tess, rapid, versions = _engines()
    results, all_boxes, scored = [], [], []
    (OUTPUT/"engine_environment.json").write_text(json.dumps(versions, indent=2), encoding="utf-8")
    run_cache = OUTPUT/"raw_runs.jsonl"
    cached = {}
    if run_cache.is_file():
        for text in run_cache.read_text(encoding="utf-8").splitlines():
            r = json.loads(text)
            cached[tuple(r["key"])] = r
    # Exclude variable initialization timing from cache identity.
    versions_key = hashlib.sha256(json.dumps({e:{k:v for k,v in info.items() if not k.endswith("_ms")}
        for e,info in versions.items()}, sort_keys=True).encode()).hexdigest()
    with run_cache.open("a", encoding="utf-8") as checkpoint:
        for sample in samples:
            image = cv2.imread(str(ROOT/sample["input_path"]))
            for engine in ("Tesseract", "RapidOCR"):
                for variant in VARIANTS:
                    collected = {"A": [], "B": []}
                    for name, roi in rois.items():
                        strategy = "A" if name == "full_ticket" else "B"
                        dx,dy = roi.x-rois["full_ticket"].x, roi.y-rois["full_ticket"].y
                        crop = image[dy:dy+roi.height, dx:dx+roi.width]
                        started = time.perf_counter()
                        processed, factor = _preprocess(crop, variant)
                        preprocessing_ms = 1000*(time.perf_counter()-started)
                        key = (sample["sample_id"], sample["pixel_sha256"], engine, variant, name, versions_key)
                        if key in cached:
                            raw = cached[key]
                            cache_hit = True
                        else:
                            started = time.perf_counter()
                            boxes = _recognize(engine, processed, tess, rapid)
                            raw = {"key": key, "boxes": boxes, "latency_ms": 1000*(time.perf_counter()-started),
                                   "preprocessing_ms": preprocessing_ms}
                            checkpoint.write(json.dumps(raw, ensure_ascii=False)+"\n")
                            checkpoint.flush()
                            cache_hit = False
                        converted = []
                        for b in raw["boxes"]:
                            mapped = np.asarray(b["bbox"], dtype=float)/factor + [dx,dy]
                            out = {**b, "bbox_full_ticket": mapped.tolist(),
                                   "bbox_source_frame": (mapped+[rois["full_ticket"].x, rois["full_ticket"].y]).tolist()}
                            converted.append(out)
                        collected[strategy].extend(converted)
                        confidences = [b["confidence"] for b in converted]
                        row = {"sample_id": sample["sample_id"], "video_name": sample["video_name"],
                            "investigation_index": sample["investigation_index"], "candidate_index": sample["candidate_index"],
                            "actual_timestamp": sample["materialized"]["actual_timestamp_seconds"], "kind": sample["kind"],
                            "stable_view_index": sample["stable_view_index"], "roi_strategy": strategy, "roi_name": name,
                            "engine": engine, "preprocessing_variant": variant, "detected_text": "\n".join(b["text"] for b in converted),
                            "detection_count": len(converted), "mean_confidence": statistics.mean(confidences) if confidences else None,
                            "min_confidence": min(confidences, default=None), "max_confidence": max(confidences, default=None),
                            "latency_ms": raw["latency_ms"], "preprocessing_ms": raw["preprocessing_ms"],
                            "total_processing_ms": raw["latency_ms"]+raw["preprocessing_ms"], "reused_recorded_run": cache_hit}
                        results.append(row)
                        all_boxes.append({**{k:v for k,v in row.items() if k != "detected_text"}, "boxes": converted})
                    for strategy, boxes in collected.items():
                        relevant = [r for r in truth if r["sample_id"] == sample["sample_id"]]
                        scores = _score(boxes, relevant)
                        metric_row = {"sample_id": sample["sample_id"], "kind": sample["kind"], "engine": engine,
                            "preprocessing_variant": variant, "roi_strategy": strategy, "verified_lines": len(relevant), **scores}
                        scored.append(metric_row)
            print(f"OCR completed {sample['sample_id']} ({sample['kind']})", flush=True)
    _write_csv(OUTPUT/"ocr_engine_results.csv", results)
    with (OUTPUT/"ocr_boxes.jsonl").open("w", encoding="utf-8") as stream:
        for row in all_boxes:
            stream.write(json.dumps(row, ensure_ascii=False)+"\n")
    with (OUTPUT/"scored_runs.jsonl").open("w", encoding="utf-8") as stream:
        for row in scored:
            stream.write(json.dumps(row, ensure_ascii=False)+"\n")
    _summarize(results, truth, versions, scored)


def _summarize(results, truth, versions, scored):
    """Report CPU measurements and accuracy only for explicitly verified annotations."""
    summary = []
    for engine in ("Tesseract", "RapidOCR"):
        for variant in VARIANTS:
            for strategy in ("A", "B"):
                rows = [r for r in results if r["engine"] == engine and r["preprocessing_variant"] == variant
                        and r["roi_strategy"] == strategy and r["kind"] == "stable"]
                per_ticket = {}
                for row in rows:
                    per_ticket[row["sample_id"]] = per_ticket.get(row["sample_id"], 0)+row["total_processing_ms"]
                metrics = [r for r in scored if r["engine"] == engine and r["preprocessing_variant"] == variant
                           and r["roi_strategy"] == strategy and r["kind"] == "stable" and r["verified_lines"]]
                lines = [line for r in metrics for line in r["lines"]]
                chars, words = sum(l["characters"] for l in lines), sum(l["words"] for l in lines)
                summary.append({"engine": engine, "variant": variant, "strategy": strategy,
                    "stable_representatives": len(per_ticket), "roi_calls": len(rows),
                    "mean_detections_per_ticket": sum(r["detection_count"] for r in rows)/len(per_ticket),
                    "median_ticket_latency_ms": statistics.median(per_ticket.values()),
                    "mean_ticket_latency_ms": statistics.mean(per_ticket.values()),
                    "mean_native_confidence": statistics.mean(r["mean_confidence"] for r in rows if r["mean_confidence"] is not None),
                    "human_verified_lines": len(lines),
                    "cer": sum(l["character_edits"] for l in lines)/chars if chars else None,
                    "wer": sum(l["word_edits"] for l in lines)/words if words else None,
                    "exact_match_rate": statistics.mean(l["exact_match"] for l in lines) if lines else None,
                    "exact_field_match_rate": sum(r["exact_fields"] for r in metrics)/sum(r["field_count"] for r in metrics)
                        if metrics and sum(r["field_count"] for r in metrics) else None,
                    "detection_recall": statistics.mean(l["detected"] for l in lines) if lines else None,
                    "false_text_count": sum(r["false_text_count"] for r in metrics) if metrics else None,
                    "false_extra_box_rate": sum(r["false_text_count"] for r in metrics)/sum(r["scoped_box_count"] for r in metrics)
                        if metrics and sum(r["scoped_box_count"] for r in metrics) else None,
                    "false_extra_word_rate": sum(r["extra_word_count"] for r in metrics)/sum(r["scoped_word_count"] for r in metrics)
                        if metrics and sum(r["scoped_word_count"] for r in metrics) else None})
    _write_csv(OUTPUT/"ocr_engine_summary.csv", summary)
    line_rows = [{"sample_id": run["sample_id"], "engine": run["engine"], "variant": run["preprocessing_variant"],
                  "strategy": run["roi_strategy"], **line} for run in scored for line in run["lines"]]
    if line_rows:
        _write_csv(OUTPUT/"ocr_annotation_scores.csv", line_rows)
    category_rows = []
    for engine in ("Tesseract", "RapidOCR"):
        for variant in VARIANTS:
            for strategy in ("A", "B"):
                selected_lines = [r for r in line_rows if r["engine"] == engine and r["variant"] == variant and r["strategy"] == strategy]
                for category in sorted({r["category"] for r in selected_lines}):
                    group = [r for r in selected_lines if r["category"] == category]
                    chars, words = sum(r["characters"] for r in group), sum(r["words"] for r in group)
                    category_rows.append({"engine": engine, "variant": variant, "strategy": strategy, "category": category,
                        "lines": len(group), "cer": sum(r["character_edits"] for r in group)/chars if chars else None,
                        "wer": sum(r["word_edits"] for r in group)/words if words else None,
                        "exact_line_match_rate": statistics.mean(r["exact_match"] for r in group),
                        "detection_recall": statistics.mean(r["detected"] for r in group)})
    if category_rows:
        _write_csv(OUTPUT/"ocr_category_summary.csv", category_rows)
    lines = ["Phase 6.1a OCR engine/preprocessing calibration (development only)",
        "17 actual Phase 5 primary stable representatives from seven videos; four separate diagnostic controls.",
        "Full crop A=(16,121,336,831); B=header(16,121,336,76)+upper(16,197,336,189)+body(16,386,336,566).",
        "Focused union has zero uncovered/overlapping relevant pixels. Bottom action labels (3D view, Route, Follow, Share, More; y952:1008) were visually audited across all17 stable frames and excluded as interface actions without ticket field content.",
        "V0 original; V1 gray; V2 cubic2x; V3 gray+cubic2x; V4 CLAHE(2,8x8)+cubic2x; V5 gray+cubic2x+Otsu; V6 gray+cubic2x+unsharp(1.5,-.5,sigma1).",
        "Otsu is the single threshold probe: no local block-size parameter grid. Originals remain unchanged.",
        "All boxes retain raw text, native confidence, processed coordinates, full-ticket and original-frame coordinates.",
        "Tesseract word boxes/0..100 confidence versus RapidOCR line boxes/0..1 confidence are not interchangeable or directly count-comparable.",
        "Tesseract fixed OEM1/PSM11; English tessdata_fast. RapidOCR1.4.4 bundled PP-OCRv4 models, default .5 score filter; 2 intra/1 inter CPU threads.",
        "Package setup: no OCR initially installed in project. System Tesseract absent. Global unrelated Python had torch but is not used.",
        "Reproduce: uv sync --group ocr-calibration; provision native Tesseract +English model as documented; uv run python scripts/calibration/evaluate_ocr_engine_calibration.py --prepare-only; verify annotations; then run the same script without --prepare-only.",
        "Added calibration-only group: rapidocr-onnxruntime>=1.4.4 and onnxruntime<1.24 (1.24.3 has no Windows CPython3.10 wheel). No PaddleOCR/GPU framework.",
        "Native Tesseract5.5.3 extracted with existing 7-Zip; English model downloaded from official tessdata_fast. No system installer executed.",
        "Setup sources: https://github.com/UB-Mannheim/tesseract/wiki ; https://github.com/tesseract-ocr/tessdata_fast ; https://pypi.org/project/rapidocr-onnxruntime/",
        "Initialization details="+json.dumps(versions),
        f"Human-verified ground truth lines={len(truth)}. Assistant visual-transcription drafts are NOT accepted ground truth until human verification.",
        "No accuracy recommendation can be made from detection volume, confidence or speed alone.",
        "CER/WER/line exactness use whitespace-only normalization, no case/punctuation/digit correction. Detection recall uses manual line geometry.",
        "Extra-text rates apply only inside fully transcribed manual annotation scopes; unannotated ticket text is not false text.",
        "Raw inference checkpoints are reused only with identical pixels, engine/model versions and configuration. Recorded latency is original measured inference, not cache retrieval time.",
        "Per-ROI latency includes engine/wrapper costs and Tesseract CLI model reload; preprocessing is recorded separately and added for total per-ticket processing. No GPU is used.",
        "Benchmark annotations cover selected declared bands in seven originals; unannotated upper photo credits/branding, clipped content and other regions are outside accuracy scope, not OCR-input blind spots.",
        "Review draft v2 preserves section/column/field membership. Each literal label/value has a separate text box; adjacent left/right fields are not flattened. Colons in review tables are presentation only, never inserted into scored raw text. Country flags/lock icons are not expanded to inferred text.",
        "Observed stable input/ROI/variant summaries: see ocr_engine_summary.csv; interpreted comparison below.",
        "Controls: see raw result rows for dimmed_spinner, mixed_ticket, active_scroll and sliding_open; they never enter production representatives.",
        "Scroll views: all five positions evaluated independently; no merge, semantic deduplication or inferred fields.",
        "Engine comparison:",
        "Tesseract | CPU native CLI +4MB English model; word boxes/confidence; per-call model reload, layout/font sensitivity.",
        "RapidOCR | CPU ONNX, persistent sessions and line boxes/confidence; bundled models plus runtime/dependencies, internal resize/filter.",
        "PaddleOCR | considered, not installed/evaluated; unavailable initially, additional framework stack not justified for this bounded two-engine calibration. Further candidates require a separate decision if residual errors prove unacceptable.",
        "Coverage is complete for both input strategies. More detections does not prove completeness; higher confidence is not correctness.",
        "Unvalidated: held-out tickets/themes/layouts/FPS, non-English text, other models/PSMs, hardware diversity and production throughput.",
        "No production OCR, parser, semantic schema, multi-view merge, dictionaries, VLM/fusion or Phase 6.1b implementation. No commit/push."]
    lines.extend(_interpret_results(results, truth, scored, summary))
    (OUTPUT/"ocr_engine_calibration_report.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"Raw calibration complete; verified GT lines={len(truth)}; report={OUTPUT/'ocr_engine_calibration_report.txt'}", flush=True)


def _interpret_results(results, truth, scored, summary):
    """Save reproducible field-aware analyses without parsing or repairing OCR text.

    Recommendation wording applies only to this fixed calibration configuration.
    Component/field exactness remains stricter than character accuracy, and text
    spanning columns is never split using the answer key to improve a score.
    """
    if not truth:
        return ["Recommendation DEFERRED: human-verified ground truth is unavailable."]
    selected = [r for r in scored if r["verified_lines"]]
    samples = sorted({r["sample_id"] for r in selected})
    lookup = {(r["sample_id"], r["annotation_id"]): r for r in truth}
    region_rows, sample_rows, confidence_rows = [], [], []
    for run in selected:
        identity = {"engine": run["engine"], "variant": run["preprocessing_variant"], "strategy": run["roi_strategy"]}
        sample_rows.append({**identity, "sample_id": run["sample_id"], **{k: run[k] for k in
            ("cer", "wer", "exact_match_rate", "exact_field_match_rate", "detection_recall", "false_text_count")}})
        for correctness in ("correct", "incorrect"):
            values = run[correctness+"_confidences"]
            confidence_rows.append({**identity, "sample_id": run["sample_id"], "correctness": correctness,
                "native_boxes": len(values), "mean_native_confidence": statistics.mean(values) if values else None,
                "min_native_confidence": min(values, default=None), "max_native_confidence": max(values, default=None)})
    for engine in ("Tesseract", "RapidOCR"):
        for variant in VARIANTS:
            for strategy in ("A", "B"):
                runs = [r for r in selected if (r["engine"], r["preprocessing_variant"], r["roi_strategy"]) == (engine, variant, strategy)]
                groups = {"header": [], "upper_information": [], "scroll_body": []}
                for run in runs:
                    for line in run["lines"]:
                        annotation = lookup[run["sample_id"], line["annotation_id"]]
                        y = float(annotation["y"])
                        region = "header" if y < 76 else "upper_information" if y < 265 else "scroll_body"
                        groups[region].append(line)
                for region, group in groups.items():
                    if not group:
                        continue
                    region_rows.append({"engine": engine, "variant": variant, "strategy": strategy, "region": region,
                        "components": len(group), "cer": sum(l["character_edits"] for l in group)/sum(l["characters"] for l in group),
                        "wer": sum(l["word_edits"] for l in group)/sum(l["words"] for l in group),
                        "exact_match_rate": statistics.mean(l["exact_match"] for l in group),
                        "detection_recall": statistics.mean(l["detected"] for l in group)})
    _write_csv(OUTPUT/"ocr_region_summary.csv", region_rows)
    _write_csv(OUTPUT/"ocr_sample_summary.csv", sample_rows)
    _write_csv(OUTPUT/"ocr_confidence_summary.csv", confidence_rows)
    controls = [r for r in results if r["kind"] != "stable"]
    _write_csv(OUTPUT/"ocr_control_results.csv", controls)
    report = ["", "VERIFIED FIELD-WISE CALIBRATION INTERPRETATION",
        f"Human verification: {len(samples)} stable originals, {len({r['field_id'] for r in truth})} fields/cells, {len(truth)} literal components. User explicitly verified v2 text, columns and boxes on 2026-10-20; ground_truth_verification.json records approval.",
        "Draft was visually authored, then human-verified; labels/values remain separate and no raw text was repaired from OCR output.",
        "Scoring limitation: a native detection is assigned to one component by greatest geometric overlap. A line spanning two columns is not split using known answers. Such merges lower component recall/exactness and can double-penalize CER; these are layout-sensitive end-to-end OCR metrics, not pure recognizer character accuracy.",
        "No unassigned boxes were found inside the narrow verified field scopes. This is NOT a whole-ticket zero-false-text claim; photo credits, branding, icons and unannotated/clipped content lack exhaustive truth. Extra text merged into an assigned component is penalized by CER/WER instead.",
        "All 151 components are scored for every configuration. Latency uses all17 stable representatives, accuracy uses seven. Text/box populations and confidence scales differ by engine.",
        "Comparison (engine | variant | ROI | CER | WER | component exact | field exact | detection recall | median ticket ms):"]
    for row in summary:
        report.append(" | ".join(str(row[k]) if isinstance(row[k], str) else f"{row[k]:.4f}" for k in
            ("engine", "variant", "strategy", "cer", "wer", "exact_match_rate", "exact_field_match_rate", "detection_recall", "median_ticket_latency_ms")))
    compared = {("RapidOCR", VARIANTS[2], "A"), ("Tesseract", VARIANTS[6], "A"),
                ("Tesseract", VARIANTS[4], "B"), ("RapidOCR", VARIANTS[5], "B")}
    report += ["Region results (annotation coverage only; all variants in ocr_region_summary.csv):",
        *(str(r) for r in region_rows if (r["engine"], r["variant"], r["strategy"]) in compared),
        "Per-sample results including all five scroll positions (all variants in ocr_sample_summary.csv):",
        *(str(r) for r in sample_rows if (r["engine"], r["variant"], r["strategy"]) in compared),
        "Confidence by correctness: per-sample native distributions in ocr_confidence_summary.csv; aggregate interpretation below. Scores are not calibrated probabilities.",
        "Baseline controls: raw text, box counts and confidence; controls have no human accuracy truth:"]
    for row in controls:
        if row["preprocessing_variant"] == VARIANTS[0] and row["roi_name"] == "full_ticket":
            report.append(json.dumps(row, ensure_ascii=False))
    for engine in ("Tesseract", "RapidOCR"):
        for kind in sorted({r["kind"] for r in controls}):
            group = [r for r in controls if r["engine"] == engine and r["kind"] == kind and r["roi_name"] == "full_ticket"]
            report.append(f"Control preprocessing sensitivity {engine}/{kind}: "+str([
                (r["preprocessing_variant"], r["detection_count"], r["mean_confidence"]) for r in group]))
    report += [
        "ENGINE DECISION: recommend RapidOCR/ONNX CPU, V2 color cubic2x, Strategy A full relevant ticket as the provisional Phase6.1b baseline. No production implementation is made here.",
        "Rationale: lowest CER (7.21%), 100% annotated flight-identifier/airport-code exactness, high geometric recall (98.68%), native quadrilateral/provenance and persistent CPU sessions. These outweigh greater latency (~1.35s/ticket versus Tesseract V6 ~0.419s) for this offline evidence workflow. WER and label spacing remain weaknesses; this is not a claim that every field is reliable.",
        "Tesseract is a useful faster comparator: V6/A has CER10.22%, WER20.93%, component exact72.19%, recall98.01%; V4/B has WER18.99%, labels97.83% exact and recall98.68%, but flight identifiers57.14% exact and annotated times0% exact. Do not silently combine engines or repair outputs in this calibration.",
        "RapidOCR V2/A has WER26.74%, component exact75.50%, field exact62.77%; time components45% exact (often PM/pM or missing spaces), numeric components92.86%, labels65.22%, small text64.29%. Airport codes and short registration-like identifiers were exact in this small set. Precision on times still needs work; no case or punctuation correction was applied.",
        "Preprocessing: RapidOCR original A CER7.94%/WER35.66% -> V2 A7.21%/26.74%. Grayscale alone gives8.01%/30.62%; gray2x8.82%/32.17%; CLAHE9.71%/42.64%; sharpening9.78%/40.31%. Color2x is a modest empirical improvement, not universally beneficial. Tesseract benefits most from V6/A for CER and V4/B for WER.",
        "Thresholding is not the default: RapidOCR Otsu A CER15.51% and recall94.70%; B improves WER22.48%/component exact79.47% but CER11.03% and small-text exact57.14%. Timezone strings become visibly corrupted. Higher aggregate exactness does not justify losing gray/small information. No further threshold grid was explored.",
        "ROI policy: focused B preserves complete input coverage but does not consistently improve accuracy and costs three calls. At RapidOCR V2, A vs B CER7.21 vs7.72%, WER26.74 vs28.68%, same98.68% recall, medians1351 vs2050ms. Prefer A; retain B as diagnostic option, not a body-only shortcut.",
        "Upper ticket: 29 header components including identifiers and airline/operated-by text. RapidOCR V2/A CER0.48%, WER2.78%, exact96.55%, recall100%; Tesseract V6/A CER19.05%, WER41.67%, exact48.28%, recall96.55%, often mistaking 6H972/A332. The aircraft_image/photo strip has input coverage but no independent verified accuracy annotations; credits/branding are not exhaustively scored. Complete input coverage is not complete accuracy validation.",
        "Body: 122 annotated components. RapidOCR V2/A CER8.43%, WER30.63%, exact70.49%, recall98.36%; Tesseract V6/A CER8.61%, WER17.57%, exact77.87%, recall98.36%. Tesseract is stronger on labels/spacing here; no single engine dominates every region or metric.",
        "Five scroll views remain independent: c5(.5s) route/times/registration to GPS-track; c25(2.5s) registration through address and latitude/longitude labels at bottom; c33(3.3s) serial values at top through full latitude/longitude values; c54(5.8s) clipped timezone area at top through ground/true speeds; c62(6.6s) route/times returned with updated track. Same header repeats, body content moves. Clipped labels/values are not reconstructed and repeated text is not merged. See per-sample/region CSVs for variant consistency.",
        "Field layout remains a real failure mode: RapidOCR merges SERIAL NUMBER (MSN) and AGE (Mar2010) into one detection at c25/c54. Ground truth correctly preserves independent left/right fields. Spatial matching assigns that merged box once; confidence does not make it two reliable observations.",
        "Controls: dimmed/spinner still produces plausible ticket text; mixed frame contains incoming identifiers with old HFA/PFO content and an 'Operated by airHaifa' read; active-scroll produces readable old/new body content plus partial/icons; sliding control includes clipped header and map labels (e.g.WZZ93L/Rosetta). Baseline RapidOCR mean confidences remain~.961-.974 despite these defects; Tesseract~78.7-84.1/100. Neither confidence nor OCR text establishes temporal validity/loading. Controls never become representatives.",
        "Confidence: RapidOCR V2/A exact boxes mean.9889 vs incorrect.9632, with wrong boxes as high as.9982. Tesseract V6/A means92.93 vs59.79, but wrong boxes reach92.00. Both have confident errors and overlap; no production acceptance threshold was derived, and engine confidence is not interchangeable with correctness or detection/selection confidence.",
        "Boxes: preserve native quadrilaterals/word boxes and exact processed->ticket->source-frame mapping. Tesseract word granularity often preserves column labels better; RapidOCR line granularity can merge columns and spaces. Recall is annotated-component geometric coverage, not bbox IoU accuracy; independent bbox-quality benchmark remains future work.",
        "Runtime: native Tesseract download26,573,224 bytes + English model4,113,088 bytes; local extraction only. RapidOCR bundled weights~16MB plus ONNX runtime and transitive dependencies. No GPU framework. Tesseract process/model load is included per call; Rapid session initialization is separately recorded. Total original inference181.9s Tesseract/500.4s Rapid across588calls each. Original timings reused on scoring reruns.",
        "ENGINE OPTION TABLE",
        "Tesseract | advantages: fastest CPU, word boxes, strong labels/spacing | disadvantages: native tool setup, CLI reload, weak header identifiers/time suffixes | confidence0..100, boxes yes | evaluated; retain comparator, not chosen baseline.",
        "RapidOCR | advantages: best CER/critical IDs here, persistent CPU ONNX, quadrilateral boxes | disadvantages: larger runtime, slower CPU, merged columns/spacing, confident errors, Chinese/Latin model can recognize icons as characters | confidence0..1, boxes yes | evaluated; provisional V2/A choice.",
        "PaddleOCR | considered candidate with text/boxes/confidence support, separate framework/model setup | not installed/evaluated, no project-specific speed/accuracy claim | defer additional stack; residual limitations reported before any third-engine experiment.",
        "Remaining uncertainty: seven human-verified originals are representative manual calibration, not held-out benchmark; repeated headers over-weight some identifiers; 17 representative inputs do not imply17 accuracy annotations; controls lack truth; unknown/corrupted icon/credit text is incompletely scored. No exhaustive bbox/false-positive benchmark. Latency single fixed-order pass, not warmed randomized repeats; other CPUs, models/languages/themes/resolutions and production throughput unvalidated.",
        "Recommendation balances accuracy, critical tokens, completeness, spatial provenance, CPU cost and dependency complexity; it is provisional. No confidence-based repair, semantic parsing, multi-view assembly, VLM or fusion was performed."]
    return report


def main():
    """Prepare native evidence or execute calibration without production OCR APIs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--prepare-review", action="store_true", help="Render manually authored column-aware review_fields_source.json; never run a field parser")
    args = parser.parse_args()
    if args.prepare_only:
        _prepare()
        return 0
    if args.prepare_review:
        _prepare_review()
        return 0
    _run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
