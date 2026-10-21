"""Evaluate production Phase 6.3c over all saved OCR representatives, without OCR.

The 94 previously human-verified field annotations provide header/airport truth
for seven crops. Expected text and pixel boxes are evaluation-only; they never
drive production extraction. Report exact text, whitespace-only differences,
source-location correctness, abstentions and unscored observations separately.
Persist the combined per-frame output and audit under data/debug/ocr/ocr_upper_ticket.
Existing lower-field CSVs are checked row-for-row and never overwritten. Existing
raw OCR, spatial/calibration artifacts, production modules and tests are read-only.
No external knowledge, normalization of production values, VLM or view fusion.
"""

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from events_diary.ocr_upper_ticket import (
    HEADER_FIELDS,
    extract_fr24_tickets_batch,
)
from scripts.evaluation.evaluate_ocr_fields import _flatten
from scripts.evaluation.evaluate_ocr_spatial import _rehydrate

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/debug/ocr/ocr_upper_ticket"
TRUTH = ROOT / "data/debug/ocr/ocr_engine_calibration/ocr_ground_truth_fields.csv"
LOWER = ROOT / "data/debug/ocr/ocr_fields/ocr_fields_results.csv"


def read(path):
    """Read UTF-8 source rows without modifying any literal OCR/reference text."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(filename, data):
    """Write a new Phase 6.3c audit table, keeping all absent/ambiguous outcomes."""
    with (OUTPUT / filename).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in data for k in r)))
        writer.writeheader()
        writer.writerows(data)


def flatten_upper(summary, upper):
    """Serialize each raw upper observation with physical and candidate provenance."""
    return [{"video_name": summary["video_name"], **asdict(upper.provenance),
        "field_name": f.field_name, "raw_text": f.raw_text, "status": f.status,
        "method": f.method, "reason": f.reason,
        "evidence": [{"detection_index": o.detection_index, "raw_text": o.text,
                      "source_polygon": o.source_polygon, "ocr_confidence": o.ocr_confidence}
                     for o in f.evidence]} for f in upper.fields]


def references():
    """Build expected semantic slots from existing human-verified text and boxes."""
    groups = defaultdict(list)
    for row in read(TRUTH):
        if row["review_status"] != "human_verified":
            raise ValueError("Ground truth must remain human verified")
        groups[row["sample_id"]].append(row)
    expected = []
    for sample, items in groups.items():
        header = [r for r in items if r["section"] == "header"]
        strip = sorted((r for r in header if json.loads(r["value_box"])[1] < 49),
                       key=lambda r: json.loads(r["value_box"])[0])
        airline = [r for r in header if 49 <= json.loads(r["value_box"])[1] < 70]
        if len(strip) != 3 or len(airline) != 1:
            raise ValueError("Expected the existing verified header layout")
        for name, row in zip(("primary_flight_id", "alternate_flight_id", "aircraft_code", "airline_name"),
                             strip + airline, strict=True):
            expected.append({"sample_id": sample, "field_id": row["field_id"], "field_name": name,
                             "expected": row["raw_value_text"], "box": json.loads(row["value_box"])})
        for row in items:
            if row["section"] != "airport_cells":
                continue
            role = {"left": "origin", "right": "destination"}[row["column"]]
            expected.append({"sample_id": sample, "field_id": row["field_id"],
                             "field_name": role + "_airport_code", "expected": row["raw_value_text"],
                             "box": json.loads(row["value_box"])})
            for part, item in zip(("city", "timezone_text"), json.loads(row["additional_text"])):
                expected.append({"sample_id": sample, "field_id": row["field_id"],
                                 "field_name": role + "_" + part, "expected": item["text"], "box": item["box"]})
    return expected


def main():
    """Run real production APIs, check lower regression, and publish new audit artifacts."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protected = [TRUTH, LOWER, ROOT / "src/events_diary/ticket_ocr.py",
                 ROOT / "src/events_diary/ocr_spatial.py", ROOT / "src/events_diary/ocr_fields.py"]
    for folder in ("ticket_ocr", "ocr_spatial", "canonical_label_calibration"):
        protected.extend((ROOT / "data/debug/ocr" / folder).glob("*.csv"))
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    records = _rehydrate()
    outputs = extract_fr24_tickets_batch(tuple(r for _, r in records))
    upper_rows, lower_rows, route_rows, time_rows, combined = [], [], [], [], []
    by_sample = {}
    for (summary, raw), output in zip(records, outputs, strict=True):
        sample = f"{Path(summary['video_name']).stem.removesuffix('_test')}_i{raw.investigation_index}_c{raw.candidate_index}"
        by_sample[sample] = output
        flattened = flatten_upper(summary, output.upper)
        upper_rows.extend(flattened)
        lower = [_flatten(summary, f) for f in output.fields]
        lower_rows.extend(lower)
        route = {"video_name": summary["video_name"], **asdict(output.upper.provenance),
                 "route_text": output.upper.route.route_text, "status": output.upper.route.status,
                 "origin_field": "origin_airport_code", "destination_field": "destination_airport_code",
                 "method": "fr24_left_origin_right_destination"}
        route_rows.append(route)
        times = [{"video_name": summary["video_name"], **asdict(t.field.provenance),
                  "canonical_label": t.field.canonical_label, "raw_value_text": t.field.raw_value_text,
                  "label_detection_index": t.field.label_detection_index, "value_detection_index": t.field.value_detection_index,
                  "status": t.field.status, "visual_side": t.visual_side, "route_role": t.route_role}
                 for t in output.time_fields]
        time_rows.extend(times)
        combined.append({"video_name": summary["video_name"], "upper_fields": flattened,
                         "route": route, "fields": lower, "time_fields": times})
    # Exact lower-output parity is an integration check, including confidences/polygons.
    saved_lower = read(LOWER)
    normalized = [{k: "" if v is None else str(v) for k, v in row.items()} for row in lower_rows]
    if normalized != saved_lower:
        raise ValueError("Combined API changed an existing Phase 6.3b field row")
    validation = []
    for ref in references():
        output = by_sample[ref["sample_id"]]
        field = output.upper.get(ref["field_name"])
        location = False
        if field.status == "extracted":
            observation = field.evidence[0]
            bounds = observation.source_bounds
            local_x = bounds.center_x - observation.ticket_roi.x
            local_y = bounds.center_y - observation.ticket_roi.y
            x, y, w, h = ref["box"]
            location = x <= local_x <= x + w and y <= local_y <= y + h
        exact = field.raw_text == ref["expected"]
        whitespace = field.raw_text is not None and "".join(field.raw_text.split()) == "".join(ref["expected"].split())
        result = ("exact" if exact and location else "correct_location_whitespace_difference" if whitespace and location
                  else "correct_location_text_difference" if location else field.status if field.status != "extracted" else "wrong_location")
        validation.append({**{k: v for k, v in ref.items() if k != "box"},
                           "extracted": field.raw_text, "status": field.status, "location_correct": location,
                           "exact": exact, "whitespace_equivalent": whitespace, "result": result,
                           "detection_index": field.evidence[0].detection_index if field.status == "extracted" else None})
    for sample, output in by_sample.items():
        code_refs = {r["field_name"]: r for r in validation if r["sample_id"] == sample and r["field_name"].endswith("airport_code")}
        if len(code_refs) != 2:
            continue
        expected = code_refs["origin_airport_code"]["expected"] + " → " + code_refs["destination_airport_code"]["expected"]
        route = output.upper.route
        correct = all(r["location_correct"] for r in code_refs.values())
        validation.append({"sample_id": sample, "field_id": "derived_from_verified_airport_pair",
                           "field_name": "route", "expected": expected, "extracted": route.route_text,
                           "status": route.status, "location_correct": correct,
                           "exact": route.route_text == expected, "whitespace_equivalent": route.route_text == expected,
                           "result": "exact" if route.route_text == expected and correct else route.status,
                           "detection_index": None})
    summaries = []
    for section, values in (("identity_header", [r for r in upper_rows if r["field_name"] in HEADER_FIELDS]),
                            ("route_airport", [r for r in upper_rows if r["field_name"] not in HEADER_FIELDS]),
                            ("derived_route", route_rows)):
        refs = [r for r in validation if (r["field_name"] in HEADER_FIELDS if section == "identity_header"
                else r["field_name"] == "route" if section == "derived_route" else r["field_name"] not in HEADER_FIELDS + ("route",))]
        counts = Counter(r["status"] for r in values)
        summaries.append({"section": section, "total_slots": len(values),
                          **{status: counts[status] for status in ("extracted", "unavailable", "ambiguous")},
                          "verified_references": len(refs), "exact": sum(r["exact"] for r in refs),
                          "correct_location": sum(r["location_correct"] for r in refs),
                          "correct_location_text_different": sum(r["location_correct"] and not r["exact"] for r in refs),
                          "verified_unavailable": sum(r["status"] == "unavailable" for r in refs),
                          "verified_ambiguous": sum(r["status"] == "ambiguous" for r in refs)})
    write_csv("upper_ticket_results.csv", [{**r, "evidence": json.dumps(r["evidence"])} for r in upper_rows])
    write_csv("route_results.csv", route_rows)
    write_csv("time_fields.csv", time_rows)
    write_csv("validation_table.csv", validation)
    write_csv("summary.csv", summaries)
    (OUTPUT / "ticket_observations.json").write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    table = ["# Phase 6.3c human-readable validation", "",
             "Expected text/boxes come from the unchanged human-verified annotations. Routes are derived from the two verified code references. Missing annotations are unscored, not assumed correct. Location correctness and literal transcription are distinct.", "",
             "| Representative | Field | Extracted raw text | Human-verified / expected | Result |",
             "|---|---|---|---|---|"]
    table += [f"| {r['sample_id']} | {r['field_name']} | {r['extracted'] or '—'} | {r['expected']} | {r['result']} |" for r in validation]
    (OUTPUT / "validation_table.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    report = ["Phase 6.3c FR24 upper-ticket extraction", f"Representatives evaluated: {len(records)}",
              f"Existing lower-field rows unchanged: {len(saved_lower)}/{len(lower_rows)}",
              "Saved OCR only; no model execution, external data or cross-view copying.",
              *[str(r) for r in summaries],
              "Known limitations: current normalized full-ticket layout and identifier shapes only; photo/content boundary and SCHEDULED anchors are provisional FR24 gates. Partial scrolls may retain timezone only. Missing/merged/competing OCR abstains. Raw TELAVIV and UTC spacing remain unchanged; no external repair. Operator/subcontractor row is outside the requested airline slot. No VLM/fusion or final ticket assembly. Accuracy is scored only where human annotations exist."]
    (OUTPUT / "upper_ticket_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    assert all(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == checksum for path, checksum in hashes.items())
    (OUTPUT / "input_provenance.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
