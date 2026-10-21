"""Generate a disposable ticket/OCR demo solely from saved pipeline artifacts.

The PNG juxtaposes two physical ticket crops, new Phase 6.3c upper semantics and
source-labelled Phase 6.3b field observations. Grouping is presentation-only: no
OCR, field extraction, semantic inference, conflict resolution or multi-view
assembly runs here. Raw values, including N/A and imperfect spacing/case, are
unchanged. Missing selected values keep their saved status. Production code and
all source artifacts are read-only; generated files go under data/visual only.
This script and its output directory may be deleted after the presentation.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageStat

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/visual/ocr_extraction_demo"
VIDEO = "ticket_scroll_test.mp4"
INVESTIGATION = 0
SIZE = (2560, 1440)
NAVY = "#142F42"
TEAL = "#007F83"
MUTED = "#516574"
AMBER = "#9A5800"
SOURCES: dict[str, str] = {}
TEXT_CHECKS = 0


def fingerprint(path: Path) -> str:
    """Compute a source hash to detect accidental changes during generation."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(relative: str) -> Path:
    """Register a read-only artifact and return its workspace path."""
    path = ROOT / relative
    SOURCES[relative] = fingerprint(path)
    return path


def rows(relative: str) -> list[dict[str, str]]:
    """Load saved CSV rows without correcting or interpreting their raw text."""
    with source(relative).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def same_candidate(row: dict[str, str], candidate: int) -> bool:
    """Match physical representative identity across recorded pipeline tables."""
    return (row["video_name"] == VIDEO and int(row["investigation_index"]) == INVESTIGATION
            and int(row["candidate_index"]) == candidate)


def put(image: Image.Image, xy: tuple[int, int], value: str, size: int = 32,
        color: str = NAVY, bold: bool = False, width: int | None = None) -> None:
    """Draw bounded, projector-readable text using an existing system font."""
    global TEXT_CHECKS
    draw = ImageDraw.Draw(image)
    filename = "arialbd.ttf" if bold else "arial.ttf"
    chosen = ImageFont.truetype(str(Path("C:/Windows/Fonts") / filename), size)
    while width and draw.textlength(value, font=chosen) > width and size > 26:
        size -= 1
        chosen = ImageFont.truetype(str(Path("C:/Windows/Fonts") / filename), size)
    if width and draw.textlength(value, font=chosen) > width:
        raise ValueError(f"Text exceeds allocated width: {value}")
    box = draw.textbbox(xy, value, font=chosen)
    assert box[0] >= 0 and box[1] >= 0 and box[2] < SIZE[0] and box[3] < SIZE[1], value
    draw.text(xy, value, font=chosen, fill=color)
    TEXT_CHECKS += 1


def panel(image: Image.Image, box: tuple[int, int, int, int], fill: str = "white") -> None:
    """Draw a simple presentation panel independently of all source pixels."""
    ImageDraw.Draw(image).rounded_rectangle(box, radius=22, fill=fill)


def shown_value(row: dict[str, str]) -> str:
    """Return exact selected OCR text or an explicit saved-status placeholder."""
    if row["status"] == "extracted":
        assert row["raw_value_text"] != ""
        return row["raw_value_text"]
    assert row["raw_value_text"] == ""
    return {"ambiguous": "— ambiguous", "special_non_text": "— non-text",
            "no_value_selected": "— unavailable"}[row["status"]]


def display_label(row: dict[str, str]) -> str:
    """Format a canonical identifier without adding airport or flight semantics."""
    label = row["canonical_label"].replace("_", " ")
    if row["canonical_label"] == "SCHEDULED":
        polygon = json.loads(row["label_source_polygon"])
        center_x = sum(point[0] for point in polygon) / len(polygon)
        crop = json.loads(row["ticket_crop"])
        side = "L" if center_x < crop["x"] + crop["width"] / 2 else "R"
        label += f" [{side}]"
    return label


def main() -> None:
    """Validate recorded sources, render the PNG and write exact field/source audits."""
    OUT.mkdir(parents=True, exist_ok=True)
    extraction_path = "data/debug/ocr/ocr_fields/ocr_fields_results.csv"
    extracted = rows(extraction_path)
    summaries = rows("data/debug/ocr/ticket_ocr/ticket_ocr_summary.csv")
    raw = rows("data/debug/ocr/ticket_ocr/ticket_ocr_results.csv")
    upper_path = "data/debug/ocr/ocr_upper_ticket/upper_ticket_results.csv"
    upper_rows = rows(upper_path)
    route_rows = rows("data/debug/ocr/ocr_upper_ticket/route_results.csv")
    selections = rows("data/debug/map_motion/representative_selection/representative_selection.csv")
    materialized = rows("data/debug/map_motion/frame_materialization/frame_materialization.csv")
    metadata = {}
    for candidate, expected_time in ((5, .5), (33, 3.3)):
        summary = next(r for r in summaries if same_candidate(r, candidate))
        selected = next(r for r in selections if same_candidate(r, candidate)
                        and r["configuration"] == "B_default")
        physical = next(r for r in materialized if same_candidate(r, candidate))
        assert float(summary["actual_timestamp"]) == expected_time
        assert summary["frame_index"] == selected["frame_index"] == physical["frame_index"]
        assert float(selected["actual_timestamp"]) == float(physical["actual_timestamp"]) == expected_time
        assert summary["resolution_status"] == selected["resolution_status"] == physical["resolution_status"] == "resolved"
        assert selected["primary_stable"] == "True"
        if candidate == 33:
            assert selected["post_scroll_stable"] == "True"
        metadata[candidate] = {"ocr": summary, "selection": selected, "materialized": physical}

    # Explicit display groups retain view provenance. They do not fuse values.
    groups = [
        ("A · Time fields", 5, ("SCHEDULED", "ACTUAL", "ESTIMATED"), 1200, 1030),
        ("A · Aircraft details", 5, ("AIRCRAFT_TYPE", "REGISTRATION", "AIRCRAFT_CATEGORY",
                                    "SERIAL_NUMBER", "AGE", "COUNTRY_OF_REGISTRATION"), 1850, 354),
        ("B · Flight metrics", 33, ("BAROMETRIC_ALTITUDE", "VERTICAL_SPEED", "GPS_ALTITUDE",
                                   "TRACK", "GROUND_SPEED", "TRUE_AIRSPEED", "INDICATED_AIRSPEED",
                                   "MACH", "WIND", "TEMPERATURE"), 1850, 644),
        ("B · ADS-B / technical", 33, ("ICAO_ADDRESS", "SQUAWK", "LATITUDE", "LONGITUDE", "FIR_UIR"), 1850, 1090),
    ]
    displayed = []
    grouped = []
    for heading, candidate, names, x, y in groups:
        matching = [r for name in names for r in extracted
                    if same_candidate(r, candidate) and r["canonical_label"] == name]
        assert {r["canonical_label"] for r in matching} == set(names)
        for row in matching:
            detections = {r["detection_index"]: r for r in raw if same_candidate(r, candidate)}
            label = detections[row["label_detection_index"]]
            assert label["text"] == row["raw_label_text"]
            assert json.loads(label["source_polygon"]) == json.loads(row["label_source_polygon"])
            assert row["actual_timestamp_seconds"] == metadata[candidate]["ocr"]["actual_timestamp"]
            if row["status"] == "extracted":
                value = detections[row["value_detection_index"]]
                assert value["text"] == row["raw_value_text"]
                assert json.loads(value["source_polygon"]) == json.loads(row["value_source_polygon"])
            displayed.append({"view": "A" if candidate == 5 else "B", "section": heading,
                              "display_label": display_label(row), "display_value": shown_value(row),
                              "source_row": row})
        grouped.append((heading, matching, x, y))
    assert len(displayed) == 25
    upper_a = [r for r in upper_rows if same_candidate(r, 5)]
    route_a = next(r for r in route_rows if same_candidate(r, 5))
    assert len(upper_a) == 10 and all(r["status"] == "extracted" for r in upper_a)
    raw_a = {r["detection_index"]: r for r in raw if same_candidate(r, 5)}
    for field in upper_a:
        evidence = json.loads(field["evidence"])
        assert len(evidence) == 1
        assert evidence[0]["raw_text"] == field["raw_text"] == raw_a[str(evidence[0]["detection_index"])]["text"]
        assert evidence[0]["source_polygon"] == json.loads(raw_a[str(evidence[0]["detection_index"])]["source_polygon"])
    by_name = {r["field_name"]: r for r in upper_a}
    assert route_a["route_text"] == by_name["origin_airport_code"]["raw_text"] + " → " + by_name["destination_airport_code"]["raw_text"]

    image = Image.new("RGB", SIZE, "#F1F5F6")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 2560, 14), fill=TEAL)
    put(image, (64, 48), "EVENTS DIARY  /  REAL PIPELINE OUTPUT", 28, TEAL, True)
    put(image, (64, 106), "OCR-based Ticket Extraction Example", 68, bold=True)
    put(image, (64, 194), f"{VIDEO}  ·  Investigation {INVESTIGATION}  ·  Two stable views, source-labelled field observations", 33, MUTED)

    for candidate, x, title, letter in ((5, 64, "Ticket view — before scroll", "A"),
                                       (33, 620, "Ticket view — after scroll", "B")):
        panel(image, (x, 272, x + 520, 1340))
        put(image, (x + 22, 292), title, 34, bold=True, width=480)
        summary = metadata[candidate]["ocr"]
        put(image, (x + 22, 341), f"{letter}  ·  {summary['actual_timestamp']} s  ·  Candidate {candidate}", 28, TEAL, True)
        path = source(f"data/debug/ocr/ocr_engine_calibration/inputs/ticket_scroll_i0_c{candidate}.png")
        ticket = Image.open(path).convert("RGB")
        geometry = json.loads(summary["crop_geometry"])
        assert ticket.size == (geometry["width"], geometry["height"]) == (336, 831)
        image.paste(ticket.resize((380, 940), Image.Resampling.LANCZOS), (x + 70, 386))

    panel(image, (1178, 272, 2496, 1340))
    put(image, (1200, 291), "Extracted OCR output", 40, bold=True)
    put(image, (1880, 300), "A / B identify the source view", 29, TEAL)
    for heading, matching, x, y in grouped:
        compact = x == 1850
        header_height, row_height = (36, 39) if compact else (47, 45)
        draw.rounded_rectangle((x, y, x + 620, y + header_height), radius=10, fill="#E5F1F1")
        put(image, (x + 12, y + 3), heading, 28, TEAL, True)
        for index, row in enumerate(matching):
            ry = y + header_height + 8 + index * row_height
            label = display_label(row)
            value = shown_value(row)
            # Values are right-aligned; source text itself is never normalized.
            value_font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 30)
            value_width = draw.textlength(value, font=value_font)
            vx = max(x + 330, round(x + 607 - value_width))
            color = NAVY if row["status"] == "extracted" else AMBER
            put(image, (x + 10, ry), label, 27, MUTED, width=vx - x - 28)
            put(image, (vx, ry - 1), value, 30, color, True, width=x + 615 - vx)
            draw.line((x + 10, ry + row_height - 2, x + 610, ry + row_height - 2), fill="#E4EAED", width=1)

    upper_groups = [
        ("A · Header / identity", 354, ("airline_name", "primary_flight_id", "alternate_flight_id", "aircraft_code")),
        ("A · Origin / destination", 685, ("origin_airport_code", "origin_city", "origin_timezone_text",
                                           "destination_airport_code", "destination_city", "destination_timezone_text")),
    ]
    for heading, y, names in upper_groups:
        draw.rounded_rectangle((1200, y, 1820, y + 47), radius=10, fill="#E5F1F1")
        put(image, (1212, y + 5), heading, 29, TEAL, True)
        for index, name in enumerate(names):
            row = by_name[name]
            ry = y + 55 + index * 45
            label = name.replace("_timezone_text", "_timezone").replace("_airport_code", "_airport").replace("_", " ").upper()
            value = row["raw_text"]
            vf = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 28)
            vx = max(1490, round(1807 - draw.textlength(value, font=vf)))
            put(image, (1210, ry), label, 26, MUTED, width=vx - 1225)
            put(image, (vx, ry), value, 28, NAVY, True, width=1815 - vx)
            draw.line((1210, ry + 42, 1810, ry + 42), fill="#E4EAED", width=1)
    panel(image, (1200, 610, 1820, 666), NAVY)
    put(image, (1220, 619), "ROUTE  " + route_a["route_text"], 33, "white", True)
    put(image, (1212, 1295), "Raw OCR preserved; [L]/[R] = visual side.", 26, MUTED, width=600)
    draw.line((64, 1367, 2496, 1367), fill="#C9D5DB", width=2)
    put(image, (64, 1385), "Real Phase 5–6.3c artifacts  ·  Route roles from FR24 position  ·  Raw values retained  ·  No VLM / multi-view fusion", 28, MUTED)

    png = OUT / "ocr_extraction_demo.png"
    image.save(png, dpi=(144, 144))
    with Image.open(png) as reopened:
        reopened.verify()
    with Image.open(png) as reopened:
        assert reopened.size == SIZE and max(ImageStat.Stat(reopened).stddev) > 15

    # Include every exact production row shown, plus all omitted duplicate rows.
    keys = {(r["source_row"]["candidate_index"], r["source_row"]["label_detection_index"],
             r["source_row"]["canonical_label"]) for r in displayed}
    omitted = [r for r in extracted if any(same_candidate(r, c) for c in (5, 33))
               and (r["candidate_index"], r["label_detection_index"], r["canonical_label"]) not in keys]
    report = ["# Disposable OCR extraction demonstration", "", "Image: [ocr_extraction_demo.png](ocr_extraction_demo.png)", "",
              f"Source video: `{VIDEO}`; Investigation {INVESTIGATION}.", "",
              "| View | Candidate / physical frame | Source time | Selection reason |", "|---|---|---|---|"]
    for c, letter in ((5, "A"), (33, "B")):
        summary = metadata[c]["ocr"]
        report.append(f"| {letter} | {c} / {summary['frame_index']} | {summary['actual_timestamp']} s | {summary['selection_reasons']} |")
    report += ["", f"Lower content comes from `{extraction_path}`. New upper content comes from `{upper_path}` and the adjacent `route_results.csv`, generated by the production Phase 6.3c combined API. Raw strings, source polygons and occurrence IDs are cross-checked against Phase 6.1b `ticket_ocr_results.csv`. Representative selection uses saved `B_default` rows, corroborated by resolved frame-materialization metadata. The full ticket PNGs were reused from saved calibration inputs, with the same recorded OCR ROI geometry.", "",
               "## Exact field occurrences displayed", "", "Display labels replace canonical underscores with spaces only. Values are literal saved OCR text. `[L]` / `[R]` are derived from source-label geometry and mean visual column, never departure/arrival. A/B headers preserve each field's view provenance. No field merging, consistency check, best-value selection, temporal reconciliation or complete-ticket assembly is implemented.", "",
               "| View | Canonical | Raw label | Exact selected raw value | Production status | Detection label → value |", "|---|---|---|---|---|---|"]
    for shown in displayed:
        r = shown["source_row"]
        value = f"`{r['raw_value_text']}`" if r["raw_value_text"] else "None"
        report.append(f"| {shown['view']} | {r['canonical_label']} | `{r['raw_label_text']}` | {value} | {r['status']} | {r['label_detection_index']} → {r['value_detection_index'] or 'None'} |")
    report += ["", "## Unavailable and omitted information", "",
               "SERIAL_NUMBER and AGE are recognized from one shared label polygon but have no selected value: both remain ambiguous. COUNTRY_OF_REGISTRATION and FIR_UIR have no textual value and keep special_non_text status. N/A is a literal extracted OCR string, not a placeholder invented by this script.", "",
               "Airline, flight identifiers, aircraft code and all six airport-block fields now come from production upper-ticket extraction and are displayed. Origin/destination are the approved FR24 left/right positional semantics. Raw TELAVIV and timezone spacing are preserved. View B has no visible route blocks and its route is unavailable; A values are displayed as A evidence, never copied into B. Operator/subcontractor text is outside the requested airline slot. No external data or corrected replacement text is used.", "",
               "Repeated metrics from A and the repeated aircraft category from B are omitted solely to avoid duplicate display rows. This is an explicit presentation subset, not deduplication or production multi-view fusion:", ""]
    report += [f"- Candidate {r['candidate_index']}: `{r['canonical_label']}` / `{r['raw_value_text']}` / {r['status']}" for r in omitted]
    report += ["", "## Validation / disposal", "", "The PNG is 2560×1440. It was reopened and checked for nonblank content; annotation text is bounded with a minimum font size of 26 px. The audit stores full exact source rows and input SHA-256 hashes. No OCR/calibration reruns, new models, dependencies, production edits or existing artifact writes are performed. Delete this directory and its standalone visualization helper after the demo if desired.", ""]
    report += ["", "## Phase 6.3c upper fields displayed from view A", "",
               "| Semantic field | Exact raw value | Status |", "|---|---|---|"]
    report += [f"| {r['field_name']} | `{r['raw_text']}` | {r['status']} |" for r in upper_a]
    report += [f"| route | {route_a['route_text']} | {route_a['status']} |", ""]
    (OUT / "ocr_extraction_demo.md").write_text("\n".join(report), encoding="utf-8")
    assert all(fingerprint(ROOT / path) == checksum for path, checksum in SOURCES.items())
    audit = {"video": VIDEO, "investigation_index": INVESTIGATION,
             "sources_sha256": SOURCES, "view_metadata": metadata,
             "upper_fields_displayed": upper_a, "route_displayed": route_a,
             "displayed_fields": displayed, "omitted_duplicate_occurrences": omitted,
             "source_hashes_unchanged": True, "png_size": SIZE,
             "png_reopen_nonblank": "passed", "text_bounds_checks": TEXT_CHECKS,
             "ocr_rerun": False, "production_changes": False, "multi_view_fusion": False}
    (OUT / "ocr_extraction_demo_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Generated {png}")
    print(f"Displayed {len(displayed)} exact field observations; omitted {len(omitted)} repeated occurrences.")
    print("PNG validation and source integrity checks passed; no OCR rerun.")


if __name__ == "__main__":
    main()
