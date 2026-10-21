"""Build a disposable Phase 6.3 presentation from recorded project evidence.

This presentation-only tool reads saved OCR polygons, production fields, verified
annotations and spatial scores. It never runs OCR, extraction or calibration,
and never changes production/debug artifacts. One recorded physical video frame
is decoded solely for context. Original ticket pixels remain visible, with
presentation overlays outside the source text where practical. All generated
files go to data/visual; this helper may also be deleted after the presentation.
Pillow and OpenCV are already available in the existing project environment.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont, ImageStat

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/visual"
DEBUG = ROOT / "data/debug/ocr"
W, H = 2560, 1440
BG = "#F4F6F7"
NAVY = "#122B3D"
MUTED = "#506574"
TEAL = "#007F83"
GREEN = "#14804A"
AMBER = "#A65C00"
BLUE = "#285FBD"
WHITE = "#FFFFFF"
SOURCES: dict[str, str] = {}
FILES: list[str] = []
TEXT_BOUNDS: list[tuple[str, tuple[float, float, float, float]]] = []


def digest(path: Path) -> str:
    """Hash a read-only input so the presentation's evidence is auditable."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(path: Path) -> Path:
    """Register an existing source artifact without modifying its contents."""
    SOURCES[str(path.relative_to(ROOT))] = digest(path)
    return path


def read_csv(relative: str) -> list[dict[str, str]]:
    """Read a recorded UTF-8 CSV, preserving literal OCR strings."""
    with source(DEBUG / relative).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    """Load an installed Windows font; do not download or install fonts."""
    filename = "consola.ttf" if mono else ("arialbd.ttf" if bold else "arial.ttf")
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / filename), size)


def text(image: Image.Image, xy: tuple[int, int], value: str, size: int = 38,
         color: str = NAVY, bold: bool = False, max_width: int | None = None,
         mono: bool = False) -> None:
    """Draw legible text, shrinking long identifiers only within explicit limits."""
    draw = ImageDraw.Draw(image)
    chosen = font(size, bold, mono)
    while max_width and draw.textlength(value, font=chosen) > max_width and size > 26:
        size -= 1
        chosen = font(size, bold, mono)
    if max_width and draw.textlength(value, font=chosen) > max_width:
        raise ValueError(f"Text too wide: {value}")
    draw.text(xy, value, font=chosen, fill=color)
    bounds = draw.textbbox(xy, value, font=chosen)
    if min(bounds[:2]) < 0 or bounds[2] > W or bounds[3] > H:
        raise ValueError(f"Text outside slide: {value}: {bounds}")
    TEXT_BOUNDS.append((value, bounds))


def panel(image: Image.Image, box: tuple[int, int, int, int], fill: str = WHITE,
          outline: str | None = None) -> None:
    """Draw a simple rounded panel to separate explanation from source evidence."""
    ImageDraw.Draw(image).rounded_rectangle(box, radius=22, fill=fill, outline=outline, width=3)


def slide(number: int, title: str, subtitle: str) -> Image.Image:
    """Create a consistent high-resolution slide with title and provenance footer."""
    image = Image.new("RGB", (W, H), BG)
    ImageDraw.Draw(image).rectangle((0, 0, W, 14), fill=TEAL)
    text(image, (76, 48), f"{number:02d} / EVENTS DIARY", 28, TEAL, True)
    text(image, (76, 103), title, 64, bold=True, max_width=2400)
    text(image, (76, 188), subtitle, 34, MUTED, max_width=2400)
    ImageDraw.Draw(image).line((76, 1340, 2484, 1340), fill="#CAD4D9", width=2)
    text(image, (76, 1365), "REAL SAVED EVIDENCE  |  Phase 6.3  |  Temporary presentation", 26, MUTED)
    text(image, (2200, 1365), f"{number} / 8", 28, TEAL, True)
    return image


def arrow(image: Image.Image, start: tuple[float, float], end: tuple[float, float],
          color: str = TEAL, width: int = 6, dashed: bool = False) -> None:
    """Draw an explanatory arrow without changing source geometry or scores."""
    draw = ImageDraw.Draw(image)
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    if dashed:
        for distance in range(0, int(length), 26):
            a, b = distance / length, min(distance + 14, length) / length
            draw.line((start[0] + a * dx, start[1] + a * dy,
                       start[0] + b * dx, start[1] + b * dy), fill=color, width=width)
    else:
        draw.line((start, end), fill=color, width=width)
    angle = math.atan2(dy, dx)
    head = [end, (end[0] - 20 * math.cos(angle - .5), end[1] - 20 * math.sin(angle - .5)),
            (end[0] - 20 * math.cos(angle + .5), end[1] - 20 * math.sin(angle + .5))]
    draw.polygon(head, fill=color)


def save(image: Image.Image, filename: str) -> None:
    """Save and reopen each PNG, checking dimensions, content and decode integrity."""
    path = OUT / filename
    image.save(path, dpi=(144, 144))
    with Image.open(path) as check:
        check.verify()
    with Image.open(path) as check:
        assert check.size == (W, H)
        assert max(ImageStat.Stat(check).stddev) > 15, filename
    FILES.append(filename)


def crop_view(image: Image.Image, candidate: int, region: tuple[int, int, int, int],
              xy: tuple[int, int], width: int):
    """Paste an unaltered ticket region and return a source-polygon coordinate map.

    Recorded polygons use original-video coordinates. Saved ticket crops begin
    at x=16,y=121; region coordinates here are relative to that ticket crop.
    Only a uniform presentation resize is applied, never inferred label boxes.
    """
    path = source(DEBUG / f"ocr_engine_calibration/inputs/ticket_scroll_i0_c{candidate}.png")
    original = Image.open(path).convert("RGB")
    piece = original.crop(region)
    scale = width / piece.width
    image.paste(piece.resize((width, round(piece.height * scale)), Image.Resampling.LANCZOS), xy)

    def transform(polygon: list[list[float]]) -> list[tuple[float, float]]:
        """Convert saved source-frame polygon vertices into presentation coordinates."""
        return [(xy[0] + (x - 16 - region[0]) * scale,
                 xy[1] + (y - 121 - region[1]) * scale) for x, y in polygon]

    return transform


def polygon(image: Image.Image, points: list[tuple[float, float]], color: str) -> None:
    """Outline the actual saved polygon, without replacing it with invented geometry."""
    ImageDraw.Draw(image).line(points + [points[0]], fill=color, width=5, joint="curve")


def center(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Return a polygon's mean vertex position for a presentation connector."""
    return (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))


def main() -> None:
    """Generate eight evidence-based visuals, Hebrew notes and an integrity manifest."""
    OUT.mkdir(parents=True, exist_ok=True)
    raw = read_csv("ticket_ocr/ticket_ocr_results.csv")
    summaries = read_csv("ticket_ocr/ticket_ocr_summary.csv")
    fields = read_csv("ocr_fields/ocr_fields_results.csv")
    verified = read_csv("ocr_fields/field_validation_table.csv")
    relations = read_csv("ocr_spatial/ocr_spatial_relations.csv")
    metrics = read_csv("ocr_fields/verified_regression_metrics.csv")
    unknown = read_csv("ocr_fields/unknown_controls.csv")
    assert len(summaries) == 17 and len(fields) == 246 and len(verified) == 46
    assert all(r["canonical_correct"] == "True" for r in verified)
    assert sum(r["location_correct"] == "True" for r in verified) == 39
    assert sum(r["raw_value_exact"] == "True" for r in verified) == 26
    assert sum(r["production_status"] == "ambiguous" for r in verified) == 4
    status_counts = {status: sum(r["status"] == status for r in fields) for status in
                     ("extracted", "ambiguous", "no_value_selected", "special_non_text")}
    assert list(status_counts.values()) == [188, 26, 14, 18]

    def detections(candidate: int) -> dict[int, dict[str, str]]:
        """Index one representative's original detections without rerunning OCR."""
        return {int(r["detection_index"]): r for r in raw
                if r["video_name"] == "ticket_scroll_test.mp4"
                and int(r["candidate_index"]) == candidate}

    def field(candidate: int, canonical: str) -> dict[str, str]:
        """Find an actual field occurrence in a single representative."""
        return next(r for r in fields if r["video_name"] == "ticket_scroll_test.mp4"
                    and int(r["candidate_index"]) == candidate and r["canonical_label"] == canonical)

    def reference(field_id: str) -> dict[str, str]:
        """Find a human-verified example by its stable annotation identifier."""
        return next(r for r in verified if r["field_id"] == field_id)

    def relation(a: int, b: int) -> dict[str, str]:
        """Read a saved spatial score for the 2.5-second technical example."""
        return next(r for r in relations if r["video_name"] == "ticket_scroll_test.mp4"
                    and r["candidate_index"] == "25" and int(r["source_index"]) == a
                    and int(r["target_index"]) == b)

    d25 = detections(25)
    summary = next(r for r in summaries if r["video_name"] == "ticket_scroll_test.mp4"
                   and r["candidate_index"] == "25")
    assert summary["frame_index"] == "25" and float(summary["actual_timestamp"]) == 2.5
    assert json.loads(summary["crop_geometry"]) == {"x": 16, "y": 121, "width": 336, "height": 831}

    # 1: architecture, with an independent future VLM consumer of the clip.
    im = slide(1, "From video to traceable OCR evidence", "Completed path: deterministic selection + OCR branch through Phase 6.3")
    stages = [("Video", "Original recording"), ("Investigation detection", "Boundaries + visual timing"),
              ("Representative selection", "Stable visual evidence"), ("Full ticket crop", "Precise ticket ROI"),
              ("2× color upscale", "Cubic preprocessing"), ("RapidOCR", "Raw text + confidence"),
              ("Detections + polygons", "Original-video coordinates"), ("Spatial relationships", "Stack / inline ranking"),
              ("Canonical labels", "Stable field identities"), ("Label/value association", "Select or abstain"),
              ("OCRFieldObservation", "Structured + traceable")]
    positions = [(90 + c * 800, 290 + r * 230) for r in range(3) for c in range(3)]
    positions += [(890, 980), (1690, 980)]
    for index, ((title, detail), (x, y)) in enumerate(zip(stages, positions, strict=True)):
        panel(im, (x, y, x + 690, y + 164), outline=TEAL if index == 10 else None)
        text(im, (x + 25, y + 23), f"{index + 1:02d}  {title}", 37, bold=True, max_width=640)
        text(im, (x + 25, y + 88), detail, 30, MUTED, max_width=640)
        if index in (0, 1, 3, 4, 6, 7, 9):
            arrow(im, (x + 706, y + 80), (x + 778, y + 80))
        elif index in (2, 5, 8):
            # Numbering and an explicit continuation avoid a crossing return path.
            text(im, (x + 25, y + 174), f"Continue at {index + 2:02d} below", 27, TEAL)
    text(im, (100, 1000), "NEXT / NOT IMPLEMENTED", 29, AMBER, True)
    text(im, (100, 1050), "Phase 6.4: independent VLM", 32, bold=True)
    text(im, (100, 1100), "Input: full Investigation core clip", 29, MUTED)
    text(im, (100, 1225), "The future VLM branch consumes the clip independently; it is not an OCR fallback.", 36, TEAL, True)
    save(im, "01_pipeline_overview.png")

    # 2: exact physical frame, decoded for context only; recorded ROI is highlighted.
    im = slide(2, "Select useful evidence before reading text", "ticket_scroll_test.mp4  |  Investigation 0  |  Candidate 25  |  2.5 s  |  Frame 25")
    video = ROOT / "data/videos/ticket_scroll_test.mp4"
    cap = cv2.VideoCapture(str(video))
    assert cap.isOpened()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 25)
    success, frame = cap.read()
    cap.release()
    assert success and frame.shape[:2] == (1080, 1920)
    frame_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    original = Image.open(source(DEBUG / "ocr_engine_calibration/inputs/ticket_scroll_i0_c25.png")).convert("RGB")
    assert frame_image.crop((16, 121, 352, 952)).tobytes() == original.tobytes()
    im.paste(frame_image.resize((1392, 783), Image.Resampling.LANCZOS), (80, 310))
    s = 1392 / 1920
    box = (80 + 16*s, 310 + 121*s, 80 + 352*s, 310 + 952*s)
    ImageDraw.Draw(im).rectangle(box, outline=TEAL, width=6)
    crop_view(im, 25, (0, 0, 336, 831), (1830, 275), 410)
    arrow(im, (box[2] + 12, box[1] + 100), (1785, 520), TEAL)
    text(im, (1485, 1155), "Ticket OCR ROI", 30, TEAL, True, max_width=300)
    text(im, (1485, 1200), "336 × 831", 30, MUTED)
    text(im, (1485, 1245), "source pixels", 29, MUTED)
    panel(im, (80, 1130, 1430, 1295))
    text(im, (110, 1155), "Selection: " + summary["selection_reasons"], 36, bold=True, max_width=1290)
    text(im, (110, 1220), "Phases 1–5 select stable evidence; OCR then reads the precise ROI.", 32, MUTED)
    save(im, "02_representative_ticket.png")

    # 3: six original detections, with raw text and confidence in a separate panel.
    im = slide(3, "RapidOCR returns evidence, not corrected fields", "Candidate 25 at 2.5 s  |  2× color preprocessing  |  Curated detections from saved OCR")
    transform = crop_view(im, 25, (36, 415, 308, 520), (100, 390), 1224)
    selected = [17, 19, 18, 20, 22, 24]
    colors = [TEAL, GREEN, TEAL, GREEN, BLUE, BLUE]
    for index, color in zip(selected, colors, strict=True):
        polygon(im, transform(json.loads(d25[index]["source_polygon"])), color)
    text(im, (105, 290), "Original ticket region + actual detection polygons", 34, bold=True)
    for row, (index, color) in enumerate(zip(selected, colors, strict=True)):
        r = d25[index]
        y = 315 + row * 125
        panel(im, (1430, y, 2470, y + 105))
        text(im, (1460, y + 12), r["text"], 40, color, True, max_width=680)
        text(im, (2170, y + 22), f"{float(r['confidence']):.3f}", 36, MUTED)
        text(im, (1460, y + 64), f"Detection {index}  |  OCR confidence", 27, MUTED)
    text(im, (110, 1020), "Raw text is preserved exactly:", 44, bold=True)
    text(im, (110, 1090), "BAROMETRICALT. stays BAROMETRICALT.", 38, TEAL)
    text(im, (110, 1150), "0ft stays 0ft — no silent text repair.", 38, TEAL)
    text(im, (110, 1250), "RapidOCR output = raw text + confidence + geometry", 40, bold=True)
    save(im, "03_raw_ocr_detections.png")

    # 4: original two-column geometry; arrows are explanatory, scores are recorded.
    im = slide(4, "Geometry separates neighboring fields", "Phase 6.2 ranks spatial relationships; Phase 6.3 selects a value or preserves ambiguity")
    transform = crop_view(im, 25, (36, 422, 308, 473), (100, 440), 1360)
    points = {i: transform(json.loads(d25[i]["source_polygon"])) for i in (17, 18, 19, 20)}
    for i, p in points.items():
        polygon(im, p, TEAL if i in (17, 18) else GREEN)
    # Draw through the whitespace between each genuine label/value polygon.
    for label, value in ((17, 19), (18, 20)):
        a, b = points[label], points[value]
        x = center(b)[0]
        arrow(im, (x, max(p[1] for p in a) + 5), (x, min(p[1] for p in b) - 4), GREEN, 5)
    a, b = center(points[17]), center(points[20])
    arrow(im, (a[0] + 220, a[1] + 55), (b[0] - 100, b[1] - 20), AMBER, 5, True)
    same = relation(17, 19)
    wrong = relation(17, 20)
    panel(im, (1550, 310, 2475, 1040))
    for n, line in enumerate(("Two separate column fields", "Labels above values", "Same-column compatibility", "Horizontal overlap / alignment")):
        text(im, (1590, 360 + n * 95), line, 37, bold=n == 0, max_width=840)
    text(im, (1590, 805), f"BARO → 0ft: {float(same['vertical_stack_score']):.3f}", 37, GREEN, True)
    text(im, (1590, 875), f"BARO → N/A: {float(wrong['vertical_stack_score']):.3f}", 37, AMBER, True)
    text(im, (1590, 950), "Actual stack scores, not probabilities", 29, MUTED)
    text(im, (105, 840), "Solid: within-column association", 38, GREEN, True)
    text(im, (105, 905), "Dashed: competing opposite-column candidate", 36, AMBER)
    text(im, (105, 1150), "Nearby text is not automatically the right value.", 46, bold=True)
    text(im, (105, 1220), "Keep the two-column field grid; rank the recorded geometry.", 36, TEAL)
    save(im, "04_spatial_relationships.png")

    # 5: all mappings come from actual production field records.
    im = slide(5, "Noisy UI strings → stable canonical identities", "Canonicalization adds meaning while preserving the original OCR text and polygon")
    text(im, (110, 290), "RAW OCR LABEL", 30, MUTED, True)
    text(im, (1290, 290), "CANONICAL IDENTITY", 30, MUTED, True)
    mapping = [(25, "BAROMETRIC_ALTITUDE"), (33, "ICAO_ADDRESS"), (33, "GROUND_SPEED"),
               (25, "VERTICAL_SPEED"), (54, "AIRCRAFT_TYPE")]
    for n, (candidate, name) in enumerate(mapping):
        r = field(candidate, name)
        y = 355 + n * 145
        panel(im, (90, y, 2470, y + 118))
        text(im, (120, y + 30), r["raw_label_text"], 42, max_width=960)
        arrow(im, (1130, y + 58), (1220, y + 58))
        text(im, (1290, y + 30), name, 43, TEAL, True, max_width=1110)
    text(im, (110, 1150), "Compound source: SERIALNUMBER(MSN)AGE(Mar2010)", 38, bold=True)
    text(im, (110, 1210), "→ SERIAL_NUMBER + AGE; qualifiers are not part of the canonical identity.", 36, TEAL)
    save(im, "05_canonical_label_mapping.png")

    # 6: location correctness and literal transcription are distinct measurements.
    im = slide(6, "Correct association does not mean perfect text", "Human-verified examples from candidates 5 / 25 / 33; each occurrence retains its own frame provenance")
    columns = [(105, "RAW LABEL"), (620, "CANONICAL"), (1280, "SELECTED RAW"), (1690, "VERIFIED"), (2130, "RESULT")]
    for x, label in columns:
        text(im, (x, 290), label, 28, MUTED, True)
    ids = ["F25", "F39", "F40", "F7", "F22", "F9"]
    for n, field_id in enumerate(ids):
        r = reference(field_id)
        y = 350 + n * 115
        panel(im, (85, y, 2480, y + 100))
        text(im, (105, y + 17), r["raw_ocr_label"], 33, max_width=470)
        text(im, (105, y + 62), field_id, 26, MUTED)
        text(im, (620, y + 27), r["canonical"], 34, TEAL, True, max_width=620)
        text(im, (1280, y + 27), r["selected_raw_value"], 37)
        text(im, (1690, y + 27), r["verified_value"], 37)
        exact = r["raw_value_exact"] == "True"
        text(im, (2130, y + 15), "Location correct", 29, GREEN, True)
        text(im, (2130, y + 56), "Text exact" if exact else "Text differs", 29, GREEN if exact else AMBER)
    panel(im, (85, 1090, 2480, 1305), NAVY)
    text(im, (120, 1110), "46/46 labels    |    39/39 selected locations correct    |    0 wrong    |    4 compound abstentions", 36, WHITE, True, max_width=2320)
    text(im, (120, 1175), "Coverage: 39/43 eligible text fields. Exact raw transcription: 26/39 (66.67%).", 35, "#7FE1D6")
    text(im, (120, 1235), "13 selected values have correct locations but differ from verified text. No text correction is applied.", 31, WHITE)
    save(im, "06_field_extraction_success.png")

    # 7: preserve the single actual compound polygon, with no fictional sub-boxes.
    im = slide(7, "Two meanings, one OCR polygon: abstain", "Candidate 25 at 2.5 s  |  Actual detection 11 contains both labels")
    transform = crop_view(im, 25, (36, 298, 308, 351), (100, 370), 1360)
    polygon(im, transform(json.loads(d25[11]["source_polygon"])), AMBER)
    text(im, (100, 285), d25[11]["text"], 40, AMBER, True)
    text(im, (100, 710), "One shared source polygon — no fabricated per-label boxes", 35, AMBER, True)
    panel(im, (1540, 290, 2470, 1030))
    text(im, (1580, 330), "Recognized identities", 38, bold=True)
    text(im, (1580, 405), "SERIAL_NUMBER", 45, TEAL, True)
    text(im, (1580, 475), "AGE", 45, TEAL, True)
    text(im, (1580, 590), "Visible candidate values", 36, bold=True)
    text(im, (1580, 660), d25[12]["text"] + "     /     " + d25[13]["text"], 45)
    text(im, (1580, 800), "Production result for both:", 35, bold=True)
    text(im, (1580, 865), "ambiguous; value = None", 41, AMBER, True)
    text(im, (110, 890), "Semantic recognition succeeds.", 46, bold=True)
    text(im, (110, 960), "Independent value association is not supported", 37)
    text(im, (110, 1015), "by the shared label geometry.", 37)
    text(im, (110, 1150), "4 verified compound occurrences abstain; they are excluded from selected-location accuracy.", 36, AMBER, True, max_width=2360)
    text(im, (110, 1220), "Later independent VLM evidence may help; it has not been implemented or evaluated yet.", 35, TEAL)
    save(im, "07_compound_label_ambiguity.png")

    # 8: a single frame's output, never assembled from several scroll views.
    im = slide(8, "Structured OCR evidence, with traceability", "One frame only: ticket_scroll_test.mp4  |  Investigation 0  |  Candidate 25  |  2.5 s")
    text(im, (110, 290), "CANONICAL FIELD", 29, MUTED, True)
    text(im, (800, 290), "RAW VALUE", 29, MUTED, True)
    text(im, (1120, 290), "STATUS", 29, MUTED, True)
    names = ["TRACK", "GROUND_SPEED", "SQUAWK", "BAROMETRIC_ALTITUDE", "VERTICAL_SPEED", "AGE", "SERIAL_NUMBER", "COUNTRY_OF_REGISTRATION"]
    for n, name in enumerate(names):
        r = field(25, name)
        y = 350 + n * 88
        panel(im, (90, y, 1650, y + 76))
        color = GREEN if r["status"] == "extracted" else AMBER
        text(im, (110, y + 20), name, 34, max_width=660)
        text(im, (800, y + 18), r["raw_value_text"] or "—", 37)
        text(im, (1120, y + 20), r["status"], 33, color, True)
    track = field(25, "TRACK")
    panel(im, (1730, 290, 2470, 1085), NAVY)
    info = [("TRACK provenance", 38, True), ("Physical frame: 25", 34, False),
            ("Source time: 2.5 s", 34, False),
            (f"Detection: {track['label_detection_index']} → {track['value_detection_index']}", 34, False),
            (f"Label OCR: {float(track['ocr_label_confidence']):.4f}", 34, False),
            (f"Value OCR: {float(track['ocr_value_confidence']):.4f}", 34, False),
            (f"Stack score: {float(track['relationship_score']):.4f}", 34, False),
            ("Source polygons retained", 31, False), ("Scores remain distinct", 31, False)]
    for n, (line, size, bold) in enumerate(info):
        text(im, (1770, 330 + n * 78), line, size, WHITE, bold, max_width=660)
    text(im, (110, 1120), "Full run: 17 representatives / 246 field observations (not 246 successful extractions)", 34, bold=True)
    text(im, (110, 1175), "188 extracted  |  26 ambiguous  |  14 no value selected  |  18 special/non-text", 33, MUTED)
    text(im, (110, 1240), "NEXT: 6.4 independent full-Investigation VLM clip analysis → 6.5 OCR ↔ VLM fusion", 35, TEAL, True, max_width=2350)
    save(im, "08_final_ocr_output.png")

    write_notes()
    purposes = ["Completed pipeline and the independent future VLM branch.",
                "One selected stable frame and its precise full-ticket OCR ROI.",
                "Actual raw text, confidences and source polygons.",
                "Two-column geometry and competing spatial relationships.",
                "Real OCR strings mapped to stable canonical labels.",
                "Verified field association versus literal OCR accuracy.",
                "A shared compound polygon and honest production abstention.",
                "A single frame's structured output, provenance and next stages."]
    readme = "# Temporary Events_Diary presentation\n\nOpen the PNGs full-screen in the numbered order below. Each is 2560 × 1440 (16:9).\n\n"
    readme += "\n".join(f"{i}. [{name}]({name}) — {purpose}" for i, (name, purpose) in enumerate(zip(FILES, purposes, strict=True), 1))
    readme += "\n\nSpeaker notes: [presentation_notes.md](presentation_notes.md), in Hebrew, approximately 10 minutes.\n\n"
    readme += "All results come from saved OCR/evaluation outputs. No OCR or calibration was rerun. One original physical frame was decoded for visual context and its ticket pixels checked against the saved crop. The final table is candidate 25 only; no multi-view assembly is implied.\n\n"
    readme += "Canonical examples preserve real raw strings, including the complete shared compound string. Verified metrics apply to seven crops; the full 17-representative run is shown separately. Source confidences and spatial scores are distinct quantities, not interchangeable probabilities.\n\n"
    readme += "This entire directory is temporary and safe to delete after the presentation. The presentation-only helper `scripts/visualization/create_phase6_demo_visuals.py` may also be deleted. It is not imported by production.\n\n"
    readme += "`presentation_validation.json` records source hashes, metric checks, example IDs and PNG validation. No production code, tests, dependencies, project docs or existing debug outputs were changed.\n"
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    assert all(digest(ROOT / path) == value for path, value in SOURCES.items())
    notes = (OUT / "presentation_notes.md").read_text(encoding="utf-8")
    assert all(name in notes for name in FILES)
    manifest = {"source_sha256": SOURCES, "video_context": {"path": str(video.relative_to(ROOT)), "frame": 25, "ticket_pixels_equal_saved_crop": True},
                "pngs": FILES, "size": [W, H], "png_decode_nonblank_checks": "passed",
                "text_bounds_checks": len(TEXT_BOUNDS), "minimum_annotation_font_px": 26,
                "representatives": len(summaries), "production_observations": len(fields),
                "production_statuses": status_counts, "verified_occurrences": len(verified),
                "verified_metrics": metrics, "unknown_control_rows": len(unknown),
                "success_slide_field_ids": ids, "ocr_rerun": False, "source_hashes_unchanged": True}
    (OUT / "presentation_validation.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Generated {len(FILES)} validated PNGs in {OUT}")
    print("Saved Hebrew notes, README and source/validation manifest. Existing source hashes unchanged.")


def write_notes() -> None:
    """Write conversational Hebrew speaker notes with the exact slide filenames."""
    notes = """# הערות דובר — Events_Diary עד שלב 6.3

משך מתוכנן: כ־10 דקות. התמונות באנגלית כדי לשמור על שמות השדות וה־OCR המקוריים; ההסבר בעל פה בעברית. מומלץ להציג כל PNG במסך מלא לפי הסדר.

### Slide/Visual 1 — תמונת המערכת (דקה)
קובץ: `01_pipeline_overview.png`

- **מה רואים:** המסלול שכבר הושלם, מהווידאו ועד תצפית OCR מובנית. המספור ממשיך משמאל לימין, שורה אחרי שורה. המסלול העתידי של VLM מוצג בנפרד.
- **מה לומר:** ״המטרה היא להפוך הקלטת מסך לראיות שאפשר לבדוק. קודם מזהים חלון Investigation ובוחרים פריימים שימושיים. אחר כך חותכים את הכרטיס, מגדילים פי שניים בצבע ומפעילים RapidOCR. הפלט שלו עדיין לא אומר איזה ערך שייך לאיזה שדה — את זה אנחנו מוסיפים בשכבות הבאות.״
- **למה השלב קיים:** לחבר בין העבודה הדטרמיניסטית בשלבים 1–5 לבין ענף הטקסט שכבר עובד.
- **החלטה טכנית:** גבולות, תזמון ובחירת ראיות נשארים באחריות הצינור הדטרמיניסטי. בעתיד VLM יקבל את קליפ ה־Investigation המלא כצרכן עצמאי; הוא לא גיבוי ל־OCR ולא תלוי בפלט שלו.
- **תוצאה מרכזית:** ענף ה־OCR הושלם עד 6.3; שלב 6.4 טרם מומש. גם מיזוג המקורות ב־6.5 והרכבת כרטיס ממספר תצוגות ב־6.6 עדיין בהמשך.
- **מעבר:** ״כדי להבין למה בחירת הראיות חשובה, נתחיל מפריים אמיתי שנבחר.״

### Slide/Visual 2 — בחירת ראיה יציבה (דקה)
קובץ: `02_representative_ticket.png`

- **מה רואים:** פריים 25 מתוך ticket_scroll_test.mp4, בזמן 2.5 שניות, Investigation מספר 0 ומועמד 25. משמאל הפריים המקורי עם מסגרת סביב ה־ROI, ומימין הכרטיס המוגדל לתצוגה.
- **מה לומר:** ״אנחנו לא שולחים כל פריים אקראי ל־OCR. המערכת כבר מצאה כאן תצוגה יציבה אחרי גלילה. רק אז לוקחים את אזור הכרטיס המדויק. לכן אפשר לקשור כל תוצאה בחזרה לפריים ולזמן המקוריים.״
- **למה השלב קיים:** לצמצם עבודה מיותרת ולהעדיף ראיות שאפשר לקרוא, תוך שמירה על עקיבות.
- **החלטה טכנית:** משתמשים בפריים פיזי שנבחר ובחיתוך המקורי; לא מרכיבים כאן תמונה מכמה פריימים. ההגדלה ב־OCR היא פי שניים בצבע, ואינה שינוי בטקסט המקור.
- **תוצאה מרכזית:** סיבות הבחירה השמורות הן stable_ticket ו־post_scroll_stable. חיתוך הכרטיס הוא 336 על 831 פיקסלים מתוך פריים 1920 על 1080.
- **מעבר:** ״עכשיו נראה מה מנוע ה־OCR מחזיר בפועל, לפני כל פרשנות שלנו.״

### Slide/Visual 3 — טקסט גולמי וגאומטריה (דקה ורבע)
קובץ: `03_raw_ocr_detections.png`

- **מה רואים:** קטע מאותו כרטיס עם שישה פוליגונים אמיתיים שנשמרו מ־RapidOCR. בצד מוצגים הטקסט הגולמי, מזהה הזיהוי וציון הביטחון שלו.
- **מה לומר:** ״RapidOCR מחזיר שלושה דברים: מה הוא קרא, איפה הוא ראה את זה, ועד כמה המנוע בטוח בזיהוי. שימו לב ל־BAROMETRICALT. בלי רווחים ול־0ft. אנחנו משאירים אותם בדיוק כך. גם 281° הוא זיהוי נפרד מ־TRACK.״
- **למה השלב קיים:** לשמר את הראיה לפני קנוניזציה או בחירת ערך, כדי שאפשר יהיה לבקר החלטות מאוחרות יותר.
- **החלטה טכנית:** לא מתקנים טקסט בשקט. ציון הביטחון של OCR אינו ציון ההתאמה הגאומטרית ואינו אמת סמנטית סופית.
- **תוצאה מרכזית:** אלה זיהויים שמורים מהריצה הקיימת, לא ריצת OCR חדשה ולא טקסט שהומצא למצגת. מוצג רק מבחר קטן כדי להישאר קריא.
- **מעבר:** ״הטקסט קיים, אבל נשארת השאלה: איזה מספר שייך לאיזו תווית?״

### Slide/Visual 4 — קשרים מרחביים (דקה וחצי)
קובץ: `04_spatial_relationships.png`

- **מה רואים:** שתי עמודות מקוריות: BAROMETRIC ALT. עם 0 ft, ו־VERTICAL SPEED עם N/A. המסגרות הן פוליגוני הזיהוי המקוריים. החצים הם הסבר למצגת, לא פלט נוסף של המנוע.
- **מה לומר:** ״כאן שני שדות נמצאים כמעט באותו גובה. אם היינו מחברים רק לפי קרבה או לפי שורה, היינו עלולים לערבב ביניהם. במקום זה שומרים את הגאומטריה ומדרגים קשרים: האם הערך באותה עמודה, האם הוא מתחת לתווית, ומה מידת החפיפה והיישור. החץ המקווקו מראה מועמד מתחרה בעמודה השנייה.״
- **למה השלב קיים:** להבין מבנה של כרטיס דו־עמודי, בלי לשטח אותו לשורת טקסט אחת.
- **החלטה טכנית:** שדות טכניים משתמשים בציון stack; תוויות זמן משתמשות בציון inline. אין כלל אוניברסלי של ״הערך תמיד מתחת״. נוסחאות שלב 6.2 נשארו ללא שינוי.
- **תוצאה מרכזית:** שני הציונים בתמונה נקראו מקובץ הקשרים הקיים. אלה ציוני דירוג גאומטריים, לא הסתברויות ולא ביטחון OCR.
- **מעבר:** ״אחרי המבנה המרחבי צריך לזהות גם את המשמעות הקבועה של התווית, למרות רעש בכתיב.״

### Slide/Visual 5 — זהות קנונית (דקה)
קובץ: `05_canonical_label_mapping.png`

- **מה רואים:** חמש המרות אמיתיות של תוויות OCR לזהויות קנוניות יציבות, ובתחתית דוגמה למחרוזת מורכבת.
- **מה לומר:** ״המשמעות של BAROMETRICALT. היא BAROMETRIC_ALTITUDE גם אם הרווחים נעלמו. באופן דומה, MSN הוא תיאור של שדה המספר הסידורי; חודש ושנה בתווית AGE אינם חלק משם השדה. לכן נשמור את המחרוזת הגולמית לצד הזהות הקנונית, ולא נחליף אותה.״
- **למה השלב קיים:** לאפשר לצרכנים הבאים לעבוד עם זהות יציבה, בלי לאבד את המקור.
- **החלטה טכנית:** מתחילים בנרמול רווחים, סימנים ורישיות, משתמשים בכללים ייעודיים מוגבלים לתוויות, ורק אם זה נכשל מאפשרים fuzzy שמרני. מחרוזת לא מוכרת יכולה להישאר לא מוכרת; לא חייבים למצוא לה תווית בכוח.
- **תוצאה מרכזית:** 46 מתוך 46 מופעי התוויות המאומתים זוהו, על פני 24 זהויות. לא היו שיוכים שגויים ב־105 דוגמאות לא־תווית מאומתות וב־16 בקרות סינתטיות. זה אינו מדד כללי לכל מסך אפשרי.
- **מעבר:** ״עכשיו אפשר לחבר את הזהות הקנונית לערך ולבדוק בנפרד את הקשר ואת איכות הקריאה.״

### Slide/Visual 6 — חילוץ ותוצאות (דקה וחצי)
קובץ: `06_field_extraction_success.png`

- **מה רואים:** דוגמאות מאומתות ממספר פריימים נפרדים, עם תווית גולמית, זהות קנונית, ערך שנבחר וערך שאדם אימת. מזהי F הם מזהי ההערות המקוריים.
- **מה לומר:** ״TRACK ו־SQUAWK כאן מדויקים גם בקשר וגם בטקסט. לעומתם, ב־0ft נבחר בדיוק המקום הנכון אבל חסר רווח; ב־2:30 pM יש הבדל ברישיות. אלה שני סוגי איכות שונים. אנחנו לא מסתירים את ההבדל ולא מתקנים את התוצאה כדי להציג הצלחה.״
- **למה השלב קיים:** למדוד אם בחרנו את הראיה הנכונה בנפרד מהשאלה אם OCR קרא כל תו בדיוק.
- **החלטה טכנית:** המדדים מחושבים רק על שבעת החיתוכים שאומתו אנושית. אין איחוד של המועמדים שמוצגים כאן לכרטיס אחד.
- **תוצאה מרכזית:** 46/46 תוויות; 39/39 מיקומי הערכים שנבחרו נכונים; אפס בחירות שגויות. הכיסוי הוא 39 מתוך 43 שדות טקסט זכאים, וארבעה מופעים מורכבים נמנעו מבחירה. הטקסט מדויק ב־26 מתוך 39, כלומר 66.67%; ב־13 המקום נכון והטקסט שונה. בטכניים: 19 מתוך 23 נבחרו נכון וארבע הימנעויות; בזמנים: 20 מתוך 20 נבחרו נכון.
- **מעבר:** ״ארבע ההימנעויות אינן תקלה מוסתרת. הנה המקרה שמסביר למה הן נחוצות.״

### Slide/Visual 7 — עמימות בתווית מורכבת (דקה)
קובץ: `07_compound_label_ambiguity.png`

- **מה רואים:** מנוע OCR החזיר זיהוי אחד למחרוזת SERIALNUMBER(MSN)AGE(Mar2010), עם פוליגון אחד. מתחתיה מופיעים 1100 ו־16 years.
- **מה לומר:** ״אנחנו יודעים לזהות שיש כאן שתי משמעויות: מספר סידורי וגיל. אבל למנוע הייתה רק גאומטריה אחת לשתיהן. אסור לנו להמציא עכשיו שתי תיבות כאילו הוא זיהה אותן בנפרד. לכן המחלץ שומר את שתי התוויות, מסמן שהגאומטריה משותפת ומשאיר את הערך לא נבחר.״
- **למה השלב קיים:** למנוע החלטה שנראית בטוחה אך אינה נתמכת בראיה שנשמרה.
- **החלטה טכנית:** הימנעות מפורשת במקום בחירה כפויה. אין שימוש בכלל של ״גיל חייב להסתיים בשנים״, ואין היוריסטיקה חדשה שמפרידה את התיבה לפי מיקום אופקי.
- **תוצאה מרכזית:** ארבעה מופעים מאומתים כאלה נשארו עמומים. המספר 39/39 אינו כולל אותם כהצלחות. ראיות VLM עצמאיות עשויות לעזור בעתיד, אך זה עדיין לא נבדק ולא מומש.
- **מעבר:** ״למרות ההימנעות, הפלט נשאר שימושי ומובנה. נראה איך הוא נראה לצרכן הבא.״

### Slide/Visual 8 — פלט מובנה והמשך הדרך (דקה)
קובץ: `08_final_ocr_output.png`

- **מה רואים:** תת־קבוצה אמיתית של שדות מפריים 25 בלבד, עם ערך וסטטוס. בצד רואים דוגמת עקיבות עבור TRACK: פריים, זמן, מזהי זיהוי, ביטחון תווית וערך וציון stack.
- **מה לומר:** ״הפלט הוא כבר לא רק טקסט. כל שדה הוא תצפית שאפשר לחזור ממנה לפריים ולפוליגון המקוריים. גם ערך שלא נבחר הוא מידע: אפשר להבדיל בין חילוץ מוצלח, עמימות ומקרה לא־טקסטואלי כמו דגל מדינה. אנחנו עדיין לא בונים רשומת Investigation סופית ולא מאחדים מספר תצוגות גלילה.״
- **למה השלב קיים:** לתת לשלב הבא ראיות מובנות וניתנות לביקורת, כולל מגבלות המקור.
- **החלטה טכנית:** משמרים ביטחון OCR, ציון התאמה קנונית וציון מרחבי בנפרד. VLM יקרא בעתיד את קליפ ה־Investigation המלא בענף עצמאי, עם הבחנה בין הכוונה דטרמיניסטית לבין מה שראה בעצמו. שני הענפים יוכלו לפעול במקביל; כשל באחד לא ימחק את הראיות מהשני. אחר כך יגיע מיזוג ב־6.5 עם טיפול מפורש בסתירות ושמירת מקור כל תצפית.
- **תוצאה מרכזית:** הריצה המלאה כוללת 17 נציגים ו־246 תצפיות שדה: 188 חילוצים, 26 מופעים מורכבים עמומים, 14 מקרים ללא ערך בגלל מועמד חלש ו־18 מקרים מיוחדים לא־טקסטואליים. אלה אינם 246 חילוצים מוצלחים, והדיוק המאומת נמדד רק בתת־הקבוצה האנושית.
- **משפט סיום:** ״עד כאן יש לנו ענף OCR עובד ושקוף עד שלב 6.3. הצעד הבא הוא ענף VLM עצמאי לקליפ המלא; לאחריו נצליב ראיות בלי למחוק בשקט מחלוקות בין המקורות.״
"""
    (OUT / "presentation_notes.md").write_text(notes, encoding="utf-8")


if __name__ == "__main__":
    main()
