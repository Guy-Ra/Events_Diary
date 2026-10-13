"""Phase 3.1d: exploration and production validation of SINGLE-frame presence.

This development utility studies visual structure that could support recordings
beginning with an already-active Investigation. Production measurement/classifier
APIs now supply per-frame evidence and presence; this utility adds no candidate,
lifecycle, aircraft identity, or OCR logic. Each
measurement uses only one frame: no pair MAD/CPR, route-change, scroll decisions,
or temporal inference is used as presence evidence.

Reuse UIConfig's unchanged canonical ticket ROIs and the existing sequential
decoder/layout validation. Explore grayscale distributions, fixed Canny edges,
histogram entropy, and local block variance with OpenCV/NumPy only. Dark/light
cutoffs and Canny settings are diagnostic probes, not classifier thresholds.
Report all-frame ranges as well as a small set of visually inspected examples.
Representative indices below are manual annotations of these two recordings,
not detection rules. In the reference recording, frame zero genuinely has no
ticket: map content and the collapsed sidebar occupy the configured ticket zone.

Nominal timestamps are index/source FPS; constant FPS and canonical layout are
assumed. Contact sheets retain all 47 startup-active frames and the reference's
first 18 frames for auditing the annotations. Eight full-frame/crop examples and
one edge comparison illustrate startup, stable tickets, dimmed switch, opening,
and no-ticket appearance. Matching local outputs overwrite on rerun; unrelated
artifacts are preserved. Decode/layout/I/O errors propagate. Conclusions concern
these recordings only; provisional production defaults are not tuned here.
"""

import csv
from pathlib import Path

import cv2
import numpy as np
from validate_phase2 import _sequential_frames

from events_diary.config import UIConfig
from events_diary.investigation_signals import (
    estimate_investigation_presence,
    is_investigation_present,
)
from events_diary.video import read_video_metadata

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/debug/map_motion/investigation_presence"
SOURCES = {
    "started_active": ROOT / "data/videos/investigation_started_active_test.mp4",
    "reference": ROOT / "data/videos/investigation_switch_test.mp4",
}
ROIS = ("ticket_zone", "ticket_header", "aircraft_image", "ticket_scroll_body",
        "ticket_bottom_actions")
# Annotations from visual inspection of the full consecutive-frame contact sheets.
REPRESENTATIVES = {
    "started_active": {0: "start_active", 10: "stable_pre_switch", 23: "switch_loading",
                       24: "new_ticket", 46: "after_switch"},
    "reference": {0: "no_ticket", 15: "opening", 17: "reference_stable"},
}


def _measure(crop):
    """Return single-crop distribution, texture, and edge probes plus edge mask.

    Grayscale intensity is 0..255. Pixel fractions use crop area. Canny is fixed
    at 50/150, dark is <64, light is >=192, very light is >=224. Local variance
    averages population variances of nonoverlapping 16px blocks, including edges.
    Histogram entropy is in bits. No semantic interpretation occurs here.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    probabilities = np.bincount(gray.ravel(), minlength=256) / gray.size
    nonzero = probabilities[probabilities > 0]
    blocks = [float(gray[y:y + 16, x:x + 16].var())
              for y in range(0, gray.shape[0], 16) for x in range(0, gray.shape[1], 16)]
    p10, p50, p90 = np.percentile(gray, [10, 50, 90])
    return {
        "mean_gray": float(gray.mean()), "std_gray": float(gray.std()),
        "p10_gray": float(p10), "median_gray": float(p50), "p90_gray": float(p90),
        "edge_fraction": float(np.count_nonzero(edges) / gray.size),
        "dark_fraction": float((gray < 64).mean()),
        "light_fraction": float((gray >= 192).mean()),
        "very_light_fraction": float((gray >= 224).mean()),
        "entropy_bits": float(-(nonzero * np.log2(nonzero)).sum()),
        "mean_local_variance": float(np.mean(blocks)),
    }, edges


def _write_image(name, image):
    """Save a diagnostic PNG and raise explicitly on an unsuccessful write."""
    path = OUTPUT / name
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not save image: {path}")


def _contact_sheet(tiles):
    """Arrange annotated ticket thumbnails chronologically in ten columns."""
    sheet = np.full((((len(tiles) + 9) // 10) * 474, 1680, 3), 235, np.uint8)
    for n, tile in enumerate(tiles):
        sheet[(n // 10) * 474:(n // 10 + 1) * 474,
              (n % 10) * 168:(n % 10 + 1) * 168] = tile
    return sheet


def _evaluate(source, path, config):
    """Decode a source, measure each ROI independently, and save annotated examples.

    Representative selection uses manual indices, while all-frame measurements
    permit checking stable ranges. This never computes frame-to-frame differences.
    The reliable no-ticket reference must exist as a real decoded frame.
    """
    metadata = read_video_metadata(path)
    if max(REPRESENTATIVES[source]) >= metadata.frame_count:
        raise ValueError(f"Annotated frames unavailable in {path}; inspect recording again.")
    rows, tiles, edge_tiles = [], [], []
    for index, frame in _sequential_frames(path, metadata, config):
        label = REPRESENTATIVES[source].get(index)
        presence = estimate_investigation_presence(frame, config)
        present = is_investigation_present(presence)
        for roi_name in ROIS:
            values, edges = _measure(getattr(config, roi_name).crop(frame))
            rows.append(dict(source=source, frame=index, time=index / metadata.fps,
                             representative=label or "", roi=roi_name, **values,
                             bottom_actions_dark_fraction=presence.bottom_actions_dark_fraction,
                             ticket_body_edge_fraction=presence.ticket_body_edge_fraction,
                             ticket_body_light_fraction=presence.ticket_body_light_fraction,
                             investigation_present=present))
            if label and roi_name == "ticket_zone":
                _write_image(f"representative_{label}.png", frame)
                _write_image(f"ticket_zone_{label}.png", config.ticket_zone.crop(frame))
                tile = np.full((474, 168), 235, np.uint8)
                tile[30:] = cv2.resize(edges, (168, 444), interpolation=cv2.INTER_NEAREST)
                cv2.putText(tile, label, (2, 18), cv2.FONT_HERSHEY_SIMPLEX, .35, 0, 1)
                edge_tiles.append(tile)
        if source == "started_active" or index <= 17:
            tile = np.full((474, 168, 3), 235, np.uint8)
            tile[30:] = cv2.resize(config.ticket_zone.crop(frame), (168, 444))
            cv2.putText(tile, f"{index} {index / metadata.fps:.1f}s", (4, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 1)
            tiles.append(tile)
    _write_image(path.stem + "_contact_sheet.png", _contact_sheet(tiles))
    return metadata, rows, edge_tiles


def _report(metadata, rows):
    """Describe observed structure, representative measurements, and limitations.

    Stable/no-ticket ranges use visually audited frame windows only for reporting;
    they never classify frames. Numeric evidence is retained for all five ROIs.
    """
    lines = ["Phase 3.1d: single-frame Investigation presence exploration",
             ("Production estimate_investigation_presence/is_investigation_present are called unchanged. "
              "No pair-change or lifecycle logic is used.")]
    for source, meta in metadata.items():
        lines.append(f"{source}: {meta.path}; {meta.width}x{meta.height}; fps={meta.fps}; "
                     f"frames={meta.frame_count}; duration={meta.duration_seconds}s")
    lines += ["\nVisual annotations:",
              ("started_active: frame 0 (0.0s) ticket already open; frame 10 (1.0s) stable pre-switch; "
              "frame 23 (2.3s) dimmed old ticket during switch; frame 24 (2.4s) new ticket; "
              "frame 46 (4.6s) stable post-switch."),
              ("reference: frame 0 (0.0s) reliable no-ticket/map-and-collapsed-sidebar reference; "
              "frame 15 (1.5s) opening animation with map still exposed; frame 17 (1.7s) open ticket."),
              "No-ticket is not an empty black rectangle: genuine map and collapsed UI occupy the ROIs."]
    for row in rows:
        if row["representative"]:
            lines.append(f"\n{row['source']} frame={row['frame']} time={row['time']:.1f}s "
                         f"{row['representative']} {row['roi']}: " + ", ".join(
                             f"{key}={value:.8g}" for key, value in row.items()
                             if key not in ("source", "frame", "time", "representative", "roi")))
    groups = {
        "no ticket reference (0..12)": [r for r in rows if r["source"] == "reference" and r["frame"] <= 12],
        "stable pre-switch (0..21)": [r for r in rows if r["source"] == "started_active" and r["frame"] <= 21],
        "stable post-switch (25..46)": [r for r in rows if r["source"] == "started_active" and r["frame"] >= 25],
    }
    lines.append("\nAll-frame baseline ranges (manual scenario windows, not detections):")
    for group, subset in groups.items():
        lines.append(group)
        for roi in ROIS:
            selected = [r for r in subset if r["roi"] == roi]
            lines.append(f"  {roi}, n={len(selected)}: " + "; ".join(
                f"{field}={min(r[field] for r in selected):.6g}..{max(r[field] for r in selected):.6g}"
                for field in ("mean_gray", "std_gray", "edge_fraction", "dark_fraction",
                              "light_fraction", "mean_local_variance")))
    frame_rows = [r for r in rows if r["roi"] == "ticket_zone"]
    lines.append("\nProduction presence validation (dark>=0.50 AND edges>=0.05; light does not vote):")
    for source in metadata:
        subset = [r for r in frame_rows if r["source"] == source]
        positive = [r for r in subset if r["investigation_present"]]
        negative = [r for r in subset if not r["investigation_present"]]
        lines.append(f"{source}: True={len(positive)}/{len(subset)}; "
                     f"False frames={[r['frame'] for r in negative]}")
        if positive:
            first = positive[0]
            lines.append(f"First True: frame {first['frame']} / {first['time']:.1f}s; "
                         f"dark={first['bottom_actions_dark_fraction']:.8g}, "
                         f"edges={first['ticket_body_edge_fraction']:.8g}, "
                         f"light={first['ticket_body_light_fraction']:.8g}")
        for row in subset:
            if row["representative"]:
                lines.append(f"{row['representative']}: frame {row['frame']} / {row['time']:.1f}s "
                             f"presence={row['investigation_present']}")
    lines += ["\nInterpretation (limited dataset):",
              ("Observed bottom-actions dark occupancy is about 0.9075 for both stable tickets, "
               "0.8581 during switch dimming, and 0.0062 without a ticket. This is the strongest "
               "single structural probe here and survives the sampled loading overlay."),
              ("Stable body light occupancy is about 0.7804 before and 0.7820 after the switch, "
               "versus 0.0711 without a ticket. Dimming reduces it to 0.1841; mean body gray "
               "falls to 171.6, below the no-ticket value 178.3. Mean brightness alone is weak. "
               "Body edges remain about 0.095-0.098 with tickets versus 0.0204 without one."),
              ("Inspect light occupancy in ticket_scroll_body together with dark occupancy of "
              "ticket_bottom_actions. They test two complementary panel structures: a bright "
              "content body and a dark anchored action toolbar. Header alone is weaker because "
              "collapsed sidebar controls also occupy the top of the no-ticket zone."),
              ("Brightness/texture/edges in the aircraft image depend on the photograph. Whole-zone "
              "statistics mix several structures and can hide local evidence. Edge density is useful "
              "support but map labels/coastlines can also produce edges."),
              ("A single toolbar-dark or body-light metric may separate these stable examples. A small combination "
              "of body occupancy and bottom-toolbar structure is more promising for varied maps. "
              "Production currently uses toolbar darkness AND body edges, not required body brightness."),
              ("Switch dimming retains ticket structure but changes brightness. Initial opening at "
              "reference frame 15 still exposes map in the body: incomplete panels are ambiguous. "
              "The evidence does not define when an opening becomes active presence."),
              ("Deterministic single-frame presence appears feasible for stable canonical tickets, "
              "including frame zero with no visible opening transition. Generalization is unproven."),
              ("Limitations: two recordings, one no-ticket background (mostly sea), two startup-active "
              "ticket photographs, manual annotations, fixed canonical layout/theme, no held-out "
              "classifier validation. Land maps, dark themes, partial occlusion, missing photos, "
              "loading/skeleton panels, and other sidebars need additional examples. Probe cutoffs "
              "and Canny parameters retain their exploration values. No OCR, identity, or "
              "lifecycle logic was implemented. Presence uses provisional single-frame defaults only.")]
    return "\n".join(lines) + "\n"


def main():
    """Measure both recordings and write purposeful PNG, CSV, and report artifacts."""
    config = UIConfig()
    for path in SOURCES.values():
        if not path.is_file():
            raise FileNotFoundError(f"Required source/reference video missing: {path}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metadata, rows, edges = {}, [], []
    for source, path in SOURCES.items():
        print(f"Input: {path}", flush=True)
        meta, measured, tiles = _evaluate(source, path, config)
        metadata[source] = meta
        rows.extend(measured)
        edges.extend(tiles)
    if not any(r["representative"] == "no_ticket" for r in rows):
        raise ValueError("No reliable annotated no-ticket reference available; stop exploration.")
    _write_image("representative_ticket_edges.png", np.hstack(edges))
    csv_path = OUTPUT / "investigation_presence_measurements.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = _report(metadata, rows)
    report_path = OUTPUT / "investigation_presence_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Measurements: {csv_path}\nReport: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
