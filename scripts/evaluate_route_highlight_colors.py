"""Phase 3.1a exploratory route/selection color characterization only.

Inspect source frames 46, 48, and 49 of investigation_switch_test.mp4: before,
transition, and after the known switch at the video's actual FPS. Reuse project
metadata, canonical ROI/layout validation, and sequential decoding. No production
code, signal logic, thresholds, or APIs change. This does not extract routes,
skeletonize, infer direction/identity, or implement an Investigation detector.

Small map-local point samples are manually annotated from visually inspected old
and new trails and red aircraft graphics. They are evaluation sampling coordinates,
not new UIConfig ROIs or automatically discovered route geometry. For each point,
take the highest-saturation pixel in its 5x5 neighborhood to sample the colored
stroke rather than the antialiased background. This intentionally biases samples
toward stroke cores and may miss pale edges. Report observed BGR/HSV distributions.
OpenCV HSV units are H=0..179, S/V=0..255, unlike hue degrees.

Derive illustrative HSV boxes from observed sample ranges, with five hue units
and twenty S/V units of padding (display margins, not production calibration).
Hue uses the shortest circular arc covering samples, splitting a wraparound range
at red's 0/179 boundary. Derivation is descriptive, not threshold optimization.
Separate route-gradient and red-highlight candidates are applied to the entire
unmodified map crop; no aircraft/geography/UI masking is used to hide false positives.

Map motion was previously measured near identity in this recording; these are
individual-frame color distributions, so no alignment is necessary or applied.
Sequential reads preserve every source position without seeking or downsampling.
Save a small set of raw crops, annotated samples, candidate masks, masked colors,
and route_highlight_color_report.txt under data/debug/map_motion/investigation_switch/.
No timeline CSV is created. Candidate PNGs encode route white and highlight red;
the report lists separate/combined pixel fractions. Visual suitability and false
positives must be reviewed; no semantic conclusions are inferred from color alone.
"""

from pathlib import Path

import cv2
import numpy as np
from evaluate_map_motion_video import _validate_layout
from validate_phase2 import _sequential_frames

from events_diary.config import UIConfig
from events_diary.video import read_video_metadata

ROOT = Path(__file__).resolve().parents[1]
VIDEO_PATH = ROOT / "data/videos/investigation_switch_test.mp4"
OUTPUT_DIR = ROOT / "data/debug/map_motion/investigation_switch"
FRAME_LABELS = {46: "before", 48: "switch", 49: "after"}
# Hand-inspected stroke samples in map-local coordinates, not a route extractor.
ROUTE_POINTS = {
    "before": [(697, 99), (722, 109), (746, 120), (770, 137),
               (791, 154), (815, 173), (840, 191), (858, 205)],
    "after": [(661, 249), (665, 295), (670, 351), (676, 408),
              (681, 466), (687, 519), (692, 558), (701, 589)],
}
HIGHLIGHT_POINTS = {"before": [(678, 90), (660, 212)], "after": [(660, 212)]}


def _sample_colors(image, points):
    """Return sampled BGR/HSV stroke cores at hand-annotated map-local points.

    image is a BGR canonical map crop; points are (x,y) visual annotations.
    Returns BGR samples, HSV samples, and chosen pixel coordinates. Each 5x5
    neighborhood picks maximum saturation; input pixels remain untouched.
    Assumes annotations hit colored strokes; no semantic verification is automatic.
    Raises ValueError for neighborhoods outside the image; OpenCV errors propagate.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    bgr_samples, hsv_samples, coordinates = [], [], []
    for x, y in points:
        if x < 2 or y < 2 or x + 2 >= image.shape[1] or y + 2 >= image.shape[0]:
            raise ValueError(f"Sample neighborhood outside map: {(x, y)}")
        patch = hsv[y-2:y+3, x-2:x+3]
        py, px = np.unravel_index(np.argmax(patch[:, :, 1]), patch.shape[:2])
        chosen = (x - 2 + int(px), y - 2 + int(py))
        coordinates.append(chosen)
        bgr_samples.append(image[chosen[1], chosen[0]])
        hsv_samples.append(hsv[chosen[1], chosen[0]])
    return np.array(bgr_samples), np.array(hsv_samples), coordinates


def _candidate_ranges(samples):
    """Derive padded descriptive HSV boxes from observed colored-core samples.

    samples is a nonempty Nx3 OpenCV HSV array. Returns inclusive lower/upper
    triples, splitting a circular hue arc at zero when needed. Assumes 180 hue
    units; padding is fixed at H=5/SV=20 and clipped to representable limits.
    No performance objective, optimization, or production threshold is used.
    Invalid/empty shapes raise ValueError; NumPy errors propagate.
    """
    if samples.ndim != 2 or samples.shape[1] != 3 or not len(samples):
        raise ValueError("Expected nonempty Nx3 HSV samples.")
    hues = np.unique(samples[:, 0]).astype(int)
    gaps = np.diff(np.r_[hues, hues[0] + 180])
    gap_index = int(np.argmax(gaps))
    start = int(hues[(gap_index + 1) % len(hues)])
    end = int(hues[gap_index])
    if end < start:
        end += 180
    lower, upper = start - 5, end + 5
    s_min, v_min = np.maximum(samples[:, 1:].min(axis=0).astype(int) - 20, 0)
    s_max, v_max = np.minimum(samples[:, 1:].max(axis=0).astype(int) + 20, 255)
    ranges = []
    for h in range(180):
        if any(lower <= h + offset <= upper for offset in (-180, 0, 180)):
            if not ranges or h != ranges[-1][1] + 1:
                ranges.append([h, h])
            else:
                ranges[-1][1] = h
    return [((int(a), int(s_min), int(v_min)), (int(b), int(s_max), int(v_max)))
            for a, b in ranges]


def _mask(hsv, ranges):
    """Return the union of inclusive exploratory HSV boxes as a uint8 mask.

    hsv is a map-local OpenCV HSV image and ranges contains lower/upper triples.
    Returns same-height/width 0/255 mask. Assumes ranges use OpenCV units; no
    morphology, spatial filtering, or content extraction occurs. cv2 errors propagate.
    """
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lower, upper in ranges:
        mask |= cv2.inRange(hsv, np.uint8(lower), np.uint8(upper))
    return mask


def _distribution(name, bgr, hsv):
    """Return readable observed min/median/max color statistics for samples.

    name labels a manual sample group; bgr/hsv are nonempty Nx3 arrays. Returns
    text lines with OpenCV channel units and hue values. These are selected stroke
    cores, not whole-map histograms or a representative population estimate.
    Empty/malformed samples fail through NumPy validation; no color model is fitted.
    """
    lines = [f"{name}: {len(bgr)} manually sampled colored-core pixels"]
    for label, values in (("BGR", bgr), ("HSV", hsv)):
        lines.append(f"  {label} min={values.min(axis=0).tolist()}, "
                     f"median={np.median(values,axis=0).tolist()}, max={values.max(axis=0).tolist()}")
    lines.append(f"  Observed hue values: {sorted({int(h) for h in hsv[:,0]})}")
    return lines


def _save(path, image):
    """Save one diagnostic PNG; path is destination and image is its pixel array.

    Returns None. Parent directory must exist; existing matching files overwrite.
    Raises OSError on a failed write; OpenCV exceptions propagate. No input changes.
    """
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write {path}")


def main():
    """Inspect three source frames and save color statistics and candidate masks.

    No parameters; repository input/output and explicit manual sample points are
    used. Returns zero on success. Metadata/decode/layout/sampling/write failures
    propagate clearly; no production behavior is changed. Source timestamps use
    index/FPS. Original frames remain unwarped for color-distribution inspection.
    """
    config = UIConfig()
    metadata = read_video_metadata(VIDEO_PATH)
    _validate_layout(metadata.width, metadata.height, config)
    crops = {}
    for index, frame in _sequential_frames(VIDEO_PATH, metadata, config):
        if index in FRAME_LABELS:
            crops[FRAME_LABELS[index]] = config.map_analysis_region.crop(frame).copy()
    if len(crops) != len(FRAME_LABELS):
        raise ValueError("Recording lacks one or more requested representative frames.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["Phase 3.1a exploratory route/highlight colors -- manual review only",
             f"Source: {VIDEO_PATH}; {metadata.width}x{metadata.height}; FPS={metadata.fps:g}",
             "HSV uses H=0..179, S/V=0..255. No motion warp for individual color samples."]
    samples = {"route": [], "highlight": []}
    for label in ("before", "after"):
        annotated = crops[label].copy()
        for group, points in (("route", ROUTE_POINTS[label]), ("highlight", HIGHLIGHT_POINTS[label])):
            bgr, hsv, chosen = _sample_colors(crops[label], points)
            samples[group].append((bgr, hsv))
            lines.extend(_distribution(f"{label} {group}", bgr, hsv))
            lines.append(f"  Annotation points={points}; sampled pixels={chosen}")
            for point in chosen:
                cv2.circle(annotated, point, 5, (255, 255, 255), 1)
        _save(OUTPUT_DIR / f"route_color_samples_{label}.png", annotated)
    candidates = {}
    for group, groups in samples.items():
        hsv = np.concatenate([item[1] for item in groups])
        candidates[group] = _candidate_ranges(hsv)
        lines.append(f"Exploratory {group} HSV boxes: {candidates[group]}")
    for index, label in FRAME_LABELS.items():
        image = crops[label]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        route, highlight = (_mask(hsv, candidates[group]) for group in ("route", "highlight"))
        combined = route | highlight
        fractions = [np.count_nonzero(mask) / mask.size for mask in (route, highlight, combined)]
        lines.append(f"{label}: frame={index}, time={index/metadata.fps:.3f}s; "
                     f"route={fractions[0]:.6f} ({100*fractions[0]:.4f}%); "
                     f"highlight={fractions[1]:.6f} ({100*fractions[1]:.4f}%); "
                     f"union={fractions[2]:.6f} ({100*fractions[2]:.4f}%)")
        display = cv2.cvtColor(route, cv2.COLOR_GRAY2BGR)
        display[highlight != 0] = (0, 0, 255)
        _save(OUTPUT_DIR / f"route_color_{label}.png", image)
        _save(OUTPUT_DIR / f"hsv_candidate_mask_{label}.png", display)
        if label != "switch":
            _save(OUTPUT_DIR / f"masked_route_{label}.png", cv2.bitwise_and(image, image, mask=combined))
    lines.extend([
        "The trail is a multishade blue/cyan/green gradient; red highlight is a separate color family.",
        "HSV separates hue from brightness more readably than a broad BGR channel box; colored-core",
        "sampling excludes pale antialiased edges and candidate masks are not calibrated.",
        "Manual review of this recording: the old slanted trail is clearly visible before/switch;",
        "the new near-vertical trail is clearly visible after. Most water/terrain/roads and yellow",
        "ordinary aircraft are suppressed. Blue map/POI icons, faint colored boundaries, and a",
        "southern blue map label remain false positives; pale/green route edges are not guaranteed.",
        "Red-highlight output is sparse/fragmentary, and the hovered aircraft is already red before",
        "the ticket changes. Red alone cannot distinguish hover from committed selection.",
        "One broad hue arc accommodates the observed trail gradient; a separate wraparound red",
        "family is needed for highlight cores. HSV masking looks promising for a small future",
        "selection-change evidence signal, but false positives and generalization need more validation.",
        "Color alone does not establish route ownership, connectivity, or selection identity.",
        "No production detector or route extraction is implemented; one recording cannot establish robustness.",
    ])
    report = "\n".join(lines) + "\n"
    (OUTPUT_DIR / "route_highlight_color_report.txt").write_text(report, encoding="utf-8")
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
