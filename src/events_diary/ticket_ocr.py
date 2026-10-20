"""Phase 6.1b raw ticket OCR execution, without field interpretation.

Phase 6.1a selected RapidOCR/ONNX CPU, full-ticket Strategy A and color cubic
upscaling by exactly two. The crop is ticket_zone above ticket_bottom_actions;
the existing layout mapper supports reference and proportional frame sizes.
No grayscale, enhancement, text correction or alternate-engine fallback occurs.

Native quadrilateral vertices are mapped from processed-crop pixels to original
source-frame pixels: source = processed / 2 + mapped_crop_origin. Vertex and
detection order are retained, but neither is declared semantic reading order.
Ticket-local coordinates are derivable by subtracting crop origin and are not
stored redundantly. Floating coordinates are not rounded or clipped. This leaves
geometry available for future spatial grouping without pairing labels/values,
assigning columns/sections, parsing fields or merging views in this module.

Confidence is the OCR engine's score, not frame validity, semantic correctness or
fusion confidence. Empty detections are a valid execution result and imply no
field-level status. Text is preserved exactly, including whitespace/punctuation.
Models hold no image arrays. Video identity belongs to the surrounding operation.

OCR dependencies are imported only when constructing RapidOCREngine. Install the
existing optional group with ``uv sync --group ocr-calibration``. No global engine
or decoder exists; callers can reuse an engine and an active VideoFrameResolver.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.metadata import version
from numbers import Integral, Real
from pathlib import Path

import cv2
import numpy as np

from events_diary.config import ROI, UIConfig
from events_diary.evidence_candidate_timeline import EvidenceCandidate
from events_diary.frame_materialization import (
    MaterializedEvidenceCandidate,
    VideoFrameResolver,
)
from events_diary.layout import map_roi_to_frame
from events_diary.representative_selection import RepresentativeSelection

Polygon = tuple[tuple[float, float], ...]
PREPROCESSING_NAME = "full_ticket_color_cubic_2x"
_OCR_ROLES = frozenset(("stable_ticket", "post_scroll_stable"))


def _real(name, value):
    """Validate finite nonboolean real metadata without correcting its value."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be real, not bool.")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")


def _index(name, value):
    """Require a nonnegative nonboolean integer index."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, not bool.")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")


@dataclass(frozen=True)
class OCRDetection:
    """Exact raw text, engine confidence and a source-frame quadrilateral.

    source_polygon contains four (x,y) pixel vertices in native engine order.
    Coordinates describe the original physical image, not the upscaled crop.
    Nested tuples keep geometry immutable; no axis-aligned approximation is made.
    confidence is engine-reported (0..1), or None if the engine supplies no score.
    """

    text: str
    confidence: float | None
    source_polygon: Polygon

    def __post_init__(self):
        """Freeze polygon storage and reject malformed engine output without repair."""
        if not isinstance(self.text, str):
            raise TypeError("OCR text must be a string.")
        if self.confidence is not None:
            _real("OCR confidence", self.confidence)
            if not 0 <= self.confidence <= 1:
                raise ValueError("OCR confidence must be within [0,1].")
        points = tuple(tuple(point) for point in self.source_polygon)
        if len(points) != 4 or any(len(point) != 2 for point in points):
            raise ValueError("OCR polygon must have four two-dimensional vertices.")
        for point in points:
            for coordinate in point:
                _real("polygon coordinate", coordinate)
        object.__setattr__(self, "source_polygon", tuple((float(x), float(y)) for x, y in points))


@dataclass(frozen=True)
class RawTicketOCRResult:
    """One full-ticket OCR execution with source-space crop/remapping metadata.

    crop_roi is the actual integer crop in source-frame pixels; upscale_factor is
    exactly 2. Detections retain engine order, without semantic reading order.
    No image, field model, inferred value or interpretation status is stored.
    """

    engine_name: str
    preprocessing_name: str
    crop_roi: ROI
    upscale_factor: int
    detections: tuple[OCRDetection, ...]

    def __post_init__(self):
        """Retain immutable detection storage and the explicit calibrated transform."""
        if not isinstance(self.crop_roi, ROI):
            raise TypeError("crop_roi must be ROI.")
        if self.upscale_factor != 2 or isinstance(self.upscale_factor, bool):
            raise ValueError("The calibrated upscale factor must be 2.")
        detections = tuple(self.detections)
        if any(not isinstance(d, OCRDetection) for d in detections):
            raise TypeError("detections must contain OCRDetection records.")
        object.__setattr__(self, "detections", detections)


@dataclass(frozen=True)
class OCRTicketResult(RawTicketOCRResult):
    """Raw execution plus one selected candidate's unchanged physical provenance.

    candidate_index addresses the caller's Investigation-local candidate sequence.
    Frame index, actual CFR source time and resolution status are copied from the
    verified materialization. Inherited crop geometry is in that physical frame.
    """

    investigation_index: int
    candidate_index: int
    frame_index: int
    actual_timestamp_seconds: float
    resolution_status: str


def _load_rapidocr():
    """Import the optional runtime on demand with an actionable dependency error."""
    try:
        from rapidocr_onnxruntime import RapidOCR
        identity = f"RapidOCR {version('rapidocr-onnxruntime')}/ONNX Runtime {version('onnxruntime')} CPU"
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "RapidOCR CPU dependencies could not load. Run `uv sync --group ocr-calibration` "
            f"and check native runtime availability. Original error: {error}"
        ) from error
    return RapidOCR, identity


class RapidOCREngine:
    """One reusable CPU-only RapidOCR session, using the calibrated bundled models.

    Initializes once, with two intra-op and one inter-op threads. CUDA and DirectML
    are explicitly disabled. No global singleton, fallback, network download or
    alternate preprocessing is provided. Runtime failures propagate to the caller.
    No explicit close API is required by the installed ONNX-backed RapidOCR API.
    """

    def __init__(self):
        """Load the optional engine and its bundled models only when OCR is invoked."""
        factory, self.engine_name = _load_rapidocr()
        self._engine = factory(intra_op_num_threads=2, inter_op_num_threads=1,
            det_use_cuda=False, cls_use_cuda=False, rec_use_cuda=False,
            det_use_dml=False, cls_use_dml=False, rec_use_dml=False)

    def run(self, image: np.ndarray):
        """Return native ordered (quadrilateral, text, confidence) entries or None.

        image is an already preprocessed BGR crop. Library timing is intentionally
        not interpreted as provenance or confidence. Call run_ticket_ocr for crop,
        preprocessing and canonical source-coordinate mapping.
        """
        detections, _ = self._engine(image)
        return detections


def ticket_ocr_roi(ui_config: UIConfig, frame_width: int, frame_height: int) -> ROI:
    """Derive Strategy A from existing ticket geometry and map it to source pixels.

    At 1920x1080 this is (16,121,336,831), containing header, aircraft-image/upper
    information and scroll body, excluding bottom actions. Use existing shared
    boundary rounding for proportional layouts; unsupported/cropped layouts fail.
    """
    if not isinstance(ui_config, UIConfig):
        raise TypeError("ui_config must be UIConfig.")
    zone, actions = ui_config.ticket_zone, ui_config.ticket_bottom_actions
    if not isinstance(zone, ROI) or not isinstance(actions, ROI):
        raise TypeError("Ticket zone and bottom actions must be ROI records.")
    if (actions.x != zone.x or actions.width != zone.width
            or not zone.y < actions.y < zone.bottom or actions.bottom != zone.bottom):
        raise ValueError("Ticket bottom actions must be the full-width bottom strip of ticket_zone.")
    full = ROI(zone.x, zone.y, zone.width, actions.y-zone.y)
    roi = map_roi_to_frame(full, ui_config, frame_width, frame_height)
    if roi.right > frame_width or roi.bottom > frame_height:
        raise ValueError("Full relevant ticket crop is outside the source frame.")
    return roi


def run_ticket_ocr(image: np.ndarray, *, ui_config: UIConfig,
                   engine: RapidOCREngine | None = None) -> RawTicketOCRResult:
    """OCR one full source frame using the calibrated color2x full-ticket pipeline.

    Accept nonempty uint8 BGR, crop before cubic upscaling, retain exact text and
    engine confidence. Return an empty tuple for no detections. Map all four native
    vertices back with (x/2+crop.x,y/2+crop.y), without rounding/clipping/reordering.
    Input pixels are not modified. Reuse engine for repeated calls.
    """
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise TypeError("image must be a uint8 NumPy array.")
    if image.ndim != 3 or image.size == 0 or image.shape[2] != 3:
        raise ValueError("image must be nonempty BGR with three channels.")
    height, width = image.shape[:2]
    roi = ticket_ocr_roi(ui_config, width, height)
    processed = cv2.resize(roi.crop(image), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    engine = RapidOCREngine() if engine is None else engine
    native = engine.run(processed)
    detections = tuple(OCRDetection(text, confidence,
        tuple((x/2+roi.x, y/2+roi.y) for x, y in polygon))
        for polygon, text, confidence in (() if native is None else native))
    return RawTicketOCRResult(engine.engine_name, PREPROCESSING_NAME, roi, 2, detections)


def _selected_candidates(investigation_index, selections, candidates, materialized):
    """Validate correspondence before I/O and retain unique eligible indexes in order."""
    _index("investigation_index", investigation_index)
    for name, sequence in (("selections", selections), ("candidates", candidates), ("materialized_candidates", materialized)):
        if not isinstance(sequence, Sequence) or isinstance(sequence, (str, bytes)):
            raise TypeError(f"{name} must be a sequence.")
    if len(candidates) != len(materialized):
        raise ValueError("Candidate/materialized sequence lengths must match.")
    for candidate, item in zip(candidates, materialized, strict=True):
        if not isinstance(candidate, EvidenceCandidate) or not isinstance(item, MaterializedEvidenceCandidate):
            raise TypeError("Expected EvidenceCandidate and MaterializedEvidenceCandidate records.")
        if candidate != item.evidence_candidate or candidate.investigation_index != investigation_index:
            raise ValueError("Candidate/materialized correspondence or Investigation ownership mismatch.")
    indexes, seen, previous_time = [], set(), None
    for selection in selections:
        if not isinstance(selection, RepresentativeSelection):
            raise TypeError("selections must contain RepresentativeSelection records.")
        index = selection.candidate_index
        _index("candidate_index", index)
        if index >= len(candidates):
            raise ValueError("Selected candidate_index is outside the candidate sequence.")
        candidate, item = candidates[index], materialized[index]
        _real("candidate timestamp", candidate.timestamp_seconds)
        if previous_time is not None and candidate.timestamp_seconds < previous_time:
            raise ValueError("Selections must be in chronological request order.")
        previous_time = candidate.timestamp_seconds
        if not isinstance(selection.selection_reasons, tuple) or any(not isinstance(r, str) for r in selection.selection_reasons):
            raise TypeError("selection_reasons must be a tuple of strings.")
        if not _OCR_ROLES.intersection(selection.selection_reasons) or index in seen:
            continue
        if item.resolution_status not in ("resolved", "video_end_adjusted"):
            raise ValueError(f"Selected ticket candidate {index} is unresolved or has unknown status.")
        _index("frame_index", item.frame_index)
        _real("actual_timestamp_seconds", item.actual_timestamp_seconds)
        if item.actual_timestamp_seconds < 0 or candidate.timestamp_seconds < 0:
            raise ValueError("Selected timestamps must be nonnegative.")
        seen.add(index)
        indexes.append(index)
    return tuple(indexes)


def run_ocr_for_representatives(*, video_path: Path, investigation_index: int,
        selections: Sequence[RepresentativeSelection], candidates: Sequence[EvidenceCandidate],
        materialized_candidates: Sequence[MaterializedEvidenceCandidate], ui_config: UIConfig,
        resolver: VideoFrameResolver | None = None, engine: RapidOCREngine | None = None,
        ) -> tuple[OCRTicketResult, ...]:
    """OCR only caller-selected stable_ticket/post_scroll_stable references once.

    Validate full index correspondence before decoding. Preserve chronological
    caller selection order; reject unsorted inputs rather than silently reorder.
    Multiple roles/duplicate indexes execute once. Distinct candidate indexes that
    share a physical frame remain distinct provenance records. Other roles alone
    never trigger OCR. Empty eligible batches do not open a decoder/load OCR.

    Reuse a caller-owned active single-video resolver/cache and engine, or own one
    per nonempty call. Resolve only eligible selected requests and verify decoded
    index/time/status against materialization; errors are explicit, with no extra
    frames/fallback OCR. Inputs and resolver-owned pixels are never modified.
    """
    indexes = _selected_candidates(investigation_index, selections, candidates, materialized_candidates)
    path = Path(video_path).resolve()
    if not isinstance(ui_config, UIConfig):
        raise TypeError("ui_config must be UIConfig.")
    if resolver is not None:
        if not isinstance(resolver, VideoFrameResolver):
            raise TypeError("resolver must be VideoFrameResolver.")
        if resolver.video_path != path:
            raise ValueError("Supplied resolver video must match video_path.")
    if not indexes:
        return ()
    engine = RapidOCREngine() if engine is None else engine
    if resolver is None:
        with VideoFrameResolver(path) as owned:
            return _run_selected(indexes, candidates, materialized_candidates, ui_config, owned, engine)
    return _run_selected(indexes, candidates, materialized_candidates, ui_config, resolver, engine)


def _run_selected(indexes, candidates, materialized, ui_config, resolver, engine):
    """Verify each loaded physical frame and attach unchanged selected provenance."""
    results = []
    for index in indexes:
        candidate, item = candidates[index], materialized[index]
        frame = resolver.resolve(candidate.timestamp_seconds)
        if (frame.image is None or frame.frame_index != item.frame_index
                or frame.actual_timestamp_seconds != item.actual_timestamp_seconds
                or frame.resolution_status != item.resolution_status):
            raise RuntimeError(f"Decoded frame disagrees with materialized ticket candidate {index}.")
        raw = run_ticket_ocr(frame.image, ui_config=ui_config, engine=engine)
        results.append(OCRTicketResult(raw.engine_name, raw.preprocessing_name, raw.crop_roi,
            raw.upscale_factor, raw.detections, candidate.investigation_index, index,
            item.frame_index, item.actual_timestamp_seconds, item.resolution_status))
    return tuple(results)
