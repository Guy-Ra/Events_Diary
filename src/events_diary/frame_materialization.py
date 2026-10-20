"""Phase 5.1b physical-frame resolution and raw ticket-body diagnostics.

For the project's constant-FPS convention, frame i has time i/FPS and nominal
duration is frame_count/FPS. Resolve arbitrary requests in [0, duration] to the
nearest index explicitly, with exact halfway ties going to the later frame.
Seek by frame index and verify the decoded position; do not delegate rounding to
millisecond seek behavior. Actual times use the established index/FPS convention,
not backend-dependent POS_MSEC. Variable-frame-rate time-nearest access is outside
this contract. Existing Phase 1 uniform sampling is unchanged.

Only an EXACT nominal-duration request gets video_end_adjusted status. It targets
the last reported frame, searching backwards only if that endpoint is undecodable
to find the last decodable frame. Other in-range requests round/clamp to physical
indexes normally; arbitrary out-of-range times are rejected, never repaired.
Failed ordinary reads remain explicit unresolved records. Inconsistent decoded
indexes or geometry raise errors instead of reporting an incorrect success.

MaterializedEvidenceCandidate carries its original semantic candidate and scalar
metadata only. Requested time and absolute error are derived properties. Video
identity belongs to the surrounding single-video operation, not every record.
ResolvedVideoFrame is a short-lived image handle supplied separately by the
resolver. Its cached BGR uint8 image has its writeable flag disabled; callers must
copy before editing. Frozen records do not make NumPy storage deeply immutable.

VideoFrameResolver owns a context-managed decoder and bounded operation-local LRU
cache keyed by (resolved video path, frame index); no global cache exists. Closing
releases the decoder and cached references. External image handles can outlive the
cache, so callers must not retain all handles if they want bounded image memory.
The helper is single-threaded. Materialization itself returns no image handles.

Quality is measured on UIConfig.ticket_scroll_body mapped with existing layout
utilities, without resizing pixels. Store raw grayscale Laplacian CV_64F ksize=1
variance, mean intensity and population standard deviation. No normalization,
quality threshold, score, ranking, filtering, deduplication or selection occurs.
High sharpness does not prove usefulness, loading completeness, readable final
content or representative quality; moving scroll can be sharper than stable views.
No ticket similarity, section grouping, OCR, VLM or Phase 5.2 logic is implemented.
"""

import math
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path

import cv2
import numpy as np

from events_diary.config import ROI, UIConfig
from events_diary.evidence_candidate_timeline import EvidenceCandidate
from events_diary.layout import map_roi_to_frame
from events_diary.video import VideoMetadata, read_video_metadata


@dataclass(frozen=True)
class ResolvedVideoFrame:
    """One request's resolution, with optional short-lived shared image access.

    Status is resolved, video_end_adjusted or unresolved. Unresolved records have
    None index/time/image. Resolved image bytes are shared read-only cache data;
    copy before editing. Actual time is decoded index/FPS under the CFR contract.
    """

    requested_timestamp_seconds: float
    resolution_status: str
    frame_index: int | None
    actual_timestamp_seconds: float | None
    image: np.ndarray | None = field(repr=False, compare=False)


@dataclass(frozen=True)
class TicketQualityFeatures:
    """Raw, unnormalized ticket-body diagnostics, never a quality decision."""

    ticket_sharpness_laplacian_variance: float
    ticket_brightness_mean: float
    ticket_contrast_std: float


@dataclass(frozen=True)
class MaterializedEvidenceCandidate:
    """Original semantic candidate plus physical-frame metadata and diagnostics.

    No image or redundant requested timestamp is stored. Source video identity is
    supplied by the materialization operation. Unresolved records have all physical
    and quality fields None. No candidate is ranked, removed or merged.
    """

    evidence_candidate: EvidenceCandidate
    resolution_status: str
    frame_index: int | None
    actual_timestamp_seconds: float | None
    ticket_sharpness_laplacian_variance: float | None
    ticket_brightness_mean: float | None
    ticket_contrast_std: float | None

    @property
    def requested_timestamp_seconds(self) -> float:
        """Expose the unchanged original request without redundant stored state."""
        return self.evidence_candidate.timestamp_seconds

    @property
    def timestamp_error_seconds(self) -> float | None:
        """Derive absolute physical-time error, or None for unresolved requests."""
        if self.actual_timestamp_seconds is None:
            return None
        return abs(self.actual_timestamp_seconds - self.requested_timestamp_seconds)


def _real(name, value):
    """Reject non-real, boolean and non-finite numeric inputs without repairing them."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be real, not bool.")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")


def _positive_integer(name, value):
    """Validate a positive nonboolean integer such as a cache bound or dimension."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, not bool.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _validate_metadata(metadata, path):
    """Require the established frame-count/FPS convention and matching source path."""
    if not isinstance(metadata, VideoMetadata):
        raise TypeError("metadata must be VideoMetadata.")
    if Path(metadata.path).resolve() != path:
        raise ValueError("metadata.path must match video_path.")
    _real("fps", metadata.fps)
    if metadata.fps <= 0:
        raise ValueError("fps must be positive.")
    for name in ("frame_count", "width", "height"):
        _positive_integer(name, getattr(metadata, name))
    _real("duration_seconds", metadata.duration_seconds)
    if metadata.duration_seconds != metadata.frame_count / metadata.fps:
        raise ValueError("duration_seconds must equal frame_count / fps.")


def _target(metadata, timestamp):
    """Choose the nearest CFR index, comparing against the midpoint to break ties later."""
    _real("requested_timestamp_seconds", timestamp)
    if not 0 <= timestamp <= metadata.duration_seconds:
        raise ValueError("Requested timestamp is outside [0, nominal duration].")
    if timestamp == metadata.duration_seconds:
        return metadata.frame_count - 1, "video_end_adjusted"
    lower = math.floor(timestamp * metadata.fps)
    # Comparing with the representable midpoint avoids bankers rounding and a
    # rounded product just below a decimal half (e.g. 1.005 * 100).
    later = timestamp >= (lower + .5) / metadata.fps
    return min(lower + int(later), metadata.frame_count - 1), "resolved"


class VideoFrameResolver:
    """Context-managed single-video nearest-frame loader with a bounded shared LRU.

    max_frames defaults to 16 (about 95 MiB for canonical BGR images). Supplied
    metadata must match the path and existing CFR convention. Missing files raise
    FileNotFoundError; invalid inputs raise TypeError/ValueError. Opening failures
    raise ValueError; failed individual reads return unresolved. Position/layout
    disagreement raises RuntimeError. Every exit releases decoder and cache.

    resolve() accepts arbitrary request order and preserves duplicate requests.
    cache_hits/cache_misses count index lookup outcomes; decode_attempts counts
    uncached read attempts, including failures. cache_size exposes current retained
    frame count. A closed helper cannot be reused. This is not a global image store.
    """

    def __init__(self, video_path: Path, *, metadata: VideoMetadata | None = None, max_frames: int = 16):
        """Validate source metadata and cache bound without opening the owned decoder."""
        _positive_integer("max_frames", max_frames)
        self.video_path = Path(video_path).resolve()
        self.metadata = read_video_metadata(self.video_path) if metadata is None else metadata
        _validate_metadata(self.metadata, self.video_path)
        if not self.video_path.is_file():
            raise FileNotFoundError(f"Video file does not exist: {self.video_path}")
        self.max_frames = max_frames
        self.cache_hits = 0
        self.cache_misses = 0
        self.decode_attempts = 0
        self._cache = OrderedDict()
        self._capture = None
        self._closed = False

    def __enter__(self):
        """Open one decoder, releasing it if opening or property inspection fails."""
        if self._closed or self._capture is not None:
            raise RuntimeError("Resolver must be entered once and must not be closed.")
        capture = cv2.VideoCapture(str(self.video_path))
        try:
            if not capture.isOpened():
                raise ValueError(f"Unable to open video: {self.video_path}")
            for prop, expected in ((cv2.CAP_PROP_FPS, self.metadata.fps),
                                   (cv2.CAP_PROP_FRAME_COUNT, self.metadata.frame_count),
                                   (cv2.CAP_PROP_FRAME_WIDTH, self.metadata.width),
                                   (cv2.CAP_PROP_FRAME_HEIGHT, self.metadata.height)):
                if capture.get(prop) != expected:
                    raise ValueError("Decoder metadata disagrees with supplied video metadata.")
            self._capture = capture
        finally:
            if self._capture is None:
                capture.release()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Release all owned resources on success or any exception."""
        self.close()

    def close(self):
        """Idempotently release decoder and cached references; external handles survive."""
        try:
            if self._capture is not None:
                self._capture.release()
        finally:
            self._capture = None
            self._cache.clear()
            self._closed = True

    @property
    def cache_size(self) -> int:
        """Return the number of retained physical frames, never above max_frames."""
        return len(self._cache)

    def _load(self, index):
        """Reuse or decode one intended frame index, verifying position and image format."""
        key = (self.video_path, index)
        if key in self._cache:
            self.cache_hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self.cache_misses += 1
        self.decode_attempts += 1
        try:
            if not self._capture.set(cv2.CAP_PROP_POS_FRAMES, index):
                return None
            success, image = self._capture.read()
        except cv2.error:
            return None
        if not success or image is None:
            return None
        reported = self._capture.get(cv2.CAP_PROP_POS_FRAMES) - 1
        if not math.isfinite(reported) or reported != index:
            raise RuntimeError(f"Decoded frame index {reported} disagrees with requested index {index}.")
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                or image.shape != (self.metadata.height, self.metadata.width, 3)):
            raise RuntimeError("Decoded image format/dimensions disagree with video metadata.")
        image.setflags(write=False)
        self._cache[key] = image
        if len(self._cache) > self.max_frames:
            self._cache.popitem(last=False)
        return image

    def resolve(self, requested_timestamp_seconds: float) -> ResolvedVideoFrame:
        """Resolve one request, retaining its time and marking any endpoint adjustment.

        Ordinary decode failure returns unresolved with None physical/image fields.
        Nominal video end searches backwards for the final decodable physical frame;
        if none decode, it too is unresolved. No other request falls back to a
        different index. An in-range request near duration can normally round to
        the final index without being an exact video-end adjustment.
        """
        if self._capture is None:
            raise RuntimeError("Resolver must be used inside its active context.")
        index, status = _target(self.metadata, requested_timestamp_seconds)
        indexes = range(index, -1, -1) if status == "video_end_adjusted" else (index,)
        for physical_index in indexes:
            image = self._load(physical_index)
            if image is not None:
                return ResolvedVideoFrame(requested_timestamp_seconds, status, physical_index,
                                          physical_index / self.metadata.fps, image)
        return ResolvedVideoFrame(requested_timestamp_seconds, "unresolved", None, None, None)


def resolve_video_frame(video_path: Path, requested_timestamp_seconds: float, *,
                        metadata: VideoMetadata | None = None) -> ResolvedVideoFrame:
    """Resolve one arbitrary request with an owned decoder released before return.

    For repeated access use VideoFrameResolver to share its bounded cache. The
    returned optional image handle outlives this short-lived resolver and must be
    copied before editing. Validation/errors follow VideoFrameResolver.resolve.
    """
    with VideoFrameResolver(video_path, metadata=metadata) as resolver:
        return resolver.resolve(requested_timestamp_seconds)


def measure_ticket_quality(image: np.ndarray, ui_config: UIConfig) -> TicketQualityFeatures:
    """Measure raw float Laplacian variance, grayscale mean/std on mapped ticket body.

    Require nonempty uint8 BGR pixels and a UIConfig with a valid ticket_scroll_body
    ROI. Existing proportional mapping supports reference and same-aspect scaled
    frames; cropped/unsupported layouts and out-of-bounds/empty crops raise
    ValueError. Invalid types/formats raise TypeError/ValueError. No pixel resizing,
    normalization, threshold, semantic completeness inference or scoring occurs.
    """
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise TypeError("image must be a uint8 NumPy array.")
    if image.size == 0 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be nonempty BGR with exactly three channels.")
    if not isinstance(ui_config, UIConfig) or not isinstance(ui_config.ticket_scroll_body, ROI):
        raise TypeError("ui_config must provide a ticket_scroll_body ROI.")
    height, width = image.shape[:2]
    roi = map_roi_to_frame(ui_config.ticket_scroll_body, ui_config, width, height)
    crop = roi.crop(image)
    if crop.size == 0:
        raise ValueError("Mapped ticket body must not be empty.")
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return TicketQualityFeatures(float(cv2.Laplacian(gray, cv2.CV_64F, ksize=1).var()),
                                 float(gray.mean()), float(gray.std()))


def materialize_evidence_candidates(*, video_path: Path, candidates: Sequence[EvidenceCandidate],
                                    ui_config: UIConfig, metadata: VideoMetadata | None = None,
                                    max_frames: int = 16, resolver: VideoFrameResolver | None = None,
                                    ) -> tuple[MaterializedEvidenceCandidate, ...]:
    """Materialize all candidates in exact input order, with no chronology requirement.

    Preserve duplicate requests and the original candidate objects. Empty batches
    return (). Invalid candidate/timestamp/config/cache inputs raise before decoding;
    failed reads produce explicit unresolved results with None physical/quality fields.
    A supplied active resolver must belong to this video and remains caller-owned;
    its metadata/cache bound govern access (metadata must match if also supplied).
    Otherwise an operation-local resolver is opened/closed here. No images survive
    in returned records. No ranking, filtering or duplicate removal occurs.
    """
    if not isinstance(candidates, Sequence):
        raise TypeError("candidates must be a sequence of EvidenceCandidate records.")
    if not isinstance(ui_config, UIConfig) or not isinstance(ui_config.ticket_scroll_body, ROI):
        raise TypeError("ui_config must provide a ticket_scroll_body ROI.")
    _positive_integer("max_frames", max_frames)
    snapshot = tuple(candidates)
    for candidate in snapshot:
        if not isinstance(candidate, EvidenceCandidate):
            raise TypeError("Every candidate must be EvidenceCandidate.")
        _real("candidate timestamp", candidate.timestamp_seconds)
    path = Path(video_path).resolve()
    if resolver is not None:
        if not isinstance(resolver, VideoFrameResolver):
            raise TypeError("resolver must be VideoFrameResolver.")
        if resolver.video_path != path or (metadata is not None and metadata != resolver.metadata):
            raise ValueError("Supplied resolver metadata/video must match this operation.")
        if resolver._capture is None:
            raise RuntimeError("Supplied resolver must be active.")
        return _materialize(snapshot, ui_config, resolver)
    if not snapshot:
        if metadata is not None:
            _validate_metadata(metadata, path)
        return ()
    with VideoFrameResolver(path, metadata=metadata, max_frames=max_frames) as owned:
        return _materialize(snapshot, ui_config, owned)


def _materialize(candidates, ui_config, resolver):
    """Validate the complete batch before reads, then attach raw scalar features."""
    for candidate in candidates:
        _target(resolver.metadata, candidate.timestamp_seconds)
    # Validate layout even if every individual read later fails.
    roi = map_roi_to_frame(ui_config.ticket_scroll_body, ui_config,
                           resolver.metadata.width, resolver.metadata.height)
    if roi.right > resolver.metadata.width or roi.bottom > resolver.metadata.height:
        raise ValueError("Mapped ticket body is outside the video frame.")
    results = []
    for candidate in candidates:
        frame = resolver.resolve(candidate.timestamp_seconds)
        quality = None if frame.image is None else measure_ticket_quality(frame.image, ui_config)
        results.append(MaterializedEvidenceCandidate(candidate, frame.resolution_status,
            frame.frame_index, frame.actual_timestamp_seconds,
            quality.ticket_sharpness_laplacian_variance if quality else None,
            quality.ticket_brightness_mean if quality else None,
            quality.ticket_contrast_std if quality else None))
    return tuple(results)
