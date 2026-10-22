"""Model-independent production contracts for one Investigation-level VLM branch.

The runtime receives one continuous video clip plus its source-video identity,
source-time bounds, and prompt version. It does not receive representative-frame
lists or OCR evidence. Runtime execution returns
raw text and execution diagnostics; a later parser can turn that response into
field observations, which are wrapped as one Investigation result. This module
defines only these data contracts and a minimal runtime Protocol: it implements
no model loading, video decoding, parsing, OCR integration, or evidence fusion.

Field names are strings so callers can use the project's existing canonical OCR
names without importing OCR models or creating a second field taxonomy. Field
status describes only the VLM's usable observation: unavailable does not mean
absent, invisible, or unreadable in the video. Execution failure is represented
as result data so the surrounding pipeline can retain other evidence.
"""

import math
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

ExecutionStatus = Literal[
    "success", "timeout", "runtime_error", "input_error", "invalid_response"
]
VLMFieldStatus = Literal["observed", "unavailable", "ambiguous"]
VisualRouteStatus = Literal["observed", "inferred", "not_visible"]
VLMParseStatus = Literal[
    "not_attempted", "valid_json", "recovered_markdown_fence", "missing_keys",
    "invalid_json", "unsupported_top_level_shape",
]


@dataclass(frozen=True)
class VLMParseDiagnostics:
    """Small parse outcome record that leaves the original execution untouched."""

    status: VLMParseStatus
    json_format: Literal["strict_json", "markdown_fence"] | None = None
    missing_keys: tuple[str, ...] = ()
    unknown_keys: tuple[str, ...] = ()
    error: str | None = None

    def __post_init__(self) -> None:
        """Freeze diagnostic key collections and validate optional error text."""
        if self.status not in (
            "not_attempted", "valid_json", "recovered_markdown_fence", "missing_keys",
            "invalid_json", "unsupported_top_level_shape",
        ):
            raise ValueError("Unsupported VLM parse status.")
        if self.json_format not in (None, "strict_json", "markdown_fence"):
            raise ValueError("Unsupported VLM JSON format.")
        for name in ("missing_keys", "unknown_keys"):
            values = tuple(getattr(self, name))
            if any(not isinstance(value, str) for value in values):
                raise TypeError(f"{name} must contain strings.")
            object.__setattr__(self, name, values)
        if self.error is not None and not isinstance(self.error, str):
            raise TypeError("error must be a string or None.")


def _index(name: str, value: int) -> None:
    """Require a nonnegative integer index while rejecting bool values."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, not bool.")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")


def _time(name: str, value: float, *, optional: bool = False) -> None:
    """Validate finite nonnegative seconds without coercing caller input."""
    if value is None and optional:
        return
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, not bool.")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative.")


def _path(name: str, value: Path) -> None:
    """Require an explicit filesystem path without checking file existence."""
    if not isinstance(value, Path):
        raise TypeError(f"{name} must be pathlib.Path.")


@dataclass(frozen=True)
class VLMRequest:
    """One request containing a continuous Investigation clip, never frame inputs.

    ``source_video`` identifies the original recording; ``video_clip_path`` points
    to the continuous clip passed to the runtime. Source bounds retain original-
    video seconds. The prompt version identifies the stable instruction/schema.
    OCR values and OCR/fused evidence are intentionally absent from this contract.
    """

    investigation_index: int
    source_video: Path
    video_clip_path: Path
    source_start_s: float
    source_end_s: float
    prompt: str
    prompt_version: str = "fr24_vlm_v1"

    def __post_init__(self) -> None:
        """Validate identity, continuous clip bounds, and explicit prompt metadata."""
        _index("investigation_index", self.investigation_index)
        _path("source_video", self.source_video)
        _path("video_clip_path", self.video_clip_path)
        _time("source_start_s", self.source_start_s)
        _time("source_end_s", self.source_end_s)
        if self.source_end_s <= self.source_start_s:
            raise ValueError("source_end_s must be greater than source_start_s.")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must be a nonempty string.")
        if not isinstance(self.prompt_version, str) or not self.prompt_version.strip():
            raise ValueError("prompt_version must be a nonempty string.")


@dataclass(frozen=True)
class RawVLMResult:
    """Direct runtime outcome before any semantic response parsing.

    Raw text may be absent or partial for failed executions. Timing and generated
    token measurements are optional because a runtime may not expose them. Model,
    prompt, Investigation, and source-clip metadata make the result reproducible.
    """

    execution_status: ExecutionStatus
    investigation_index: int
    source_video: Path
    video_clip_path: Path
    source_start_s: float
    source_end_s: float
    model_name: str
    model_revision: str | None
    prompt_version: str
    raw_text: str | None = None
    inference_time_s: float | None = None
    ttft_s: float | None = None
    generated_tokens: int | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        """Validate status, source provenance, identity and optional diagnostics."""
        if self.execution_status not in ("success", "timeout", "runtime_error", "input_error", "invalid_response"):
            raise ValueError("Unsupported VLM execution_status.")
        _index("investigation_index", self.investigation_index)
        _path("source_video", self.source_video)
        _path("video_clip_path", self.video_clip_path)
        _time("source_start_s", self.source_start_s)
        _time("source_end_s", self.source_end_s)
        if self.source_end_s <= self.source_start_s:
            raise ValueError("source_end_s must be greater than source_start_s.")
        for name in ("model_name", "prompt_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string.")
        if self.model_revision is not None and not isinstance(self.model_revision, str):
            raise TypeError("model_revision must be a string or None.")
        if self.raw_text is not None and not isinstance(self.raw_text, str):
            raise TypeError("raw_text must be a string or None.")
        _time("inference_time_s", self.inference_time_s, optional=True)
        _time("ttft_s", self.ttft_s, optional=True)
        if self.generated_tokens is not None:
            _index("generated_tokens", self.generated_tokens)
        if self.error is not None and not isinstance(self.error, str):
            raise TypeError("error must be a string or None.")


@dataclass(frozen=True)
class VLMFieldObservation:
    """One field-level VLM observation with source and model provenance.

    ``raw_value`` preserves the parser's value or abstention. ``unavailable``
    means no reliable usable value was provided; it does not assert that the
    field was absent, invisible, or unreadable. ``evidence_time_s`` is optional
    and, when provided, uses source-video time.
    """

    field_name: str
    raw_value: str | bool | None
    status: VLMFieldStatus
    investigation_index: int
    source_video: Path
    source_start_s: float
    source_end_s: float
    model_name: str
    model_revision: str | None
    prompt_version: str
    evidence_time_s: float | None = None

    def __post_init__(self) -> None:
        """Validate field identity and ensure optional evidence time is in range."""
        if not isinstance(self.field_name, str) or not self.field_name.strip():
            raise ValueError("field_name must be a nonempty canonical-name string.")
        if self.raw_value is not None and not isinstance(self.raw_value, (str, bool)):
            raise TypeError("raw_value must be a string, bool, or None.")
        if self.status not in ("observed", "unavailable", "ambiguous"):
            raise ValueError("Unsupported VLM field status.")
        if self.status == "observed" and (
            self.raw_value is None
            or (isinstance(self.raw_value, str) and not self.raw_value.strip())
        ):
            raise ValueError("observed status requires a nonempty raw_value.")
        if self.status == "unavailable" and self.raw_value is not None:
            raise ValueError("unavailable status must not carry a selected raw_value.")
        _index("investigation_index", self.investigation_index)
        _path("source_video", self.source_video)
        _time("source_start_s", self.source_start_s)
        _time("source_end_s", self.source_end_s)
        if self.source_end_s <= self.source_start_s:
            raise ValueError("source_end_s must be greater than source_start_s.")
        for name in ("model_name", "prompt_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string.")
        if self.model_revision is not None and not isinstance(self.model_revision, str):
            raise TypeError("model_revision must be a string or None.")
        if self.evidence_time_s is not None:
            _time("evidence_time_s", self.evidence_time_s)
            if not self.source_start_s <= self.evidence_time_s <= self.source_end_s:
                raise ValueError("evidence_time_s must lie within the source clip bounds.")


@dataclass(frozen=True)
class VLMVisualRouteClaim:
    """One map-derived route claim with explicit observation/inference status."""

    value: str | None
    status: VisualRouteStatus

    def __post_init__(self) -> None:
        """Keep unsupported or invisible route claims explicitly empty."""
        if self.status not in ("observed", "inferred", "not_visible"):
            raise ValueError("Unsupported visual route status.")
        if self.value is not None and (not isinstance(self.value, str) or not self.value.strip()):
            raise ValueError("Visual route value must be nonempty text or None.")
        if self.status == "not_visible" and self.value is not None:
            raise ValueError("not_visible visual route claims must have a null value.")
        if self.status in ("observed", "inferred") and self.value is None:
            raise ValueError("Visible or inferred visual route claims require a value.")


@dataclass(frozen=True)
class VLMVisualRouteObservation:
    """Source-specific visual map/route claims from one full-video VLM pass."""

    direction_sector: VLMVisualRouteClaim
    direction_text: VLMVisualRouteClaim
    route_description: VLMVisualRouteClaim
    departure_context: VLMVisualRouteClaim
    investigation_index: int
    source_video: Path
    source_start_s: float
    source_end_s: float
    model_name: str
    model_revision: str | None
    prompt_version: str

    def __post_init__(self) -> None:
        """Validate each independent route claim and match its Pass 1 provenance."""
        for name in ("direction_sector", "direction_text", "route_description", "departure_context"):
            if not isinstance(getattr(self, name), VLMVisualRouteClaim):
                raise TypeError(f"{name} must be VLMVisualRouteClaim.")
        if self.direction_sector.value is not None and self.direction_sector.value.upper() not in {
            "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
        }:
            raise ValueError("direction_sector must be a 16-point compass sector or None.")
        _index("investigation_index", self.investigation_index)
        _path("source_video", self.source_video)
        _time("source_start_s", self.source_start_s)
        _time("source_end_s", self.source_end_s)
        if self.source_end_s <= self.source_start_s:
            raise ValueError("source_end_s must be greater than source_start_s.")
        for name in ("model_name", "prompt_version"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty string.")
        if self.model_revision is not None and not isinstance(self.model_revision, str):
            raise TypeError("model_revision must be a string or None.")


@dataclass(frozen=True)
class VLMInvestigationResult:
    """Complete raw and parsed VLM output for one Investigation.

    Runtime status and execution provenance are exposed through the nested raw
    result instead of duplicated fields. Failed execution results must contain no
    field observations, but remain valid data for downstream pipeline handling.
    """

    raw_result: RawVLMResult
    observations: tuple[VLMFieldObservation, ...] = ()
    parse_diagnostics: VLMParseDiagnostics | None = None
    visual_route: VLMVisualRouteObservation | None = None

    def __post_init__(self) -> None:
        """Freeze observation storage and reject cross-Investigation provenance."""
        if not isinstance(self.raw_result, RawVLMResult):
            raise TypeError("raw_result must be RawVLMResult.")
        if self.parse_diagnostics is not None and not isinstance(self.parse_diagnostics, VLMParseDiagnostics):
            raise TypeError("parse_diagnostics must be VLMParseDiagnostics or None.")
        if self.visual_route is not None and not isinstance(self.visual_route, VLMVisualRouteObservation):
            raise TypeError("visual_route must be VLMVisualRouteObservation or None.")
        observations = tuple(self.observations)
        if any(not isinstance(item, VLMFieldObservation) for item in observations):
            raise TypeError("observations must contain VLMFieldObservation records.")
        if self.raw_result.execution_status != "success" and (observations or self.visual_route is not None):
            raise ValueError("Failed VLM execution must have empty observations.")
        for item in observations:
            if (item.investigation_index != self.raw_result.investigation_index
                    or item.source_video != self.raw_result.source_video
                    or item.source_start_s != self.raw_result.source_start_s
                    or item.source_end_s != self.raw_result.source_end_s
                    or item.model_name != self.raw_result.model_name
                    or item.model_revision != self.raw_result.model_revision
                    or item.prompt_version != self.raw_result.prompt_version):
                raise ValueError("Observation provenance must match its raw Investigation result.")
        if self.visual_route is not None:
            route = self.visual_route
            if (route.investigation_index != self.raw_result.investigation_index
                    or route.source_video != self.raw_result.source_video
                    or route.source_start_s != self.raw_result.source_start_s
                    or route.source_end_s != self.raw_result.source_end_s
                    or route.model_name != self.raw_result.model_name
                    or route.model_revision != self.raw_result.model_revision
                    or route.prompt_version != self.raw_result.prompt_version):
                raise ValueError("Visual route provenance must match its raw Investigation result.")
        object.__setattr__(self, "observations", observations)

    @property
    def investigation_index(self) -> int:
        """Return the Investigation identity from the raw execution record."""
        return self.raw_result.investigation_index

    @property
    def execution_status(self) -> ExecutionStatus:
        """Expose runtime status without storing a duplicate value."""
        return self.raw_result.execution_status

    @property
    def model_name(self) -> str:
        """Expose the model identity from the raw execution record."""
        return self.raw_result.model_name

    @property
    def model_revision(self) -> str | None:
        """Expose the model revision from the raw execution record."""
        return self.raw_result.model_revision

    @property
    def prompt_version(self) -> str:
        """Expose the prompt version from the raw execution record."""
        return self.raw_result.prompt_version

    @property
    def source_start_s(self) -> float:
        """Expose the original-video start bound from the raw execution record."""
        return self.raw_result.source_start_s

    @property
    def source_end_s(self) -> float:
        """Expose the original-video end bound from the raw execution record."""
        return self.raw_result.source_end_s

    @property
    def inference_time_s(self) -> float | None:
        """Expose optional measured inference time from the raw execution record."""
        return self.raw_result.inference_time_s

    @property
    def error(self) -> str | None:
        """Expose optional execution error from the raw execution record."""
        return self.raw_result.error


@runtime_checkable
class VLMRuntime(Protocol):
    """Minimal model-independent interface for running one continuous-video request."""

    def run(self, request: VLMRequest) -> RawVLMResult:
        """Execute the request and return raw output or a represented failure."""
        ...
