# Events Diary — System Design

## 1. Purpose

Events Diary is a local visual-intelligence system for analyzing short Flightradar24 screen recordings and converting them into a structured aircraft event journal.

The system is designed to understand what an operator investigated on screen, extract relevant information from the Flightradar24 interface, infer missing information only when evidence supports it, and produce both machine-readable and human-readable outputs.

The initial target is short recordings of approximately 30–40 seconds, but the architecture should remain scalable to longer sessions.

---

## 2. Core Product Concept

The fundamental unit of operator activity is an **Investigation Event**.

An Investigation Event represents a period during which the operator focuses on a selected aircraft and inspects its information.

The same aircraft may be selected more than once during a video. These separate Investigation Events can later be grouped under a single **Tracked Aircraft Entity**.

Example:

```text
event_001 → UAL84
event_002 → DL152
event_003 → UAL84
```

This represents:

- 3 Investigation Events
- 2 Tracked Aircraft Entities

The system must preserve both perspectives:

1. chronological operator activity;
2. aircraft-level aggregation.

---

## 3. Main Inputs and Outputs

### Input

Initial MVP input:

```text
Short Flightradar24 screen recording
```

Typical characteristics:

- approximately 30–40 seconds;
- north-up map;
- no map rotation;
- zoom and pan may occur;
- many aircraft may be visible simultaneously;
- one aircraft is selected at a time;
- the information ticket appears on the left side;
- the ticket can be vertically scrolled;
- the right side of the map is useful visual evidence;
- no advertisement region needs to be excluded.

### Outputs

The system produces one canonical session representation and renders multiple user-facing outputs from it.

Planned outputs:

```text
session.json
operational_log.txt
detailed_report.txt
optional debug artifacts
```

The JSON is the source of truth. Text outputs are renderings of that structured record.

---

## 4. Design Principles

### 4.1 Evidence-first

The system must distinguish clearly between:

- information directly visible on screen;
- semantic enrichment;
- inference.

The system must never silently invent missing information.

Unknown values are valid outputs.

### 4.2 Deterministic methods first

Computer vision, OCR, geometry, temporal reasoning, and explicit system rules should handle tasks that can be solved reliably without a VLM.

A VLM is used selectively where it adds value.

### 4.3 VLM as assistant, not authority

The VLM can help with:

- difficult visual text;
- semantic enrichment;
- aircraft/livery interpretation;
- ambiguous visual evidence;
- final Investigation-level cross-checking.

It must not freely rewrite the structured record.

### 4.4 Modular implementation

Each phase should have a narrow responsibility and clear inputs/outputs.

Avoid:

- large catch-all modules;
- speculative abstractions;
- unnecessary class hierarchies;
- premature support for hypothetical models.

### 4.5 Configuration-driven behavior

Thresholds, ROI coordinates, sampling parameters, model settings, and other tunable behavior should live in configuration rather than being scattered through code.

### 4.6 Incremental validation

Every phase must be tested independently and then tested as part of the cumulative pipeline.

---

# 5. Evidence Model

Each meaningful extracted field should carry provenance and confidence information.

## 5.1 Field Status

Conceptual statuses:

```text
OBSERVED
ENRICHED
INFERRED
EXPLICIT_NA
UNREADABLE
NOT_PRESENT
AMBIGUOUS
CONFLICT
```

Meaning:

- **OBSERVED** — directly supported by visual evidence.
- **ENRICHED** — semantic expansion of observed information.
- **INFERRED** — derived from indirect evidence.
- **EXPLICIT_NA** — UI explicitly displays an unavailable value.
- **UNREADABLE** — field appears present but cannot be read reliably.
- **NOT_PRESENT** — field is not visible in available evidence.
- **AMBIGUOUS** — multiple plausible values remain.
- **CONFLICT** — strong evidence sources disagree.

## 5.2 Confidence

User-facing confidence should remain qualitative:

```text
HIGH
MEDIUM
LOW
```

Avoid presenting fake numerical precision.

## 5.3 Provenance

Possible evidence/source classes include:

```text
OCR
GEMMA_VISUAL
GEMMA_SEMANTIC
CV_GEOMETRY
ROUTE_INFERENCE
AIRCRAFT_IMAGE
SYSTEM_RULE
```

Observed information always outranks unsupported inference.

---

# 6. Ticket Model

The Flightradar24 ticket on the left side is a dynamic scrollable viewport.

The outer panel may remain spatially stable, but the content inside it can change due to scrolling.

Therefore the system must not assume that every ticket field exists in a fixed vertical ROI.

## 6.1 Scroll behavior

Ticket scrolling:

- does not create a new Investigation Event;
- should trigger dense evidence sampling;
- should preserve stable scroll stops;
- may reveal additional fields that were not visible in the initial ticket view.

Two frames with different ticket scroll positions are not duplicates if they reveal different information.

## 6.2 Ticket information

The canonical record should contain normalized core fields and allow additional ticket data.

Examples of useful ticket fields:

```text
flight number
airline
origin
destination
scheduled time
estimated time
registration
aircraft type
altitude
vertical speed
ground speed
track
ICAO 24-bit address
squawk
latitude / longitude
data source
aircraft serial
aircraft age
aircraft category
```

The system should not discard fields simply because they are not part of the concise operational log.

A flexible `ticket_sections` structure should retain additional information.

---

# 7. Phase Architecture

## Phase 0 — Project Foundation

Purpose:

Build a clean, reproducible Python project before implementing visual logic.

Responsibilities:

- Python environment;
- `uv`;
- package structure;
- pytest;
- Ruff;
- configuration foundation;
- logging foundation;
- runtime/hardware/model profile skeleton;
- first application entry point.

No Flightradar24 visual logic should be implemented here.

---

## Phase 1 — Video Decode & Time-Based Sampling

Purpose:

Create a reliable representation of the source video and sample it in time.

Responsibilities:

- decode metadata;
- duration;
- source FPS;
- resolution;
- frame count;
- timestamps;
- time-based frame sampling.

Timestamps are primary.

Frame index divided by FPS is only an approximation.

The sampling rate should be empirically selected based on the shortest meaningful operator action that must be detected.

Candidate test rates:

```text
1 FPS
2 FPS
4 FPS
8 FPS
```

Important cases:

- variable frame rate;
- changing video resolution;
- corrupted frame;
- video begins during an active Investigation;
- video ends during an active Investigation.

Frames should not be permanently stored unless needed as evidence or debug output.

---

## Phase 2 — UI Regions & Change Features

Status: COMPLETE (2.1-2.5). Phase 3, Investigation Event Detection, is next.
Phase 2 measures visual evidence and interprets active scroll motion; it does not
make Investigation decisions or extract semantic ticket/aircraft information.

### UI / ROI layer (2.1)

`config.py` defines frozen `UIConfig` and rectangular `ROI` objects against the
canonical 1920x1080 layout. `ROI.crop()` uses exclusive right/bottom boundaries
and rejects out-of-frame crops. Existing canonical regions include:

| Region | x | y | Width | Height |
|---|---:|---:|---:|---:|
| `map_analysis_region` | 352 | 121 | 1568 | 919 |
| `ticket_header` | 16 | 121 | 336 | 76 |
| `aircraft_image` | 16 | 197 | 336 | 189 |
| `ticket_scroll_body` | 16 | 386 | 336 | 566 |

`layout.py` classifies exact-reference, proportional-scale, and partial/cropped
layouts. `map_roi_to_frame()` maps compatible layouts proportionally and rejects
partial/cropped layouts. Explicit `LayoutTransform` scale/offsets can instead be
supplied to `map_roi_with_transform()`, which retains unrounded bounds, a visible
crop, and FULL/PARTIAL/OUTSIDE visibility. This supports supplied transforms for
cropped layouts without claiming automatic layout recovery. Current motion APIs
with canonical UIConfig and real-video evaluators require exact reference geometry.

`UIConfig.map_ignore_masks` contains four full-frame rectangles: top FR24 strip
`(352,121,1568,60)`, right controls `(1856,284,60,218)`, bottom toolbar
`(746,958,428,70)`, and attribution/footer `(1480,1024,440,16)`. These exclude SIFT
feature centers only. They do not mask raw or aligned pixel metrics, and there
are no aircraft/route/geographic exclusion categories.

### Basic visual change layer (2.2)

`change_metrics.py` accepts nonempty matching uint8 grayscale or three-channel BGR
arrays. BGR is converted to grayscale, and OpenCV absolute difference avoids
unsigned subtraction wraparound. No resizing, filtering, or normalization occurs.

- `compute_mad()` returns mean absolute grayscale difference in intensity units
  (0-255).
- `compute_changed_pixel_ratio()` returns the fraction of pixels whose difference
  is strictly greater than `pixel_threshold`. Its current default is **10**, an
  experimental working value; callers may explicitly override it within 0-255.

These measurements carry no semantic interpretation. Equal-grayscale color changes
may be missed, while noise and motion can cause changes. SSIM is intentionally
deferred and is not part of the completed Phase 2 production implementation.

### Map motion compensation (2.3)

`estimate_map_motion()` in `map_motion.py` returns frozen `MapMotionEstimate`
evidence, including validity/reason, B-to-A dx/dy/scale/rotation, feature/match/inlier
counts, inlier ratio, raw MAD/CPR, and optional aligned MAD/CPR.

The deterministic pipeline detects SIFT features, matches B descriptors against A
with BF L2 KNN (`k=2`), applies Lowe's ratio test, and fits a RANSAC partial-affine
transform. Working values are Lowe ratio 0.75, minimum 10 good matches, 8 inliers,
inlier ratio 0.50, scale range [0.5,2.0], and rotation tolerance 3 degrees. RANSAC
uses a 3-pixel reprojection threshold, 2000 maximum iterations, 0.99 confidence,
and 10 refinement iterations. These are provisional, not production calibration.

From the fitted matrix `[[a,-b,tx],[b,a,ty]]`, derive scale `hypot(a,b)`, rotation
`degrees(atan2(b,a))`, dx `tx`, and dy `ty`. The intended north-up model supports
translation plus uniform scale. Rotation is retained diagnostically and checked
for validity; accepted rotation is excluded from the alignment warp.

When `ui_config` is supplied, full-frame ignore rectangles subtract the map origin
(352,121) to build a map-local uint8 SIFT mask (255 allowed / 0 excluded). Inputs
must match canonical crop dimensions. Generic calls without configuration remain
unmasked. Input image pixels are never changed by detection masks.

Map B is warped into Map A using `[[scale,0,dx],[0,scale,dy]]` with linear
interpolation. An all-ones support image is warped identically; only pixels with
full source support enter aligned MAD/CPR through the existing metric APIs.
Artificial padding is excluded. Raw metrics cover the whole crop, so their domain
differs from aligned overlap. Invalid fits retain raw metrics and an explanatory
reason, with aligned metrics unavailable. Missing descriptors, weak matches or
inliers, failed/non-finite transforms, implausible scale/rotation, and empty overlap
are rejected rather than forced into a result.

Engineering conclusion: **dense consecutive-source-frame analysis is the preferred
production approach for map motion**. Large distant-frame displacement can reduce
geographic feature overlap; static overlays also previously dominated the distant
pan fit. SIFT masking removes static UI feature centers but cannot recover missing
geographic overlap. Adjacent-frame comparisons preserve overlap better.
This is an input-selection conclusion, not a new detector or fallback algorithm.

Real-video validation confirmed meaningful pan translation with near-unit scale
and coherent zoom scale changes. Some weak zoom fits may be rejected. Residual MAD may rise
after interpolation/re-rendering even when CPR improves; validity does not prove
that global motion explains every visual change.

### Ticket scroll analysis (2.4)

`estimate_ticket_scroll_evidence(frame_a, frame_b, ui_config)` accepts full matching
canonical uint8 grayscale/BGR frames. Frozen `TicketScrollEvidence` contains:
`header_mad`, `header_cpr`, `aircraft_image_mad`, `aircraft_image_cpr`, `body_mad`,
`body_cpr`, `map_mad`, `map_cpr`, `dx`, `dy`, and `motion_response`.

All four ROIs reuse existing MAD/default CPR. Only `ticket_scroll_body` supplies
motion: grayscale float32 arrays, a Hann window, and `cv2.phaseCorrelate()`.
Displacement is **A-to-B**, right/down positive, opposite to map compensation's
B-to-A convention. dx/dy are not clipped; response is diagnostic without a
reliability threshold. Repetitive or textureless content and large shifts can
produce unreliable motion even with finite values.

`is_vertical_scroll_evidence()` requires all inclusive working conditions:
body CPR >=0.10, |dy| >=2 pixels, |dy| >=3*|dx|, and each header/image/map CPR <=0.01.
These defaults are provisional, informed by one recording and simple margins;
vertical dominance is especially uncalibrated. Non-finite decision inputs fail.
A positive pair is evidence only, not a temporal event or Investigation decision.

`analyze_ticket_scroll_sequence()` consumes those records without recomputing
pixels or motion. It returns one `TicketScrollState` per input pair:
`QUIET="quiet"`, `SCROLLING_UP="scrolling_up"`, or
`SCROLLING_DOWN="scrolling_down"`. Provisional defaults are window_size=5,
min_positive_pairs=2, max_gap_pairs=1. Positive pairs form candidate spans across
at most one False gap; longer gaps split spans. Centered offline windows are
clipped to each span and sequence boundary, require two positives, and use only
positive dy signs for direction. Ties are QUIET; no leading/trailing quiet extension
occurs. Future pairs are used, so this is not a causal streaming API.

Scrolling states represent detected active motion in the current context, not the
entire human scrolling interaction. Short or longer pauses may legitimately be
QUIET between bursts. A future "recently scrolling" cooldown or broader interaction
context belongs to a higher-level consumer such as Phase 3 Investigation Detection,
rather than artificial extension of these states. No scroll-event objects,
start/end timestamps, or Investigation decisions are created.

### Real-video validation and closure (2.5)

`scripts/validate_phase2.py` exercises both pipelines on consecutive decoded frames
at source FPS, with explicit EOF/count/layout validation.
Manual evaluation scripts retain source indices and nominal index/FPS times;
constant-FPS assumptions and metadata-count limitations remain explicit.

Closure validation confirmed these architectural conclusions:

- Pan estimation captures meaningful translation while scale remains near one.
- Zoom estimation shows coherent scale changes, with occasional rejected weak fits.
- Ticket-scroll evidence and temporal states capture the correct vertical direction
  during active motion; quiet intervals and pauses remain quiet.
- Cross-phenomenon checks showed no problematic confusion between map motion and
  ticket scrolling in the representative recordings.
- Tests and lint passed at closure; no new production issue was observed.

This representative manual validation is not an accuracy benchmark. Detailed
measurements belong in the local validation report. Videos, CSVs, and debug images
remain local ignored evaluation evidence, not committed architecture artifacts.

### Phase 2 output contract for Phase 3

Phase 2 provides UI-region change signals, geometric map-motion estimates with
validity/rejection diagnostics, raw/aligned visual metrics, pair-level ticket-scroll
evidence, and temporal active-scroll state. These validated signals can support
inclusion, exclusion, or contextual reasoning by Phase 3. They are not semantic
proof of aircraft identity, route changes, or an Investigation boundary. Consumers
must preserve unavailable estimates, coordinate conventions, and metric domains.

Known limitations remain: provisional thresholds, possible rejected zoom pairs,
interpolation residuals, active-motion rather than whole-interaction scroll state,
frame-rate-dependent temporal context, and no automatic layout recovery. Phase 2
is COMPLETE; **Phase 3 - Investigation Event Detection is next planned**, where
broader context and Investigation decisions will be designed separately.

---

## Phase 3 — Investigation Event Detection

Purpose:

Convert Phase 2 signals into a timeline of operator Investigation Events.

Conceptual states:

```text
NO_INVESTIGATION
CANDIDATE_INVESTIGATION
ACTIVE_INVESTIGATION
```

Example:

```text
ACTIVE A
→ candidate B
→ ACTIVE B
```

This allows direct switching between aircraft without requiring the ticket to close.

### Strong signals

Potential strong evidence:

- ticket header change;
- origin/destination change;
- aircraft details change;
- route change after map compensation.

### Supporting signals

Potential supporting evidence:

- aircraft image change;
- selected-aircraft visual change;
- timing patterns;
- other ticket changes.

### Negative/explanatory signals

Examples:

- map pan/zoom;
- ticket scroll.

Ticket-only vertical scrolling with stable aircraft identity must not produce a new Investigation Event.

### Temporal stabilization

A single-frame change must not confirm a new Investigation.

Store separately:

```text
candidate_detected_at
confirmed_at
```

Possible end reasons:

```text
new_selection
ticket_closed
video_end
```

Possible start reason:

```text
video_started_active
```

Signal availability should be represented explicitly:

```text
NOT_AVAILABLE
LOW
MEDIUM
HIGH
```

Unavailable is not equivalent to weak evidence.

---

## Phase 4 — Investigation Extraction & Adaptive Sampling

Purpose:

Once an Investigation is detected, recover richer evidence from the original video.

The relevant interval should include context before and after the confirmed event.

Conceptual regions:

```text
PRE_TRANSITION
TRANSITION
STABLE
EXIT
```

Sampling should adapt to visual activity.

Use denser sampling during:

- aircraft selection transitions;
- ticket loading;
- ticket scrolling;
- map/route transitions.

Use sparser sampling during stable periods.

Ticket scrolling should trigger dense sampling until the scroll settles.

Each stable ticket scroll position can become candidate evidence.

The goal is not maximum frame count.

The goal is maximum information coverage with minimal redundancy.

---

## Phase 5 — Role-Based Frame & Ticket-Section Selection

Purpose:

Select evidence frames based on what downstream tasks need.

There is no single universal “best frame”.

Possible evidence roles:

```text
ticket_header
ticket_route_times
ticket_aircraft_details
ticket_flight_state
ticket_data_source
ticket_additional_section
route
aircraft_image
context
```

### Ticket evidence

Ticket frame quality can consider:

- sharpness;
- UI stability;
- readable content;
- completeness;
- absence of transition blur.

N/A is valid content and must not be treated as failed loading.

Ticket selection should maximize section coverage.

### Route evidence

Consider:

- route visibility;
- selected aircraft visibility;
- viewport coverage;
- map stability;
- route clipping.

Preserve meaningful diversity, such as:

- a wider route view;
- a closer route view.

### Aircraft image evidence

Consider:

- image loaded;
- sharpness;
- crop completeness;
- stability;
- occlusion;
- placeholder detection.

A placeholder image means image-based aircraft/livery inference is unavailable.

### Context evidence

A context frame should show the Investigation in a broad, understandable way.

### Deduplication

Near-identical frames may be deduplicated using perceptual similarity such as pHash or SSIM.

However, ticket frames from different scroll positions must remain if they expose new content.

---

## Phase 6 — OCR + Complete Ticket Extraction

Purpose:

Extract all useful ticket information across multiple scroll states.

Workflow:

```text
stable ticket views
→ section-aware OCR
→ label/value parsing
→ normalization
→ multi-frame consensus
→ complete Investigation ticket record
```

### OCR strategy

OCR should operate on semantic ticket regions rather than the full screen.

Light preprocessing variants may be used where helpful.

Raw OCR observations must be preserved.

### Multi-frame consensus

A field may appear across multiple frames.

Resolve it using evidence such as:

```text
OCR confidence
frame quality
ROI quality
independent observation count
agreement between frames
```

Possible resolution states:

```text
RESOLVED
AMBIGUOUS
UNAVAILABLE
```

### Dynamic fields

Values such as altitude or speed may change over time.

These should be retained with timestamps rather than incorrectly treated as conflicting static fields.

### VLM assistance

Use a VLM when:

- OCR confidence is poor;
- parsing is difficult;
- the value is visually readable but OCR fails.

The VLM should receive targeted crops and constrained prompts.

It is not automatically the authoritative source.

---

## Phase 7 — Route Extraction & Direction

Purpose:

Use visible route/map geometry to determine movement direction and cautiously infer missing route information.

### Direction

Direction is a computed field with 8 classes:

```text
N
NE
E
SE
S
SW
W
NW
```

Because the map is north-up, direction can initially be calculated deterministically from route-tail geometry.

### Route extraction candidate pipeline

```text
Map ROI
→ color-space segmentation
→ morphological cleanup
→ connected components
→ candidate filtering
→ selected-aircraft endpoint association
→ skeletonization
→ branch pruning
→ ordered route polyline
```

Do not assume the largest connected component is the route.

If route association is unclear:

```text
route_status = AMBIGUOUS
```

### Route clipping

If the historical endpoint reaches the map border, the route may be clipped.

This must reduce the strength of origin inference.

### Direction fitting

Fit recent route geometry using a relative tail window rather than a fixed pixel count.

Possible methods:

- PCA;
- linear fit.

Compare short and medium route windows.

Strong disagreement may indicate a turn and should reduce confidence.

### Origin/destination inference

Inference must use a conservative hierarchy.

Example origin hierarchy:

```text
UNKNOWN
→ directional region
→ geographic region/city
→ exact airport
```

Exact airport inference requires strong visual evidence.

Destination inference should be even more conservative because a historical route trail does not necessarily reveal the future destination.

The VLM must not infer airports from “typical airline routes” unless visually supported.

---

## Phase 8 — VLM Assistance & Evidence Fusion

Purpose:

Fuse deterministic evidence, OCR results, semantic enrichment, and VLM assistance into one Investigation-level record.

### Field-specific priorities

Examples:

```text
flight text:
multi-frame OCR consensus > single VLM read

direction:
deterministic route geometry > VLM direction guess

origin/destination:
observed ticket value > map inference

airline:
observed text > semantic expansion from code > livery inference

aircraft type:
observed code/text > semantic expansion > image inference
```

### Semantic enrichment

The VLM can expand observed codes without changing the observed value.

Examples:

```text
EWR → Newark, New York
B789 → Boeing 787-9
```

Keep both:

```text
observed_code
enriched_name
```

No local airport/airline dictionary is required for the initial MVP.

### Conflict handling

Rules:

- observed beats inferred;
- repeated independent evidence beats equivalent single evidence;
- deterministic geometry beats VLM direction guessing;
- strong contradictions remain visible as conflicts.

### Final Investigation cross-check

At the end of each Investigation, the VLM may receive:

- selected ticket views;
- route/map context;
- structured observations;
- targeted VLM results.

It should return findings such as:

```text
confirmed_fields
suspected_errors
conflicts
missed_visible_information
```

It should not freely regenerate the final JSON.

---

## Phase 9 — Tracked Aircraft Entity Matching

Purpose:

Determine whether Investigation Events refer to the same operational target.

Strong evidence:

```text
registration exact match
flight number / callsign
route continuity
plausible position progression
```

Medium evidence:

```text
origin/destination
aircraft type
compatible route context
```

Weaker evidence:

```text
airline
time proximity
```

Missing evidence is neutral.

Possible decisions:

```text
MATCH
AMBIGUOUS
NO_MATCH
```

Unknown aircraft may receive internal IDs such as:

```text
unknown_aircraft_001
```

### Entity merge rules

Do not blindly overwrite fields.

Static-ish fields can accumulate evidence:

```text
flight number
airline
registration
aircraft type
origin
destination
```

Time-varying fields belong on a timeline:

```text
direction
position
altitude
speed
ETA
route state
```

Strong contradictory identity evidence can reopen a previous match decision.

---

## Phase 10 — Canonical Output & Renderers

Purpose:

Create the canonical session schema and derive human-readable outputs from it.

### Canonical session

Top-level structure may include:

```text
schema_version
session
investigations
entities
processing_warnings
processing_errors
```

Every Investigation remains distinct.

Retain both:

```text
candidate_detected_at
confirmed_at
```

For operational chronology, `candidate_detected_at` can be preferred when reliable.

### Field representation

Meaningful fields should preserve:

```text
value
status
confidence
primary_source
evidence_refs
```

### Route representation

Separate:

- operational route information;
- analysis/debug diagnostics.

### Record quality

Possible Investigation quality labels:

```text
COMPLETE
PARTIAL
LOW_INFORMATION
CONFLICTED
```

### Operational log

The operational log should remain concise.

Example:

```text
[00:04–00:10]
UAL84 — United Airlines
EWR (Newark, New York) → TLV (Tel Aviv)
Aircraft: B789 (Boeing 787-9)
Direction: SE
```

Inferred origin/destination should be visibly marked.

Revisits should be identifiable.

### Detailed report

The detailed report can include:

- confidence;
- provenance;
- conflicts;
- additional ticket sections;
- inference reasoning summaries;
- evidence references.

### Renderer rule

Renderers format existing structured data only.

They do not:

- infer;
- enrich;
- resolve conflicts;
- call models.

---

# 8. Hardware and Model Adaptation

The runtime should adapt to both the machine and the selected model.

Conceptual relationship:

```text
HardwareProfile + ModelProfile → ExecutionProfile
```

## HardwareProfile

Possible information:

```text
GPU available
GPU name
VRAM
system RAM
CPU
```

## ModelProfile

Capability-oriented description:

```text
visual input
visual text reading
semantic enrichment
structured output reliability
livery interpretation
multi-image reasoning
map reasoning
full-Investigation cross-check
```

Do not infer capability only from model name or parameter count.

A compact fixed capability test suite should be run when adding a new model/backend.

## Complexity ladder

Example task difficulty:

```text
1. semantic enrichment
2. visual crop reading
3. aircraft/livery interpretation
4. multi-image reasoning
5. complex map/geographic inference
```

Unsupported capabilities should degrade gracefully.

Examples:

```text
full-context VLM cross-check
→ targeted single-image VLM
→ deterministic OCR/CV
```

or:

```text
exact airport inference
→ regional inference
→ direction only
→ UNKNOWN
```

---

# 9. Testing Strategy

Testing is integrated into every phase.

For each phase:

1. define responsibility and I/O contract;
2. implement the smallest correct version;
3. write unit/component tests;
4. test a representative real visual example;
5. inspect debug/intermediate outputs;
6. run the cumulative pipeline from Phase 1 to the current phase;
7. fix integration failures;
8. add regression/golden tests;
9. refactor and document;
10. proceed when stable.

Planned test structure:

```text
tests/
├── unit/
├── integration/
├── fixtures/
├── mocks/
└── conftest.py
```

These directories should be created only when needed.

### Golden examples

Real representative cases should be preserved to prevent regressions.

Important future examples:

```text
normal selection
ticket scroll
aircraft switch
aircraft revisit
private / N/A flight
poor OCR
route clipped
map pan/zoom
video starts mid-Investigation
video ends mid-Investigation
```

Final full-system validation is human review:

> Does the generated event journal accurately describe what occurred visually?

---

# 10. Implementation Principles

- Use Python 3.10 for the current project baseline.
- Use `uv` for environment and dependency management.
- Use pytest for testing.
- Use Ruff for linting/formatting.
- Keep production code under `src/events_diary`.
- Avoid catch-all `utils.py`.
- Create modules and abstractions only when their responsibility is understood.
- Keep mocks inside tests.
- Do not create speculative future model implementations.
- Prefer functions when classes do not add value.
- Use type hints where they improve clarity.
- Keep thresholds and model/runtime settings configurable.
- Preserve raw evidence before normalization/fusion.
- Make intermediate pipeline results inspectable.
- Optimize first for correctness, debuggability, and maintainability rather than premature performance.

---

# 11. Current MVP Scope

Included:

```text
Flightradar24 screen recordings
short offline/post-hoc analysis
north-up map
ticket scrolling
OCR
computer vision
route geometry
selective VLM assistance
Investigation detection
aircraft revisit/entity matching
structured JSON
operational log
detailed report
```

Not required for the initial MVP:

```text
real-time processing
map rotation
audio fusion
global persistent aircraft identity across unrelated sessions
large local airport/airline databases
fully generic support for arbitrary flight-tracking UIs
full cloud deployment
```

These can be added later if justified by product needs.
