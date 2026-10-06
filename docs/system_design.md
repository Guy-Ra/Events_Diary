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

Purpose:

Measure visual changes in meaningful regions without deciding yet whether an Investigation Event occurred.

Possible signals:

- mean absolute difference;
- changed-pixel ratio;
- SSIM;
- grayscale/downscaled change measures.

Semantic regions include:

```text
ticket panel
map
route
selected aircraft
aircraft image
browser/static controls masks
```

The full map, including the right side, is potentially useful evidence.

Static browser chrome and irrelevant controls should be masked rather than blindly analyzed.

### Map motion

Because pan and zoom can occur, raw pixel difference is insufficient.

The system should estimate global map motion and distinguish it from meaningful content changes.

For the initial UI:

- map orientation is north-up;
- rotation is not expected;
- translation and scale are the relevant global transforms.

### Ticket scroll motion

A dedicated signal should detect dominant vertical content motion inside the ticket while the selected aircraft identity remains stable.

This signal explains ticket scrolling and helps prevent false Investigation boundaries.

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
