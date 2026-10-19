# Events Diary — Project Plan

## 1. Goal

Build a maintainable local system that converts Flightradar24 screen recordings into a structured chronological aircraft event journal.

The implementation will proceed phase by phase.

Before each phase, we define:

- exact responsibility;
- inputs and outputs;
- algorithms;
- configuration;
- files/modules;
- tests;
- edge cases;
- definition of done.

Only then do we implement it.

---

# 2. Development Approach

The project will be developed incrementally.

For every phase:

```text
Plan
→ Implement minimal correct version
→ Unit/component tests
→ Real-example test
→ Inspect intermediate outputs
→ Integration test
→ Regression/golden tests
→ Refactor/document
→ Move forward
```

Do not implement several major phases simultaneously.

Do not create architecture for future requirements unless current requirements justify it.

---

# 3. Milestones

## Milestone A — Visual Foundation

Includes:

```text
Phase 0
Phase 1
Phase 2
```

Outcome:

The project is reproducible, video input can be decoded reliably, and meaningful UI-region change signals can be measured.

---

## Milestone B — Operator Activity Understanding

Includes:

```text
Phase 3
Phase 4
Phase 5
```

Outcome:

The system can detect operator Investigation Events and select useful visual evidence for each event.

---

## Milestone C — Semantic Flight Understanding

Includes:

```text
Phase 6
Phase 7
Phase 8
```

Outcome:

The system can extract ticket information, analyze route geometry, compute direction, perform conservative inference, and fuse deterministic + VLM evidence.

---

## Milestone D — Complete MVP

Includes:

```text
Phase 9
Phase 10
```

Outcome:

Repeated aircraft investigations are linked when appropriate, and the full canonical JSON + operational log + detailed report are generated.

---

# 4. Phase 0 — Project Foundation

Status: Completed

## Objective

Create a professional but minimal project foundation before implementing visual logic.

## Planned work

### 0.1 Repository setup

- Git repository
- GitHub remote
- `.gitignore`
- first commits

Status: completed.

### 0.2 Python environment

- Python 3.10
- `uv`
- `.venv`
- `.python-version`

Status: completed.

### 0.3 Project metadata and dependency management

- `pyproject.toml`
- `uv.lock`
- package build configuration
- development dependencies

Current development tools:

```text
pytest
ruff
```

Status: completed.

### 0.4 Source layout

Current package structure:

```text
src/
└── events_diary/
    └── __init__.py
```

Tests:

```text
tests/
```

Status: completed.

### 0.5 Documentation foundation

Create:

```text
README.md
docs/system_design.md
docs/project_plan.md
```

Status: in progress.

### 0.6 Configuration foundation

Before implementation, define the minimal configuration mechanism required by upcoming phases.

Do not introduce a large configuration framework yet.

Expected future categories:

```text
video sampling
UI regions
change thresholds
Investigation detection
OCR
route extraction
VLM/runtime
output
```

### 0.7 Logging foundation

Introduce basic structured logging when the first pipeline execution exists.

Requirements:

- readable console logging;
- phase/module context;
- useful warnings/errors;
- no excessive debug noise by default.

### 0.8 Runtime profile foundation

Define a minimal way to inspect the execution environment.

Future concepts:

```text
HardwareProfile
ModelProfile
ExecutionProfile
```

Do not fully model all future VLM capabilities during Phase 0.

Only create what is needed to expose runtime information cleanly.

### 0.9 Application entry point

Choose and implement the first clean invocation method.

Possible future UX:

```text
uv run events-diary ...
```

or

```text
uv run python -m events_diary ...
```

The final choice should be made just before implementation based on the first real pipeline requirements.

### 0.10 Foundation tests

Verify:

```text
package imports
pytest runs
ruff runs
configuration loads
basic entry point works
```


### Completed work

- Repository setup
- Python 3.10 environment with uv
- Project metadata and dependency management
- Source package structure
- pytest and Ruff setup
- Documentation foundation
- Application configuration foundation
- Logging foundation
- Runtime environment inspection
- Application entry point
- Foundation tests

### Definition of Done

Phase 0 is complete because:

- `uv sync` works
- `events_diary` imports correctly
- `uv run python -m events_diary` runs successfully
- pytest passes
- Ruff passes
- configuration foundation exists
- logging foundation exists
- runtime inspection exists
- README and design documentation exist
- no Flightradar24 analysis logic has been introduced yet

---

# 5. Phase 1 — Video Decode & Time-Based Sampling

## Status

Completed.

## Objective

Reliably decode source video, inspect its metadata, and expose timestamped
video samples using time-based sampling.

## Implemented

### Video Metadata

Implemented structured video metadata extraction containing:

```text
path
duration_seconds
fps
frame_count
width
height
```

Metadata is read from the source video using OpenCV.

Validation includes:

- missing file detection;
- video open failure;
- invalid FPS detection.

### Time-Based Sampling

Implemented timestamp generation based on a requested sampling rate.

Sampling is defined in time rather than by fixed frame intervals.

Example:

```text
2 FPS
→ 0.0, 0.5, 1.0, 1.5, ...
```

This keeps downstream processing independent of the source video's FPS.

### Sampled Frames

Each sampled frame preserves:

```text
requested_timestamp_seconds
actual_timestamp_seconds
frame_index
image
```

This allows downstream stages to retain both the requested sampling time
and the actual frame returned by the video decoder.

### Seek-Based Sampling

The current coarse sampler uses OpenCV seek operations through:

```text
CAP_PROP_POS_MSEC
```

The actual frame position is tracked using:

```text
CAP_PROP_POS_FRAMES
```

This approach is used for sparse time-based sampling in Phase 1.

Sequential frame reading may be introduced in later phases for dense
temporal analysis.

### Sampling Rate Protection

If the requested sampling FPS exceeds the source video FPS:

```text
requested FPS > source FPS
→ cap to source FPS
→ log warning
→ continue processing
```

This prevents unnecessary duplicate frame requests.

### Failure Handling

Individual failed frame reads are logged and skipped rather than causing
the complete video sampling operation to fail.

Video resources are always released after processing.

## Validation

Phase 1 was validated using both automated tests and a real
Flightradar24 screen recording.

Test video:

```text
middle_east1.mp4
```

Observed metadata:

```text
Duration: 33.9 seconds
FPS: 10.0
Frame count: 339
Resolution: 1906x984
```

At a sampling rate of 2 FPS:

```text
Samples: 68
Unique frame indices: 68
Monotonic frame progression: True
```

Requested and actual timestamps matched correctly in the validated video.

Visual sanity checks were also performed at several timestamps across the
recording and the extracted frames matched the expected positions in the
source video.

## Tests

Phase 1 completed with:

```text
13 pytest tests passing
Ruff checks passing
Real-video validation passing
Visual sanity check passing
```

## Definition of Done

Phase 1 is complete because:

- video files can be opened and validated;
- metadata is extracted correctly;
- sampling is based on time;
- sampled frames preserve timing and frame-index information;
- invalid sampling rates are handled;
- sampling above the source FPS is capped with a warning;
- frame-read failures are handled gracefully;
- automated tests pass;
- Ruff passes;
- metadata was validated against a real video;
- sampling was numerically validated;
- sampled frames were visually validated.

# 6. Phase 2 — UI Regions & Change Features

## Status

**Phase 2: COMPLETE.** Subphases 2.1-2.5 are complete and closure validation
has been reviewed and accepted. Phase 3 now builds on this completed evidence layer.

## Completed outcomes

### 2.1 UI Regions & Layout Mapping - COMPLETE

Frozen UIConfig/ROI geometry uses a canonical 1920x1080 reference. Existing ticket
header, aircraft image, scroll body, and map analysis regions are explicitly
mapped. Layout classification supports exact/proportional geometry; supplied
scale/offset transforms preserve partial/outside visibility for cropped frames.
Automatic layout recovery is deferred. Four map UI ignore rectangles exclude
SIFT features only, without masking metric pixels.

### 2.2 Basic ROI Change Metrics - COMPLETE

Public MAD and Changed Pixel Ratio compare matching uint8 grayscale/BGR arrays.
Safe grayscale difference is reused without implicit resizing or normalization.
CPR defaults to the experimental intensity threshold 10, with explicit overrides.
SSIM remains intentionally deferred; metrics do not interpret events.

### 2.3 Map Motion Compensation - COMPLETE

SIFT -> BF L2 KNN -> Lowe ratio -> RANSAC partial-affine estimation produces
B-to-A translation/scale evidence, rotation diagnostics, validity/reasons, and
raw versus overlap-aware aligned MAD/CPR. Static UI masks affect detection only;
rotation is checked but omitted from warping. Weak evidence is rejected.

Engineering conclusion: **dense consecutive-frame analysis is the preferred
production approach for map motion**, because large distant-frame displacement
can reduce geographic feature overlap. Canonical pan video achieved 124/124 valid
estimates with meaningful translation and near-unit scale. Zoom achieved 86/88
valid estimates and coherent below-/above-one scale changes. Rejected zoom pairs
and potentially increased interpolation residual MAD remain known limitations.
Thresholds remain provisional; no fallback or automatic tuning was added.

### 2.4 Ticket Scroll Detection - COMPLETE

Implemented pair-level measurement plus temporal scroll-state interpretation.
TicketScrollEvidence reuses four-ROI MAD/CPR and estimates A-to-B body translation
with float32 grayscale phase correlation and a Hann window. The pair helper uses
provisional activity/displacement/dominance/surrounding-stability criteria.

Offline temporal analysis returns QUIET, SCROLLING_UP, or SCROLLING_DOWN using
short centered context and bounded gap bridging. States describe active motion,
not the entire human interaction; pauses can be QUIET. No scroll events, start/end
timestamps, Investigation suppression, or broader cooldown context is implemented.
Those consumer-level responsibilities remain for later phases.

### 2.5 Real-Video Validation & Closure - COMPLETE

`scripts/validate_phase2.py` combines real-video regression with limited
cross-phenomenon checks using unchanged production APIs at source-frame density.

- Pan: 124/124 valid map estimates; translation with scale approximately one.
- Zoom: 86/88 valid; coherent scale changes below and above one.
- Ticket scroll: 18/83 pair positives; 65 QUIET / 11 UP / 7 DOWN states. Active
  states stay inside known scroll periods; quiet gaps are preserved.
- Cross-check: ticket-scroll logic yields 0/124 positives on pan and 0/88 on zoom,
  with all temporal states QUIET. Map motion during ticket scrolling is near
  identity (maximum translation 0.0252 pixels).
- No unexpected cross-talk or new production issue was observed.
- Closure verification: **365 pytest tests passed; Ruff passed**.

Videos, CSVs, debug PNGs, and `data/debug/map_motion/phase2_validation_report.txt`
remain local ignored evaluation evidence; no data artifacts are committed.

## Outcome and next phase

Phase 2 supplies validated region-change measurements, map-motion evidence,
raw/aligned metrics, ticket-scroll evidence, and active temporal scroll context
for later inclusion/exclusion/context reasoning. Working thresholds remain
provisional; these signals do not themselves decide Investigation Events.

Phase 3 consumes this completed evidence layer through separate Investigation
signals, candidates, temporal confirmation, and lifecycle construction. Phase 4
consumes those outputs; Phase 5 is the next planned phase.

---

# 7. Phase 3 — Investigation Event Detection

## Status

**Phase 3 - Investigation Event Detection: COMPLETE.** Subphases 3.1-3.6 are complete.
Phase 4 consumes its lifecycle intervals and effective presence.

## Completed outcomes

### 3.1 Investigation Signals & Rules - COMPLETE

Single-frame presence measures ticket structure independently of pair-level
change. `InvestigationPresenceEvidence` and `InvestigationSignals` retain their
separate meanings: current ticket appearance versus observed change/context.
Pair aggregation reuses ROI metrics, aligned map residual, route replacement, and
ticket-scroll APIs. Availability remains explicit; strength is not confidence.
Thresholds remain provisional and based on a small real-video calibration set.

### 3.2 Candidate Detection - COMPLETE

Presence transitions and meaningful ticket changes produce OPEN, SWITCH, CLOSE,
or NONE hypotheses. OPEN needs multiple ticket-change signals; SWITCH needs
bidirectional route replacement plus ticket change. Scroll is context rather than
a global veto; generic map residual/motion does not vote positively.

### 3.3 Temporal Confirmation - COMPLETE

Offline seconds-based look-ahead confirms consistent presence with sufficient
future coverage. Any later non-NONE candidate inside the provisional 0.7-second
window supersedes the earlier candidate. This filters the validated short
accidental opening without requiring repeated candidates or presence disappearance.
NONE is neutral; short endings remain insufficient. Event time stays separate from
confirmation time. No equivalent-candidate merging or online confirmation exists.

### 3.4 Investigation Lifecycle - COMPLETE

Confirmed semantic events build ordered Investigation intervals using candidate
boundary times. Startup-active videos begin at video start without synthetic OPEN.
NO + SWITCH starts a new interval; ACTIVE + OPEN replaces the current one. CLOSE
creates inactive gaps; any final active interval ends at video end. Inactive CLOSE
produces a diagnostic anomaly rather than a crash. Equal-time events preserve order.

### 3.5 Real-Video Validation & Phase 3 Closure - COMPLETE

The original closure covered three manual recordings; `scripts/validate_phase3.py`
now also includes the combined interaction regression and writes a concise closure
report and interval CSV. Opening
and direct switch, startup-active footage, and close/no-ticket/reopen scenarios
passed. Cross-layer checks verified presence, candidate/confirmation consistency,
candidate-time boundaries, superseded-event exclusion, inactive gaps, interval
ordering, and absence of unexpected lifecycle anomalies. Tests and Ruff passed.

The accidental OPEN was superseded by a later SWITCH and created no lifecycle
interval. These are representative/manual checks, not an accuracy benchmark;
working thresholds and the confirmation window remain provisional. Local videos
and validation/debug artifacts remain ignored evidence, not committed assets.

### 3.6 Lifecycle Regression Hardening - COMPLETE

During meaningful map motion (translation greater than 1 pixel or scale delta
greater than .01), SWITCH requires bidirectional route replacement plus two of
three MEDIUM/HIGH ticket signals. Outside that context one ticket signal remains
sufficient. This suppresses the observed false pan SWITCH without a blanket veto.

Bounded transient absence can be bridged up to a provisional .15 seconds, while
raw presence remains separately available. Effective presence feeds candidates,
confirmation, and stable views. The combined regression preserves actual aircraft
replacement, and close/reopen retains the real CLOSE boundary. Simultaneous
selection during substantial motion and genuine very short close/reopen remain
validation gaps.

## Output and next phase

Phase 3 outputs frozen Investigation intervals with chronological indexes,
start/end times and reasons, plus diagnostic anomalies. Presence/change separation,
pair hypotheses, temporal superseding, lifecycle construction, startup-active
initialization, and close/no-ticket/reopen support are complete for the validated MVP.
Aircraft identity, OCR, content extraction, and final canonical JSON remain later work.

The Jira high-level category **Visual Change & Interaction Analysis** (Phase 2.2-2.5
and Phase 3) can now be marked COMPLETE; this does not change Jira automatically.

Phase 4 builds evidence requests and annotations from these outputs.

---

# 8. Phase 4 — Investigation Extraction & Adaptive Sampling

## Status

**Phase 4 - Investigation Extraction & Adaptive Sampling: COMPLETE.**
Subphases 4.1-4.5 are complete. Next planned: Phase 5 (NOT STARTED).

### 4.1 Investigation Window Extraction - COMPLETE

Immutable windows preserve lifecycle core boundaries and add provisional .5-second
pre/post context, clipped to video bounds. Overlapping windows remain separate.
No synthetic Investigation behavior or decoding is added to production.

### 4.2 Adaptive Sampling - COMPLETE

Existing pair signals directly trigger dense sampling: ticket scroll, route visual
change, MEDIUM/HIGH header/image/body changes, valid translation greater than
1 pixel, or scale delta greater than .01. Map residual alone does not trigger;
there is no ActivityLevel abstraction. Dense expansion is .5 seconds before/after,
with touching/overlapping intervals merged as hysteresis. Sparse rate is 1 FPS;
dense rate is source FPS. Semantic window/core anchors remain exact. Long real
activity can legitimately remain dense; requests are not decoded frame identities.

### 4.3 Stable Ticket View Detection - COMPLETE

Effective presence, no detected scroll, and LOW ticket-body change must persist
continuously for a provisional .5-second dwell. Every qualifying maximal span is
retained, including intermediate pauses. Post-scroll provenance and right-censored
final spans remain explicit. Stability complements moving evidence and does not
establish OCR completeness or reading intent.

### 4.4 Evidence Candidate Timeline - COMPLETE

Every requested timestamp becomes exactly one immutable candidate. Sparse/dense
classification, dense reasons, coincident anchors, and stable annotations coexist.
No ranking, filtering, synthetic timestamps, or CV reruns occur. Stable ownership
must be unambiguous and every span must have existing candidate coverage.

### 4.5 End-to-End Validation, Documentation & Closure - COMPLETE

`scripts/validate_phase4.py` reuses production APIs and shared evaluation
preparation for seven lifecycle, scroll, pan, zoom, and combined recordings. It
checks preserved boundaries, exact candidate mapping, anchors, sampling reasons,
stable coverage/provenance, moving evidence, and Phase 3 regressions. Per-window
metrics and detailed validation results stay in local closure reports.

The full test suite, Ruff, Phase 3 regression validation, all required Phase 4
evaluators, and the closure invariants passed. Combined interaction retained the
correct lifecycle while scroll moving evidence, stable pauses, pan density,
post-pan sparse stability, and actual replacement remained covered. Pan was
translation-led; zoom remained dense with scale activity and route-mask cross-talk.
No algorithms or provisional defaults were changed for closure.

## Limitations and next phase

This is representative manual validation, not a benchmark; defaults remain
provisional. Footage is mostly canonical 1920x1080, 10 FPS, and one FR24 layout.
Higher FPS and other layouts, selection during substantial map motion, very short
close/reopen, cumulative slow ticket drift, and reversals without stable pauses
need further validation. LOW body change does not prove semantic completeness.
Dense candidate volume may be high during prolonged activity.

Phase 4 outputs windows, requested timestamps, stable views, and unified evidence
candidates. Resolving requests to physical frames is separate downstream work.
Final ranking/selection, OCR, semantic deduplication, and ticket-section extraction
are not implemented. **Next: Phase 5 - Role-Based Frame & Ticket-Section Selection**,
which remains NOT STARTED.

---

# 9. Phase 5 — Role-Based Frame & Ticket-Section Selection

Status: NEXT PLANNED; NOT STARTED.

## Objective

Select the best evidence for each downstream task.

## Planned roles

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

Roles may be refined after inspecting real videos.

## Ticket selection

Evaluate:

```text
sharpness
stability
readability
loaded state
section coverage
```

N/A is valid loaded data.

## Route selection

Evaluate:

```text
route visibility
selected aircraft visibility
map stability
viewport coverage
route clipping
```

## Aircraft image selection

Evaluate:

```text
image loaded
sharpness
crop completeness
placeholder
occlusion
```

## Deduplication

Near-identical visual evidence can be removed.

Different ticket scroll states must remain if they reveal new information.

## Done when

The selected evidence set is compact but covers all materially useful visible information.

---

# 10. Phase 6 — OCR + Complete Ticket Extraction

## Objective

Convert ticket evidence into structured observations.

## Pipeline

```text
ticket evidence
→ OCR
→ parsing
→ normalization
→ cross-frame consensus
→ structured ticket record
```

## Requirements

- OCR semantic regions, not full screen;
- retain raw OCR;
- support multiple ticket scroll states;
- distinguish static and dynamic fields;
- preserve additional ticket fields;
- use VLM only as targeted fallback.

## Dynamic fields

Examples:

```text
altitude
speed
vertical speed
track
position
```

These may legitimately differ across timestamps.

## Done when

All readable ticket sections can be merged into one Investigation-level structured record with uncertainty and provenance preserved.

---

# 11. Phase 7 — Route Extraction & Direction

## Objective

Extract route geometry and derive reliable direction.

## Direction classes

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

## Candidate CV pipeline

```text
map crop
→ route color segmentation
→ cleanup
→ component analysis
→ aircraft endpoint association
→ skeletonization
→ pruning
→ ordered polyline
```

## Important rules

- do not assume largest component;
- detect clipping;
- prefer deterministic geometry for direction;
- reduce confidence during turns;
- keep origin/destination inference conservative.

## Done when

Direction is consistently correct on representative route cases and route ambiguity is surfaced rather than hidden.

---

# 12. Phase 8 — VLM Assistance & Evidence Fusion

## Objective

Resolve Investigation-level information from all evidence sources.

## VLM responsibilities

Potential tasks:

```text
semantic enrichment
difficult visual text
livery/aircraft image interpretation
ambiguous visual evidence
final Investigation cross-check
```

## Fusion priorities

Examples:

```text
observed > inferred
multi-frame OCR > single visual guess
route geometry > VLM direction guess
observed origin/destination > inferred origin/destination
```

## Final cross-check

The model should flag:

```text
confirmed fields
suspected errors
conflicts
missed visible information
```

It should not rewrite the canonical record freely.

## Runtime adaptation

Use model capability + hardware capability to determine which VLM tasks are enabled.

## Done when

VLM usage improves ambiguous cases without degrading deterministic high-confidence fields.

---

# 13. Phase 9 — Tracked Aircraft Entity Matching

## Objective

Link revisits to the same operational target.

## Evidence strength

Very strong:

```text
registration exact match
```

Strong:

```text
flight number
callsign
route continuity
position progression
```

Medium:

```text
origin/destination
aircraft type
compatible route
```

Weak:

```text
airline
time proximity
```

## Decisions

```text
MATCH
AMBIGUOUS
NO_MATCH
```

## Entity model

Static-ish information accumulates evidence.

Time-varying information remains a timeline.

## Done when

Revisits are linked correctly without forcing uncertain matches.

---

# 14. Phase 10 — Canonical Output & Renderers

## Objective

Produce the final structured session record and user-facing reports.

## Outputs

```text
session.json
operational_log.txt
detailed_report.txt
```

Optional:

```text
debug/
```

## Canonical record

Expected top-level areas:

```text
schema_version
session
investigations
entities
processing_warnings
processing_errors
```

## Operational log

Concise, chronological, readable.

## Detailed report

Includes:

```text
provenance
confidence
conflicts
additional ticket fields
evidence references
analysis notes
```

## Renderer rule

Renderers format only.

No inference or model calls.

## Done when

A human reviewer can compare the generated outputs against the video and confirm that the system accurately describes the operator’s activity and visible flight information.

---

# 15. Planned Test Organization

The repository may evolve toward:

```text
tests/
├── unit/
├── integration/
├── fixtures/
├── mocks/
└── conftest.py
```

Do not create all directories before they are needed.

Mocks belong under tests, including any mock VLM implementation.

---

# 16. Current Repository Foundation

Current intended structure:

```text
Events_Diary/
├── docs/
│   ├── system_design.md
│   └── project_plan.md
├── src/
│   └── events_diary/
│       └── __init__.py
├── tests/
├── .gitignore
├── .python-version
├── pyproject.toml
├── README.md
└── uv.lock
```

Current baseline:

```text
Python 3.10
uv
pytest
ruff
```

The project should remain intentionally lean as implementation begins.
