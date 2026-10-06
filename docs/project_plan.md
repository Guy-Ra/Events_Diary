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

## Definition of Done

Phase 0 is complete when:

- fresh clone + `uv sync` works;
- `events_diary` imports correctly;
- pytest runs successfully;
- Ruff runs successfully;
- configuration foundation exists;
- basic logging/runtime inspection exists;
- first entry point runs;
- README and core design docs exist;
- no Flightradar24 analysis logic has been introduced yet.

---

# 5. Phase 1 — Video Decode & Time-Based Sampling

## Objective

Reliably decode source video and expose time-based samples.

## Main requirements

Read:

```text
duration
source FPS
resolution
frame count
timestamps
```

Sampling must be based on time rather than fixed frame intervals.

Timestamp is the primary temporal representation.

## Calibration task

Compare several sampling rates on real footage:

```text
1 FPS
2 FPS
4 FPS
8 FPS
```

Determine the lowest rate that still captures the shortest meaningful operator activity.

## Expected outputs

A structured video metadata object and sampled frames with timestamps.

Exact data models should be defined immediately before implementation.

## Tests

- normal MP4;
- variable frame rate if available;
- short video;
- video starts during active ticket;
- video ends during active ticket;
- corrupted/unreadable frame behavior.

## Done when

The same source video produces reliable timestamped samples and inspectable metadata.

---

# 6. Phase 2 — UI Regions & Change Features

## Objective

Measure visual changes in semantic regions.

No Investigation decisions yet.

## Main work

Define reference-resolution UI regions for:

```text
ticket
map
route-related area
aircraft image
selected-aircraft context
static UI masks
```

The ticket outer panel can be spatially stable while its internal content scrolls.

The full map, including the right side, is useful.

Browser chrome/static controls should be ignored.

## Candidate features

- MAD;
- changed-pixel ratio;
- SSIM;
- grayscale/downscaled comparison;
- map motion estimate;
- ticket vertical scroll estimate.

## Map compensation

Estimate:

```text
translation
scale
```

Rotation is not required for the current north-up UI.

## Ticket scroll feature

Detect vertical motion within the ticket while aircraft identity/header remains stable.

## Tests

Create representative frame pairs for:

```text
no change
ticket text update
new selected aircraft
ticket scroll
map pan
map zoom
route change
aircraft image load
```

## Done when

The system produces interpretable change features that distinguish meaningful UI changes from pan/zoom/scroll artifacts.

---

# 7. Phase 3 — Investigation Event Detection

## Objective

Detect periods in which the operator investigates a selected aircraft.

## State model

```text
NO_INVESTIGATION
CANDIDATE_INVESTIGATION
ACTIVE_INVESTIGATION
```

## Detection strategy

Use:

- strong evidence;
- supporting evidence;
- negative/explanatory evidence;
- temporal stabilization.

Strong candidate signals:

```text
header change
origin/destination change
aircraft detail change
route change after map compensation
```

Ticket scroll must not trigger a new Investigation.

## Required timestamps

Store:

```text
candidate_detected_at
confirmed_at
```

## End reasons

```text
new_selection
ticket_closed
video_end
```

## Tests

Include:

```text
single aircraft
direct aircraft switch
brief accidental click
ticket scrolling
map pan during Investigation
video begins active
video ends active
```

## Done when

Detected Investigation boundaries match human review on representative examples.

---

# 8. Phase 4 — Investigation Extraction & Adaptive Sampling

## Objective

Re-read relevant Investigation windows from the original video and capture richer evidence.

## Context window

Include time before detection and after event end.

Conceptual segments:

```text
PRE_TRANSITION
TRANSITION
STABLE
EXIT
```

## Adaptive strategy

Dense sampling during:

```text
selection transition
ticket loading
ticket scrolling
map/route transition
```

Sparse sampling during stable periods.

Use hysteresis rather than constantly switching sampling modes.

## Ticket scroll requirement

Each stable scroll stop that reveals new ticket content should be retained as candidate evidence.

## Done when

Each Investigation has enough candidate frames to reconstruct useful ticket/map/aircraft evidence without retaining excessive redundant frames.

---

# 9. Phase 5 — Role-Based Frame & Ticket-Section Selection

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
