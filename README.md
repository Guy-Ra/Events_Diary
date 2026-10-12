# Events Diary

Local visual intelligence system for analyzing Flightradar24 screen recordings using OCR, computer vision, and VLMs to generate structured aircraft event logs.

## Project Status

**Phase 2 - UI Regions & Change Features is complete.**

**Next planned: Phase 3 - Investigation Event Detection.**

The visual foundation now includes UI/layout mapping, ROI change metrics, map
motion compensation, and pair-level/temporal ticket-scroll analysis, validated on
real videos with cross-phenomenon checks. Investigation detection remains planned.
See the [project plan](docs/project_plan.md) and [system design](docs/system_design.md)
for implementation details and provisional-threshold limitations.

## Goals

The system is intended to:

- detect aircraft investigation events in Flightradar24 screen recordings;
- extract structured information from the aircraft ticket;
- analyze route geometry and movement direction;
- use OCR, computer vision, and selective VLM assistance;
- identify repeated investigations of the same aircraft;
- generate a canonical structured session record;
- render a concise operational log and a detailed report.

## Project Structure

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

## Development Environment

Current baseline:

- Python 3.10
- uv
- pytest
- Ruff

## Setup

Clone the repository:

```bash
git clone https://github.com/Guy-Ra/Events_Diary.git
cd Events_Diary
```

Install the project and development dependencies:

```bash
uv sync
```

Verify the package:

```bash
uv run python -c "import events_diary; print(events_diary)"
```

Run tests:

```bash
uv run pytest
```

Run Ruff:

```bash
uv run ruff check .
```

## Documentation

Detailed system design:

```text
docs/system_design.md
```

Implementation roadmap:

```text
docs/project_plan.md
```

## Development Approach

The project is developed incrementally.

Each phase follows the same process:

```text
Plan
→ Implement
→ Unit tests
→ Real-example validation
→ Integration test
→ Regression tests
→ Refactor and document
```

The goal is to keep the implementation modular, testable, and evidence-driven without introducing unnecessary abstractions.