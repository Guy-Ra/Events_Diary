"""Command-line entry point for Events Diary.

Running:

    uv run python -m events_diary

executes the main function defined in this module.
"""

import logging

from events_diary.config import AppConfig
from events_diary.logging_config import configure_logging
from events_diary.runtime import get_runtime_profile

logger = logging.getLogger(__name__)


def main() -> None:
    """Initialize the Events Diary application.

    At this stage of the project, the entry point initializes
    application configuration, logging, and basic runtime inspection.

    Future pipeline execution will be connected here as the project
    progresses through its implementation phases.
    """

    config = AppConfig()
    configure_logging(config)

    runtime = get_runtime_profile()

    logger.info("Events Diary initialized successfully.")
    logger.info(
        "Runtime: %s | Python %s | %s",
        runtime.operating_system,
        runtime.python_version,
        runtime.machine,
    )


if __name__ == "__main__":
    main()