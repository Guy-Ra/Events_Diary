"""Logging configuration for Events Diary.

This module is responsible for configuring Python's logging system
using values provided by the application configuration.
"""

import logging

from events_diary.config import AppConfig


def configure_logging(config: AppConfig) -> None:
    """Configure application-wide logging.

    Args:
        config:
            Application configuration containing the desired
            logging level.

    The current configuration sends log messages to the console using
    a consistent timestamped format.
    """

    logging.basicConfig(
        level=config.log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )