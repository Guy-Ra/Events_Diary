"""Application configuration definitions.

This module contains configuration objects used by Events Diary.
Configuration values are kept separate from runtime behavior so that
settings can be managed centrally and extended as the project grows.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    """Application-level configuration.

    Attributes:
        log_level:
            Logging level used by the application, such as
            "DEBUG", "INFO", "WARNING", or "ERROR".

    The configuration is immutable after creation to prevent accidental
    modification during application execution.
    """

    log_level: str = "INFO"