"""Tests for Events Diary logging configuration."""

import logging

from events_diary.config import AppConfig
from events_diary.logging_config import configure_logging


def test_configure_logging_uses_configured_level(monkeypatch) -> None:
    """Verify that logging configuration uses values from AppConfig."""

    captured = {}

    def fake_basic_config(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(logging, "basicConfig", fake_basic_config)

    config = AppConfig(log_level="DEBUG")
    configure_logging(config)

    assert captured["level"] == "DEBUG"
    assert captured["format"] == (
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )