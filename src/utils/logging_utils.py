"""Configuração de logging."""

from __future__ import annotations

import logging

from src.config import LoggingConfig


def configure_logging(logging_config: LoggingConfig) -> None:
    logging.basicConfig(
        level=getattr(logging, logging_config.level.upper(), logging.INFO),
        format=logging_config.format,
    )
