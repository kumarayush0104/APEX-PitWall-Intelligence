"""
Loguru-based logging configuration for the APEX system.

Design decisions:
- Colorised console output with compact timestamp (HH:MM:SS for readability)
- Rotating file logs with automatic compression and retention
- InterceptHandler redirects stdlib logging (uvicorn, httpx, transformers) to loguru
- Diagnose mode enabled in DEBUG for rich exception tracebacks
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from app.config import Settings


class _InterceptHandler(logging.Handler):
    """
    Bridge between Python's stdlib ``logging`` and Loguru.

    All libraries that use ``logging.getLogger(__name__)`` (uvicorn, httpx,
    transformers, etc.) will have their output re-routed through Loguru after
    this handler is installed.  This gives us a single, consistently formatted
    log stream.
    """

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D102
        # Map stdlib level name → loguru level name
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Walk up the call stack to find the original caller (skip logging internals)
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # type: ignore[assignment]
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def configure_logging(settings: "Settings") -> None:
    """
    Configure the application-wide logging pipeline.

    Call this **once** at application startup (inside the lifespan handler),
    *before* any other module imports that might emit log records.

    Args:
        settings: The resolved application settings instance.
    """
    # Remove the default loguru sink (writes raw to stderr without format)
    logger.remove()

    # ------------------------------------------------------------------
    # Console sink — colourised, compact format
    # ------------------------------------------------------------------
    _CONSOLE_FORMAT = (
        "<green>{time:HH:mm:ss}</green> "
        "<dim>│</dim> "
        "<level>{level: <8}</level> "
        "<dim>│</dim> "
        "<cyan>{name}</cyan><dim>:</dim><cyan>{line}</cyan> "
        "<dim>│</dim> "
        "{message}"
    )
    logger.add(
        sys.stderr,
        format=_CONSOLE_FORMAT,
        level=settings.LOG_LEVEL,
        colorize=True,
        backtrace=settings.DEBUG,
        diagnose=settings.DEBUG,   # show local variables in tracebacks
        catch=True,
    )

    # ------------------------------------------------------------------
    # File sink — structured, with rotation + compression
    # ------------------------------------------------------------------
    if settings.LOG_TO_FILE:
        _FILE_FORMAT = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} "
            "│ {level: <8} "
            "│ {name}:{line} "
            "│ {message}"
        )
        log_file_pattern = str(settings.LOG_DIR / "apex_{time:YYYY-MM-DD}.log")
        logger.add(
            log_file_pattern,
            format=_FILE_FORMAT,
            level=settings.LOG_LEVEL,
            rotation=settings.LOG_ROTATION,
            retention=settings.LOG_RETENTION,
            compression="zip",
            backtrace=True,
            diagnose=True,
            encoding="utf-8",
            catch=True,
        )

    # ------------------------------------------------------------------
    # Intercept stdlib logging from third-party libraries
    # ------------------------------------------------------------------
    _INTERCEPTED_LOGGERS = [
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "fastapi",
        "httpx",
        "httpcore",
        "multipart",
    ]
    # Install our bridge at the root level
    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)

    for name in _INTERCEPTED_LOGGERS:
        lib_logger = logging.getLogger(name)
        lib_logger.handlers = [_InterceptHandler()]
        lib_logger.propagate = False

    logger.debug(
        "Logging configured | level={} | file_sink={} | debug={}",
        settings.LOG_LEVEL,
        settings.LOG_TO_FILE,
        settings.DEBUG,
    )
