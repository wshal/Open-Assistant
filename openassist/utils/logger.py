"""
Logging configuration - console (Rich) + file (RotatingFileHandler).
FIXED: Robust encoding fallbacks for Windows terminals.
"""

import os
import sys
import logging
import io
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Track if we've already set up the root logger
_initialized = False


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Get or create a logger with both console and file output.
    """
    global _initialized

    log_level = getattr(logging, level.upper(), logging.INFO)

    # Set up root logger once
    if not _initialized:
        _setup_root_logger(log_level)
        _initialized = True

    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    return logger


def _setup_root_logger(level: int):
    """Configure the root logger with console + file handlers."""
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    # Handler 1: Console (Rich / Fallback)
    _console_success = False
    try:
        from rich.logging import RichHandler
        from rich.console import Console

        # On Windows, try to force UTF-8 for the console
        _file = None
        if sys.platform == "win32":
            try:
                if hasattr(sys.stdout, "reconfigure"):
                    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
            try:
                # Wrap stdout buffer in a UTF-8 wrapper with 'replace' error handling
                _file = io.TextIOWrapper(
                    sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
                )
            except (AttributeError, io.UnsupportedOperation):
                _file = None

        _console = Console(
            file=_file,
            highlight=False,
            markup=True,
            soft_wrap=True
        )

        console_handler = RichHandler(
            console=_console,
            rich_tracebacks=True,
            markup=True,
            show_time=True,
            show_path=False,
        )
        console_handler.setLevel(level)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(console_handler)
        _console_success = True

    except Exception as e:
        # Fallback to plain logging if Rich fails
        _console_success = False

    if not _console_success:
        # Final safety fallback for problematic terminals
        # Uses sys.stdout with 'ignore' or 'replace' to avoid UnicodeEncodeError
        try:
            stream = io.TextIOWrapper(sys.stdout.buffer, encoding='ascii', errors='replace')
            console_handler = logging.StreamHandler(stream)
        except Exception:
            console_handler = logging.StreamHandler()
            
        console_handler.setLevel(level)
        console_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(message)s",
            datefmt="%H:%M:%S"
        ))
        root.addHandler(console_handler)

    # Handler 2: File (Detailed)
    try:
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            filename=log_dir / "openassist.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)-25s | %(funcName)-20s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        root.addHandler(file_handler)
    except Exception as e:
        root.warning(f"Could not create log file: {e}")

    # Suppress noise
    noisy = ["urllib3", "httpx", "httpcore", "aiohttp", "chromadb", "onnxruntime", "sentence_transformers", "faster_whisper", "ctranslate2", "torch", "PIL", "pynput"]
    for _name in noisy:
        logging.getLogger(_name).setLevel(logging.WARNING)


def get_log_file_path() -> str:
    return str(Path("logs") / "openassist.log")