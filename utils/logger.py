"""
Logging configuration - console (Rich) + file (RotatingFileHandler).
"""

import os
import sys
import logging
import io
from logging.handlers import RotatingFileHandler
from pathlib import Path

_initialized = False


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    global _initialized
    log_level = getattr(logging, level.upper(), logging.INFO)
    if not _initialized:
        _setup_root_logger(log_level)
        _initialized = True
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    return logger


def _setup_root_logger(level: int):
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    # Console handler with Rich
    try:
        from rich.logging import RichHandler
        from rich.console import Console

        _file = None
        if sys.platform == "win32":
            try:
                _file = io.TextIOWrapper(
                    sys.stdout.buffer, encoding="utf-8", errors="replace"
                )
            except Exception:
                _file = None

        _console = Console(file=_file, highlight=False, markup=False, soft_wrap=True)
        console_handler = RichHandler(
            console=_console,
            rich_tracebacks=False,
            markup=False,
            show_time=True,
            show_path=False,
        )
        console_handler.setLevel(level)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(console_handler)
    except Exception:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
            )
        )
        root.addHandler(console_handler)

    # File handler
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
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)-25s | %(funcName)-20s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root.addHandler(file_handler)
    except Exception as e:
        root.warning(f"Could not create log file: {e}")

    # Suppress noise
    for _name in [
        "urllib3",
        "httpx",
        "httpcore",
        "aiohttp",
        "chromadb",
        "onnxruntime",
        "sentence_transformers",
        "faster_whisper",
        "torch",
        "PIL",
    ]:
        logging.getLogger(_name).setLevel(logging.WARNING)


def get_log_file_path() -> str:
    return str(Path("logs") / "openassist.log")
