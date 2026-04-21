# data/__init__.py
"""Data directory management."""

from pathlib import Path


def ensure_data_dirs():
    """Create all data directories."""
    dirs = [
        "data/vectordb",
        "data/cache",
        "logs",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


ensure_data_dirs()