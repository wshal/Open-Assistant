"""Knowledge base module — document storage and indexing."""

from pathlib import Path


def ensure_knowledge_dirs():
    """Create knowledge base directories."""
    dirs = [
        "knowledge/documents",
        "knowledge/templates",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


# Auto-create on import
ensure_knowledge_dirs()