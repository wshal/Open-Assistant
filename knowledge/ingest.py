"""
Knowledge Ingestion Utility — OpenAssist v4.2
==============================================
Indexes PDFs and Q&A files into the RAG (ChromaDB) vector database.

Supported input formats:
  • PDF  — any PDF in knowledge/documents/   (uses PyMuPDF — already installed)
  • JSON — Q&A pairs:  [{"q": "...", "a": "..."}]  or  {"question": ..., "answer": ...}
  • TXT  — plain Q&A: lines of "Q: ...\nA: ...\n"
  • MD   — Markdown docs (already handled by add_directory)

Usage (standalone):
    python -m knowledge.ingest                      # index everything in knowledge/documents/
    python -m knowledge.ingest path/to/file.pdf
    python -m knowledge.ingest path/to/qa.json

Called automatically at app warmup via app._background_warmup().
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Tuple

from utils.logger import setup_logger
from core.constants import DOCS_DIR

logger = setup_logger(__name__)

_DOCUMENTS_DIR = Path(DOCS_DIR)


# ---------------------------------------------------------------------------
# PDF extractor
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from a PDF using PyMuPDF (fitz). Returns plain text."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(pdf_path))
        pages = []
        for page in doc:
            pages.append(page.get_text())
        doc.close()
        return "\n\n".join(pages)
    except Exception as e:
        logger.warning(f"[Ingest] PDF extraction failed for {pdf_path.name}: {e}")
        return ""


# ---------------------------------------------------------------------------
# Q&A extractor  (JSON + plain TXT)
# ---------------------------------------------------------------------------

def extract_qa_pairs(file_path: Path) -> List[Tuple[str, str]]:
    """
    Parse a Q&A file and return [(question, answer), ...].

    Supported formats:
      JSON array:   [{"q": "...", "a": "..."}]
                    [{"question": "...", "answer": "..."}]
      JSON object:  {"pairs": [...]}
      Plain TXT:    Q: ...\nA: ...\n  (blank-line separated blocks)
    """
    suffix = file_path.suffix.lower()
    pairs: List[Tuple[str, str]] = []

    if suffix == ".json":
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data = data.get("pairs") or data.get("qa") or []
            for item in (data or []):
                q = item.get("q") or item.get("question") or ""
                a = item.get("a") or item.get("answer") or ""
                if q and a:
                    pairs.append((q.strip(), a.strip()))
        except Exception as e:
            logger.warning(f"[Ingest] JSON parse failed for {file_path.name}: {e}")

    elif suffix == ".txt":
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        # Split into blocks separated by blank lines
        blocks = re.split(r"\n\s*\n", text.strip())
        for block in blocks:
            lines = block.strip().splitlines()
            q_lines, a_lines = [], []
            for line in lines:
                if re.match(r"^[Qq]\s*[:\.]\s*", line):
                    q_lines.append(re.sub(r"^[Qq]\s*[:\.]\s*", "", line).strip())
                elif re.match(r"^[Aa]\s*[:\.]\s*", line):
                    a_lines.append(re.sub(r"^[Aa]\s*[:\.]\s*", "", line).strip())
            if q_lines and a_lines:
                pairs.append((" ".join(q_lines), " ".join(a_lines)))

    return pairs


def qa_pairs_to_chunks(pairs: List[Tuple[str, str]]) -> List[str]:
    """
    Convert Q&A pairs into retrieval-optimised text chunks.

    Format: "Q: {question}\nA: {answer}"
    This format means the question text is included in the chunk, so
    when a user asks something similar, the vector search will find it
    even if they phrase it differently.
    """
    return [f"Q: {q}\nA: {a}" for q, a in pairs if q and a]


# ---------------------------------------------------------------------------
# Main ingestion entry-point (called by app warmup + standalone)
# ---------------------------------------------------------------------------

def ingest_all(rag_engine, documents_dir: Path = _DOCUMENTS_DIR) -> None:
    """
    Index everything in the documents directory into the RAG engine.

    Steps:
      1. TXT/MD/code files  → add_directory() (already implemented)
      2. PDF files          → extract text → add as chunks
      3. JSON/TXT Q&A files → extract pairs → add as "Q: ...\nA: ..." chunks
    """
    if not rag_engine or not getattr(rag_engine, "enabled", False):
        logger.debug("[Ingest] RAG disabled — skipping ingestion")
        return

    documents_dir = Path(documents_dir)
    documents_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Standard text/code files (existing logic in add_directory)
    rag_engine.add_directory(str(documents_dir))

    # Step 2 & 3: PDFs and Q&A files (new — not handled by add_directory)
    rag_engine._ensure_loaded()
    if not getattr(rag_engine, "enabled", False) or rag_engine.collection is None:
        return

    import hashlib

    for path in documents_dir.rglob("*"):
        suffix = path.suffix.lower()

        # PDF → extract text, chunk, index
        if suffix == ".pdf":
            text = extract_text_from_pdf(path)
            if not text.strip():
                continue
            chunks = rag_engine._chunk_text(text, rag_engine.chunk_size, rag_engine.chunk_overlap)
            _index_chunks(rag_engine, chunks, source=str(path), label=path.stem)

        # JSON / TXT Q&A files → extract pairs, index as Q+A chunks
        elif suffix in {".json", ".txt"} and _looks_like_qa_file(path):
            pairs = extract_qa_pairs(path)
            if not pairs:
                continue
            chunks = qa_pairs_to_chunks(pairs)
            _index_chunks(rag_engine, chunks, source=str(path), label=f"qa:{path.stem}")
            logger.info(f"[Ingest] Indexed {len(chunks)} Q&A pairs from {path.name}")


def _looks_like_qa_file(path: Path) -> bool:
    """Heuristic: peek at the file to see if it looks like a Q&A source."""
    try:
        head = path.read_text(encoding="utf-8", errors="ignore")[:500]
        return bool(
            re.search(r"\bquestion\b|\b\"q\"\s*:", head, re.IGNORECASE)
            or re.search(r"^[Qq]\s*[:\.]\s*", head, re.MULTILINE)
        )
    except Exception:
        return False


def _index_chunks(rag_engine, chunks: List[str], source: str, label: str) -> None:
    """Add a list of text chunks to the RAG collection, skipping duplicates."""
    import hashlib
    if not chunks or rag_engine.collection is None:
        return

    documents, metadatas, ids = [], [], []
    for i, chunk in enumerate(chunks):
        content_hash = hashlib.sha256(chunk.encode()).hexdigest()[:8]
        chunk_id = f"{label}_{i}_{content_hash}"
        documents.append(chunk)
        metadatas.append({"source": source, "chunk": i})
        ids.append(chunk_id)

    # Skip existing IDs (same dedup logic as add_directory)
    try:
        existing = rag_engine.collection.get(ids=ids)
        existing_ids = set(existing.get("ids", []))
    except Exception:
        existing_ids = set()

    new_docs = [(d, m, i) for d, m, i in zip(documents, metadatas, ids) if i not in existing_ids]
    if not new_docs:
        logger.debug(f"[Ingest] All chunks from '{label}' already indexed — skipping")
        return

    nd, nm, ni = zip(*new_docs)
    embeddings = rag_engine._embed_fn(list(nd))
    rag_engine.collection.add(
        documents=list(nd),
        metadatas=list(nm),
        ids=list(ni),
        embeddings=embeddings,
    )
    logger.info(f"[Ingest] ✅ {len(nd)} new chunks indexed from '{label}'")


# ---------------------------------------------------------------------------
# Standalone entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Allow: python -m knowledge.ingest [optional_path]
    from core.config import Config
    from ai.rag import RAGEngine

    config = Config()
    rag = RAGEngine(config)

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else _DOCUMENTS_DIR
    if target.is_file():
        rag._ensure_loaded()
        suffix = target.suffix.lower()
        if suffix == ".pdf":
            text = extract_text_from_pdf(target)
            chunks = rag._chunk_text(text, rag.chunk_size, rag.chunk_overlap)
            _index_chunks(rag, chunks, source=str(target), label=target.stem)
        elif suffix in {".json", ".txt"}:
            pairs = extract_qa_pairs(target)
            chunks = qa_pairs_to_chunks(pairs)
            _index_chunks(rag, chunks, source=str(target), label=f"qa:{target.stem}")
        else:
            rag.add_directory(str(target.parent))
    else:
        ingest_all(rag, target)

    print(f"Done. Total chunks in DB: {rag.collection.count() if rag.collection else 0}")
