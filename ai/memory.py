"""
Long-Term Semantic Memory — P2.4
Uses ChromaDB (local, persistent, encrypted-at-rest via OS file permissions)
to store and retrieve past session Q&A pairs as vector embeddings.

Storage: C:\\Users\\<user>\\.openassist\\chroma_db
Privacy: ChromaDB runs fully offline — no data leaves the machine.
"""

import os
import pathlib
import threading
import time
from typing import List, Optional, Tuple

from utils.logger import setup_logger
from utils.platform_utils import PlatformInfo

logger = setup_logger(__name__)

_DB_DIR = (
    PlatformInfo.get_app_data_dir() / "chroma_db"
    if PlatformInfo.IS_FROZEN
    else pathlib.Path("data") / "chroma_db"
)
_COLLECTION_NAME = "session_memory"
_MAX_RESULTS = 3
_RELEVANCE_THRESHOLD = 0.55   # cosine similarity floor (0=unrelated, 1=identical)


class LongTermMemory:
    """
    P2.4: Persistent semantic memory backed by ChromaDB.

    Thread-safe — all ChromaDB calls are protected by a lock so it can be
    used from both the Qt main thread and the async AI loop.
    """

    def __init__(self, config):
        self._config = config
        self._enabled: bool = bool(config.get("ai.memory.enabled", True))
        self._max_results: int = int(
            config.get("ai.memory.max_results", _MAX_RESULTS)
        )
        self._threshold: float = float(
            config.get("ai.memory.relevance_threshold", _RELEVANCE_THRESHOLD)
        )
        self._lock = threading.Lock()
        self._client = None
        self._collection = None
        self._ready = False

        if self._enabled:
            threading.Thread(
                target=self._initialize,
                daemon=True,
                name="memory-init",
            ).start()
        else:
            logger.info("[LongTermMemory] Disabled via config — skipping init")

    # ------------------------------------------------------------------
    # Initialisation (background)
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        try:
            import chromadb  # noqa: F401

            db_path = pathlib.Path(
                self._config.get("ai.memory.db_path", str(_DB_DIR))
            )
            db_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"[LongTermMemory] Initialising ChromaDB at: {db_path}")

            client = chromadb.PersistentClient(path=str(db_path))
            collection = client.get_or_create_collection(
                name=_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )

            with self._lock:
                self._client = client
                self._collection = collection
                self._ready = True

            count = collection.count()
            logger.info(
                f"[LongTermMemory] ✅ Ready — {count} memories in store "
                f"(db={db_path})"
            )
        except ImportError:
            logger.warning(
                "[LongTermMemory] chromadb not installed — "
                "run: pip install chromadb"
            )
        except Exception as e:
            self._ready = False
            self._enabled = False
            logger.warning(f"[LongTermMemory] Init skipped: {e}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        return self._ready

    def store(
        self,
        session_id: str,
        query: str,
        response: str,
        mode: str = "general",
    ) -> None:
        """
        Persist a Q&A pair to long-term memory.
        Called asynchronously — fails silently so it never blocks the AI loop.
        """
        if not self._enabled or not self._ready:
            return
        if not query or not response:
            return

        def _do_store():
            try:
                doc = f"Q: {query.strip()}\nA: {response.strip()}"
                doc_id = f"{session_id}_{int(time.time() * 1000)}"
                with self._lock:
                    if not self._collection:
                        return
                    self._collection.upsert(
                        documents=[doc],
                        ids=[doc_id],
                        metadatas=[{
                            "session_id": session_id,
                            "mode": mode,
                            "ts": str(int(time.time())),
                            "query_preview": query[:80],
                        }],
                    )
                logger.debug(
                    f"[LongTermMemory] Stored memory id={doc_id} "
                    f"({len(doc)} chars, mode={mode})"
                )
            except Exception as e:
                logger.warning(f"[LongTermMemory] Store failed: {e}")

        threading.Thread(target=_do_store, daemon=True, name="memory-store").start()

    def query(self, query: str, mode: str = "") -> List[str]:
        """
        Retrieve the top-k most semantically relevant past memories for a query.
        Returns a list of formatted strings ready to inject into the system prompt.
        """
        if not self._enabled or not self._ready:
            return []
        if not query:
            return []

        try:
            with self._lock:
                if not self._collection:
                    return []
                count = self._collection.count()

            if count == 0:
                logger.debug("[LongTermMemory] No memories stored yet")
                return []

            n = min(self._max_results, count)
            where = {"mode": mode} if mode else None

            with self._lock:
                results = self._collection.query(
                    query_texts=[query],
                    n_results=n,
                    where=where,
                    include=["documents", "distances", "metadatas"],
                )

            docs = (results.get("documents") or [[]])[0]
            distances = (results.get("distances") or [[]])[0]
            metas = (results.get("metadatas") or [[]])[0]

            relevant: List[str] = []
            for doc, dist, meta in zip(docs, distances, metas):
                # ChromaDB cosine distance: 0=identical, 2=opposite; similarity = 1 - dist/2
                similarity = max(0.0, 1.0 - dist / 2.0)
                if similarity < self._threshold:
                    logger.debug(
                        f"[LongTermMemory] Skipping memory (similarity={similarity:.2f} "
                        f"< threshold={self._threshold})"
                    )
                    continue
                ts_str = meta.get("ts", "?")
                try:
                    ts_str = time.strftime(
                        "%Y-%m-%d", time.localtime(int(ts_str))
                    )
                except Exception:
                    pass
                relevant.append(
                    f"[Memory from {ts_str}]\n{doc}"
                )
                logger.debug(
                    f"[LongTermMemory] Hit: similarity={similarity:.2f}, "
                    f"preview='{meta.get('query_preview', '')[:60]}'"
                )

            logger.info(
                f"[LongTermMemory] Query returned {len(relevant)}/{n} "
                f"relevant memories for: '{query[:60]}'"
            )
            return relevant

        except Exception as e:
            logger.warning(f"[LongTermMemory] Query failed: {e}")
            return []

    def count(self) -> int:
        """Return total number of stored memories."""
        try:
            with self._lock:
                if not self._collection:
                    return 0
                return self._collection.count()
        except Exception:
            return 0

    def clear(self) -> None:
        """Delete all stored memories (e.g., on factory reset)."""
        try:
            with self._lock:
                if self._client and self._collection:
                    self._client.delete_collection(_COLLECTION_NAME)
                    self._collection = self._client.get_or_create_collection(
                        name=_COLLECTION_NAME,
                        metadata={"hnsw:space": "cosine"},
                    )
            logger.info("[LongTermMemory] All memories cleared")
        except Exception as e:
            logger.warning(f"[LongTermMemory] Clear failed: {e}")
