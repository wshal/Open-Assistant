"""Response history with encrypted persistence and navigation."""

import json
import threading
import time
import inspect
import uuid
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field, asdict
from utils.crypto import SecureStorage
from utils.logger import setup_logger
from core.constants import HISTORY_DIR

logger = setup_logger(__name__)


@dataclass
class HistoryEntry:
    query: str
    response: str
    provider: str
    mode: str = "general"
    latency: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


class ResponseHistory:
    def __init__(
        self,
        max_entries: int = 500,
        max_sessions: int = 15,
        history_dir: str = HISTORY_DIR,
    ):
        self.entries: List[HistoryEntry] = []
        self.max = max_entries
        self.max_sessions = max_sessions
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(parents=True, exist_ok=True)

        self.index_storage = SecureStorage(str(self.history_dir / "index.enc"))
        self.current_session_id = self._new_session_id()
        self.current_storage = SecureStorage(
            str(self.history_dir / f"{self.current_session_id}.enc")
        )
        self.current_index = -1
        self.screen_analyses: List[dict] = []
        self._lock = threading.Lock()

        # Load existing sessions index if any
        self.sessions = self.index_storage.get("sessions_index") or []
        # No longer auto-starting session here. Wait for start_new_session.

        # GAP 4: Pre-load the last session's tail entries so follow-up resolution
        # and the 3-turn history block work across app restarts.
        # Only the last 10 entries are loaded — enough for context without bloat.
        self._preload_last_session(max_entries=10)

    def _new_session_id(self) -> str:
        return f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"

    def _preload_last_session(self, max_entries: int = 10) -> None:
        """GAP 4: Seed in-memory history with tail entries from the last session.

        This enables cross-session follow-up resolution and the 3-turn history
        block in prompts, so 'give me an example of that' works after a restart.

        Entries are marked read-only via metadata so they are not re-saved into
        the new session's permanent record.
        """
        if not self.sessions:
            return
        last_session = self.sessions[0]  # most recent is always index 0
        last_id = last_session.get("id", "")
        if not last_id or last_id == self.current_session_id:
            return
        try:
            prev_entries = self.read_session(last_id)
            if prev_entries:
                tail = prev_entries[-max_entries:]
                for e in tail:
                    # Mark as prior-session context so engine doesn't re-cache them
                    e.metadata = {**e.metadata, "_prior_session": True}
                self.entries = tail
                self.current_index = len(self.entries) - 1
                logger.debug(
                    "[GAP4] Preloaded %d entries from prior session %s for follow-up context",
                    len(tail), last_id[:16],
                )
        except Exception as exc:
            logger.debug("[GAP4] Could not preload prior session: %s", exc)

    def start_new_session(self):
        """Starts a fresh session and archives the current one."""
        self.save()  # Ensure current is saved

        self.current_session_id = self._new_session_id()
        self.current_storage = SecureStorage(
            str(self.history_dir / f"{self.current_session_id}.enc")
        )
        self.entries = []
        self.current_index = -1
        self.screen_analyses = []
        logger.info(f"🆕 Started new session: {self.current_session_id}")

    def _ensure_current_session_meta(self):
        for session in self.sessions:
            if session["id"] == self.current_session_id:
                return session

        session_meta = {
            "id": self.current_session_id,
            "created_at": time.time(),
            "snippet": "New Conversation",
            "entry_count": 0,
        }
        self.sessions.insert(0, session_meta)
        if len(self.sessions) > self.max_sessions:
            expired = self.sessions[self.max_sessions :]
            self.sessions = self.sessions[: self.max_sessions]
            for old in expired:
                try:
                    old_file = self.history_dir / f"{old['id']}.enc"
                    if old_file.exists():
                        old_file.unlink()
                except Exception as e:
                    logger.debug(f"Failed to delete old session file {old['id']}: {e}")
        return session_meta

    def start_new_session(self):
        """Starts a fresh session and archives the current one."""
        self.save()  # Ensure current is saved

        self.current_session_id = self._new_session_id()
        self.current_storage = SecureStorage(
            str(self.history_dir / f"{self.current_session_id}.enc")
        )
        self.entries = []
        self.current_index = -1
        self.screen_analyses = []
        logger.info(f"🆕 Started new session: {self.current_session_id}")

    def add(
        self,
        query: str,
        response: str,
        provider: str,
        mode: str = "general",
        latency: float = 0.0,
        metadata: dict = None,
    ):
        """Add entry to current session."""
        entry = HistoryEntry(
            query=query,
            response=response,
            provider=provider,
            mode=mode,
            latency=latency,
            metadata=metadata or {},
        )
        with self._lock:
            self._ensure_current_session_meta()
            self.entries.append(entry)
            if len(self.entries) > self.max:
                self.entries = self.entries[-self.max :]

            # Auto-set current_index to latest entry
            self.current_index = len(self.entries) - 1

            # Update index snippet with first query
            if len(self.entries) == 1:
                for s in self.sessions:
                    if s["id"] == self.current_session_id:
                        s["snippet"] = query[:50] + ("..." if len(query) > 50 else "")
                        break

            # Update entry count in index
            for s in self.sessions:
                if s["id"] == self.current_session_id:
                    s["entry_count"] = len(self.entries)
                    break

            self.index_storage.set("sessions_index", self.sessions)
            self._save_unlocked()

    def move_prev(self) -> bool:
        """Moves current_index back one entry. Returns True if index changed."""
        if self.current_index > 0:
            self.current_index -= 1
            return True
        return False

    def move_next(self) -> bool:
        """Moves current_index forward one entry. Returns True if index changed."""
        if self.current_index < len(self.entries) - 1:
            self.current_index += 1
            return True
        return False

    def get_state(self):
        """Returns (index, total, entry) for UI synchronization."""
        entry = self.get_at(self.current_index) if self.current_index >= 0 else None
        return self.current_index, len(self.entries), entry

    def get_at(self, index: int) -> Optional[dict]:
        """Convert dataclass to dict for UI consumption."""
        if 0 <= index < len(self.entries):
            return asdict(self.entries[index])
        return None

    def get_last(self, n: int = 10) -> List[HistoryEntry]:
        with self._lock:
            return self.entries[-n:]

    def add_screen_analysis(
        self,
        prompt: str,
        response: str,
        provider: str,
        metadata: dict = None,
    ):
        with self._lock:
            self._ensure_current_session_meta()
            self.screen_analyses.append(
                {
                    "prompt": prompt,
                    "response": response,
                    "provider": provider,
                    "timestamp": time.time(),
                    "metadata": metadata or {},
                }
            )
            self.index_storage.set("sessions_index", self.sessions)
            self._save_unlocked()

    def get_screen_analyses(self) -> List[dict]:
        with self._lock:
            return list(self.screen_analyses)

    def load_session(self, session_id: str):
        """Loads a specific session into memory."""
        self.save()  # Save current first
        target_path = self.history_dir / f"{session_id}.enc"
        if target_path.exists():
            with self._lock:
                self.current_session_id = session_id
                self.current_storage = SecureStorage(str(target_path))
                self._load_unlocked()
        else:
            logger.error(f"Session file not found: {session_id}")

    def read_session(self, session_id: str) -> List[HistoryEntry]:
        """Read a session without mutating the active in-memory session."""
        target_path = self.history_dir / f"{session_id}.enc"
        if not target_path.exists():
            logger.error(f"Session file not found: {session_id}")
            return []

        try:
            storage = SecureStorage(str(target_path))
            return self._entries_from_storage(storage)
        except Exception as e:
            logger.warning(f"Failed to read session {session_id}: {e}")
            return []

    def read_session_bundle(self, session_id: str) -> dict:
        """Read both chat entries and screen analyses for a session."""
        target_path = self.history_dir / f"{session_id}.enc"
        if not target_path.exists():
            logger.error(f"Session file not found: {session_id}")
            return {"entries": [], "screen_analyses": []}

        try:
            storage = SecureStorage(str(target_path))
            return {
                "entries": self._entries_from_storage(storage),
                "screen_analyses": self._screen_analyses_from_storage(storage),
            }
        except Exception as e:
            logger.warning(f"Failed to read session bundle {session_id}: {e}")
            return {"entries": [], "screen_analyses": []}

    def load(self):
        """Load current session from its encrypted storage."""
        with self._lock:
            self._load_unlocked()

    def _load_unlocked(self):
        """Internal load logic, resilient to schema changes."""
        try:
            self.entries = self._entries_from_storage(self.current_storage)
            self.current_index = len(self.entries) - 1
            analyses = self.current_storage.get("screen_analysis_data")
            self.screen_analyses = analyses if isinstance(analyses, list) else []
        except Exception as e:
            logger.warning(f"Failed to load history: {e}")

    def _entries_from_storage(self, storage: SecureStorage) -> List[HistoryEntry]:
        data = storage.get("history_data")
        if not data or not isinstance(data, list):
            return []

        valid_fields = set(inspect.signature(HistoryEntry).parameters.keys())
        entries = []
        for e in data:
            filtered = {k: v for k, v in e.items() if k in valid_fields}
            entries.append(HistoryEntry(**filtered))
        return entries

    @staticmethod
    def _screen_analyses_from_storage(storage: SecureStorage) -> List[dict]:
        data = storage.get("screen_analysis_data")
        if not data or not isinstance(data, list):
            return []
        return [item for item in data if isinstance(item, dict)]

    def save(self):
        """Save current session to its encrypted storage."""
        with self._lock:
            self._save_unlocked()

    def _save_unlocked(self):
        if not self.entries and not self.screen_analyses:
            return
        try:
            data = [asdict(e) for e in self.entries]
            self.current_storage.set("history_data", data)
            self.current_storage.set("screen_analysis_data", self.screen_analyses)
        except Exception as e:
            logger.error(f"Failed to save history: {e}")

    def clear(self):
        """Emergency Wipe: Purge memory and ALL session files from disk."""
        self.entries.clear()
        self.current_index = -1
        self.screen_analyses = []

        # RESTORATION: Total Disk Purge logic
        logger.warning("🚨 EMERGENCY DISK PURGE INITIATED")
        try:
            # 1. Delete all .enc files in history directory
            for enc_file in self.history_dir.glob("*.enc"):
                try:
                    enc_file.unlink()
                    logger.debug(f"Removed trace: {enc_file.name}")
                except Exception as e:
                    logger.error(f"Failed to remove {enc_file}: {e}")

            # 2. Re-initialize empty state
            self.sessions = []
            self.index_storage.set("sessions_index", [])
            self.current_session_id = self._new_session_id()
            self.current_storage = SecureStorage(
                str(self.history_dir / f"{self.current_session_id}.enc")
            )

        except Exception as e:
            logger.error(f"Global purge failed: {e}")

        logger.info("🗑️ Global history and disk traces cleared")

    def search(self, keyword: str) -> List[HistoryEntry]:
        kw = keyword.lower()
        return [
            e for e in self.entries if kw in e.query.lower() or kw in e.response.lower()
        ]

    def export_markdown(self, filepath: str):
        lines = ["# OpenAssist AI — Response History\n"]
        for e in self.entries:
            ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(e.timestamp))
            lines.append(f"## {ts} [{e.mode}] via {e.provider}\n")
            lines.append(f"**Q:** {e.query}\n")
            lines.append(f"**A:** {e.response}\n")
            lines.append("---\n")
        Path(filepath).write_text("\n".join(lines), encoding="utf-8")
