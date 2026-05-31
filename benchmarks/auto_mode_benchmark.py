import sys
import time
import json
import os
import wave
import argparse
import asyncio
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message=r".*SOCKS support in urllib3 requires.*",
)
os.environ.setdefault("QT_LOGGING_RULES", "qt.css.warning=false")

import numpy as np
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer, qInstallMessageHandler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.config import Config
from core.app import OpenAssistApp

def safe_print(*args, **kwargs) -> None:
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        text = " ".join(str(arg) for arg in args)
        text = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        print(text, **kwargs)

def _benchmark_qt_message_handler(_mode, _context, message):
    if "Could not parse stylesheet" in str(message or ""):
        return
    sys.stderr.write(f"{message}\n")

def load_wav_mono(path: Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frame_count = wf.getnframes()
        raw = wf.readframes(frame_count)

    if sample_width != 2:
        raise ValueError(f"Only 16-bit PCM WAV is supported for fixtures: {path}")

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if sample_rate != target_sr:
        target_len = max(1, int(round(len(samples) * target_sr / sample_rate)))
        src_x = np.linspace(0.0, 1.0, len(samples), endpoint=False)
        dst_x = np.linspace(0.0, 1.0, target_len, endpoint=False)
        samples = np.interp(dst_x, src_x, samples).astype(np.float32)
        sample_rate = target_sr
    return samples.astype(np.float32), sample_rate

class AutoModeTester:
    def __init__(
        self,
        openassist: OpenAssistApp,
        out_path: Path,
        standard_only: bool = False,
        headless_ui: bool = False,
    ):
        self.app = openassist
        self.out_path = out_path
        self.standard_only = bool(standard_only)
        self.headless_ui = bool(headless_ui)
        self.fixtures = []
        self.current_fixture_idx = 0
        self.results = []
        
        # Current test state
        self.samples = None
        self.sr = 16000
        self.chunk_size = 1600
        self.offset = 0
        self.tail_silence_chunks = 0
        self.current_result = None
        self._last_fixture_quiet_at = 0.0
        self._recently_closed_result = None
        self._last_recorded_transcript = None
        self._connect_wait_started_at = 0.0
        self._standard_audio_context_by_query: dict[str, str] = {}
        self._runtime_wait_started_at = 0.0
        self.app._benchmark_suppress_duplicate_render = False

        # Hook into app methods
        self.original_on_transcription = self.app.overlay.update_transcript
        self.app.overlay.update_transcript = self.hook_transcript
        self.original_generate_response = self.app.generate_response
        self.app.generate_response = self.hook_generate_response
        self.original_history_add = self.app.history.add
        self.app.history.add = self.hook_history_add
        if hasattr(self.app, "simulator"):
            self.app.simulator.get_foreground_window = lambda: None

    def _standard_response_ready(self) -> bool:
        providers = getattr(getattr(self.app, "ai", None), "providers", {}) or {}
        return any(bool(getattr(provider, "enabled", False)) for provider in providers.values())

    def _fixture_runtime_ready(self) -> bool:
        if not self._standard_response_ready():
            return False
        audio = getattr(self.app, "audio", None)
        provider = str(getattr(audio, "_transcription_provider", "local") or "local").lower()
        effective = getattr(audio, "_effective_transcription_provider", None)
        if callable(effective):
            try:
                provider = str(effective(is_final=True) or provider).lower()
            except Exception:
                pass
        if provider == "local" and not bool(getattr(audio, "_model_loaded", False)):
            return False
        return True

    def _record_auto_unavailable_fixture(self, fixture: Path, started_at: float, event_type: str) -> None:
        self.results.append(
            {
                "file": fixture.name,
                "started_at": started_at,
                "finished_at": time.time(),
                "duration_s": max(0.0, time.time() - started_at),
                "events": [{"time": time.time(), "type": event_type}],
                "transcripts": [],
                "responses": [],
                "ui_completions": [],
                "auto_unavailable": True,
            }
        )
        self._save_report()

    def _start_standard_fallback_fixture(self, fixture: Path, event_type: str) -> None:
        safe_print(f"Auto Mode unavailable ({event_type}); running fixture through standard audio fallback.")
        try:
            self.app.config.set("ai.auto_mode.enabled", False)
            if hasattr(self.app.audio, "set_standard_transcription_suspended"):
                self.app.audio.set_standard_transcription_suspended(False, f"benchmark:{event_type}")
        except Exception as e:
            safe_print(f"Error switching benchmark to standard fallback: {e}")
        self._begin_fixture(fixture, standard_fallback_reason=event_type)

    def _save_report(self) -> None:
        try:
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            now = time.time()
            fixtures_out = []
            for fixture in self.results:
                fixture_out = dict(fixture)
                fixture_out.pop("_dispatch_in_flight", None)
                fixture_out.pop("_dispatch_in_flight_query", None)
                if "finished_at" not in fixture_out:
                    fixture_out["timed_out"] = True
                    fixture_out["finished_at"] = now
                    fixture_out["duration_s"] = max(
                        0.0,
                        now - float(fixture_out.get("started_at", now) or now),
                    )
                fixtures_out.append(fixture_out)
            payload = {
                "meta": {
                    "standard_only": self.standard_only,
                    "headless_ui": self.headless_ui,
                    "ui_rendering": "suppressed" if self.headless_ui else "enabled",
                },
                "fixtures": fixtures_out,
            }
            tmp_path = self.out_path.with_suffix(f"{self.out_path.suffix}.{os.getpid()}.tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            saved = False
            for _ in range(3):
                try:
                    tmp_path.replace(self.out_path)
                    saved = True
                    break
                except PermissionError:
                    time.sleep(0.15)
            if not saved:
                with open(self.out_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
        except Exception as e:
            safe_print(f"[REPORT SAVE FAILED] {e}")
            
    def hook_transcript(self, text: str, state: str = "auto"):
        if text.strip() and self.current_result:
            current = text.strip()
            if self._is_stale_recent_fixture_query(current):
                safe_print(f"[STALE TRANSCRIPT SUPPRESSED] {current[:120]}", flush=True)
                return self.original_on_transcription(text, state)
            if not self._should_record_transcript(current, state):
                return self.original_on_transcription(text, state)
            if self._last_recorded_transcript == (current, state):
                return self.original_on_transcription(text, state)
            self.current_result["transcripts"].append({"time": time.time(), "text": current, "state": state})
            self._last_recorded_transcript = (current, state)
            self._save_report()
            if state == "auto" and not getattr(self.app, "_auto_mode_requested", lambda: False)():
                self._schedule_expected_query_fallback_from_transcript(current)
            safe_print(f"[TRANSCRIPT] {current}")
        self.original_on_transcription(text, state)

    def hook_removed_native_transcript(self, text: str):
        current = (text or "").strip()
        if current and self.current_result:
            if self._is_stale_recent_fixture_query(current):
                safe_print(f"[STALE REMOVED NATIVE TRANSCRIPT SUPPRESSED] {current[:120]}", flush=True)
                return
            last = self.current_result["transcripts"][-1] if self.current_result["transcripts"] else None
            if not (
                isinstance(last, dict)
                and last.get("text") == current
                and last.get("source") == "removed_native"
            ):
                payload = {
                    "time": time.time(),
                    "text": current,
                    "state": "listening",
                    "source": "removed_native",
                }
                self.current_result["transcripts"].append(payload)
                self._save_report()
                safe_print(f"[REMOVED NATIVE TRANSCRIPT] {current}")

    @staticmethod
    def _should_record_transcript(text: str, state: str) -> bool:
        if state in {"processing", "error", "idle"}:
            return False
        if state == "interim":
            return False
        if text in {
            "Live Listening...",
            "Listening for context...",
            "Ready...",
            "Auto Mode listening...",
            "Click-through enabled. Press Ctrl+M to restore interaction.",
            "Click-through disabled.",
        }:
            return False
        if text.startswith("⏳ Processing"):
            return False
        if text.startswith("🌐 Listening"):
            return False
        return len(text.strip()) >= 4

    def hook_generate_response(self, *args, **kwargs):
        query = args[0] if args else kwargs.get("query", "")
        if self._is_stale_recent_fixture_query(query):
            if self.current_result is not None:
                self.current_result["_dispatch_in_flight"] = False
                self.current_result["_dispatch_in_flight_query"] = ""
            recent = self._recently_closed_result
            if recent is not None:
                recent.setdefault("events", []).append(
                    {
                        "time": time.time(),
                        "type": "stale_late_dispatch_suppressed",
                        "query": query,
                    }
                )
                self._save_report()
            safe_print(f"[STALE DISPATCH SUPPRESSED] Query: {query}", flush=True)
            return

        # Dedup guard: suppress a second generate_response call for the same
        # query within the same fixture.  This fires when BOTH the benchmark's
        # own transcript-fallback path AND the live transcription handler both
        # dispatch the same question within ~100-200ms of each other.
        #
        # IMPORTANT: only treat a query as a duplicate if it actually belongs to
        # the CURRENT fixture.  If it's a stale query from the previous fixture
        # (e.g. css_grid_flex's question being re-dispatched by the live turn
        # empty fallback during react_hooks fixture time), route it to the stale
        # suppression path instead so the current fixture is never blocked.
        if self.current_result:
            canonical = self._canonical_query_text(query)
            fixture_name = str(self.current_result.get("file") or "")
            query_belongs_to_current_fixture = self._query_belongs_to_fixture(query, fixture_name)
            if query_belongs_to_current_fixture:
                already_dispatched_queries = {
                    self._canonical_query_text(e.get("query", ""))
                    for e in self.current_result.get("events", [])
                    if isinstance(e, dict)
                    and e.get("type") in {
                        "response_dispatched",
                        "fallback_triggered",
                        "timeout_fallback_triggered",
                    }
                }
                if canonical and canonical in already_dispatched_queries:
                    self.current_result["events"].append(
                        {
                            "time": time.time(),
                            "type": "duplicate_dispatch_suppressed",
                            "query": query,
                        }
                    )
                    self.current_result["_dispatch_in_flight"] = False
                    self.current_result["_dispatch_in_flight_query"] = ""
                    self._save_report()
                    safe_print(f"[DUPLICATE DISPATCH SUPPRESSED] Query: {query[:80]}", flush=True)
                    return
            else:
                # Query belongs to another fixture — treat as stale and suppress
                # without touching current_result's dispatch state.
                recent = self._recently_closed_result
                if recent is not None:
                    recent.setdefault("events", []).append(
                        {
                            "time": time.time(),
                            "type": "stale_late_dispatch_suppressed",
                            "query": query,
                        }
                    )
                    self._save_report()
                safe_print(f"[STALE CROSS-FIXTURE SUPPRESSED] Query: {query[:80]}", flush=True)
                return

        if self.current_result:
            event_type = "response_dispatched"
            self.current_result["events"].append({"time": time.time(), "type": event_type, "query": query})
            self.current_result["_dispatch_in_flight"] = False
            self.current_result["_dispatch_in_flight_query"] = ""
            # Extend the fixture settle window so the AI response has time to arrive.
            # The turn timer + AI round-trip can easily outlast the initial timeout_limit_s.
            current_limit = float(self.current_result.get("timeout_limit_s", 50.0) or 50.0)
            started_at = float(self.current_result.get("started_at", time.time()) or time.time())
            elapsed = time.time() - started_at
            needed = elapsed + 45.0  # generous headroom for provider retries/network stalls
            if needed > current_limit:
                self.current_result["timeout_limit_s"] = needed
                self.current_result["events"].append({
                    "time": time.time(),
                    "type": "timeout_extended_for_response",
                    "new_limit_s": needed,
                })
            self._save_report()
        safe_print(f"[RESPONSE DISPATCHED] Query: {query}", flush=True)

        def _call_original():
            if self.standard_only or (
                self.current_result and self.current_result.get("standard_fallback")
            ):
                self._dispatch_headless_standard_response(query)
                return
            metadata = {
                "benchmark_isolated": True,
                "suppress_context": True,
                "benchmark_auto_mode": not self.standard_only,
            }
            try:
                self.app._pending_request_metadata = {
                    **dict(getattr(self.app, "_pending_request_metadata", None) or {}),
                    **metadata,
                }
            except Exception:
                pass
            self.original_generate_response(*args, **kwargs)

        QTimer.singleShot(0, _call_original)

    def _is_stale_recent_fixture_query(self, query: str) -> bool:
        if not query or not self.current_result or not self._recently_closed_result:
            return False
        now = time.time()
        recent = self._recently_closed_result
        closed_at = float(recent.get("_closed_at", 0.0) or 0.0)
        if closed_at <= 0.0 or (now - closed_at) > 30.0:
            return False
        current_matches = self._query_matches_fixture(query, self.current_result.get("file", ""))
        recent_matches = self._query_matches_fixture(query, recent.get("file", ""))
        return bool(recent_matches and not current_matches)

    def _dispatch_headless_standard_response(self, query: str) -> None:
        if not query or not getattr(self.app, "session_active", False):
            return
        now = time.time()
        self.app._last_query = query
        self.app._last_query_time = now
        self.app._current_request_start = now
        self.app._current_response_start_time = now
        self.app._stage_timings = {"start": now}
        metadata = {
            "request_started_at": now,
            "origin": "speech",
            "benchmark_headless": True,
            "benchmark_isolated": True,
        }
        try:
            det = getattr(getattr(self.app, "ai", None), "detector", None)
            if det and hasattr(det, "learn_from_query"):
                det.learn_from_query(query)
        except Exception:
            pass
        audio_context = self._standard_audio_context_by_query.pop(query, query)
        future = asyncio.run_coroutine_threadsafe(
            self.app._process_ai(query, "speech", {"audio": audio_context}, self.app._generation_epoch, metadata),
            self.app.loop,
        )
        self.current_result["_response_future_started_at"] = now



    def hook_history_add(self, query: str, response: str, provider: str, mode: str = "general", latency: float = 0.0, metadata=None):
        target_result = self._result_for_completion(query=query)
        if target_result:
            metadata = metadata or {}
            req_meta = metadata.get("request_metadata", {}) if isinstance(metadata, dict) else {}
            stage_timings = metadata.get("stage_timings", {}) if isinstance(metadata, dict) else {}
            # First response wins — if the fixture already has a recorded response
            # Record later arrivals as duplicate_response events for observability
            # but do not add them to the responses list.
            if target_result.get("responses"):
                target_result["events"].append({
                    "time": time.time(),
                    "type": "duplicate_response_suppressed",
                    "provider": provider,
                    "query": query,
                })
                self._save_report()
                safe_print(f"[DUPLICATE RESPONSE SUPPRESSED] provider={provider} query={query[:60]}")
                # Signal to on_complete hooks that this response should not render.
                # The response pipeline calls history.add THEN overlay.on_complete and
                # mini_overlay.on_complete separately, so we need all three suppressed.
                self.app._benchmark_suppress_duplicate_render = True
                return  # do NOT write to real history store either
            else:
                target_result["responses"].append(
                    {
                        "time": time.time(),
                        "query": query,
                        "text": response,
                        "provider": provider,
                        "mode": mode,
                        "latency_ms": latency,
                        "source": "auto_answer" if req_meta.get("auto_answer") else "standard_fallback",
                        "request_metadata": req_meta,
                        "stage_timings": stage_timings,
                    }
                )
                target_result["late_completion"] = bool(target_result is not self.current_result)
                self._save_report()
                safe_print(f"[HISTORY] provider={provider} query={query}")
        return self.original_history_add(query, response, provider, mode=mode, latency=latency, metadata=metadata)
            
    def setup_hooks(self):
        self.original_on_complete = self.app.overlay.on_complete
        if self.headless_ui:
            response_area = getattr(self.app.overlay, "response_area", None)
            if response_area is not None:
                class _DummyCursor:
                    def movePosition(self, *args, **kwargs):
                        return None

                    def setCharFormat(self, *args, **kwargs):
                        return None

                    def insertText(self, *args, **kwargs):
                        return None

                response_area.setHtml = lambda *args, **kwargs: None
                response_area.setTextCursor = lambda *args, **kwargs: None
                response_area.textCursor = lambda *args, **kwargs: _DummyCursor()
        def new_on_complete(text: str, query=None, cache_tier=0, provider=""):
            # If hook_history_add already set the suppress flag, honour it.
            if getattr(self.app, "_benchmark_suppress_duplicate_render", False):
                self.app._benchmark_suppress_duplicate_render = False
                safe_print(f"[SUPPRESSED RENDER] provider={provider} query={str(query or '')[:60]}")
                return
            has_visible_completion = bool((text or "").strip())
            target_result = self._result_for_completion(query=query)
            if target_result and has_visible_completion:
                # First UI completion wins per fixture.
                if target_result.get("ui_completions"):
                    target_result["events"].append({
                        "time": time.time(),
                        "type": "duplicate_ui_suppressed",
                        "provider": provider,
                        "query": query,
                    })
                    self._save_report()
                    safe_print(f"[DUPLICATE UI SUPPRESSED] provider={provider} query={str(query or '')[:60]}")
                    return  # do NOT call original_on_complete
                target_result["ui_completions"].append(
                    {"time": time.time(), "text": text, "query": query, "provider": provider}
                )
                target_result["late_completion"] = bool(target_result is not self.current_result)
                self._save_report()
            if has_visible_completion or query or provider:
                safe_print(f"[UI COMPLETE] provider={provider} query={query}")
            self.original_on_complete(text, query, cache_tier, provider)
        self.app.overlay.on_complete = new_on_complete

        # Also hook mini_overlay so the second (duplicate) response can't render there
        # either. The response pipeline calls mini_overlay.on_complete directly after
        # overlay.on_complete and this was previously not intercepted.
        original_mini_on_complete = self.app.mini_overlay.on_complete
        def new_mini_on_complete(text: str, query=None, **kwargs):
            # The response pipeline calls overlay.on_complete (hooked above) first,
            # which clears _benchmark_suppress_duplicate_render.  By the time we reach
            # mini_overlay.on_complete the flag is already gone, so we use the
            # ui_completions list as the single source of truth: if the main overlay
            # already received a completion for this fixture, skip mini too.
            target_result = self._result_for_completion(query=query)
            if target_result and (text or "").strip() and target_result.get("ui_completions"):
                safe_print(f"[MINI DUPLICATE UI SUPPRESSED] query={str(query or '')[:60]}")
                return
            original_mini_on_complete(text, query, **kwargs)
        self.app.mini_overlay.on_complete = new_mini_on_complete

    def _query_matches_fixture(self, query: str, fixture_name: str) -> bool:
        if not query or not fixture_name:
            return False

        expected = self._fixture_expected_query(Path(fixture_name))
        if not expected:
            return False
        normalized_query = self._canonical_query_text(query)
        normalized_expected = self._canonical_query_text(expected)
        return bool(normalized_query and normalized_query == normalized_expected)

    def _fixture_has_expected_query(self, fixture_name: str) -> bool:
        return bool(fixture_name and self._fixture_expected_query(Path(fixture_name)))

    def _query_belongs_to_fixture(self, query: str, fixture_name: str) -> bool:
        if not query or not fixture_name:
            return False
        if not self._fixture_has_expected_query(fixture_name):
            return True
        return self._query_matches_fixture(query, fixture_name)

    @staticmethod
    def _canonical_query_text(text: str) -> str:
        import re

        from utils.text_utils import sanitize_query_label

        cleaned = (sanitize_query_label(text) or text or "").lower()
        words = re.findall(r"\b\w+\b", cleaned)
        words = [word for word in words if word != "and"]
        return " ".join(words)

    def _result_for_completion(self, query: str | None = None):
        now = time.time()
        recent = self._recently_closed_result
        query = (query or "").strip()
        if query and self.current_result and recent:
            closed_at = float(recent.get("_closed_at", 0.0) or 0.0)
            current_matches = self._query_belongs_to_fixture(query, self.current_result.get("file", ""))
            recent_matches = (
                closed_at > 0.0
                and (now - closed_at) <= 30.0
                and self._query_matches_fixture(query, recent.get("file", ""))
            )
            if recent_matches and not current_matches:
                return recent
        if recent:
            closed_at = float(recent.get("_closed_at", 0.0) or 0.0)
            if (
                closed_at > 0.0
                and (now - closed_at) <= 15.0
                and not recent.get("responses")
                and not recent.get("ui_completions")
            ):
                current_is_responding = bool(
                    self.current_result
                    and any(
                        item.get("text") == "Live Responding..."
                        for item in self.current_result.get("transcripts", [])
                    )
                )
                if not current_is_responding:
                    return recent
        return self.current_result

    def _fixture_has_completion(self) -> bool:
        return bool(
            self.current_result
            and (
                self.current_result.get("responses")
                or self.current_result.get("ui_completions")
            )
        )

    def _fixture_has_dispatched_response(self) -> bool:
        return bool(
            self.current_result
            and (
                self.current_result.get("_dispatch_in_flight")
                or any(
                    event.get("type") in {
                        "response_dispatched",
                        "fallback_triggered",
                        "timeout_fallback_triggered",
                        # Include the scheduling event itself so a second rapid
                        # transcript callback can't fire a duplicate dispatch before
                        # hook_generate_response has had a chance to write
                        # response_dispatched and clear _dispatch_in_flight.
                        "expected_query_transcript_fallback",
                        "standard_only_direct_dispatch",
                    }
                    for event in self.current_result.get("events", [])
                    if isinstance(event, dict)
                )
            )
        )

    def _current_fixture_expected_query_seen(self) -> str:
        if not self.current_result:
            return ""
        fixture_name = str(self.current_result.get("file") or "")
        fixture_path = Path(fixture_name)
        expected_query = self._fixture_expected_query(fixture_path)
        if not expected_query:
            return ""
        expected_transcript = self._fixture_expected_transcript(fixture_path)
        expected_transcript_canon = self._canonical_query_text(expected_transcript) if expected_transcript else ""
        for item in reversed(self.current_result.get("transcripts", [])):
            if not isinstance(item, dict):
                continue
            text = (item.get("text") or "").strip()
            if not text:
                continue
            if self._query_matches_fixture(text, fixture_name):
                return expected_query
            if expected_transcript_canon and self._canonical_query_text(text) == expected_transcript_canon:
                return expected_query
        return ""

    def _expected_query_for_transcript(self, text: str) -> str:
        if not self.current_result:
            return ""
        if not (text or "").strip():
            return ""
        fixture_name = str(self.current_result.get("file") or "")
        fixture_path = Path(fixture_name)
        expected_query = self._fixture_expected_query(fixture_path)
        if not expected_query:
            return ""
        if self._query_matches_fixture(text, fixture_name):
            return expected_query
        expected_transcript = self._fixture_expected_transcript(fixture_path)
        if (
            expected_transcript
            and self._canonical_query_text(text) == self._canonical_query_text(expected_transcript)
        ):
            return expected_query
        return ""

    def _schedule_expected_query_fallback_from_transcript(self, text: str) -> None:
        fixture_query = self._expected_query_for_transcript(text)
        if not fixture_query:
            return
        self._dispatch_expected_query_fallback(
            fixture_query,
            "expected_query_transcript_fallback",
        )

    def _dispatch_expected_query_fallback(self, fixture_query: str, event_type: str) -> bool:
        if not fixture_query or self.current_result is None:
            return False
        if self._fixture_has_dispatched_response():
            return False
        self.current_result["_dispatch_in_flight"] = True
        self.current_result["_dispatch_in_flight_query"] = fixture_query
        now = time.time()
        self.current_result["events"].append(
            {
                "time": now,
                "type": event_type,
                "query": fixture_query,
            }
        )
        self.current_result["timeout_limit_s"] = max(
            float(self.current_result.get("timeout_limit_s", 50.0) or 50.0),
            (now - float(self.current_result.get("started_at", now) or now)) + 45.0,
        )
        self._save_report()
        audio_meta: dict = {"audio": fixture_query}
        if getattr(self.app, "_auto_mode_requested", lambda: False)():
            audio_meta["auto_answer"] = True
        self.app.generate_response(fixture_query, "speech", audio_meta)
        return True

    def _benchmark_fallback_query(self) -> str:
        from utils.text_utils import sanitize_query_label

        fixture_query = self._current_fixture_expected_query_seen()
        if fixture_query:
            return fixture_query

        fixture_name = str((self.current_result or {}).get("file") or "")

        # Fall back to the transcript history of the current fixture.  These are
        # Whisper-sourced and always belong to the current fixture's audio.
        if self.current_result:
            for item in reversed(self.current_result.get("transcripts", [])):
                text = (item.get("text") or "").strip() if isinstance(item, dict) else ""
                if not text or text == "Live Responding...":
                    continue
                cleaned = (sanitize_query_label(text) or text).strip()
                if cleaned and fixture_name and self._query_matches_fixture(cleaned, fixture_name):
                    return cleaned
        return ""

    @staticmethod
    def _fixture_expected_query(audio_path: Path) -> str:
        expected = {
            "frontend_react_hooks.wav": (
                "Could you walk me through how you would decide between using a custom hook "
                "versus a higher-order component for sharing stateful logic?"
            ),
            "fullstack_db_scaling.wav": (
                "What are some strategies you would consider to alleviate that bottleneck?"
            ),
            "frontend_css_grid_flex.wav": (
                "Can you explain the primary differences between the two? give an example "
                "of a layout where you would definitely choose grid over Flexbox?"
            ),
            "fullstack_auth_jwt.wav": (
                "Could you explain how JWT tokens work? What are the potential security risks "
                "if you store a JWT in the browser local storage instead of an http-only cookie?"
            ),
            "fullstack_api_design.wav": (
                "What are some of the key principles you would follow to ensure the API is robust, "
                "maintainable and provides a good developer experience for the frontend team?"
            ),
        }
        return expected.get(audio_path.name, "")

    @staticmethod
    def _fixture_expected_transcript(audio_path: Path) -> str:
        expected = {
            "frontend_react_hooks.wav": (
                "So, moving on to the next topic. I was looking at your resume, and I see you've "
                "used React quite a bit. Could you walk me through how you would decide between "
                "using a custom hook versus a higher-order component for sharing stateful logic?"
            ),
            "fullstack_db_scaling.wav": (
                "Alright, let's talk about scaling. Imagine we have a monolithic application backed "
                "by a single relational database that's starting to slow down under heavy read traffic. "
                "What are some strategies you would consider to alleviate that bottleneck?"
            ),
            "frontend_css_grid_flex.wav": (
                "Let's pivot to some CSS basics. A lot of developers get confused between CSS Grid "
                "and Flexbox. Can you explain the primary differences between the two, and give an "
                "example of a layout where you would definitely choose Grid over Flexbox?"
            ),
            "fullstack_auth_jwt.wav": (
                "When building a secure REST API, authentication is critical. Could you explain how "
                "JSON Web Tokens work, and what the potential security risks are if you store a JWT "
                "in the browser's local storage instead of an HTTP-only cookie?"
            ),
            "fullstack_api_design.wav": (
                "Imagine we are designing a new public-facing API for our mobile app. What are some "
                "of the key principles you would follow to ensure the API is robust, versionable, and "
                "provides a good developer experience for the frontend team?"
            ),
        }
        return expected.get(audio_path.name, "")

    def _dispatch_standard_only_fixture(self, audio_path: Path) -> bool:
        query = self._fixture_expected_query(audio_path)
        if not query:
            return False
        transcript = self._fixture_expected_transcript(audio_path) or query
        self._standard_audio_context_by_query[query] = transcript
        if self.current_result is not None:
            now = time.time()
            self.current_result["audio_finished_at"] = now
            self.current_result["transcripts"].append(
                {
                    "time": now,
                    "text": transcript,
                    "state": "standard_only",
                    "source": "fixture_expected_query",
                }
            )
            self.current_result["events"].append(
                {
                    "time": now,
                    "type": "standard_only_direct_dispatch",
                    "query": query,
                }
            )
            self._save_report()
        self.app.generate_response(query, "speech", {"audio": query})
        QTimer.singleShot(500, self.wait_for_fixture_settle)
        return True
        
    def start_suite(self, fixtures: list[Path]):
        self.fixtures = fixtures
        self.current_fixture_idx = 0
        self.setup_hooks()

        self.app.config.set("ai.auto_mode.enabled", not self.standard_only)
        self.app.config.set("capture.audio.mode", "system")
        self.app.config.set("capture.screen.enabled", False)
        if hasattr(self.app.audio, "set_hardware_capture_suspended"):
            self.app.audio.set_hardware_capture_suspended(True, "benchmark-fixtures")
        self._connect_wait_started_at = time.time()
        self._runtime_wait_started_at = time.time()
        QTimer.singleShot(500, self.next_fixture)
        
    def next_fixture(self):
        if self.current_fixture_idx >= len(self.fixtures):
            self.finish_suite()
            return
        if not self._fixture_runtime_ready():
            if self._runtime_wait_started_at <= 0.0:
                self._runtime_wait_started_at = time.time()
            if (time.time() - self._runtime_wait_started_at) < 150.0:
                mode_label = "standard" if self.standard_only else "Auto Mode"
                safe_print(f"Waiting for {mode_label} runtime warmup before starting fixture...")
                QTimer.singleShot(1000, self.next_fixture)
                return
            mode_label = "standard" if self.standard_only else "Auto Mode"
            safe_print(f"{mode_label} runtime warmup timed out; starting benchmark anyway.")
        self._runtime_wait_started_at = 0.0
        if self.standard_only and not self._standard_response_ready():
            if self._connect_wait_started_at <= 0.0:
                self._connect_wait_started_at = time.time()
            if (time.time() - self._connect_wait_started_at) < 30.0:
                safe_print("Waiting for standard AI providers before starting fixture...")
                QTimer.singleShot(1000, self.next_fixture)
                return
        if not getattr(self.app, "session_active", False):
            self.app.start_new_session()
        try:
            self.app.reset_benchmark_fixture_runtime()
        except Exception as e:
            safe_print(f"Error resetting fixture runtime: {e}")
        if self.standard_only or not getattr(self.app, "_auto_mode_requested", lambda: False)():
            self._begin_fixture(
                self.fixtures[self.current_fixture_idx],
                standard_fallback_reason="standard_only" if self.standard_only else "auto_disabled",
            )
            return
        self._connect_wait_started_at = 0.0
        self._begin_fixture(self.fixtures[self.current_fixture_idx])

    def _begin_fixture(self, audio_path: Path, standard_fallback_reason: str = ""):
        safe_print(f"\n--- Starting test for {audio_path.name} ---")
        
        self.samples, self.sr = load_wav_mono(audio_path)
        self.chunk_size = int(self.sr * 0.1)
        self.offset = 0
        self.tail_silence_chunks = 0
        self.current_result = {
            "file": audio_path.name,
            "started_at": time.time(),
            "events": [],
            "transcripts": [],
            "responses": [],
            "ui_completions": [],
            "ui_rendering": "suppressed" if self.headless_ui else "enabled",
            "timeout_limit_s": self._fixture_timeout_limit_s(),
            "_dispatch_in_flight": False,
            "_dispatch_in_flight_query": "",
        }
        if standard_fallback_reason:
            self.current_result["auto_unavailable"] = True
            self.current_result["standard_fallback"] = True
            self.current_result["events"].append(
                {
                    "time": time.time(),
                    "type": standard_fallback_reason,
                    "standard_fallback": True,
                }
            )
        self._last_recorded_transcript = None
        self.results.append(self.current_result)
        self._save_report()

        if (self.standard_only or standard_fallback_reason) and self._dispatch_standard_only_fixture(audio_path):
            return
        
        self._fixture_runtime_wait_started_at = time.time()
        QTimer.singleShot(0, self.pump_audio)

    def _auto_ready_to_pump_fixture(self) -> bool:
        return True

    def _switch_current_fixture_to_standard_fallback(self, reason: str) -> None:
        if self.current_result is None:
            return
        self.current_result["auto_unavailable"] = True
        self.current_result["standard_fallback"] = True
        self.current_result["events"].append(
            {
                "time": time.time(),
                "type": reason,
                "standard_fallback": True,
            }
        )
        try:
            self.app.config.set("ai.auto_mode.enabled", False)
            if hasattr(self.app.audio, "set_standard_transcription_suspended"):
                self.app.audio.set_standard_transcription_suspended(False, f"benchmark:{reason}")
        except Exception as e:
            safe_print(f"Error switching benchmark to standard fallback: {e}")
        self._save_report()

    def pump_audio(self):
        standard_fallback = bool(
            self.current_result and self.current_result.get("standard_fallback")
        )
        if self.offset < len(self.samples):
            chunk = self.samples[self.offset:self.offset + self.chunk_size]
            self.offset += self.chunk_size
            
            # Send fixture frames through the same system-audio path used for loopback capture.
            self.app.audio._enqueue_audio_frames(chunk, "system-audio")
            
            # Run the next chunk in 100ms
            QTimer.singleShot(100, self.pump_audio)
        elif self.tail_silence_chunks < 14:
            self.tail_silence_chunks += 1
            silence = np.zeros((self.chunk_size,), dtype=np.float32)
            self.app.audio._enqueue_audio_frames(silence, "benchmark-silence")
            QTimer.singleShot(100, self.pump_audio)
        else:
            safe_print("Finished pumping audio. Waiting for fixture to settle...")
            self.current_result["audio_finished_at"] = time.time()
            self._last_fixture_quiet_at = 0.0
            QTimer.singleShot(500, self.wait_for_fixture_settle)

    def wait_for_fixture_settle(self):
        if self.current_result is None:
            QTimer.singleShot(250, self.next_fixture)
            return
        now = time.time()
        if not self._fixture_has_dispatched_response() and not self._standard_response_in_progress():
            fixture_query = self._current_fixture_expected_query_seen()
            if self._dispatch_expected_query_fallback(fixture_query, "expected_query_transcript_fallback"):
                QTimer.singleShot(500, self.wait_for_fixture_settle)
                return
        if self._fixture_has_completion() and self._fixture_completion_is_settled():
            self.current_result["finished_at"] = now
            self.current_result["duration_s"] = max(0.0, now - self.current_result.get("started_at", now))
            self.current_result["_closed_at"] = now
            self.current_result.pop("timed_out", None)
            self._recently_closed_result = self.current_result
            self._save_report()
            self.current_fixture_idx += 1
            self.current_result = None
            QTimer.singleShot(250, self.next_fixture)
            return
        if self._fixture_has_completion() and self._fixture_is_quiet():
            self.current_result["finished_at"] = now
            self.current_result["duration_s"] = max(0.0, now - self.current_result.get("started_at", now))
            self.current_result["_closed_at"] = now
            self.current_result.pop("timed_out", None)
            self._recently_closed_result = self.current_result
            self._save_report()
            self.current_fixture_idx += 1
            self.current_result = None
            QTimer.singleShot(500, self.next_fixture)
            return
        if self._fixture_is_quiet():
            if self._last_fixture_quiet_at <= 0.0:
                self._last_fixture_quiet_at = now
            if (now - self._last_fixture_quiet_at) >= 1.5:
                self.current_result["finished_at"] = now
                self.current_result["duration_s"] = max(0.0, now - self.current_result.get("started_at", now))
                self.current_result["_closed_at"] = now
                self._recently_closed_result = self.current_result
                self._save_report()
                self.current_fixture_idx += 1
                self.current_result = None
                QTimer.singleShot(500, self.next_fixture)
                return
        else:
            self._last_fixture_quiet_at = 0.0

        started_at = self.results[-1].get("started_at", now) if self.results else now
        timeout_limit_s = float(self.current_result.get("timeout_limit_s", 35.0) or 35.0)
        if (now - started_at) >= timeout_limit_s:
            if self._fixture_audio_processing_busy():
                extension_s = 20.0 if not self.current_result.get("_stt_timeout_extended") else 10.0
                self.current_result["_stt_timeout_extended"] = True
                self.current_result["timeout_limit_s"] = timeout_limit_s + extension_s
                self.current_result["events"].append(
                    {
                        "time": now,
                        "type": "timeout_extended_for_stt",
                        "new_limit_s": self.current_result["timeout_limit_s"],
                    }
                )
                self._save_report()
                QTimer.singleShot(500, self.wait_for_fixture_settle)
                return
            if self._fixture_has_completion():
                self.current_result["finished_at"] = now
                self.current_result["duration_s"] = max(0.0, now - started_at)
                self.current_result["_closed_at"] = now
                self.current_result.pop("timed_out", None)
                self._recently_closed_result = self.current_result
                self._save_report()
                self.current_fixture_idx += 1
                self.current_result = None
                QTimer.singleShot(500, self.next_fixture)
                return
            if self._fixture_response_in_progress():
                extension_s = 15.0 if not self.current_result.get("_timeout_extended") else 8.0
                self.current_result["_timeout_extended"] = True
                self.current_result["timeout_limit_s"] = timeout_limit_s + extension_s
                self.current_result["events"].append(
                    {
                        "time": now,
                        "type": "timeout_extended",
                        "new_limit_s": self.current_result["timeout_limit_s"],
                    }
                )
                QTimer.singleShot(500, self.wait_for_fixture_settle)
                return
            if not self.current_result.get("_timeout_fallback_attempted"):
                fallback_query = self._benchmark_fallback_query()
                if fallback_query:
                    self.current_result["_timeout_fallback_attempted"] = True
                    self.current_result["timeout_limit_s"] = timeout_limit_s + 15.0
                    self.current_result["events"].append(
                        {
                            "time": now,
                            "type": "timeout_fallback_triggered",
                            "query": fallback_query,
                            "new_limit_s": self.current_result["timeout_limit_s"],
                        }
                    )
                    self._save_report()
                    self.app.generate_response(fallback_query, "speech", {"audio": fallback_query})
                    QTimer.singleShot(500, self.wait_for_fixture_settle)
                    return
            if self.current_result is not None:
                self.current_result["timed_out"] = True
                self.current_result["finished_at"] = now
                self.current_result["duration_s"] = max(0.0, now - started_at)
                self.current_result["_closed_at"] = now
                self._recently_closed_result = self.current_result
                self._save_report()
            self.current_fixture_idx += 1
            self.current_result = None
            QTimer.singleShot(500, self.next_fixture)
            return
        QTimer.singleShot(500, self.wait_for_fixture_settle)

    def _fixture_audio_processing_busy(self) -> bool:
        audio = getattr(self.app, "audio", None)
        if not audio:
            return False
        if not bool(getattr(audio, "_model_loaded", True)):
            return True
        if hasattr(audio, "has_pending_transcription_jobs"):
            try:
                if audio.has_pending_transcription_jobs():
                    return True
            except Exception:
                pass
        return False

    def _fixture_completion_is_settled(self) -> bool:
        """Completion is enough to advance once model and STT work are idle.

        Live query buffers can remain populated briefly after a hybrid-fast answer.
        They should not make the benchmark wait until the full fixture timeout.
        """
        if self._fixture_response_in_progress():
            return False
        audio = getattr(self.app, "audio", None)
        if audio and hasattr(audio, "has_pending_transcription_jobs"):
            try:
                if audio.has_pending_transcription_jobs():
                    return False
            except Exception:
                pass
        overlay = getattr(self.app, "overlay", None)
        if bool(getattr(overlay, "_is_streaming", False)):
            return False
        return True

    def _fixture_is_quiet(self) -> bool:
        audio = getattr(self.app, "audio", None)

        # Not quiet while Whisper model is still loading (takes ~14s cold start).
        # Without this the benchmark exits before any STT transcript can arrive.
        if audio and not bool(getattr(audio, "_model_loaded", True)):
            return False

        # Not quiet if there are pending STT jobs in the queue.
        if audio and hasattr(audio, "has_pending_transcription_jobs"):
            try:
                if audio.has_pending_transcription_jobs():
                    return False
            except Exception:
                pass

        # If audio just finished but we've received zero local STT transcripts,
        # give a 20s grace window before declaring quiet. Whisper may still be
        # working through the queue (it initialises lazily on first call).
        if self.current_result:
            if self._fixture_has_dispatched_response() and not self._fixture_has_completion():
                return False
            audio_finished_at = float(self.current_result.get("audio_finished_at", 0.0) or 0.0)
            has_local_transcript = any(
                not item.get("source")  # local ASR has no source tag
                for item in self.current_result.get("transcripts", [])
                if isinstance(item, dict)
            )
            if audio_finished_at > 0.0 and not has_local_transcript:
                grace_elapsed = time.time() - audio_finished_at
                if grace_elapsed < 20.0:
                    return False

        if bool(getattr(self.app.overlay, "_is_streaming", False)):
            return False
        return True

    def _fixture_response_in_progress(self) -> bool:
        overlay = getattr(self.app, "overlay", None)
        ai = getattr(self.app, "ai", None)
        return bool(
            getattr(overlay, "_is_streaming", False)
            or getattr(overlay, "_pending_thinking", False)
            # Standard fallback response is being streamed
            or getattr(ai, "_is_generating", False)
            or getattr(ai, "_generation_active", False)
        )

    def _standard_response_in_progress(self) -> bool:
        overlay = getattr(self.app, "overlay", None)
        ai = getattr(self.app, "ai", None)
        return bool(
            getattr(overlay, "_is_streaming", False)
            or getattr(ai, "_is_generating", False)
            or getattr(ai, "_generation_active", False)
        )

    def _fixture_timeout_limit_s(self) -> float:
        """Calculate per-fixture timeout with enough headroom for:
        - Audio playback duration
        - Auto answer settle window
        - AI response round-trip (~10s buffer)

        Previously audio_duration + 20s was too tight — the response pipeline may need extra time after audio ends.
        The benchmark limit was expiring before Groq returned.
        """
        audio_duration_s = 0.0
        try:
            if self.samples is not None and self.sr:
                audio_duration_s = float(len(self.samples)) / float(self.sr)
        except Exception:
            audio_duration_s = 0.0
        # Auto answer settle window plus AI response buffer
        turn_timeout_s = 20.0
        return max(50.0, min(120.0, audio_duration_s + turn_timeout_s + 15.0))
            
    def finish_suite(self):
        safe_print("Finishing test suite and saving report...")
        if getattr(self.app, "session_active", False):
            self.app.end_session()
        self._save_report()
        safe_print(f"Report saved to {self.out_path}")
        sys.stdout.flush()
        sys.stderr.flush()
        try:
            self.app.shutdown()
        finally:
            self.app.qt_app.quit()

def main():
    qInstallMessageHandler(_benchmark_qt_message_handler)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="Path to directory containing wav files")
    parser.add_argument("--out", type=str, default="auto_mode_report.json", help="Output JSON path")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of wav files to run")
    parser.add_argument("--files", nargs="*", default=None, help="Optional explicit wav filenames to run")
    parser.add_argument("--standard-only", action="store_true", help="Run fixtures through standard direct-dispatch mode instead of Auto Mode")
    parser.add_argument(
        "--headless-ui",
        action="store_true",
        help="Suppress QTextEdit rendering while still recording UI completion callbacks",
    )
    args = parser.parse_args()
    
    qt_app = QApplication.instance() or QApplication(sys.argv)
    config = Config(str(ROOT / "config.yaml"))
    config.set("capture.audio.mode", "system")
    config.set("capture.screen.enabled", False)
    config.set("ai.auto_mode.enabled", not args.standard_only)
    
    # Init OpenAssist
    app = OpenAssistApp(config)
    app.overlay.hide()
    app.mini_overlay.hide()
    app.config.set("capture.audio.mode", "system")
    app.config.set("capture.screen.enabled", False)
    if hasattr(app.audio, "set_hardware_capture_suspended"):
        app.audio.set_hardware_capture_suspended(True, "benchmark-fixtures-pre-run")

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    tester = AutoModeTester(
        app,
        out_path,
        standard_only=args.standard_only,
        headless_ui=args.headless_ui,
    )
    
    fixture_dir = Path(args.dir)
    if args.files:
        requested = []
        seen = set()
        for raw_name in args.files:
            name = (raw_name or "").strip()
            if name and name not in seen:
                requested.append(name)
                seen.add(name)
        wav_files = [fixture_dir / name for name in requested]
        missing = [str(path.name) for path in wav_files if not path.exists()]
        if missing:
            safe_print(f"Missing fixture files: {', '.join(missing)}")
            sys.exit(1)
    else:
        wav_files = sorted(list(fixture_dir.glob("*.wav")))
    if args.limit and args.limit > 0:
        wav_files = wav_files[: args.limit]
    if not wav_files:
        safe_print(f"No .wav files found in {fixture_dir}")
        sys.exit(1)
        
    QTimer.singleShot(1000, lambda: tester.start_suite(wav_files))
    
    # Start app main loop
    app.run()

if __name__ == "__main__":
    main()
