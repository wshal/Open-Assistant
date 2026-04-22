"""
AI Orchestration Engine - OpenAssist v4.2 (ModeProfile Edition).
RESTORED: Async Task Control, RAG caching, and Cloud-ASR Correction.
FIXED: Removed internal loops; App Master Loop now controls all AI execution.
FIXED: Added latency tracking for performance monitoring.
NEW: ModeManager wired in — engine reads live Mode objects, not bare strings.
"""

import asyncio
import re
import time
import inspect
from functools import lru_cache
from typing import Optional, List, Dict, Any
from collections import OrderedDict
from PyQt6.QtCore import QObject, pyqtSignal

from ai.providers import init_providers
from ai.prompts import PromptBuilder
from ai.history import ResponseHistory
from ai.detectors.question_detector import QuestionDetector
from ai.router import SmartRouter
from ai.parallel import ParallelInference
from utils.logger import setup_logger

logger = setup_logger(__name__)


class AIEngine(QObject):
    """
    Handles model selection, context synthesis, and async response streaming.
    Consolidated to run on the App's background asyncio loop.
    """

    response_chunk = pyqtSignal(str)
    response_complete = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    provider_status = pyqtSignal(dict)  # Emits health for UI dashboard

    def __init__(self, config, history: ResponseHistory, rag=None, mode_manager=None):
        super().__init__()
        self.config = config
        self.history = history
        self.rag = rag
        self.prompts = PromptBuilder()
        self.detector = QuestionDetector(config)
        # ModeManager: first-class dependency — resolves live Mode objects per request
        self._mode_manager = mode_manager

        self._providers = {}
        self._active_provider_id = config.get("ai.fixed_provider", "")
        self._is_cancelled = False
        self._health_task = None
        self._loop = None
        self._router = None
        self._selected_tier = "balanced"
        self._parallel = None

        # Provider speed ranking (fastest first)
        self._provider_priority = ["groq", "cerebras", "gemini", "together", "ollama"]

        # Query complexity patterns
        self._simple_patterns = [
            r"^(hi|hello|hey|how are you|thanks|thank you)\b",
            r"^(what|who|where|when|why|how)\s+(is|are|do|does|can)\b",
            r"^(define|explain|summarize|tell me about)\s+\w+\??$",
        ]
        self._complex_patterns = [
            r"\b(analyze|compare|evaluate|design|architect)\b",
            r"\b(code|function|algorithm|implement|optimize)\b",
            r"\b(reasoning|logic|mathematical|calculate)\b",
            r"\b(essay|article|document|write a|report)\b",
            r"\b(interview|prepare|coach)\b",
        ]
        self._reasoning_patterns = [
            r"\b(why|because|reason|explain|logic)\b",
            r"\b(prove|demonstrate|show that)\b",
            r"\b(if\s+.+\s+then|therefore|thus|hence)\b",
        ]

        # ── RAG Cache (Midnight Restoration) ─────────────────────────────────
        self._rag_cache = OrderedDict()   # query → (context, expiry)
        self._cache_ttl = 60
        self._max_cache_size = 100

        # ── Background RAG Prefetch ───────────────────────────────────────────
        # Stores pre-fetched RAG results keyed by a short context fingerprint.
        # Populated by prefetch_rag() before the user submits a query.
        self._rag_prefetch: dict = {}          # fingerprint → (context_str, expiry)
        self._prefetch_ttl: int = 90           # seconds a prefetch stays valid
        self._prefetch_lock = asyncio.Lock()   # prevent concurrent prefetch storms

        # ── Session Context (user-defined persona / instructions) ─────────────
        # Set by app.py from AppState.session_context each time it changes.
        # Injected as the top block of every system prompt for the session.
        self._session_context: str = ""

    @staticmethod
    def _cooldown_seconds_for_error(exc: Exception) -> int:
        """
        Best-effort cooldown for transient provider failures so we don't keep
        hammering an overloaded endpoint on the next request.
        """
        msg = str(exc or "").lower()
        if any(
            k in msg
            for k in [
                "503",
                "unavailable",
                "overloaded",
                "high demand",
                "service unavailable",
            ]
        ):
            return 60
        if any(k in msg for k in ["429", "rate limit", "ratelimit", "too many requests"]):
            return 20
        if any(k in msg for k in ["timeout", "timed out", "connection", "connectorerror"]):
            return 10
        return 0

    def _maybe_cooldown_provider(self, provider, exc: Exception) -> None:
        seconds = self._cooldown_seconds_for_error(exc)
        if seconds <= 0:
            return
        try:
            if provider and hasattr(provider, "disable"):
                provider.disable(seconds=seconds)
        except Exception:
            # Cooldown is best-effort; never fail the request flow due to it.
            pass

    def set_session_context(self, text: str):
        """Update the active session context injected into all system prompts."""
        self._session_context = (text or "").strip()
        logger.debug(f"Session context updated ({len(self._session_context)} chars)")

    def warmup(self):
        """Initializes AI providers and checks availability."""
        try:
            # If warmup is called multiple times (e.g. hot-apply settings), ensure
            # we clean up any long-lived network resources (aiohttp sessions, etc.)
            # from the previous provider instances to prevent leak warnings.
            if self._providers:
                self.close_providers()

            self._providers = init_providers(self.config)
            logger.info(f"AI Engine initialized: {list(self._providers.keys())}")

            # Initialize router
            self._router = SmartRouter(self.config, self._providers)

            # Initialize parallel inference if enabled
            if self.config.get("ai.parallel.enabled", False):
                self._parallel = ParallelInference(self.config, self._router)
                logger.info("Parallel inference enabled")

            # Auto-select best provider if none fixed
            # Smart Provider Selection: Speed-first with complexity routing
            # Priority: groq (fastest) > gemini > ollama (local, no API)
            configured = self.config.get("ai.fixed_provider", "")
            if (
                configured
                and configured in self._providers
                and self._providers[configured].enabled
            ):
                self._active_provider_id = configured
                logger.info(f"Using configured provider: {configured}")
            else:
                # Auto-select: groq > cerebras > gemini based on speed
                for p in ["groq", "cerebras", "gemini", "together", "ollama"]:
                    if p in self._providers and self._providers[p].enabled:
                        self._active_provider_id = p
                        logger.info(f"Auto-selected provider: {p} (fastest available)")
                        break

            if not self._active_provider_id and self._providers:
                self._active_provider_id = list(self._providers.keys())[0]

            logger.info(f"Active Provider: {self._active_provider_id}")
        except Exception as e:
            logger.error(f"AI Warmup Error: {e}")

    async def aclose_providers(self):
        """Async best-effort provider cleanup (aiohttp sessions, etc.)."""
        providers = list((self._providers or {}).values())
        for prov in providers:
            close_fn = getattr(prov, "close", None)
            if not close_fn:
                continue
            try:
                if inspect.iscoroutinefunction(close_fn):
                    await close_fn()
                else:
                    close_fn()
            except Exception:
                # Cleanup is best-effort; never fail app flow due to it.
                pass

    def close_providers(self, timeout_s: float = 2.0):
        """Sync wrapper to close providers on the engine's asyncio loop when possible."""
        try:
            loop = self._loop
            if loop and loop.is_running():
                fut = asyncio.run_coroutine_threadsafe(self.aclose_providers(), loop)
                try:
                    fut.result(timeout=timeout_s)
                except Exception:
                    pass
                return
        except Exception:
            # Fall back to thread-local attempt below.
            pass

        # Fallback: try to run cleanup in this thread (may not match the original loop).
        try:
            asyncio.run(self.aclose_providers())
        except RuntimeError:
            # Already in an event loop in this thread; schedule and move on.
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.aclose_providers())
            except Exception:
                pass
        except Exception:
            pass

    def cancel(self):
        """Standardized cancellation flag for prioritized overrides."""
        self._is_cancelled = True
        logger.info("AI: Generation cancel flag set")

    async def generate_response(
        self,
        query: str,
        nexus_snapshot: Dict[str, Any],
        screen_context: Optional[str] = None,
        audio_context: Optional[str] = None,
        origin: Optional[str] = None,
        request_metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        RESTORED: Main async generation task.
        Control flow managed by the App Master Loop.
        """
        self._is_cancelled = False
        start_time = time.time()
        stage_timings = {}
        first_token_time = None
        request_metadata = dict(request_metadata or {})
        request_started_at = request_metadata.get("request_started_at", start_time)
        stage_timings["request_to_ai_start_ms"] = (start_time - request_started_at) * 1000

        # Resolve the live Mode object — use ModeManager if available
        mode_obj = self._mode_manager.current if self._mode_manager else None
        mode_id = mode_obj.name if mode_obj else self.config.get("ai.mode", "general")

        # Smart routing: analyse query complexity and select provider
        complexity = self._analyze_query_complexity(query)
        preferred_providers = None
        if origin in {"manual", "speech"}:
            cfg_text = self.config.get("ai.text.preferred_providers", None)
            if isinstance(cfg_text, list) and cfg_text:
                # Respect explicit user priority order for manual/audio queries.
                preferred_providers = [p for p in cfg_text if isinstance(p, str) and p.strip()]

        if not preferred_providers:
            preferred_providers = self._preferred_providers_for_complexity(
                complexity, mode_obj or mode_id
            )

        # Use parallel inference if enabled
        use_parallel = self._parallel and self.config.get("ai.parallel.enabled", False)

        if use_parallel:
            # Build prompts first for parallel — pass Mode object for profile-aware limits
            _mode_arg = mode_obj or mode_id
            sys_prompt = self.prompts.system(_mode_arg)
            user_msg = self.prompts.user(
                query=query,
                screen=screen_context or "",
                audio=audio_context or nexus_snapshot.get("recent_audio", ""),
                rag="",
                mode=_mode_arg,
                origin=origin,
                nexus=nexus_snapshot
            )

            try:
                full_response = await self._parallel.generate(
                    sys_prompt, user_msg, task=mode_id
                )

                if self._is_cancelled:
                    return

                latency_ms = (time.time() - start_time) * 1000
                self.history.add(
                    query,
                    full_response,
                    provider="parallel",
                    mode=mode_id,
                    latency=latency_ms,
                    metadata={
                        "stage_timings": {
                            **stage_timings,
                            "request_to_complete_ms": (time.time() - request_started_at) * 1000,
                        },
                        "request_metadata": request_metadata,
                    },
                )

                # P0.4 FIX: Emit chunks so the UI shows streaming feedback.
                # Parallel returns a full string — stream it in word-level batches
                # (~50 chars each) so append_response() fires and the user sees
                # text appearing progressively rather than a blank-then-flash.
                words = full_response.split(" ")
                batch, batch_size = [], 50
                for word in words:
                    batch.append(word)
                    chunk = " ".join(batch) + " "
                    if len(chunk) >= batch_size:
                        self.response_chunk.emit(chunk)
                        batch = []
                        await asyncio.sleep(0)  # yield to event loop between batches
                if batch:
                    self.response_chunk.emit(" ".join(batch))

                self.response_complete.emit(full_response)
                logger.info(f"Parallel AI: Response complete ({latency_ms:.0f}ms)")
                return
            except Exception as e:
                logger.warning(
                    f"Parallel inference failed, falling back to single provider: {e}"
                )

        provider, tier = self._select_provider(mode_obj or mode_id, complexity, preferred_providers)
        self._selected_tier = tier
        if not provider:
            self.error_occurred.emit("No available AI provider found.")
            return

        self._active_provider_id = provider.name
        logger.debug(
            f"Provider selection -> complexity={complexity}, provider={provider.name}, tier={tier}"
        )

        try:
            # 1 & 2.  RAG + Refinement — run concurrently to hide latency
            # ─────────────────────────────────────────────────────────────────
            # Previously sequential: wait(RAG) + wait(refine) = ~600ms worst-case.
            # Now parallel:           wait(max(RAG, refine))   = ~400ms worst-case.
            # Cache / prefetch hits remain zero-latency (no coroutine created).

            # ── RAG: resolve from cache or build coroutine for parallel exec ─────────
            rag_context = ""
            rag_coro = None
            if self.rag:
                now = time.time()
                cache_key = query.lower().strip()[:100]

                # Prefetch hit: zero-latency, no coroutine needed
                prefetch_fp = self._rag_prefetch_fingerprint(
                    screen_context or nexus_snapshot.get("latest_ocr", ""),
                    audio_context or nexus_snapshot.get("recent_audio", ""),
                )
                prefetch_hit = self._rag_prefetch.get(prefetch_fp)
                if prefetch_hit and now < prefetch_hit[1]:
                    rag_context = prefetch_hit[0]
                    logger.debug("RAG: Prefetch Hit (zero-latency)")
                # Regular cache hit: also zero-latency
                elif cache_key in self._rag_cache and now < self._rag_cache[cache_key][1]:
                    rag_context = self._rag_cache[cache_key][0]
                    self._rag_cache.move_to_end(cache_key)
                    logger.debug("RAG: Cache Hit")
                else:
                    # Miss — schedule for parallel execution
                    rag_coro = self.rag.query(query)

            # ── Refinement: decide whether to run, build coroutine if so ────────
            refined_audio = audio_context
            refine_coro = None
            is_general_knowledge = (
                origin == "manual"
                and self.prompts._is_general_knowledge_query(query)
            )
            skip_refinement = complexity == "simple" or is_general_knowledge
            needs_refinement = (
                audio_context
                and not skip_refinement
                and len(audio_context.split()) > 5
                and self.config.get("capture.audio.correct_transcript", True)
            )
            if needs_refinement:
                refiner_id = self.config.get("capture.audio.correction_provider", "groq")
                refiner = self._providers.get(refiner_id)
                if refiner:
                    refine_coro = self._refine_transcript(audio_context, refiner)
            elif skip_refinement:
                logger.debug(
                    f"Refinement skipped: complexity={complexity}, "
                    f"origin={origin}, general_knowledge={is_general_knowledge}"
                )

            # ── Parallel execution: gather only what needs a network call ────────
            parallel_start = time.time()
            if rag_coro and refine_coro:
                # Both need a live call — run concurrently
                rag_results, refined_text = await asyncio.gather(rag_coro, refine_coro)
                if rag_results:
                    rag_context = "\n".join(rag_results)
                    now2 = time.time()
                    self._rag_cache[cache_key] = (rag_context, now2 + self._cache_ttl)
                    if len(self._rag_cache) > self._max_cache_size:
                        self._rag_cache.popitem(last=False)
                if refined_text:
                    refined_audio = refined_text
                logger.debug(
                    f"Parallel RAG+Refine: {(time.time() - parallel_start)*1000:.0f}ms"
                )
            elif rag_coro:
                # Only RAG needed
                rag_results = await rag_coro
                if rag_results:
                    rag_context = "\n".join(rag_results)
                    now2 = time.time()
                    self._rag_cache[cache_key] = (rag_context, now2 + self._cache_ttl)
                    if len(self._rag_cache) > self._max_cache_size:
                        self._rag_cache.popitem(last=False)
            elif refine_coro:
                # Only refinement needed
                result = await refine_coro
                if result:
                    refined_audio = result

            stage_timings["rag_refine_parallel_ms"] = (time.time() - start_time) * 1000


            # 3. Prompt Synthesis — pass Mode object for profile-aware context limits
            _mode_arg = mode_obj or mode_id
            sys_prompt = self.prompts.system(
                _mode_arg,
                session_context=getattr(self, "_session_context", ""),
            )

            # Resolve follow-up queries using recent history so short ambiguous
            # queries like "give me an example" inherit the previous topic context.
            resolved_query = self._resolve_followup_query(query)

            # Build conversation history block from last 3 turns (capped at 1500 chars).
            # Injected into the prompt so the model understands what was discussed.
            history_block = self._build_history_block(max_turns=3, max_chars=1500)

            user_msg = self.prompts.user(
                query=resolved_query,
                screen=screen_context or nexus_snapshot.get("latest_ocr", ""),
                audio=refined_audio or nexus_snapshot.get("full_audio_history", ""),
                rag=rag_context,
                mode=_mode_arg,
                origin=origin,
                nexus=nexus_snapshot,
                history=history_block,
            )

            # Optional low-latency "race" for text: run top providers concurrently and use the first success.
            # Note: we race non-streaming calls, so the UI won't see token-by-token streaming.
            if (
                origin in {"manual", "speech"}
                and bool(self.config.get("ai.text.race_enabled", False))
                and isinstance(preferred_providers, list)
                and len(preferred_providers) >= 2
            ):
                candidates = []
                for pid in preferred_providers:
                    p = self._providers.get(pid)
                    if p and getattr(p, "enabled", False) and p.check_rate():
                        candidates.append(p)
                    if len(candidates) >= 2:
                        break

                if len(candidates) >= 2:
                    tasks = {
                        asyncio.create_task(p.generate(sys_prompt, user_msg, tier)): p
                        for p in candidates
                    }
                    try:
                        while tasks:
                            done, _pending = await asyncio.wait(
                                tasks.keys(), return_when=asyncio.FIRST_COMPLETED
                            )
                            for t in done:
                                p = tasks.pop(t, None)
                                try:
                                    raced = (t.result() or "").strip()
                                except Exception:
                                    continue

                                if self._is_cancelled:
                                    return

                                for other in list(tasks.keys()):
                                    other.cancel()
                                tasks.clear()

                                latency_ms = (time.time() - start_time) * 1000
                                providers_tried = [pp.name for pp in candidates]
                                self.history.add(
                                    query,
                                    raced,
                                    provider=p.name if p else "race",
                                    mode=mode_id,
                                    latency=latency_ms,
                                    metadata={
                                        "stage_timings": {
                                            **stage_timings,
                                            "request_to_complete_ms": (time.time() - request_started_at) * 1000,
                                        },
                                        "request_metadata": request_metadata,
                                        "providers_tried": providers_tried,
                                        "race": True,
                                        "had_screen": bool((nexus_snapshot or {}).get("latest_ocr", "").strip()),
                                        "had_audio": bool((nexus_snapshot or {}).get("full_audio_history", "").strip()),
                                        "had_rag": bool(rag_context.strip()) if rag_context else False,
                                    },
                                )
                                if raced:
                                    self.response_chunk.emit(raced)
                                self.response_complete.emit(raced)
                                return
                    finally:
                        for t in list(tasks.keys()):
                            try:
                                t.cancel()
                            except Exception:
                                pass

            # 4. Streamed Generation — with provider fallback chain (P1.4)
            # If the primary provider fails mid-stream, we transparently try the
            # next available provider from the preferred list before giving up.
            full_response = ""
            providers_tried = [provider.name]
            generation_succeeded = False

            # Build a fallback queue: preferred list minus already-selected primary
            fallback_queue = [
                pid for pid in preferred_providers
                if pid != provider.name
                and pid in self._providers
                and getattr(self._providers[pid], "enabled", False)
            ]

            current_provider = provider
            while True:
                try:
                    async for chunk in current_provider.generate_stream(sys_prompt, user_msg):
                        if self._is_cancelled:
                            logger.warning("AI: Stream cancelled mid-token")
                            return

                        full_response += chunk
                        if not first_token_time:
                            first_token_time = time.time()
                            stage_timings["first_token_ms"] = (first_token_time - start_time) * 1000
                            stage_timings["request_to_first_token_ms"] = (
                                first_token_time - request_started_at
                            ) * 1000
                        self.response_chunk.emit(chunk)
                    generation_succeeded = True
                    break  # Stream completed successfully

                except Exception as stream_err:
                    logger.warning(
                        f"AI: Provider '{current_provider.name}' failed during stream "
                        f"({stream_err}). Trying fallback..."
                    )
                    self._maybe_cooldown_provider(current_provider, stream_err)
                    full_response = ""  # Discard any partial output

                    if not fallback_queue:
                        logger.error("AI: All providers exhausted — no response generated.")
                        self.error_occurred.emit(
                            f"All providers failed. Last error: {stream_err}"
                        )
                        return

                    next_id = fallback_queue.pop(0)
                    current_provider = self._providers[next_id]
                    providers_tried.append(next_id)
                    logger.info(f"AI: Falling back to '{next_id}'")

            # 5. Finalize
            if not self._is_cancelled:
                latency_ms = (time.time() - start_time) * 1000
                self.history.add(
                    query,
                    full_response,
                    provider=current_provider.name,  # actual provider that served the response
                    mode=mode_id,
                    latency=latency_ms,
                    metadata={
                        "stage_timings": {
                            **stage_timings,
                            "request_to_complete_ms": (time.time() - request_started_at) * 1000,
                        },
                        "request_metadata": request_metadata,
                        "providers_tried": providers_tried,
                        # P2.7: Provenance flags — which context sources contributed
                        "had_screen": bool((nexus_snapshot or {}).get("latest_ocr", "").strip()),
                        "had_audio": bool((nexus_snapshot or {}).get("full_audio_history", "").strip()),
                        "had_rag": bool(rag_context.strip()) if rag_context else False,
                    }
                )
                self.response_complete.emit(full_response)
                logger.info(f"AI: Response complete ({latency_ms:.0f}ms) | {stage_timings}")

        except Exception as e:
            logger.error(f"AI Engine Runtime Error: {e}")
            self.error_occurred.emit(str(e))

    # ── Conversation memory helpers ─────────────────────────────────────────────

    def _build_history_block(self, max_turns: int = 3, max_chars: int = 1500) -> str:
        """Serialise the last N conversation turns for injection into the prompt.

        Returns an empty string when there is no history yet (first query).
        Each turn is formatted as a compact Q/A pair. The total char budget is
        shared across all turns to avoid inflating the prompt on long sessions.
        """
        entries = self.history.get_last(max_turns)
        if not entries:
            return ""

        per_turn_budget = max_chars // max(len(entries), 1)
        lines = []
        for e in entries:
            q_short = e.query[:200]
            a_budget = max(100, per_turn_budget - len(q_short) - 20)
            a_short = e.response[:a_budget]
            if len(e.response) > a_budget:
                a_short += "..."
            lines.append(f"User: {q_short}\nAssistant: {a_short}")

        block = "\n\n".join(lines)
        return block[:max_chars]

    # Follow-up query patterns — short queries that need prior context to make sense
    _FOLLOWUP_PATTERNS = [
        "give me an example", "show me an example", "example code",
        "example of that", "can you show", "can you give",
        "more examples", "another example", "same for",
        "now for", "and also", "what about", "how about",
        "explain more", "elaborate", "tell me more", "go deeper",
        "expand on that", "continue", "next step", "and then",
        "how would i", "how do i do that", "how to do that",
    ]

    def _resolve_followup_query(self, query: str) -> str:
        """Expand ambiguous follow-up queries with the previous conversation topic.

        Example:
            Previous Q: "what is react?"
            Current Q:  "give me an example code"
            Resolved:   "give me an example code [continuing from: what is react?]"
        """
        q_lower = query.lower().strip()
        is_short = len(query.split()) <= 10
        is_followup = any(p in q_lower for p in self._FOLLOWUP_PATTERNS)

        if not (is_short and is_followup):
            return query

        entries = self.history.get_last(1)
        if not entries:
            return query

        prev_query = entries[-1].query
        if prev_query.lower().strip() == q_lower:
            return query

        topic = prev_query[:80] + ("..." if len(prev_query) > 80 else "")
        resolved = f"{query} [continuing from: {topic}]"
        logger.debug(f"Follow-up resolved: '{query}' -> '{resolved}'")
        return resolved

    async def _refine_transcript(self, raw_text: str, provider) -> str:

        """Midnight Restoration: Uses Cloud-Speed models to fix ASR typos."""
        prompt = f"Fix common ASR typos in this text. Keep meaning exact. Output ONLY fixed text.\n\n{raw_text}"
        try:
            fixed = ""
            async for chunk in provider.generate_stream(
                "You are a text correction engine.", prompt
            ):
                if self._is_cancelled:
                    return raw_text
                fixed += chunk
            return fixed.strip() if fixed else raw_text
        except:
            return raw_text

    async def generate_quick_response(
        self,
        nexus_snapshot: Dict[str, Any],
        screen_context: str = "",
        audio_context: str = "",
    ) -> str:
        """
        Fast path for hotkey-triggered context answers.
        Now fully mode-aware: uses mode.quick_answer_query and mode.quick_answer_format.
        """
        self._is_cancelled = False
        start_time = time.time()

        # Resolve Mode object for profile-aware behaviour
        mode_obj = self._mode_manager.current if self._mode_manager else None
        mode_id = mode_obj.name if mode_obj else self.config.get("ai.mode", "general")

        # Use the mode's own quick-answer query — not a generic fallback
        if mode_obj and hasattr(mode_obj, "quick_answer_query") and mode_obj.quick_answer_query:
            query = mode_obj.quick_answer_query
        else:
            query = (
                "Using the latest live context, give a quick answer. "
                "First summarise the current audio briefly, then give the most useful "
                "immediate response in 2-4 bullets."
            )

        # For quick answers, prefer the mode's own fast providers
        if mode_obj and hasattr(mode_obj, "preferred_providers") and mode_obj.preferred_providers:
            quick_preferred = mode_obj.preferred_providers
        else:
            quick_preferred = ["groq", "cerebras", "gemini", "together", "ollama"]

        provider, tier = self._select_provider(
            mode_obj or mode_id,
            "simple",
            quick_preferred,
        )
        if not provider:
            self.error_occurred.emit("No available AI provider found.")
            return ""

        summarized_audio = audio_context or nexus_snapshot.get("recent_audio", "")
        if len(summarized_audio.split()) > 40:
            summarized_audio = await self._summarize_audio_fast(summarized_audio)

        _mode_arg = mode_obj or mode_id
        sys_prompt = self.prompts.system(_mode_arg)
        user_msg = self.prompts.user(
            query=query,
            screen=screen_context or nexus_snapshot.get("latest_ocr", ""),
            audio=summarized_audio,
            rag="",
            mode=_mode_arg,
            origin="quick",
            nexus=nexus_snapshot,
        )

        try:
            full_response = ""
            first_token_time = None
            async for chunk in provider.generate_stream(sys_prompt, user_msg):
                if self._is_cancelled:
                    return ""
                if first_token_time is None:
                    first_token_time = time.time()
                full_response += chunk
                self.response_chunk.emit(chunk)

            latency_ms = (time.time() - start_time) * 1000
            stage_timings = {"request_to_complete_ms": latency_ms}
            if first_token_time is not None:
                stage_timings["request_to_first_token_ms"] = (first_token_time - start_time) * 1000

            had_screen = bool((screen_context or nexus_snapshot.get("latest_ocr", "")).strip())
            had_audio = bool((summarized_audio or "").strip())
            self.history.add(
                query,
                full_response,
                provider=provider.name,
                mode=mode_id,
                latency=latency_ms,
                metadata={
                    "quick": True,
                    "stage_timings": stage_timings,
                    "providers_tried": [provider.name],
                    "had_screen": had_screen,
                    "had_audio": had_audio,
                    "had_rag": False,
                },
            )
            self.response_complete.emit(full_response)
            return full_response
        except Exception as e:
            logger.error(f"Quick response error: {e}")
            self.error_occurred.emit(str(e))
            return ""

    async def _summarize_audio_fast(self, audio_text: str) -> str:
        """Compress recent audio into a short summary using the fastest available provider."""
        # Use mode's preferred fast provider if available
        mode_obj = self._mode_manager.current if self._mode_manager else None
        fast_preferred = (
            mode_obj.preferred_providers
            if mode_obj and getattr(mode_obj, "preferred_providers", None)
            else ["groq", "cerebras", "gemini", "together", "ollama"]
        )
        provider, tier = self._select_provider(
            mode_obj or "general",
            "simple",
            fast_preferred,
        )
        if not provider:
            return audio_text

        prompt = (
            "Summarize this recent audio in 1-2 short bullets. "
            "Keep only the key ask, decision, or topic. No filler.\n\n"
            f"{audio_text}"
        )
        try:
            summary = ""
            async for chunk in provider.generate_stream(
                "You compress spoken context into very short, accurate notes.",
                prompt,
            ):
                if self._is_cancelled:
                    return audio_text
                summary += chunk
            return summary.strip() or audio_text
        except Exception:
            return audio_text

    async def analyze_image_response(
        self,
        query: str,
        image_bytes: bytes,
        nexus_snapshot: Dict[str, Any],
        screen_context: str = "",
        audio_context: str = "",
    ) -> str:
        """Run screenshot analysis through the best available vision-capable provider."""
        self._is_cancelled = False
        start_time = time.time()
        mode_id = self.config.get("ai.mode", "general")
        sys_prompt = self.prompts.system(mode_id)
        user_msg = self.prompts.user(
            query=query,
            screen=screen_context or nexus_snapshot.get("latest_ocr", ""),
            audio=audio_context or nexus_snapshot.get("full_audio_history", ""),
            rag="",
            mode=mode_id,
            origin="screen_analysis",
            nexus=nexus_snapshot,
        )

        preferred = []
        fixed = self.config.get("ai.fixed_provider", "")
        if fixed:
            preferred.append(fixed)

        cfg_order = self.config.get("ai.vision.preferred_providers", None)
        if isinstance(cfg_order, list) and cfg_order:
            for name in cfg_order:
                if name and name not in preferred:
                    preferred.append(name)
        else:
            for name in ["gemini", "ollama"]:
                if name not in preferred:
                    preferred.append(name)

        if bool(self.config.get("ai.vision.local_only", False)):
            preferred = ["ollama"]
        elif self.config.get("ai.vision.allow_paid_fallback", False):
            if "openai" not in preferred:
                preferred.append("openai")
        else:
            preferred = [p for p in preferred if p != "openai"]

        candidates = []
        for name in preferred:
            provider = self._providers.get(name)
            if (
                provider
                and provider.enabled
                and provider.check_rate()
                and getattr(provider, "supports_vision", lambda: False)()
            ):
                candidates.append(provider)

        if not candidates:
            raise Exception("No vision-capable provider available for screenshot analysis.")

        last_error = None
        providers_tried = []

        # Optional low-latency "race": run providers concurrently and use the first success.
        # Note: we race non-streaming calls to keep the implementation simple and deterministic.
        # Allow providers to use a dedicated vision-tier model if configured.
        vision_tier = "vision"

        async def _analyze_one(p):
            """Call provider vision method with best-effort tier support."""
            try:
                return await p.analyze_image(
                    sys_prompt,
                    user_msg,
                    image_bytes,
                    mime_type="image/png",
                    tier=vision_tier,
                )
            except TypeError as exc:
                # Backward-compatible: some providers/test doubles don't accept tier.
                if "unexpected keyword argument" in str(exc) and "tier" in str(exc):
                    return await p.analyze_image(
                        sys_prompt, user_msg, image_bytes, mime_type="image/png"
                    )
                raise

        async def _analyze_one_stream(p):
            """Stream provider vision output with best-effort tier support."""
            try:
                async for chunk in p.analyze_image_stream(
                    sys_prompt,
                    user_msg,
                    image_bytes,
                    mime_type="image/png",
                    tier=vision_tier,
                ):
                    yield chunk
                return
            except TypeError as exc:
                if "unexpected keyword argument" in str(exc) and "tier" in str(exc):
                    async for chunk in p.analyze_image_stream(
                        sys_prompt,
                        user_msg,
                        image_bytes,
                        mime_type="image/png",
                    ):
                        yield chunk
                    return
                raise

        if (
            bool(self.config.get("ai.vision.race_enabled", False))
            and not bool(self.config.get("ai.vision.local_only", False))
            and len(candidates) >= 2
        ):
            # Staggered race: start primary immediately, then start the next provider
            # after a short delay if we haven't gotten a success yet. This keeps
            # latency low while avoiding hammering every provider on every request.
            stagger_ms = int(self.config.get("ai.vision.race_stagger_ms", 900))
            stagger_s = max(stagger_ms / 1000.0, 0.0)

            providers_tried = []

            async def _run_one(p):
                if getattr(p, "supports_vision_stream", lambda: False)():
                    # Consume stream into a full string for parity with non-streaming.
                    out = ""
                    async for chunk in _analyze_one_stream(p):
                        if self._is_cancelled:
                            return ""
                        out += chunk
                    return out
                return await _analyze_one(p)

            tasks = {}
            next_idx = 0

            def _launch_one(idx: int):
                nonlocal next_idx
                p = candidates[idx]
                tasks[asyncio.create_task(_run_one(p))] = p
                providers_tried.append(p.name)
                next_idx = idx + 1

            _launch_one(0)
            next_launch_at = time.time() + stagger_s
            try:
                # Keep going until we have no running tasks AND no remaining providers to launch.
                while tasks or next_idx < len(candidates):
                    if not tasks and next_idx < len(candidates):
                        # All running attempts failed quickly; launch the next immediately.
                        _launch_one(next_idx)
                        next_launch_at = time.time() + stagger_s

                    timeout = None
                    if next_idx < len(candidates):
                        timeout = max(0.0, next_launch_at - time.time())

                    done, _pending = await asyncio.wait(
                        tasks.keys(),
                        timeout=timeout,
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    if not done:
                        # Time to launch the next provider in the race.
                        if next_idx < len(candidates):
                            _launch_one(next_idx)
                            next_launch_at = time.time() + stagger_s
                            continue
                        # No more providers to launch; keep waiting for completion.
                        continue

                    for t in done:
                        p = tasks.pop(t, None)
                        try:
                            response = t.result() or ""
                        except Exception as exc:
                            last_error = exc
                            self._maybe_cooldown_provider(p, exc)
                            continue

                        if self._is_cancelled:
                            return ""

                        if not response.strip():
                            last_error = Exception("Empty response from vision provider")
                            continue

                        # Cancel remaining attempts
                        for other in list(tasks.keys()):
                            other.cancel()
                        tasks.clear()

                        latency_ms = (time.time() - start_time) * 1000
                        self.history.add(
                            query,
                            response,
                            provider=p.name if p else "vision",
                            mode=mode_id,
                            latency=latency_ms,
                            metadata={
                                "vision": True,
                                "vision_race": True,
                                "stage_timings": {"request_to_complete_ms": latency_ms},
                                "providers_tried": providers_tried,
                                "had_screen": True,
                                "had_audio": bool((nexus_snapshot or {}).get("full_audio_history", "").strip()),
                                "had_rag": False,
                            },
                        )
                        self.history.add_screen_analysis(
                            query,
                            response,
                            provider=p.name if p else "vision",
                            metadata={
                                "vision": True,
                                "vision_race": True,
                                "providers_tried": providers_tried,
                                "had_screen": True,
                                "had_audio": bool((nexus_snapshot or {}).get("full_audio_history", "").strip()),
                                "had_rag": False,
                            },
                        )
                        # Emit a single chunk so the UI doesn't look blank.
                        if response:
                            self.response_chunk.emit(response)
                        self.response_complete.emit(response)
                        logger.info(
                            f"Vision race complete via {p.name if p else 'vision'} ({latency_ms:.0f}ms)"
                        )
                        return response
            finally:
                for t in list(tasks.keys()):
                    try:
                        t.cancel()
                    except Exception:
                        pass

            # If we raced and still got no result, do not retry the same providers
            # sequentially; bubble up the last error to the caller for OCR fallback.
            raise Exception(str(last_error) if last_error else "Vision analysis failed.")

        for provider in candidates:
            emitted_partial = False
            providers_tried.append(provider.name)
            try:
                response = ""
                if getattr(provider, "supports_vision_stream", lambda: False)():
                    async for chunk in _analyze_one_stream(provider):
                        if self._is_cancelled:
                            return ""
                        response += chunk
                        emitted_partial = True
                        self.response_chunk.emit(chunk)
                else:
                    response = await _analyze_one(provider)
                if self._is_cancelled:
                    return ""

                latency_ms = (time.time() - start_time) * 1000
                self.history.add(
                    query,
                    response,
                    provider=provider.name,
                    mode=mode_id,
                    latency=latency_ms,
                    metadata={
                        "vision": True,
                        "stage_timings": {"request_to_complete_ms": latency_ms},
                        "providers_tried": list(providers_tried),
                        "had_screen": True,
                        "had_audio": bool((nexus_snapshot or {}).get("full_audio_history", "").strip()),
                        "had_rag": False,
                    },
                )
                self.history.add_screen_analysis(
                    query,
                    response,
                    provider=provider.name,
                    metadata={
                        "vision": True,
                        "providers_tried": list(providers_tried),
                        "had_screen": True,
                        "had_audio": bool((nexus_snapshot or {}).get("full_audio_history", "").strip()),
                        "had_rag": False,
                    },
                )
                self.response_complete.emit(response)
                return response
            except Exception as exc:
                if emitted_partial:
                    self.response_complete.emit("")
                last_error = exc
                self._maybe_cooldown_provider(provider, exc)
                logger.warning(f"Vision analysis failed on {provider.name}: {exc}")

        raise Exception(str(last_error) if last_error else "Vision analysis failed.")

    def _chunk_response(self, text: str, chunk_size: int = 3):
        """
        Yield text in small character-level chunks for streaming effect.
        Default chunk_size=3 characters gives fluid perceived streaming.
        """
        for i in range(0, len(text), chunk_size):
            yield text[i:i + chunk_size]

    # ── Background RAG Prefetch ────────────────────────────────────────────────

    @staticmethod
    def _rag_prefetch_fingerprint(screen_text: str, audio_text: str) -> str:
        """
        Build a short fingerprint from current context to key the prefetch cache.
        Uses the first 150 chars of screen + last 100 chars of audio.
        """
        screen_snippet = (screen_text or "")[:150].strip()
        audio_snippet = (audio_text or "")[-100:].strip()
        return f"{screen_snippet}||{audio_snippet}"

    async def prefetch_rag(self, screen_text: str = "", audio_text: str = "") -> None:
        """
        Pre-fetch RAG context in the background before the user submits a query.
        Called from app.py whenever screen or audio context meaningfully changes.
        Results are stored in _rag_prefetch keyed by a context fingerprint.
        When generate_response() runs, it checks this cache first (zero-latency hit).
        """
        if not self.rag:
            return

        # Build a representative query hint from available context
        context_hint = " ".join([
            (screen_text or "")[:200],
            (audio_text or "")[-150:],
        ]).strip()
        if not context_hint or len(context_hint) < 10:
            return

        fingerprint = self._rag_prefetch_fingerprint(screen_text, audio_text)
        now = time.time()

        # Don’t re-prefetch if we have a fresh result for this context
        existing = self._rag_prefetch.get(fingerprint)
        if existing and now < existing[1]:
            logger.debug("RAG Prefetch: already fresh, skipping")
            return

        # Debounce: only one prefetch at a time
        if self._prefetch_lock.locked():
            logger.debug("RAG Prefetch: lock busy, skipping this cycle")
            return

        async with self._prefetch_lock:
            try:
                results = await self.rag.query(context_hint)
                if results:
                    context_str = "\n".join(results)
                    self._rag_prefetch[fingerprint] = (context_str, now + self._prefetch_ttl)
                    # Also seed the regular cache with a likely query key
                    cache_key = context_hint.lower()[:100]
                    self._rag_cache[cache_key] = (context_str, now + self._cache_ttl)
                    logger.debug(f"RAG Prefetch: stored ({len(results)} results)")
            except Exception as e:
                logger.debug(f"RAG Prefetch error (non-fatal): {e}")

    def clear_rag_prefetch(self) -> None:
        """Clear prefetch cache on session end to avoid stale context bleed."""
        self._rag_prefetch.clear()

    def _analyze_query_complexity(self, query: str) -> str:
        """Analyze query complexity to route to appropriate provider/model.

        Returns: 'simple', 'moderate', 'complex', or 'reasoning'

        Results are cached on the first 80 chars of the lowercased query so
        repeated or near-duplicate queries skip all regex work entirely.
        """
        key = query.lower().strip()[:80]
        return self._complexity_cached(key)

    @lru_cache(maxsize=256)
    def _complexity_cached(self, query_key: str) -> str:
        """LRU-cached inner implementation. Called with the normalised key."""
        lower = query_key

        # Check for reasoning patterns first (highest priority)
        for pattern in self._reasoning_patterns:
            if re.search(pattern, lower):
                return "reasoning"

        # Check for complex patterns
        complexity_score = 0
        for pattern in self._complex_patterns:
            if re.search(pattern, lower):
                complexity_score += 2

        # Check for simple patterns
        for pattern in self._simple_patterns:
            if re.match(pattern, lower):
                complexity_score -= 1

        # Length factor (approximate from key, which is capped at 80 chars)
        word_count = len(lower.split())
        if word_count > 12:    # ~50 words full query maps to ~12 in 80-char key
            complexity_score += 1

        if complexity_score >= 3:
            return "complex"
        elif complexity_score >= 1:
            return "moderate"
        return "simple"

    def _preferred_providers_for_complexity(self, complexity: str, mode=None) -> List[str]:
        """
        Return provider preferences without bypassing the router.
        Now reads mode.preferred_providers from the Mode object when available.

        OFFLINE-FIRST: When offline_first is enabled in config and the query is
        simple+general, Ollama is moved to the front so trivial queries don't
        burn cloud API rate-limits and avoid internet round-trip.
        """
        # Resolve mode name for fallback path
        mode_name = (
            mode.name if hasattr(mode, "name") else str(mode or "general")
        ).lower()

        # Offline-first fast-path: simple general query → prefer local Ollama
        offline_first = self.config.get("ai.offline_first", False)
        ollama_ready = (
            "ollama" in self._providers
            and getattr(self._providers.get("ollama"), "enabled", False)
        )
        if (
            offline_first
            and ollama_ready
            and complexity == "simple"
            and mode_name in {"general", "meeting"}
        ):
            logger.debug("Provider: offline-first routing simple query to Ollama")
            return ["ollama"] + [p for p in self._provider_priority if p != "ollama"]

        # If we have a Mode object with explicit provider preferences, honour them
        if mode and hasattr(mode, "preferred_providers") and mode.preferred_providers:
            base = list(mode.preferred_providers)
            # For complex/reasoning queries, ensure quality providers are ahead
            if complexity in {"complex", "reasoning"}:
                quality_first = [p for p in ["gemini", "together", "sambanova"] if p in base]
                rest = [p for p in base if p not in quality_first]
                return quality_first + rest
            return base

        # Fallback: string-based mode heuristics (backward compat)
        if mode_name in {"general", "meeting"}:
            if complexity in {"simple", "moderate"}:
                return ["groq", "cerebras", "together", "gemini", "ollama"]
            return ["groq", "cerebras", "gemini", "together", "ollama"]
        if mode_name == "interview":
            if complexity in {"simple", "moderate"}:
                return ["groq", "cerebras", "gemini", "together", "ollama"]
            return ["gemini", "groq", "cerebras", "together", "ollama"]
        if complexity == "simple":
            return ["groq", "cerebras", "gemini", "together", "ollama"]
        if complexity == "moderate":
            return ["groq", "gemini", "cerebras", "together", "ollama"]
        if complexity == "complex":
            return ["gemini", "groq", "cerebras", "together", "ollama"]
        if complexity == "reasoning":
            return ["gemini", "groq", "cerebras", "together", "ollama"]
        return list(self._provider_priority)

    def _select_provider(self, mode, complexity: str, preferred: List[str]):
        """
        Select a provider through the router when available, otherwise use ranked fallback.
        Accepts a Mode object or a string mode name.
        When a Mode object is given, its preferred_tier is passed to the router.
        """
        # P2: Local-only mode (Ollama) overrides routing for privacy/offline use.
        if bool(self.config.get("ai.text.local_only", False)):
            provider = self._providers.get("ollama")
            if provider and getattr(provider, "enabled", False) and provider.check_rate():
                mode_name = mode.name if hasattr(mode, "name") else str(mode or "general")
                tier = self._router._tier_for_task(mode_name) if self._router else "balanced"
                return provider, tier
            return None, ""

        prefer_speed = complexity in {"simple", "moderate"}
        prefer_quality = complexity in {"complex", "reasoning"}

        # Resolve tier hint from Mode profile
        if mode and hasattr(mode, "preferred_tier") and mode.preferred_tier:
            tier_hint = mode.preferred_tier
        else:
            tier_hint = None

        mode_name = mode.name if hasattr(mode, "name") else str(mode or "general")

        if self._router:
            provider, tier = self._router.select(
                task=mode_name,
                prefer_speed=prefer_speed,
                prefer_quality=prefer_quality,
                preferred=preferred,
                tier=tier_hint,
            )
            if provider:
                return provider, tier

        for provider_id in preferred:
            provider = self._providers.get(provider_id)
            if provider and getattr(provider, "enabled", False):
                tier = self._router._tier_for_task(mode_name) if self._router else "balanced"
                return provider, tier

        provider = self._providers.get(self._active_provider_id)
        if provider and getattr(provider, "enabled", False):
            return provider, tier_hint or "balanced"
        return None, ""

    async def poll_provider_health_loop(self):
        """Midnight Restoration: Background health monitor (60s interval)."""
        while True:
            try:
                await self.poll_provider_health()
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health Monitor Error: {e}")
                await asyncio.sleep(10)
        self._health_task = None

    def ensure_health_monitor(self, loop: asyncio.AbstractEventLoop):
        """Start the provider health monitor once and reuse it for future calls."""
        if not loop or not loop.is_running():
            return None
        self._loop = loop
        if self._health_task and not self._health_task.done():
            return self._health_task

        def _create():
            if self._health_task and not self._health_task.done():
                return self._health_task
            self._health_task = loop.create_task(self.poll_provider_health_loop())
            return self._health_task

        future = asyncio.run_coroutine_threadsafe(self._call_in_loop(_create), loop)
        try:
            return future.result(timeout=2)
        except Exception as e:
            logger.error(f"Failed to start provider health monitor: {e}")
            return None

    async def stop_health_monitor(self):
        """Cancel the provider health monitor during emergency erase or shutdown."""
        task = self._health_task
        self._health_task = None
        if not task:
            return

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"Provider health monitor shutdown error: {e}")

    async def poll_provider_health(self):
        """Polls providers and emits status to UI badges."""
        results = self._router.get_provider_health() if self._router else {}
        for pid, prov in list(self._providers.items()):
            info = dict(results.get(pid, {}))
            state = info.get("state", "unknown")
            locked_state = state in {"cooldown", "rate_limited", "disabled"}
            try:
                if hasattr(prov, "check_availability"):
                    is_ok = await asyncio.wait_for(prov.check_availability(), timeout=5)
                    if is_ok and not locked_state:
                        state = "active"
                    elif getattr(prov, "state", "") == "missing":
                        state = "missing"
                    elif getattr(prov, "state", "") == "unavailable":
                        state = "down"
                    elif state not in {"cooldown", "rate_limited", "disabled"}:
                        state = "down"
            except Exception:
                if not locked_state:
                    state = "down"

            info["state"] = state
            info["selected"] = pid == self._active_provider_id
            info["usable"] = state in {"active", "cooldown"}
            results[pid] = info

        self.provider_status.emit(results)

    async def _call_in_loop(self, fn):
        return fn()
