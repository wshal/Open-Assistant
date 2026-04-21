"""
AI Orchestration Engine - OpenAssist v4.1 (Midnight Hardened).
RESTORED: Async Task Control, RAG caching, and Cloud-ASR Correction.
FIXED: Removed internal loops; App Master Loop now controls all AI execution.
FIXED: Added latency tracking for performance monitoring.
"""

import asyncio
import time
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

    def __init__(self, config, history: ResponseHistory, rag=None):
        super().__init__()
        self.config = config
        self.history = history
        self.rag = rag
        self.prompts = PromptBuilder()
        self.detector = QuestionDetector(config)

        self._providers = {}
        self._active_provider_id = config.get("ai.fixed_provider", "")
        self._is_cancelled = False
        self._health_task = None
        self._router = None
        self._selected_tier = "balanced"
        self._parallel = None

        # Provider speed ranking (fastest first)
        # groq/cerebras: ~2100 tok/s (fastest)
        # gemini: Excellent quality, moderate speed
        # together: Variable speed
        # ollama: Local (no API latency but limited model)
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

        # RAG Cache (Midnight Restoration)
        self._rag_cache = OrderedDict()  # query -> (context, expiry)
        self._cache_ttl = 60
        self._max_cache_size = 100
    def warmup(self):
        """Initializes AI providers and checks availability."""
        try:
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

        mode_id = self.config.get("ai.mode", "general")

        # Smart routing: analyze query complexity and select provider
        complexity = self._analyze_query_complexity(query)
        preferred_providers = self._preferred_providers_for_complexity(complexity)

        # Use parallel inference if enabled
        use_parallel = self._parallel and self.config.get("ai.parallel.enabled", False)

        if use_parallel:
            # Build prompts first for parallel
            sys_prompt = self.prompts.system(mode_id)
            user_msg = self.prompts.user(
                query=query,
                screen=screen_context or "",
                audio=audio_context or nexus_snapshot.get("recent_audio", ""),
                rag="",
                mode=mode_id,
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
                # Parallel mode returns a fully-materialized answer, so emit only completion.
                # Streaming chunks after completion duplicates content in the UI layer.
                self.response_complete.emit(full_response)

                logger.info(f"Parallel AI: Response complete ({latency_ms:.0f}ms)")
                return
            except Exception as e:
                logger.warning(
                    f"Parallel inference failed, falling back to single provider: {e}"
                )

        provider, tier = self._select_provider(mode_id, complexity, preferred_providers)
        self._selected_tier = tier
        if not provider:
            self.error_occurred.emit("No available AI provider found.")
            return

        self._active_provider_id = provider.name
        logger.debug(
            f"Provider selection -> complexity={complexity}, provider={provider.name}, tier={tier}"
        )

        try:
            # 1. RAG Cache / Fetch
            rag_context = ""
            if self.rag:
                now = time.time()
                cache_key = query.lower().strip()[:100]
                if cache_key in self._rag_cache and now < self._rag_cache[cache_key][1]:
                    rag_context = self._rag_cache[cache_key][0]
                    self._rag_cache.move_to_end(cache_key)
                    logger.debug("RAG: Cache Hit")
                else:
                    results = await self.rag.query(query)
                    if results:
                        rag_context = "\n".join(results)
                        self._rag_cache[cache_key] = (rag_context, now + self._cache_ttl)
                        if len(self._rag_cache) > self._max_cache_size:
                            self._rag_cache.popitem(last=False)
                stage_timings["rag_query_ms"] = (time.time() - start_time) * 1000

            # 2. Transcription Refinement (OPTIMIZED)
            refined_audio = audio_context
            simple_query = complexity == "simple"
            
            if audio_context and self.config.get("capture.audio.correct_transcript", True) and not simple_query:
                # Bypass refinement for simple queries or short transcripts to save ~500ms
                if len(audio_context.split()) > 5:
                    refiner_id = self.config.get("capture.audio.correction_provider", "groq")
                    refiner = self._providers.get(refiner_id)
                    if refiner:
                        refined_audio = await self._refine_transcript(audio_context, refiner)
                stage_timings["refinement_ms"] = (time.time() - start_time) * 1000 - stage_timings.get("rag_query_ms", 0)

            # 3. Prompt Synthesis
            sys_prompt = self.prompts.system(mode_id)
            user_msg = self.prompts.user(
                query=query,
                screen=screen_context or nexus_snapshot.get("latest_ocr", ""),
                audio=refined_audio or nexus_snapshot.get("full_audio_history", ""),
                rag=rag_context,
                mode=mode_id,
                origin=origin,
                nexus=nexus_snapshot
            )

            # 4. Streamed Generation
            full_response = ""
            async for chunk in provider.generate_stream(sys_prompt, user_msg):
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

            # 5. Finalize
            if not self._is_cancelled:
                latency_ms = (time.time() - start_time) * 1000
                self.history.add(
                    query,
                    full_response,
                    provider=provider.name,
                    mode=mode_id,
                    latency=latency_ms,
                    metadata={
                        "stage_timings": {
                            **stage_timings,
                            "request_to_complete_ms": (time.time() - request_started_at) * 1000,
                        },
                        "request_metadata": request_metadata,
                    }
                )
                self.response_complete.emit(full_response)
                logger.info(f"AI: Response complete ({latency_ms:.0f}ms) | {stage_timings}")

        except Exception as e:
            logger.error(f"AI Engine Runtime Error: {e}")
            self.error_occurred.emit(str(e))

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
        """Fast path for hotkey-triggered context answers using cached context first."""
        self._is_cancelled = False
        start_time = time.time()
        mode_id = self.config.get("ai.mode", "general")
        query = (
            "Using the latest live context, give a quick answer. "
            "First summarize the current audio briefly, then give the most useful immediate response in 2-4 bullets."
        )

        provider, tier = self._select_provider(
            mode_id,
            "simple",
            ["groq", "cerebras", "gemini", "together", "ollama"],
        )
        if not provider:
            self.error_occurred.emit("No available AI provider found.")
            return ""

        summarized_audio = audio_context or nexus_snapshot.get("recent_audio", "")
        if len(summarized_audio.split()) > 40:
            summarized_audio = await self._summarize_audio_fast(summarized_audio)

        sys_prompt = self.prompts.system(mode_id)
        user_msg = self.prompts.user(
            query=query,
            screen=screen_context or nexus_snapshot.get("latest_ocr", ""),
            audio=summarized_audio,
            rag="",
            mode=mode_id,
            origin="quick",
            nexus=nexus_snapshot,
        )

        try:
            full_response = ""
            async for chunk in provider.generate_stream(sys_prompt, user_msg):
                if self._is_cancelled:
                    return ""
                full_response += chunk
                self.response_chunk.emit(chunk)

            latency_ms = (time.time() - start_time) * 1000
            self.history.add(
                query,
                full_response,
                provider=provider.name,
                mode=mode_id,
                latency=latency_ms,
                metadata={"quick": True},
            )
            self.response_complete.emit(full_response)
            return full_response
        except Exception as e:
            logger.error(f"Quick response error: {e}")
            self.error_occurred.emit(str(e))
            return ""

    async def _summarize_audio_fast(self, audio_text: str) -> str:
        """Compress recent audio into a short summary using the fastest provider."""
        provider, tier = self._select_provider(
            "general",
            "simple",
            ["groq", "cerebras", "gemini", "together", "ollama"],
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
        for name in ["gemini", "ollama"]:
            if name not in preferred:
                preferred.append(name)
        if self.config.get("ai.vision.allow_paid_fallback", False):
            for name in ["openai"]:
                if name not in preferred:
                    preferred.append(name)

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
        for provider in candidates:
            emitted_partial = False
            try:
                response = ""
                if getattr(provider, "supports_vision_stream", lambda: False)():
                    async for chunk in provider.analyze_image_stream(
                        sys_prompt, user_msg, image_bytes, mime_type="image/png"
                    ):
                        if self._is_cancelled:
                            return ""
                        response += chunk
                        emitted_partial = True
                        self.response_chunk.emit(chunk)
                else:
                    response = await provider.analyze_image(
                        sys_prompt, user_msg, image_bytes, mime_type="image/png"
                    )
                if self._is_cancelled:
                    return ""

                latency_ms = (time.time() - start_time) * 1000
                self.history.add(
                    query,
                    response,
                    provider=provider.name,
                    mode=mode_id,
                    latency=latency_ms,
                    metadata={"vision": True},
                )
                self.history.add_screen_analysis(
                    query,
                    response,
                    provider=provider.name,
                    metadata={"vision": True},
                )
                self.response_complete.emit(response)
                return response
            except Exception as exc:
                if emitted_partial:
                    self.response_complete.emit("")
                last_error = exc
                logger.warning(f"Vision analysis failed on {provider.name}: {exc}")

        raise Exception(str(last_error) if last_error else "Vision analysis failed.")

    def _chunk_response(self, text: str, chunk_size: int = 20):
        """Split response into chunks for streaming effect."""
        words = text.split()
        for i in range(0, len(words), chunk_size):
            yield " ".join(words[i : i + chunk_size])

    def _analyze_query_complexity(self, query: str) -> str:
        """Analyze query complexity to route to appropriate provider/model.

        Returns: 'simple', 'moderate', 'complex', or 'reasoning'
        """
        import re

        lower = query.lower()

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

        # Length factor
        word_count = len(query.split())
        if word_count > 50:
            complexity_score += 1
        elif word_count > 100:
            complexity_score += 2

        if complexity_score >= 3:
            return "complex"
        elif complexity_score >= 1:
            return "moderate"
        return "simple"

    def _preferred_providers_for_complexity(self, complexity: str) -> List[str]:
        """Return provider preferences without bypassing the router."""
        if complexity == "simple":
            return ["groq", "cerebras", "gemini", "together", "ollama"]
        if complexity == "moderate":
            return ["groq", "gemini", "cerebras", "together", "ollama"]
        if complexity == "complex":
            return ["gemini", "groq", "cerebras", "together", "ollama"]
        if complexity == "reasoning":
            return ["gemini", "groq", "cerebras", "together", "ollama"]
        return list(self._provider_priority)

    def _select_provider(self, mode_id: str, complexity: str, preferred: List[str]):
        """Select a provider through the router when available, otherwise use ranked fallback."""
        prefer_speed = complexity in {"simple", "moderate"}
        prefer_quality = complexity in {"complex", "reasoning"}

        if self._router:
            provider, tier = self._router.select(
                task=mode_id,
                prefer_speed=prefer_speed,
                prefer_quality=prefer_quality,
                preferred=preferred,
            )
            if provider:
                return provider, tier

        for provider_id in preferred:
            provider = self._providers.get(provider_id)
            if provider and getattr(provider, "enabled", False):
                tier = self._router._tier_for_task(mode_id) if self._router else "balanced"
                return provider, tier

        provider = self._providers.get(self._active_provider_id)
        if provider and getattr(provider, "enabled", False):
            return provider, "balanced"
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
        results = {}
        for pid, prov in list(self._providers.items()):
            try:
                is_ok = False
                if hasattr(prov, "check_availability"):
                    is_ok = await asyncio.wait_for(prov.check_availability(), timeout=5)
                else:
                    is_ok = True
                results[pid] = {"state": "active" if is_ok else "down"}
            except Exception:
                results[pid] = {"state": "down"}

        self.provider_status.emit(results)

    async def _call_in_loop(self, fn):
        return fn()
