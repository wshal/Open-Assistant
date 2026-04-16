"""
AI Orchestration Engine - OpenAssist v4.1 (Midnight Hardened).
RESTORED: Async Task Control, RAG caching, and Cloud-ASR Correction.
FIXED: Removed internal loops; App Master Loop now controls all AI execution.
"""

import asyncio
import time
from typing import Optional, List, Dict
from PyQt6.QtCore import QObject, pyqtSignal, QTimer

from ai.providers import init_providers
from ai.prompts import PromptBuilder
from ai.history import ResponseHistory
from ai.detectors.question_detector import QuestionDetector
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
        
        # RAG Cache (Midnight Restoration)
        self._rag_cache = {}  # query -> (context, expiry)
        self._cache_ttl = 60

    def warmup(self):
        """Initializes AI providers and checks availability."""
        try:
            self._providers = init_providers(self.config)
            logger.info(f"AI Engine initialized: {list(self._providers.keys())}")
            
            # Auto-select best provider if none fixed
            if not self._active_provider_id:
                for p in ["groq", "cerebras", "gemini", "ollama"]:
                    if p in self._providers:
                        self._active_provider_id = p
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
        screen_context: Optional[str] = None,
        audio_context: Optional[str] = None,
    ):
        """
        RESTORED: Main async generation task.
        Control flow managed by the App Master Loop.
        """
        self._is_cancelled = False
        
        provider = self._providers.get(self._active_provider_id)
        if not provider:
            self.error_occurred.emit(f"Provider '{self._active_provider_id}' not found.")
            return

        try:
            mode_id = self.config.get("ai.mode", "general")
            
            # 1. RAG Cache / Fetch
            rag_context = ""
            if self.rag:
                now = time.time()
                cache_key = query.lower().strip()[:100]
                if cache_key in self._rag_cache and now < self._rag_cache[cache_key][1]:
                    rag_context = self._rag_cache[cache_key][0]
                    logger.debug("RAG: Cache Hit")
                else:
                    results = await self.rag.query(query)
                    if results:
                        rag_context = "\n".join(results)
                        self._rag_cache[cache_key] = (rag_context, now + self._cache_ttl)

            # 2. Transcription Refinement
            refined_audio = audio_context
            if audio_context and self.config.get("capture.audio.correct_transcript", True):
                refiner_id = self.config.get("capture.audio.correction_provider", "groq")
                refiner = self._providers.get(refiner_id)
                if refiner:
                    refined_audio = await self._refine_transcript(audio_context, refiner)

            # 3. Prompt Synthesis
            sys_prompt = self.prompts.system(mode_id)
            user_msg = self.prompts.user(
                query=query,
                screen=screen_context or "",
                audio=refined_audio or "",
                rag=rag_context,
                mode=mode_id
            )

            # 4. Streamed Generation
            full_response = ""
            async for chunk in provider.generate_stream(sys_prompt, user_msg):
                if self._is_cancelled:
                    logger.warning("AI: Stream cancelled mid-token")
                    return
                
                full_response += chunk
                self.response_chunk.emit(chunk)

            # 5. Finalize
            if not self._is_cancelled:
                self.history.add(query, full_response, provider=self._active_provider_id, mode=mode_id)
                self.response_complete.emit(full_response)
                logger.info("AI: Response complete")

        except Exception as e:
            logger.error(f"AI Engine Runtime Error: {e}")
            self.error_occurred.emit(str(e))

    async def _refine_transcript(self, raw_text: str, provider) -> str:
        """Midnight Restoration: Uses Cloud-Speed models to fix ASR typos."""
        prompt = f"Fix common ASR typos in this text. Keep meaning exact. Output ONLY fixed text.\n\n{raw_text}"
        try:
            fixed = ""
            async for chunk in provider.generate_stream("You are a text correction engine.", prompt):
                if self._is_cancelled: return raw_text
                fixed += chunk
            return fixed.strip() if fixed else raw_text
        except:
            return raw_text

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
