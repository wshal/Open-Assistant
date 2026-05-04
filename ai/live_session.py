"""Realtime live-audio session manager for Phase 1 live mode."""

from __future__ import annotations

import asyncio
import re
from typing import Any, Optional

from PyQt6.QtCore import QObject, pyqtSignal

from ai.prompts import PromptBuilder
from utils.logger import setup_logger

logger = setup_logger(__name__)


class LiveSessionManager(QObject):
    """Manage one Gemini Live session for low-latency audio-first interaction."""

    DEFAULT_MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"
    LEGACY_MODEL_ALIASES = {
        "gemini-live-2.5-flash-preview": DEFAULT_MODEL,
        "models/gemini-live-2.5-flash-preview": DEFAULT_MODEL,
    }

    text_delta = pyqtSignal(str)
    turn_complete = pyqtSignal(str)
    turn_empty = pyqtSignal()
    transcript_update = pyqtSignal(str)
    status_changed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.prompts = PromptBuilder()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._session_task: Optional[asyncio.Task] = None
        self._session = None
        self._connected = False
        self._response_buffer = ""
        self._response_started = False
        self._running = False
        self._stopping = False
        self._resume_handle: Optional[str] = None

    @property
    def is_connected(self) -> bool:
        return bool(self._connected and self._session is not None)

    def is_enabled(self) -> bool:
        if not bool(self.config.get("ai.live_mode.enabled", False)):
            return False
        key = str(self.config.get_api_key("gemini") or "").strip()
        return bool(key)

    def start(self, loop: asyncio.AbstractEventLoop, mode=None, session_context: str = "") -> bool:
        """Start the live session on the app async loop."""
        if not self.is_enabled():
            return False
        if not loop or not loop.is_running():
            return False
        if self._session_task and not self._session_task.done():
            return True

        self._loop = loop
        self._running = True
        self._stopping = False
        self._resume_handle = None
        system_prompt = self._build_live_system_prompt(
            self.prompts.system(mode=mode, session_context=session_context)
        )
        self.status_changed.emit("Connecting Live Mode...")

        async def _spawn():
            self._session_task = asyncio.create_task(
                self._session_main(system_prompt),
                name="gemini-live-session",
            )

        asyncio.run_coroutine_threadsafe(_spawn(), loop)
        return True

    def stop(self) -> None:
        self._running = False
        self._stopping = True
        session = self._session
        self._connected = False
        self._session = None
        self._response_buffer = ""
        self._response_started = False
        loop = self._loop
        if not loop or not loop.is_running():
            return

        async def _stop():
            task = self._session_task
            self._session_task = None
            await self._close_specific_session(session)
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass

        asyncio.run_coroutine_threadsafe(_stop(), loop)

    def send_audio_chunk(self, pcm_bytes: bytes, sample_rate: int = 16000) -> None:
        if not pcm_bytes or not self.is_connected or not self._loop or not self._loop.is_running():
            return
        asyncio.run_coroutine_threadsafe(
            self._send_audio_chunk_now(pcm_bytes, sample_rate),
            self._loop,
        )

    def end_audio_turn(self) -> None:
        if not self.is_connected or not self._loop or not self._loop.is_running():
            return
        asyncio.run_coroutine_threadsafe(self._end_audio_turn_now(), self._loop)

    async def _session_main(self, system_prompt: str) -> None:
        self._connected = False
        self._response_buffer = ""
        self._response_started = False
        key = str(self.config.get_api_key("gemini") or "").strip()
        configured_model = str(
            self.config.get("ai.live_mode.model", self.DEFAULT_MODEL)
            or self.DEFAULT_MODEL
        ).strip()
        models_to_try = self._candidate_models(configured_model)
        try:
            from google import genai

            client = genai.Client(api_key=key)
            config = {
                "response_modalities": ["AUDIO"],
                "input_audio_transcription": {},
                "output_audio_transcription": {},
                "context_window_compression": {"sliding_window": {}},
                "system_instruction": system_prompt,
            }
            last_error: Exception | None = None
            for idx, model in enumerate(models_to_try):
                while self._running and not self._stopping:
                    try:
                        connect_config = dict(config)
                        if self._resume_handle:
                            connect_config["session_resumption"] = {
                                "handle": self._resume_handle
                            }
                            logger.info("Resuming Gemini Live session with saved handle")
                        async with client.aio.live.connect(model=model, config=connect_config) as session:
                            self._session = session
                            self._connected = True
                            self._persist_model_if_needed(configured_model, model)
                            self.status_changed.emit("Live Mode Connected")
                            logger.info("Live mode connected via Gemini model=%s", model)
                            await self._receive_loop(session)

                        if self._running and not self._stopping:
                            self._connected = False
                            self._session = None
                            self._response_buffer = ""
                            self._response_started = False
                            logger.warning(
                                "Live mode session ended unexpectedly; reconnecting model=%s",
                                model,
                            )
                            self.status_changed.emit("Reconnecting Live Mode...")
                            await asyncio.sleep(0.75)
                            continue
                        return
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        last_error = e
                        if self._stopping and self._is_normal_close_error(e):
                            logger.info("Live mode session closed cleanly during shutdown")
                            return
                        logger.warning("Live mode session error (%s): %s", model, e)
                        should_retry = idx < len(models_to_try) - 1 and self._looks_like_model_compat_error(e)
                        if should_retry:
                            fallback = models_to_try[idx + 1]
                            self.status_changed.emit("Updating Live Mode model...")
                            logger.info("Retrying Gemini Live with fallback model=%s", fallback)
                            break
                        raise
            if last_error:
                raise last_error
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if self._stopping and self._is_normal_close_error(e):
                logger.info("Live mode shutdown completed without error")
                return
            logger.warning("Live mode unavailable after retries: %s", e)
            self.error_occurred.emit(self._format_error_message(e))
        finally:
            self._connected = False
            self._session = None
            self._stopping = False
            if self._running:
                self.status_changed.emit("Live Mode Offline")

    async def _receive_loop(self, session) -> None:
        async for message in session.receive():
            self._update_session_resumption(message)
            self._maybe_log_go_away(message)

            transcript = self._extract_input_transcript(message)
            if transcript:
                self.transcript_update.emit(transcript)

            delta = self._extract_output_text(message)
            if delta:
                if not self._response_started:
                    self._response_started = True
                    self.status_changed.emit("Live Responding...")
                # Buffer raw live output and publish only the cleaned final answer
                # on turn completion so users don't see planning/thinking prefaces.
                self._response_buffer += delta

            if self._is_turn_complete(message):
                had_visible_response = self._response_started or bool(self._response_buffer.strip())
                final_text = self._clean_live_response(self._response_buffer)
                if final_text:
                    self.turn_complete.emit(final_text)
                elif had_visible_response:
                    self.turn_empty.emit()
                self._response_buffer = ""
                self._response_started = False
                self.status_changed.emit("Live Listening...")

    async def _send_audio_chunk_now(self, pcm_bytes: bytes, sample_rate: int) -> None:
        session = self._session
        if not session:
            return
        from google.genai import types

        await session.send_realtime_input(
            audio=types.Blob(
                data=pcm_bytes,
                mime_type=f"audio/pcm;rate={int(sample_rate)}",
            )
        )

    async def _end_audio_turn_now(self) -> None:
        session = self._session
        if not session:
            return
        await session.send_realtime_input(audio_stream_end=True)

    async def _close_specific_session(self, session) -> None:
        if not session:
            return
        try:
            close_fn = getattr(session, "close", None)
            if close_fn is None:
                return
            result = close_fn()
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            pass

    async def _close_session(self) -> None:
        session = self._session
        self._connected = False
        self._session = None
        self._response_buffer = ""
        self._response_started = False
        await self._close_specific_session(session)

    @staticmethod
    def _extract_server_content(message: Any) -> Any:
        return getattr(message, "server_content", None) or getattr(message, "serverContent", None)

    @classmethod
    def _extract_input_transcript(cls, message: Any) -> str:
        server_content = cls._extract_server_content(message)
        if not server_content:
            return ""
        input_tx = (
            getattr(server_content, "input_transcription", None)
            or getattr(server_content, "inputTranscription", None)
        )
        text = getattr(input_tx, "text", None)
        if isinstance(text, str):
            return text.strip()
        return ""

    @classmethod
    def _extract_output_text(cls, message: Any) -> str:
        server_content = cls._extract_server_content(message)
        if server_content:
            output_tx = (
                getattr(server_content, "output_transcription", None)
                or getattr(server_content, "outputTranscription", None)
            )
            out_text = getattr(output_tx, "text", None)
            if isinstance(out_text, str) and out_text.strip():
                return out_text

        model_turn = None
        if server_content:
            model_turn = getattr(server_content, "model_turn", None) or getattr(server_content, "modelTurn", None)
        parts = getattr(model_turn, "parts", None) if model_turn else None
        if not parts:
            direct = getattr(message, "text", None)
            if isinstance(direct, str) and direct:
                return direct
            return ""

        chunks = []
        for part in parts:
            text = getattr(part, "text", None)
            if isinstance(text, str) and text:
                chunks.append(text)
        if chunks:
            return "".join(chunks)

        return ""

    @classmethod
    def _is_turn_complete(cls, message: Any) -> bool:
        server_content = cls._extract_server_content(message)
        if not server_content:
            return False
        return bool(
            getattr(server_content, "turn_complete", False)
            or getattr(server_content, "turnComplete", False)
            or getattr(server_content, "generation_complete", False)
            or getattr(server_content, "generationComplete", False)
        )

    @classmethod
    def _candidate_models(cls, configured_model: str) -> list[str]:
        raw = (configured_model or "").strip() or cls.DEFAULT_MODEL
        normalized = cls.LEGACY_MODEL_ALIASES.get(raw, raw.removeprefix("models/"))
        models = [normalized]
        if cls.DEFAULT_MODEL not in models:
            models.append(cls.DEFAULT_MODEL)
        return models

    @classmethod
    def _looks_like_model_compat_error(cls, error: Exception) -> bool:
        text = str(error or "").lower()
        return (
            "not found" in text
            or "not supported for bidigeneratecontent" in text
            or "unsupported model" in text
        )

    @classmethod
    def _is_normal_close_error(cls, error: Exception) -> bool:
        text = str(error or "").lower().strip()
        return (
            text == "1000 none."
            or text == "1000 none"
            or "normal closure" in text
            or "close code 1000" in text
        )

    @classmethod
    def _format_error_message(cls, error: Exception) -> str:
        if cls._looks_like_model_compat_error(error):
            return (
                "Live mode unavailable: this Gemini Live model is no longer supported. "
                "The app retried with the current default, but Gemini rejected the session."
            )
        return f"Live mode unavailable: {error}"

    @classmethod
    def _build_live_system_prompt(cls, base_prompt: str) -> str:
        live_rules = (
            "LIVE RESPONSE RULES:\n"
            "- Give only the final answer to the user.\n"
            "- Never narrate your reasoning, planning, or thought process.\n"
            "- Never include planning headers such as 'Defining', 'Clarifying', 'I'm now', 'My goal is', or 'I am going to'.\n"
            "- Never say what you are about to explain; just explain it.\n"
            "- Start with the answer in the first sentence.\n"
            "- For 'what is', 'explain', or definition questions: give one direct definition first, then 2-4 short bullets only if helpful.\n"
            "- Keep answers compact and spoken-assistant friendly."
        )
        return f"{base_prompt}\n\n{live_rules}".strip()

    @classmethod
    def _clean_live_response(cls, content: str) -> str:
        text = (content or "").strip()
        if not text:
            return text

        patterns = [
            r"^\s*(?:defining|clarifying|pinpointing|distinguishing|listing)\b[^.?!]*[.?!]\s*",
            r"^\s*i(?:'m| am| have|['’]?ve)\b[^.?!]*[.?!]\s*",
            r"^\s*my\s+(?:goal|plan)\s+is\b[^.?!]*[.?!]\s*",
            r"^\s*now\s+i(?:'ve| have)\b[^.?!]*[.?!]\s*",
        ]
        changed = True
        while changed:
            changed = False
            for pattern in patterns:
                updated = re.sub(pattern, "", text, count=1, flags=re.IGNORECASE)
                if updated != text:
                    text = updated.lstrip(" \n\r\t-:*")
                    changed = True

        sentence_parts = [
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+", text)
            if part.strip()
        ]
        while len(sentence_parts) >= 2:
            first = sentence_parts[0]
            word_count = len(re.findall(r"[A-Za-z0-9_'-]+", first))
            if cls._looks_like_meta_paragraph(first) or word_count <= 3:
                sentence_parts.pop(0)
                continue
            break
        if sentence_parts:
            text = " ".join(sentence_parts)

        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        filtered = [p for p in paragraphs if not cls._looks_like_meta_paragraph(p)]
        if filtered:
            text = "\n\n".join(filtered)

        return text.strip() or (content or "").strip()

    @classmethod
    def _looks_like_meta_paragraph(cls, text: str) -> bool:
        lowered = (text or "").strip().lower()
        if not lowered:
            return False

        meta_starts = (
            "defining ",
            "clarifying ",
            "pinpointing ",
            "distinguishing ",
            "listing ",
            "i'm now ",
            "i am now ",
            "i'm currently ",
            "i am currently ",
            "i'm working on ",
            "i am working on ",
            "i've refined ",
            "i have refined ",
            "i've crafted ",
            "i have crafted ",
            "i've honed ",
            "i have honed ",
            "i've homed ",
            "i have homed ",
            "i'm focusing ",
            "i am focusing ",
            "i'm aiming ",
            "i am aiming ",
            "i've established ",
            "i have established ",
            "now i've ",
            "now i have ",
            "my goal is ",
            "my plan is ",
            "i am going to ",
            "i'm going to ",
        )
        if lowered.startswith(meta_starts):
            return True

        meta_markers = (
            "my goal is",
            "i am going to omit",
            "i'm now focusing",
            "i am now focusing",
            "i'm now formulating",
            "i am now formulating",
            "before discussing",
            "before i discuss",
            "i'm aiming to articulate",
            "i will be sure to",
        )
        answer_markers = (
            " is ",
            " are ",
            " refers to ",
            " means ",
            " allows ",
            " useful for",
            " mechanism",
            "function that",
        )
        if any(marker in lowered for marker in meta_markers) and not any(
            marker in lowered for marker in answer_markers
        ):
            return True

        return False

    def _update_session_resumption(self, message: Any) -> None:
        update = (
            getattr(message, "session_resumption_update", None)
            or getattr(message, "sessionResumptionUpdate", None)
        )
        if not update:
            return
        resumable = bool(getattr(update, "resumable", False))
        new_handle = getattr(update, "new_handle", None) or getattr(update, "newHandle", None)
        if resumable and isinstance(new_handle, str) and new_handle.strip():
            self._resume_handle = new_handle.strip()

    def _maybe_log_go_away(self, message: Any) -> None:
        go_away = getattr(message, "go_away", None) or getattr(message, "goAway", None)
        if not go_away:
            return
        time_left = getattr(go_away, "time_left", None) or getattr(go_away, "timeLeft", None)
        if time_left is not None:
            logger.info("Gemini Live GoAway received; time_left=%s", time_left)

    def _persist_model_if_needed(self, configured_model: str, resolved_model: str) -> None:
        original = (configured_model or "").strip()
        if not resolved_model or original == resolved_model:
            return
        try:
            self.config.set("ai.live_mode.model", resolved_model)
            if hasattr(self.config, "save"):
                self.config.save()
        except Exception:
            pass
