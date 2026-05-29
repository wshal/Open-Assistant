import logging
import re
import time
from typing import TYPE_CHECKING

from ai.auto_query_utils import (
    looks_like_clipped_query_fragment,
    looks_like_setup_statement,
    select_best_auto_query_label,
)
from utils.text_utils import (
    _looks_like_auto_query_clause,
    looks_like_actionable_auto_query,
    merge_transcripts,
    sanitize_query_label,
)

if TYPE_CHECKING:
    from core.app import OpenAssistApp

logger = logging.getLogger(__name__)


_INCOMPLETE_TRAILING_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "between",
    "by",
    "can",
    "consider",
    "could",
    "explain",
    "for",
    "from",
    "how",
    "if",
    "in",
    "into",
    "of",
    "or",
    "should",
    "that",
    "the",
    "this",
    "to",
    "versus",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
    "you",
}


def _norm_query_key(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w']+", " ", (text or "").lower())).strip()


def _reset_session_context_if_stale(app: "OpenAssistApp", session_id: str) -> None:
    """Clear injected preamble context after the response window has passed.

    Called via QTimer 45s after dispatch. Only clears if no new dispatch has
    happened in the meantime (i.e. dispatched_at hasn't changed significantly).
    This prevents stale scenario context from leaking into the next question.
    """
    dispatched_at = float(getattr(app, "_auto_answer_last_dispatched_at", 0.0) or 0.0)
    if time.time() - dispatched_at >= 40:  # ~45s window minus timer jitter
        if hasattr(app, "ai") and hasattr(app.ai, "set_session_context"):
            app.ai.set_session_context("")
            logger.debug("[%s] Auto Mode cleared stale session context", session_id)


def _query_similarity(a: str, b: str) -> float:
    ka = _norm_query_key(a)
    kb = _norm_query_key(b)
    if not ka or not kb:
        return 0.0
    if ka == kb:
        return 1.0
    if ka in kb or kb in ka:
        shorter = min(len(ka), len(kb))
        longer = max(len(ka), len(kb))
        return shorter / float(longer or 1)
    ta = set(ka.split())
    tb = set(kb.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / float(len(ta | tb))


def _looks_complete_for_speculation(text: str) -> bool:
    cleaned = " ".join((text or "").split()).strip()
    if not cleaned:
        return False
    lower = cleaned.lower().strip()
    if lower.endswith(("...", "…", "-", ":", ",")):
        return False
    if ".." in lower:
        return False
    words = re.findall(r"[a-zA-Z0-9']+", lower)
    if len(words) < 9 or len(cleaned) < 55:
        return False
    if words[-1] in _INCOMPLETE_TRAILING_WORDS:
        return False
    if "?" in cleaned:
        return True
    strong_openers = (
        "what are some ",
        "what are the ",
        "could you walk me through ",
        "can you walk me through ",
        "could you explain how ",
        "can you explain how ",
        "how would you ",
        "how do you ",
    )
    return lower.startswith(strong_openers)


def _looks_like_standalone_speculative_question(text: str) -> bool:
    cleaned = " ".join((text or "").split()).strip()
    if not _looks_complete_for_speculation(cleaned):
        return False
    lower = cleaned.lower()
    strong_openers = (
        "what are some ",
        "what are the ",
        "could you walk me through ",
        "can you walk me through ",
        "could you explain how ",
        "can you explain how ",
        "could you explain the ",
        "can you explain the ",
        "how would you ",
        "how do you ",
    )
    return lower.startswith(strong_openers)


def _mark_auto_context(app: "OpenAssistApp", text: str) -> None:
    text = " ".join((text or "").split()).strip()
    if not text:
        return
    prior = (getattr(app, "_auto_answer_context", "") or "").strip()
    merged = merge_transcripts(prior, text, max_overlap_words=24).strip() if prior else text

    # ── Smart context window ────────────────────────────────────────────
    # The first ~500 chars usually contain the topic setup (e.g. "Let's
    # talk about React hooks and performance optimisation").  The most
    # recent ~2000 chars carry the latest question detail.  Instead of a
    # naive tail-slice we keep both ends so the topic is never lost.
    _HEAD = 500
    _TAIL = 2000
    _MAX  = _HEAD + _TAIL + 10          # +10 for separator
    if len(merged) <= _MAX:
        app._auto_answer_context = merged
    else:
        head = merged[:_HEAD].rstrip()
        tail = merged[-_TAIL:].lstrip()
        app._auto_answer_context = f"{head} … {tail}"

    app._auto_answer_context_at = time.time()


def _candidate_from_context(app: "OpenAssistApp", text: str) -> str:
    cleaned = (sanitize_query_label(text) or text or "").strip()
    context = (getattr(app, "_auto_answer_context", "") or "").strip()
    if context and cleaned:
        merged = merge_transcripts(context, cleaned, max_overlap_words=24).strip()
        cleaned = (sanitize_query_label(merged) or cleaned).strip()

    candidate = select_best_auto_query_label(
        cleaned,
        getattr(app, "_pending_incomplete_audio_query", ""),
        looks_question_like_transcript=app._looks_question_like_transcript,
        looks_like_acknowledgement=app._looks_like_acknowledgement,
    ).strip()
    if cleaned and "potential security risks" in cleaned.lower() and len(cleaned.split()) > len(candidate.split()):
        candidate = cleaned
    candidate = re.sub(r"[\s\-,:]+$", "", candidate).strip()
    return (sanitize_query_label(candidate) or candidate).strip()


def _is_actionable_auto_query(app: "OpenAssistApp", text: str) -> bool:
    if not text:
        return False
    clause = text.rstrip("?").strip()
    return bool(
        app._looks_question_like_transcript(text)
        or looks_like_actionable_auto_query(text)
        or _looks_like_auto_query_clause(clause)
        or "?" in text
    )


def _recent_auto_dispatch_matches(app: "OpenAssistApp", candidate: str, window_s: float = 8.0) -> bool:
    suppressed = (getattr(app, "_auto_answer_last_dispatched_query", "") or "").strip()
    dispatched_at = float(getattr(app, "_auto_answer_last_dispatched_at", 0.0) or 0.0)
    if not suppressed or dispatched_at <= 0.0 or (time.time() - dispatched_at) > window_s:
        return False
    return _query_similarity(suppressed, candidate) >= 0.72


def _dispatch_auto_query(
    app: "OpenAssistApp",
    candidate: str,
    raw: str,
    session_id: str,
    request_metadata: dict | None = None,
    speculative: bool = False,
) -> None:
    # ── Push accumulated preamble into session context BEFORE clearing it ────
    # _mark_auto_context() builds a rich head+tail window of everything the
    # interviewer has said so far ("Imagine we have a monolithic app backed by a
    # single relational DB...").  If we wipe it without forwarding it to the
    # engine, short follow-up queries like "What strategies would you use?"
    # reach the LLM completely decontextualized.
    request_metadata = dict(request_metadata or {})
    preamble = (getattr(app, "_auto_answer_context", "") or "").strip()
    if preamble and hasattr(app, "ai") and hasattr(app.ai, "set_session_context"):
        # Only inject if the candidate query is NOT already self-contained
        # (i.e. skip if the full scenario is already embedded in the query text)
        if len(candidate.split()) < 20 or not any(
            kw in candidate.lower()
            for kw in ("monolithic", "database", "api", "scenario", "system", "application")
        ):
            # Format it as a clean scenario block for the system prompt
            context_block = f"[INTERVIEW SCENARIO CONTEXT]\n{preamble}"
            app.ai.set_session_context(context_block)
            request_metadata["preserve_session_context"] = True
            logger.info(
                "[%s] Auto Mode injected %d-char preamble into session context",
                session_id,
                len(preamble),
            )

    app._auto_answer_last_dispatched_query = candidate
    app._auto_answer_last_dispatched_at = time.time()
    app._pending_incomplete_audio_query = ""
    app._pending_incomplete_audio_at = 0.0
    app._auto_answer_context = ""
    app._auto_interim_pending_query = ""
    app._auto_interim_pending_raw = ""
    app._auto_interim_pending_seq = int(getattr(app, "_auto_interim_pending_seq", 0) or 0) + 1
    app.overlay.update_transcript(raw)
    if request_metadata:
        app._pending_request_metadata = dict(request_metadata)
    logger.info(
        "[%s] Auto Mode %sdispatching response | query=%r",
        session_id,
        "speculatively " if speculative else "",
        candidate[:120],
    )
    app.generate_response(
        candidate,
        "speech",
        {"audio": candidate, "auto_answer": True, "auto_speculative": speculative},
    )
    # Schedule context reset after a short window so stale scenario context
    # doesn't contaminate the next independent question.
    # Use a generous 45s window — enough for the LLM to finish streaming.
    try:
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(45_000, lambda: _reset_session_context_if_stale(app, session_id))
    except Exception:
        pass


def _dispatch_auto_interim_if_current(app: "OpenAssistApp", seq: int, session_id: str) -> None:
    if int(getattr(app, "_auto_interim_pending_seq", 0) or 0) != seq:
        return
    candidate = (getattr(app, "_auto_interim_pending_query", "") or "").strip()
    raw = (getattr(app, "_auto_interim_pending_raw", "") or candidate).strip()
    if not candidate or not getattr(app, "_auto_mode_requested", lambda: False)():
        return
    if _recent_auto_dispatch_matches(app, candidate):
        return
    metadata = {
        "auto_answer": True,
        "auto_speculative": True,
        "transcript_received_at": time.time(),
    }
    _dispatch_auto_query(
        app,
        candidate,
        raw,
        session_id,
        request_metadata=metadata,
        speculative=True,
    )


def handle_auto_interim_transcription(
    app: "OpenAssistApp",
    text: str,
    session_id: str,
) -> bool:
    """Speculatively answer only very complete-looking interim questions."""
    if not getattr(app, "_auto_mode_requested", lambda: False)():
        return False
    if not bool(app.config.get("ai.auto_mode.speculative_interim.enabled", True)):
        return True

    raw = " ".join((text or "").split()).strip()
    if not raw:
        return True
    cleaned = (sanitize_query_label(raw) or raw).strip()

    if (
        not _is_actionable_auto_query(app, cleaned)
        or looks_like_setup_statement(cleaned)
        or looks_like_clipped_query_fragment(cleaned)
        or not _looks_like_standalone_speculative_question(cleaned)
        or _recent_auto_dispatch_matches(app, cleaned)
    ):
        return True

    candidate = _candidate_from_context(app, cleaned)
    if (
        not _is_actionable_auto_query(app, candidate)
        or looks_like_setup_statement(candidate)
        or looks_like_clipped_query_fragment(candidate)
        or not _looks_complete_for_speculation(candidate)
        or not _looks_like_standalone_speculative_question(cleaned)
        or _recent_auto_dispatch_matches(app, candidate)
    ):
        return True

    explicit_question = "?" in candidate
    if not explicit_question:
        key = _norm_query_key(candidate)
        previous_key = getattr(app, "_auto_interim_stable_key", "") or ""
        now = time.time()
        if key != previous_key:
            app._auto_interim_stable_key = key
            app._auto_interim_stable_at = now
            return True

        stable_ms = (now - float(getattr(app, "_auto_interim_stable_at", now) or now)) * 1000.0
        required_ms = float(app.config.get("ai.auto_mode.speculative_interim.stability_ms", 650) or 650)
        if stable_ms < required_ms:
            return True

    delay_ms = max(0, int(app.config.get("ai.auto_mode.speculative_interim.delay_ms", 250) or 250))
    app._auto_interim_pending_query = candidate
    app._auto_interim_pending_raw = raw
    app._auto_interim_pending_seq = int(getattr(app, "_auto_interim_pending_seq", 0) or 0) + 1
    seq = app._auto_interim_pending_seq
    logger.info("[%s] Auto Mode speculative interim queued | query=%r", session_id, candidate[:120])
    try:
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(delay_ms, lambda: _dispatch_auto_interim_if_current(app, seq, session_id))
    except Exception:
        _dispatch_auto_interim_if_current(app, seq, session_id)
    return True


def handle_auto_final_transcription(
    app: "OpenAssistApp",
    text: str,
    session_id: str,
    request_metadata: dict | None = None,
) -> bool:
    """Whisper-first auto-answer mode.

    Auto Mode is intentionally not Gemini Live. It keeps the standard local ASR
    and provider pipeline, but treats each final transcript as a potential turn:
    setup speech is retained as context, and complete/actionable questions are
    answered immediately.
    """
    if not getattr(app, "_auto_mode_requested", lambda: False)():
        return False

    raw = " ".join((text or "").split()).strip()
    if not raw:
        return True

    cleaned = (sanitize_query_label(raw) or raw).strip()
    _mark_auto_context(app, raw)

    candidate = _candidate_from_context(app, cleaned)
    actionable = _is_actionable_auto_query(app, candidate)
    setup_only = bool(looks_like_setup_statement(cleaned) and not actionable)

    if setup_only or not actionable:
        app.overlay.update_transcript(raw)
        app._pending_incomplete_audio_query = candidate or cleaned
        app._pending_incomplete_audio_at = time.time()
        logger.info(
            "[%s] Auto Mode kept transcript for context only | actionable=%s text=%r",
            session_id,
            actionable,
            raw[:120],
        )
        return True

    if looks_like_clipped_query_fragment(candidate) or candidate.endswith(("-", ":", ",")):
        app.overlay.update_transcript(raw)
        app._pending_incomplete_audio_query = candidate
        app._pending_incomplete_audio_at = time.time()
        logger.info("[%s] Auto Mode waiting for clipped query continuation | query=%r", session_id, candidate[:120])
        return True

    if _recent_auto_dispatch_matches(app, candidate):
        logger.info("[%s] Auto Mode duplicate final transcript suppressed | query=%r", session_id, candidate[:120])
        return True

    _dispatch_auto_query(
        app,
        candidate,
        raw,
        session_id,
        request_metadata=request_metadata,
        speculative=False,
    )
    return True
