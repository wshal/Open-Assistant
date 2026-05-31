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

# Lazy-loaded intent classifier (embedding-based)
_intent_classifier = None
_intent_classifier_loaded = False


def _get_intent_classifier():
    """Lazy singleton accessor for the embedding-based IntentClassifier."""
    global _intent_classifier, _intent_classifier_loaded
    if _intent_classifier_loaded:
        return _intent_classifier
    try:
        from ai.intent_classifier import IntentClassifier
        _intent_classifier = IntentClassifier()
        _intent_classifier_loaded = True
    except Exception:
        _intent_classifier = None
        _intent_classifier_loaded = True
    return _intent_classifier


def _auto_learn(text: str, intent: str, *, scores=None, app=None) -> None:
    """Teach the intent classifier from a live session outcome.

    Only learns when:
      1. Learning is enabled in config (ai.intent_classifier.learning_enabled)
      2. The embedding classifier is available
      3. Regex and embeddings agree on the classification
    """
    # Check config; learning is opt-in, off by default.
    if app is not None:
        config = getattr(app, "config", None)
        if config and not bool(config.get("ai.intent_classifier.learning_enabled", False)):
            return

    classifier = _get_intent_classifier()
    if classifier is None:
        return
    try:
        confidence = 0.0
        if scores is not None:
            best = scores.best_intent
            if best == intent and scores.is_confident:
                confidence = scores.best_score
            else:
                return
        else:
            scores = classifier.classify(text)
            if scores is None:
                return
            if scores.best_intent == intent and scores.is_confident:
                confidence = scores.best_score
            else:
                return

        classifier.learn(text, intent, confidence=confidence, source="auto")
    except Exception as e:
        logger.debug("Auto-learn failed: %s", e)

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
    regex_says = bool(
        app._looks_question_like_transcript(text)
        or looks_like_actionable_auto_query(text)
        or _looks_like_auto_query_clause(clause)
        or "?" in text
    )

    # Use embedding classifier to augment regex decision
    classifier = _get_intent_classifier()
    if classifier is not None:
        return classifier.is_likely_question(text, regex_says_question=regex_says)

    return regex_says


def _recent_auto_dispatch_matches(app: "OpenAssistApp", candidate: str, window_s: float = 8.0) -> bool:
    suppressed = (getattr(app, "_auto_answer_last_dispatched_query", "") or "").strip()
    dispatched_at = float(getattr(app, "_auto_answer_last_dispatched_at", 0.0) or 0.0)
    if not suppressed or dispatched_at <= 0.0 or (time.time() - dispatched_at) > window_s:
        return False
    return _query_similarity(suppressed, candidate) >= 0.72


def _audio_processing_busy(app: "OpenAssistApp") -> bool:
    audio = getattr(app, "audio", None)
    if audio and hasattr(audio, "has_pending_transcription_jobs"):
        try:
            return bool(audio.has_pending_transcription_jobs())
        except Exception:
            return False
    return False


def _should_defer_for_followup(raw: str, candidate: str) -> bool:
    """Delay broad command-style prompts briefly so nearby follow-ups can merge.

    In interview scenarios, a single spoken prompt often arrives as two separate
    transcripts:
      T1: "Tell me about a time when you disagreed with a senior engineer."
      T2: "How did you approach the situation and what was the outcome?"

    Without deferral, T1 dispatches immediately and T2 either creates a
    duplicate response or arrives too late to be contextualised.

    Strategy: check for open-ended patterns FIRST (they always defer regardless
    of punctuation), then short-circuit on explicit ``?`` only for non-open-ended
    questions.
    """
    cleaned = " ".join((candidate or raw or "").split()).strip()
    if not cleaned:
        return False

    words = re.findall(r"[A-Za-z0-9']+", cleaned)
    if len(words) < 8:
        return False

    lower = cleaned.lower().rstrip(".?!")

    # ── Open-ended prompts: ALWAYS defer regardless of punctuation ──────────
    # Whisper may or may not add a `?`; we must not depend on it.
    OPEN_ENDED_STARTS = (
        "tell me about ",
        "tell us about ",
        "describe ",
        "discuss ",
        "walk me through ",
        "walk us through ",
        "take me through ",
        "explain how you ",
        "explain a time ",
        "outline how ",
        "give me an example ",
        "give an example ",
        "share ",
    )
    OPEN_ENDED_PHRASES = (
        " about a time ",
        " about a situation ",
        " situation where ",
        " situation when ",
        " scenario where ",
        " experience with ",
        " experience where ",
    )
    is_open_ended = (
        lower.startswith(OPEN_ENDED_STARTS)
        or any(p in f" {lower} " for p in OPEN_ENDED_PHRASES)
    )
    if is_open_ended:
        return True

    # ── Non-open-ended with explicit ? → treat as complete, don't defer ────
    raw_cleaned = " ".join((raw or "").split()).strip()
    if "?" in raw_cleaned:
        return False

    # ── Catch remaining behavioral / outcome patterns without ? ─────────────
    return bool(
        " outcome" in lower
        or " result" in lower
        or " what happened" in lower
        or lower.endswith("and how")
        or lower.endswith("and what")
        or lower.endswith("and why")
    )


def _needs_long_followup_debounce(text: str) -> bool:
    cleaned = " ".join((text or "").split()).strip().lower().rstrip(".?!")
    if not cleaned:
        return False
    behavioral_starts = (
        "tell me about ",
        "tell us about ",
        "explain a time ",
        "give me an example ",
        "give an example ",
        "share ",
    )
    behavioral_phrases = (
        " about a time ",
        " about a situation ",
        " situation where ",
        " situation when ",
        " scenario where ",
        " experience with ",
        " experience where ",
    )
    if cleaned.startswith(("describe ", "discuss ")):
        return any(phrase in f" {cleaned} " for phrase in behavioral_phrases)
    return cleaned.startswith(behavioral_starts) or any(
        phrase in f" {cleaned} " for phrase in behavioral_phrases
    )


def _final_dispatch_delay_ms(app: "OpenAssistApp", raw: str, candidate: str) -> int:
    default_delay_ms = max(0, int(app.config.get("ai.auto_mode.final_dispatch_debounce_ms", 900) or 900))
    if _needs_long_followup_debounce(candidate or raw):
        return max(
            default_delay_ms,
            int(app.config.get("ai.auto_mode.open_ended_followup_debounce_ms", 5200) or 5200),
        )
    return default_delay_ms


def _clear_pending_final_dispatch(app: "OpenAssistApp") -> None:
    app._auto_final_pending_query = ""
    app._auto_final_pending_raw = ""
    app._auto_final_pending_metadata = None
    app._auto_final_pending_session_id = ""
    app._auto_final_pending_at = 0.0


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
        context_block = (
            "[RECENT SPOKEN CONTEXT]\n"
            "Use this only when it helps disambiguate the user's current question. "
            "Ignore it if the current question is clearly unrelated.\n"
            f"{preamble}"
        )
        app.ai.set_session_context(context_block)
        request_metadata["preserve_session_context"] = True
        logger.info(
            "[%s] Auto Mode injected %d-char spoken context into session context",
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
    _clear_pending_final_dispatch(app)
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


def _dispatch_pending_final_if_current(app: "OpenAssistApp", seq: int) -> None:
    if int(getattr(app, "_auto_final_pending_seq", 0) or 0) != seq:
        return
    candidate = (getattr(app, "_auto_final_pending_query", "") or "").strip()
    raw = (getattr(app, "_auto_final_pending_raw", "") or candidate).strip()
    if not candidate or not getattr(app, "_auto_mode_requested", lambda: False)():
        _clear_pending_final_dispatch(app)
        return

    pending_at = float(getattr(app, "_auto_final_pending_at", 0.0) or 0.0)
    max_wait_ms = int(app.config.get("ai.auto_mode.final_dispatch_max_wait_ms", 6500) or 6500)
    if _audio_processing_busy(app) and pending_at > 0 and (time.time() - pending_at) * 1000.0 < max_wait_ms:
        try:
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(500, lambda: _dispatch_pending_final_if_current(app, seq))
            return
        except Exception:
            logger.debug(
                "[%s] Auto Mode could not reschedule busy final dispatch; dispatching now",
                getattr(app, "_auto_final_pending_session_id", "auto") or "auto",
            )

    metadata = dict(getattr(app, "_auto_final_pending_metadata", None) or {})
    session_id = getattr(app, "_auto_final_pending_session_id", "") or "auto"
    if _recent_auto_dispatch_matches(app, candidate):
        _clear_pending_final_dispatch(app)
        return
    _dispatch_auto_query(
        app,
        candidate,
        raw,
        session_id,
        request_metadata=metadata,
        speculative=False,
    )


def _schedule_final_dispatch(
    app: "OpenAssistApp",
    candidate: str,
    raw: str,
    session_id: str,
    request_metadata: dict | None,
    *,
    delay_ms: int | None = None,
) -> None:
    app._auto_final_pending_query = candidate
    app._auto_final_pending_raw = raw
    app._auto_final_pending_metadata = dict(request_metadata or {})
    app._auto_final_pending_session_id = session_id
    app._auto_final_pending_at = time.time()
    app._auto_final_pending_seq = int(getattr(app, "_auto_final_pending_seq", 0) or 0) + 1
    seq = app._auto_final_pending_seq
    if delay_ms is None:
        delay_ms = _final_dispatch_delay_ms(app, raw, candidate)
    delay_ms = max(0, int(delay_ms or 0))
    app.overlay.update_transcript(raw)
    logger.info(
        "[%s] Auto Mode queued final dispatch for follow-up merge | delay_ms=%d query=%r",
        session_id,
        delay_ms,
        candidate[:120],
    )
    try:
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(delay_ms, lambda: _dispatch_pending_final_if_current(app, seq))
    except Exception:
        _dispatch_pending_final_if_current(app, seq)


def _merge_pending_final_dispatch(
    app: "OpenAssistApp",
    candidate: str,
    raw: str,
    session_id: str,
    request_metadata: dict | None,
) -> bool:
    pending = (getattr(app, "_auto_final_pending_query", "") or "").strip()
    if not pending:
        return False
    pending_at = float(getattr(app, "_auto_final_pending_at", 0.0) or 0.0)
    max_wait_ms = int(app.config.get("ai.auto_mode.final_dispatch_max_wait_ms", 6500) or 6500)
    if pending_at <= 0 or (time.time() - pending_at) * 1000.0 > max_wait_ms:
        _dispatch_pending_final_if_current(app, int(getattr(app, "_auto_final_pending_seq", 0) or 0))
        return False

    prior_raw = (getattr(app, "_auto_final_pending_raw", "") or pending).strip()
    merged_raw = merge_transcripts(prior_raw, raw, max_overlap_words=24).strip()
    merged_candidate = (sanitize_query_label(merge_transcripts(pending, candidate, max_overlap_words=24)) or candidate).strip()
    metadata = dict(getattr(app, "_auto_final_pending_metadata", None) or {})
    metadata.update(dict(request_metadata or {}))
    followup_delay_ms = max(0, int(app.config.get("ai.auto_mode.final_dispatch_debounce_ms", 900) or 900))
    _schedule_final_dispatch(
        app,
        merged_candidate,
        merged_raw,
        session_id,
        metadata,
        delay_ms=followup_delay_ms,
    )
    logger.info("[%s] Auto Mode merged follow-up into pending final dispatch | query=%r", session_id, merged_candidate[:120])
    return True


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

    Auto Mode keeps the standard local ASR and provider pipeline, but treats
    each final transcript as a potential turn:
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
    setup_only_regex = bool(looks_like_setup_statement(cleaned))

    # Use embedding classifier to refine setup vs question decision
    classifier = _get_intent_classifier()
    if classifier is not None:
        setup_only = classifier.is_likely_setup(
            cleaned, regex_says_setup=(setup_only_regex and not actionable)
        )
    else:
        setup_only = setup_only_regex and not actionable

    if setup_only or not actionable:
        app.overlay.update_transcript(raw)
        app._pending_incomplete_audio_query = candidate or cleaned
        app._pending_incomplete_audio_at = time.time()
        # Learn: this transcript was identified as setup context
        if setup_only_regex and setup_only:
            _auto_learn(cleaned, "setup", app=app)
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

    if _merge_pending_final_dispatch(app, candidate, raw, session_id, request_metadata):
        return True

    if _recent_auto_dispatch_matches(app, candidate):
        logger.info("[%s] Auto Mode duplicate final transcript suppressed | query=%r", session_id, candidate[:120])
        return True

    if _should_defer_for_followup(raw, candidate):
        _schedule_final_dispatch(app, candidate, raw, session_id, request_metadata)
        return True

    # Learn: this transcript is being dispatched as a question
    _auto_learn(candidate, "question", app=app)

    _dispatch_auto_query(
        app,
        candidate,
        raw,
        session_id,
        request_metadata=request_metadata,
        speculative=False,
    )
    return True
