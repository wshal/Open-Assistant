"""Runtime state helpers for Whisper-first Auto Mode."""

from __future__ import annotations

import time


def init_auto_mode_state(app) -> None:
    app._auto_answer_context = ""
    app._auto_answer_context_at = 0.0
    app._auto_answer_last_dispatched_query = ""
    app._auto_answer_last_dispatched_at = 0.0
    app._auto_interim_stable_key = ""
    app._auto_interim_stable_at = 0.0
    app._auto_interim_pending_query = ""
    app._auto_interim_pending_raw = ""
    app._auto_interim_pending_seq = 0
    app._pending_incomplete_audio_query = ""
    app._pending_incomplete_audio_at = 0.0


def reset_auto_mode_turn_state(app, *, cancel_pending_interim: bool = True) -> None:
    app._auto_answer_context = ""
    app._auto_answer_context_at = 0.0
    app._auto_answer_last_dispatched_query = ""
    app._auto_answer_last_dispatched_at = 0.0
    app._auto_interim_stable_key = ""
    app._auto_interim_stable_at = 0.0
    app._auto_interim_pending_query = ""
    app._auto_interim_pending_raw = ""
    if cancel_pending_interim:
        app._auto_interim_pending_seq = int(getattr(app, "_auto_interim_pending_seq", 0) or 0) + 1
    app._pending_incomplete_audio_query = ""
    app._pending_incomplete_audio_at = 0.0


def auto_mode_requested(app) -> bool:
    return bool(app.config.get("ai.auto_mode.enabled", False))


def start_auto_mode(app) -> None:
    app.config.set("ai.auto_mode.enabled", True)
    if hasattr(app.audio, "set_standard_transcription_suspended"):
        app.audio.set_standard_transcription_suspended(False, "auto-mode")
    update = getattr(getattr(app, "overlay", None), "update_auto_mode_state", None)
    if callable(update):
        update(auto_mode_requested(app), False, reconnecting=False, fallback=False)
    if getattr(app, "session_active", False):
        app.overlay.update_transcript("Auto Mode listening...", state="listening")


def toggle_auto_mode(app) -> bool:
    new_state = not auto_mode_requested(app)
    app.config.set("ai.auto_mode.enabled", new_state)
    try:
        if hasattr(app.config, "save"):
            app.config.save()
    except Exception:
        pass
    if getattr(app, "session_active", False):
        if new_state:
            start_auto_mode(app)
        else:
            if hasattr(app.audio, "set_standard_transcription_suspended"):
                app.audio.set_standard_transcription_suspended(False, "auto-mode-off")
            app.overlay.update_transcript(
                "Auto Mode disabled - using standard audio pipeline.",
                state="idle",
            )
    update = getattr(getattr(app, "overlay", None), "update_auto_mode_state", None)
    if callable(update):
        update(new_state, False, reconnecting=False, fallback=False)
    if hasattr(app, "overlay") and hasattr(app.overlay, "refresh_standby_state"):
        app.overlay.refresh_standby_state()
    return new_state


def mark_auto_context_timestamp(app) -> None:
    app._auto_answer_context_at = time.time()
