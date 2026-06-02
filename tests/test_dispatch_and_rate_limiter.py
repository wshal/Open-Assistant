import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PyQt6.QtCore import QCoreApplication

from core.state import AppState
from utils.rate_limiter import RateLimiter


def _process_events_until(predicate, timeout_s=1.0):
    app = QCoreApplication.instance() or QCoreApplication([])
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    app.processEvents()
    return predicate()


def test_app_state_dispatches_worker_thread_setters_to_owner_thread():
    state = AppState()
    owner_thread = state.thread()
    observed = []
    state.mode_changed.connect(lambda mode: observed.append((mode, state.thread())))

    worker = threading.Thread(target=lambda: setattr(state, "mode", "coding"))
    worker.start()
    worker.join(timeout=1.0)

    assert _process_events_until(lambda: state.mode == "coding")
    assert observed == [("coding", owner_thread)]


def test_hotkey_dispatch_uses_signal_bridge_instead_of_thread_local_timer():
    from core.hotkeys import HotkeyManager

    calls = []
    app = SimpleNamespace(
        toggle_overlay=lambda: calls.append("toggle"),
        cancel_generation=lambda: None,
        quick_answer=lambda: None,
        analyze_current_screen=lambda: None,
        paste_as_context=lambda: None,
        history_prev=lambda: None,
        history_next=lambda: None,
        scroll_up=lambda: None,
        scroll_down=lambda: None,
        move_left=lambda: None,
        move_right=lambda: None,
        move_up=lambda: None,
        move_down=lambda: None,
        toggle_audio=lambda: None,
        emergency_erase=lambda: None,
        toggle_mini_mode=lambda: None,
        switch_mode=lambda: None,
        toggle_click_through=lambda: None,
        start_move=lambda direction: None,
        stop_move=lambda: None,
    )
    config = SimpleNamespace(get=lambda path, default=None: {})

    with patch("core.hotkeys.QApplication.instance") as qapp_instance:
        qapp_instance.return_value.focusChanged.connect = lambda callback: None
        manager = HotkeyManager(config, app)

    with patch("core.hotkeys.QTimer.singleShot") as single_shot:
        manager._handle_trigger("toggle")

    assert calls == ["toggle"]
    single_shot.assert_not_called()


def test_rate_limiter_rejects_invalid_limits():
    limiter = RateLimiter()
    with pytest.raises(ValueError, match="rpm"):
        limiter.configure("groq", rpm=0)
    with pytest.raises(ValueError, match="rpd"):
        limiter.configure("groq", rpd=0)


def test_rate_limiter_wait_respects_short_deadline():
    async def run():
        limiter = RateLimiter()
        limiter.configure("groq", rpm=1, rpd=1)
        await limiter.wait_if_needed("groq", max_wait_s=0.01)
        started = time.monotonic()
        with pytest.raises(Exception):
            await limiter.wait_if_needed("groq", max_wait_s=0.05)
        assert time.monotonic() - started < 0.5

    asyncio.run(run())
