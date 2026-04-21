from pynput import keyboard
from PyQt6.QtCore import QObject, pyqtSignal, QTimer
from PyQt6.QtWidgets import QApplication
import ctypes
from ctypes import wintypes
import sys
import threading
import time
from utils.logger import setup_logger

logger = setup_logger(__name__)


class HotkeySignals(QObject):
    triggered = pyqtSignal(str)
    started = pyqtSignal(str)
    stopped = pyqtSignal(str)


class NativeHotkeyThread(threading.Thread):
    """Windows-specific RegisterHotKey polling thread."""

    MOD_NOREPEAT = 0x4000

    def __init__(self, manager):
        super().__init__(name="NativeHotkeyThread", daemon=True)
        self.manager = manager
        self.thread_id = None
        self._registered_ids = []
        self.WM_HOTKEY = 0x0312

    def run(self):
        self.thread_id = ctypes.windll.kernel32.GetCurrentThreadId()
        user32 = ctypes.windll.user32

        # AGGRESSIVE CLEANUP: Flush all potential IDs (100-200) to ensure a clean slate
        # NOTE: RegisterHotKey is thread-specific, but we clean up to be safe.
        for i in range(100, 200):
            user32.UnregisterHotKey(None, i)

        current_id = 100
        for action, key_str in self.manager.config.get("hotkeys", {}).items():
            if action.startswith("move_") or action.startswith("scroll_"):
                continue
            mods, vk = self.manager._parse_key_native(key_str)
            if action == "toggle":
                mods |= self.MOD_NOREPEAT
            if vk:
                if user32.RegisterHotKey(None, current_id, mods, vk):
                    self.manager._id_to_action[current_id] = action
                    self._registered_ids.append(current_id)
                    logger.info(f"🔑 Registered (Native): {action} -> {key_str}")
                else:
                    logger.warning(f"⚠️ Native Hotkey Failed: {action}")
                current_id += 1

        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) != 0:
            if msg.message == self.WM_HOTKEY:
                self.manager._handle_native_trigger(msg.wParam)
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))

    def stop(self):
        # Unregister all before dying
        user32 = ctypes.windll.user32
        for hid in self._registered_ids:
            user32.UnregisterHotKey(None, hid)
            
        if self.thread_id:
            # Post WM_QUIT to exit the GetMessage loop
            ctypes.windll.user32.PostThreadMessageW(self.thread_id, 0x0012, 0, 0)
            logger.debug("🛑 Native Hotkey thread termination signal sent")


class HotkeyManager:
    MODIFIER_GROUPS = {
        "ctrl": {"key.ctrl", "key.ctrl_l", "key.ctrl_r"},
        "shift": {"key.shift", "key.shift_l", "key.shift_r"},
        "alt": {"key.alt", "key.alt_l", "key.alt_r", "key.alt_gr"},
    }

    def __init__(self, config, app):
        self.config = config
        self.app = app
        self.signals = HotkeySignals()
        self.active_keys = {}
        self._id_to_action = {}
        self.signals.triggered.connect(self._handle_trigger)
        self.signals.started.connect(
            lambda n: (
                self.app.start_move(n.replace("move_", ""))
                if n.startswith("move_")
                else None
            )
        )
        self.signals.stopped.connect(
            lambda n: self.app.stop_move() if n.startswith("move_") else None
        )

        self.cleanup_timer = QTimer()
        self.cleanup_timer.timeout.connect(self._cleanup_stuck_keys)
        self.cleanup_timer.start(5000)

        # GHOST GUARD: Reset state if application loses focus (Alt-Tab prevention)
        QApplication.instance().focusChanged.connect(self._on_focus_changed)

    def _on_focus_changed(self, old, new):
        if new is None:
            logger.debug("⌨️ Focus Lost: Resetting hotkey state.")
            self.reset_state()

    def _cleanup_stuck_keys(self):
        now = time.time()
        stuck = [k for k, t in self.active_keys.items() if now - t > 10.0]
        if stuck:
            for k in stuck:
                del self.active_keys[k]
            self.app.stop_move()

    def reset_state(self):
        self.active_keys.clear()
        self.app.stop_move()

    def restart(self):
        """Cleanly rebuild listeners with updated config. Layer 6 stability fix."""
        logger.info("🔄 Restarting Hotkey Engine...")
        self.stop()
        time.sleep(0.3)  # Critical settle time for Windows OS
        self.start()

    def _handle_trigger(self, action):
        bridges = {
            "toggle": self.app.toggle_overlay,
            "quick_answer": self.app.quick_answer,
            "analyze_screen": self.app.analyze_current_screen,
            "history_prev": self.app.history_prev,
            "history_next": self.app.history_next,
            "scroll_up": self.app.scroll_up,
            "scroll_down": self.app.scroll_down,
            "move_left": self.app.move_left,
            "move_right": self.app.move_right,
            "move_up": self.app.move_up,
            "move_down": self.app.move_down,
            "toggle_audio": self.app.toggle_audio,
            "emergency_erase": self.app.emergency_erase,
            "mini_mode": self.app.toggle_mini_mode,
            "switch_mode": self.app.switch_mode,
            "stealth": self.app.toggle_stealth_mode,
            "toggle_click_through": self.app.toggle_click_through,
        }
        if action in bridges:
            QTimer.singleShot(0, bridges[action])

    def _handle_native_trigger(self, hotkey_id):
        action = self._id_to_action.get(hotkey_id)
        if action:
            self.signals.triggered.emit(action)

    def start(self):
        if sys.platform == "win32":
            self.native_thread = NativeHotkeyThread(self)
            self.native_thread.start()
        self.listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )
        self.listener.start()
        logger.info("⌨️  Multi-Engine Hotkey System Active")

    def stop(self):
        """Harden cleanup to prevent race-condition crashes."""
        if hasattr(self, "native_thread"):
            self.native_thread.stop()
            self.native_thread.join(timeout=1.0)  # Ensure it's dead

        if hasattr(self, "listener"):
            try:
                self.listener.stop()
            except Exception as e:
                logger.debug(f"Hotkey listener stop: {e}")

        self.active_keys.clear()
        self._id_to_action.clear()

    def _on_press(self, key):
        k = self._normalize_key(key)
        self.active_keys[k] = time.time()
        matched_actions = []
        for action, key_str in self.config.get("hotkeys", {}).items():
            if not (action.startswith("move_") or action.startswith("scroll_")):
                continue
            req = self._parse_key_pynput(key_str)
            if not req:
                continue
            if self._is_exact_hotkey_match(req) and any(k in group for group in req):
                matched_actions.append((len(req), action))

        for _, action in sorted(matched_actions, reverse=True):
            logger.debug(f"🔥 Hotkey detected: {action}")
            if action.startswith("move_"):
                self.signals.started.emit(action)
            else:
                self.signals.triggered.emit(action)
            break

    def _on_release(self, key):
        k = self._normalize_key(key)
        if k in self.active_keys:
            del self.active_keys[k]
            for action, key_str in self.config.get("hotkeys", {}).items():
                if action.startswith("move_"):
                    req = self._parse_key_pynput(key_str)
                    if any(k in g for g in req):
                        self.signals.stopped.emit(action)

    def _is_exact_hotkey_match(self, required_groups):
        active_keys = set(self.active_keys.keys())
        if not all(any(alias in active_keys for alias in group) for group in required_groups):
            return False

        active_modifiers = set()
        for name, aliases in self.MODIFIER_GROUPS.items():
            if active_keys & aliases:
                active_modifiers.add(name)

        required_modifiers = set()
        for group in required_groups:
            for name, aliases in self.MODIFIER_GROUPS.items():
                if group <= aliases:
                    required_modifiers.add(name)
                    break

        return active_modifiers == required_modifiers

    def _normalize_key(self, key):
        if isinstance(key, keyboard.Key):
            return str(key).lower()
        if isinstance(key, keyboard.KeyCode):
            if key.vk is not None:
                return f"vk_{key.vk}"
            if key.char:
                return key.char.lower()
        return str(key).lower()

    def _parse_key_native(self, key_str):
        MOD_CON, MOD_SHF, MOD_ALT, MOD_WIN = 0x0002, 0x0004, 0x0001, 0x0008
        VK_MAP = {
            "space": 0x20,
            "enter": 0x0D,
            "esc": 0x1B,
            "tab": 0x09,
            "backspace": 0x08,
            "up": 0x26,
            "down": 0x28,
            "left": 0x25,
            "right": 0x27,
            "pageup": 0x21,
            "pagedown": 0x22,
            "home": 0x24,
            "end": 0x23,
            "ins": 0x2D,
            "del": 0x2E,
            "[": 0xDB,
            "]": 0xDD,
            "\\": 0xDC,
            "/": 0xBF,
            "`": 0xC0,
            "m": 0x4D,
            "a": 0x41,
            "s": 0x53,
            "z": 0x5A,
            "n": 0x4E,
            "e": 0x45,
            "f1": 0x70,
            "f2": 0x71,
            "f3": 0x72,
            "f4": 0x73,
            "f5": 0x74,
            "f6": 0x75,
            "f7": 0x76,
            "f8": 0x77,
            "f9": 0x78,
            "f10": 0x79,
            "f11": 0x7A,
            "f12": 0x7B,
        }
        parts = key_str.lower().split("+")
        m, v = 0, 0
        for p in parts:
            p = p.strip()
            if p == "ctrl":
                m |= MOD_CON
            elif p == "shift":
                m |= MOD_SHF
            elif p == "alt":
                m |= MOD_ALT
            elif p == "win":
                m |= MOD_WIN
            elif p in VK_MAP:
                v = VK_MAP[p]
            elif len(p) == 1:
                v = ord(p.upper())
        return m, v

    def _parse_key_pynput(self, s):
        parts = s.lower().split("+")
        groups = []
        for p in parts:
            p = p.strip()
            if p == "ctrl":
                groups.append({"key.ctrl", "key.ctrl_l", "key.ctrl_r"})
            elif p == "shift":
                groups.append({"key.shift", "key.shift_l", "key.shift_r"})
            elif p == "alt":
                groups.append({"key.alt", "key.alt_l", "key.alt_r", "key.alt_gr"})
            elif p == "up":
                groups.append({"key.up"})
            elif p == "down":
                groups.append({"key.down"})
            elif p == "left":
                groups.append({"key.left"})
            elif p == "right":
                groups.append({"key.right"})
            elif len(p) == 1:
                groups.append({p, f"vk_{ord(p.upper())}"})
        return groups
