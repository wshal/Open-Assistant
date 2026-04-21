"""Active window detection and smart region capture."""

import time
import threading
from typing import Optional, Tuple, Dict
from PyQt6.QtCore import QObject, pyqtSignal
from utils.platform_utils import ProcessUtils, PlatformInfo
from utils.logger import setup_logger

logger = setup_logger(__name__)


class WindowContext:
    """Represents the current active window state."""
    def __init__(self):
        self.title: str = ""
        self.app_name: str = ""
        self.rect: Optional[Tuple[int, int, int, int]] = None
        self.category: str = "unknown"  # ide, browser, terminal, meeting, document, chat
        self.timestamp: float = 0.0

    def __repr__(self):
        return f"<Window '{self.title[:40]}' cat={self.category}>"


# App name â category mapping
APP_CATEGORIES: Dict[str, str] = {
    # IDEs & Editors
    "code": "ide", "vscode": "ide", "visual studio": "ide",
    "pycharm": "ide", "intellij": "ide", "webstorm": "ide",
    "sublime": "ide", "atom": "ide", "notepad++": "ide",
    "vim": "ide", "neovim": "ide", "emacs": "ide",
    "cursor": "ide", "windsurf": "ide",

    # Browsers
    "chrome": "browser", "firefox": "browser", "safari": "browser",
    "edge": "browser", "brave": "browser", "opera": "browser", "arc": "browser",

    # Terminals
    "terminal": "terminal", "cmd": "terminal", "powershell": "terminal",
    "iterm": "terminal", "warp": "terminal", "alacritty": "terminal",
    "windows terminal": "terminal", "konsole": "terminal",

    # Meeting apps
    "zoom": "meeting", "teams": "meeting", "meet": "meeting",
    "webex": "meeting", "slack": "meeting", "discord": "meeting",
    "google meet": "meeting", "gotomeeting": "meeting",

    # Document editors
    "word": "document", "docs": "document", "pages": "document",
    "notion": "document", "obsidian": "document", "google docs": "document",
    "libreoffice": "document", "overleaf": "document",

    # Chat
    "chatgpt": "chat", "claude": "chat", "whatsapp": "chat",
    "telegram": "chat", "signal": "chat", "messenger": "chat",

    # Design
    "figma": "design", "photoshop": "design", "illustrator": "design",
    "sketch": "design", "canva": "design",

    # Spreadsheets
    "excel": "spreadsheet", "sheets": "spreadsheet", "numbers": "spreadsheet",

    # Email
    "outlook": "email", "gmail": "email", "thunderbird": "email", "mail": "email",
}

# Title keywords â category overrides
TITLE_CATEGORIES: Dict[str, str] = {
    "interview": "meeting",
    "coding challenge": "ide",
    "leetcode": "ide",
    "hackerrank": "ide",
    "codility": "ide",
    "codesignal": "ide",
    "exam": "browser",
    "quiz": "browser",
    "test": "browser",
    "assessment": "browser",
}


class WindowDetector(QObject):
    """Monitor active window changes and detect context."""

    window_changed = pyqtSignal(object)  # WindowContext
    category_changed = pyqtSignal(str)  # category name

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._running = False
        self._thread = None
        self._last_title = ""
        self._last_category = ""
        self.current = WindowContext()
        self._poll_interval = 1.0  # seconds

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("ðª Window detector started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def get_current(self) -> WindowContext:
        """Get current window context (non-blocking)."""
        return self.current

    def detect_now(self) -> WindowContext:
        """Force immediate detection."""
        return self._detect()

    def _monitor_loop(self):
        while self._running:
            try:
                ctx = self._detect()
                if ctx.title != self._last_title:
                    self._last_title = ctx.title
                    self.current = ctx
                    self.window_changed.emit(ctx)

                    if ctx.category != self._last_category:
                        self._last_category = ctx.category
                        self.category_changed.emit(ctx.category)
                        logger.debug(f"Window context: {ctx}")
            except Exception as e:
                logger.debug(f"Window detect error: {e}")

            time.sleep(self._poll_interval)

    def _detect(self) -> WindowContext:
        ctx = WindowContext()
        ctx.timestamp = time.time()

        # Get window title
        ctx.title = ProcessUtils.get_active_window_title()
        if not ctx.title:
            return ctx

        # Get window rect
        ctx.rect = ProcessUtils.get_active_window_rect()

        # Extract app name from title
        ctx.app_name = self._extract_app_name(ctx.title)

        # Categorize
        ctx.category = self._categorize(ctx.title, ctx.app_name)

        return ctx

    @staticmethod
    def _extract_app_name(title: str) -> str:
        """Extract application name from window title."""
        # Common patterns: "File - App Name", "App Name - File"
        parts = title.split(" - ")
        if len(parts) >= 2:
            # Usually the app name is the last part
            return parts[-1].strip()
        parts = title.split(" â ")
        if len(parts) >= 2:
            return parts[-1].strip()
        return title.strip()

    @staticmethod
    def _categorize(title: str, app_name: str) -> str:
        """Determine the category of the active window."""
        title_lower = title.lower()
        app_lower = app_name.lower()

        # Check title keywords first (more specific)
        for keyword, category in TITLE_CATEGORIES.items():
            if keyword in title_lower:
                return category

        # Check app name
        for app_keyword, category in APP_CATEGORIES.items():
            if app_keyword in app_lower or app_keyword in title_lower:
                return category

        return "unknown"

    def suggest_mode(self) -> Optional[str]:
        """Suggest an assistant mode based on current window context."""
        category = self.current.category
        mode_map = {
            "ide": "coding",
            "terminal": "coding",
            "meeting": "meeting",
            "document": "writing",
            "email": "writing",
            "browser": None,  # Could be anything
            "chat": "general",
        }

        # Special title-based overrides
        title = self.current.title.lower()
        if any(kw in title for kw in ["interview", "behavioral", "technical screen"]):
            return "interview"
        if any(kw in title for kw in ["leetcode", "hackerrank", "codility", "codesignal"]):
            return "coding"
        if any(kw in title for kw in ["exam", "quiz", "assessment", "test"]):
            return "exam"

        return mode_map.get(category)