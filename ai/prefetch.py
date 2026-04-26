"""
Predictive Prefetch Engine — P3.1
Monitors the active window and screen text to detect when the user is
looking at a function/class definition in an IDE, then fires a background
RAG prefetch so docs are ready before the user even types a question.

Heuristic (from the spec):
  if cursor_in_function_body: prefetch("explain " + function_name)
"""

import re
import time
import threading
from typing import Optional, Set, Callable, Dict

from utils.logger import setup_logger

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# IDE process/window-title fingerprints
# ---------------------------------------------------------------------------

_IDE_PROCESS_NAMES: Set[str] = {
    # ── Classic desktop IDEs ──────────────────────────────────────────────
    "code",               # VS Code / VS Code Insiders
    "code - insiders",
    "cursor",             # Cursor AI
    "pycharm64",
    "pycharm",
    "idea64",             # IntelliJ IDEA
    "webstorm64",
    "clion64",
    "rider64",
    "datagrip64",
    "goland64",
    "sublime_text",
    "atom",
    "nvim",
    "vim",
    "gvim",
    "notepad++",
    "fleet",              # JetBrains Fleet
    "zed",               # Zed editor
    "helix",
    # ── Visual Studio (not Code) ──────────────────────────────────────────
    "devenv",             # Visual Studio IDE process
    # ── Other common editors ──────────────────────────────────────────────
    "thonny",             # Popular beginner Python IDE
    "idle",              # Python IDLE
    "eclipse",
    "android studio",
    "studio64",           # Android Studio process
    "brackets",
    "kate",
    "gedit",
    "mousepad",
    "kwrite",
    # ── Notepad variants ─────────────────────────────────────────────────
    "notepad",            # Windows Notepad (plain)
}

_IDE_TITLE_FRAGMENTS: tuple = (
    # ── VS Code / Cursor ──────────────────────────────────────────────────
    " — code",            # "file.py — Code"
    " - code",
    "visual studio code",
    "cursor —",
    "cursor -",
    # ── JetBrains family ─────────────────────────────────────────────────
    "pycharm",
    "intellij",
    "webstorm",
    "clion",
    "rider",
    "goland",
    "datagrip",
    "fleet",
    # ── Other desktop IDEs ────────────────────────────────────────────────
    "sublime text",
    "notepad++",
    "vim",
    "nvim",
    "helix",
    "zed —",
    "visual studio",      # MS Visual Studio (devenv)
    "thonny",
    "idle ",               # Python IDLE ("IDLE Shell 3.x")
    "eclipse",
    "android studio",
    # ── Notepad (plain Windows) ───────────────────────────────────────────
    # Only match when filename ends .py/.js/.ts etc. to avoid generic text files
    "notepad",            # catches "Untitled - Notepad" and "main.py - Notepad"
    # ── Online editors / interview platforms ──────────────────────────────
    "replit",
    "codepen",
    "jsfiddle",
    "stackblitz",
    "codesandbox",
    "glitch",
    "github.dev",          # VS Code in browser (github.dev / vscode.dev)
    "vscode.dev",
    "codespaces",          # GitHub Codespaces
    "gitpod",
    # ── Interview / competitive coding platforms ──────────────────────────
    "leetcode",
    "hackerrank",
    "codeforces",
    "coderpad",
    "coderbyte",
    "hackerearth",
    "topcoder",
    "kattis",
    "interviewing.io",
    "pramp",
    "codesignal",
    "codewars",
    # ── Online JS / Frontend editors (user-specified + common) ──────────────
    # Programiz  — https://www.programiz.com/javascript/online-compiler/
    "programiz",
    # OneCompiler — https://onecompiler.com/javascript
    "onecompiler",
    # RunJS      — https://runjs.app/play
    "runjs",
    # PlayCode   — https://playcode.io
    "playcode",
    # W3Schools  — https://www.w3schools.com/tryit/
    "w3schools",
    "w3schools tryit",
    # Plunker    — https://plnkr.co
    "plunker",
    "plnkr",
    # JS.do      — https://js.do
    "js.do",
    # JS Bin     — https://jsbin.com
    "jsbin",
    "js bin",
    # TypeScript Playground — https://www.typescriptlang.org/play
    "typescript playground",
    "typescriptlang.org/play",
    # Vue SFC Playground — https://play.vuejs.org
    "vue playground",
    "play.vuejs",
    # Svelte REPL — https://svelte.dev/repl
    "svelte repl",
    "svelte.dev/repl",
    # React.new (redirects to codesandbox)
    "react.new",
    # Scrimba    — https://scrimba.com (very common in frontend interviews)
    "scrimba",
    # Expo Snack — https://snack.expo.dev  (React Native demos)
    "snack.expo",
    "expo snack",
    # CodeSandbox project titles typically show 'codesandbox'
    "codesandbox",
    # StackBlitz (already present but keep for clarity)
    "stackblitz",
)

# ---------------------------------------------------------------------------
# Code symbol patterns
# ---------------------------------------------------------------------------

# Python / JS / TS / Java / C++ function/class definitions
_SYMBOL_PATTERNS = [
    # Python function/method
    re.compile(r"^\s*(?:async\s+)?def\s+([a-zA-Z_]\w*)\s*\(", re.MULTILINE),
    # Python class
    re.compile(r"^\s*class\s+([a-zA-Z_]\w*)\s*[\(:]", re.MULTILINE),
    # JS/TS function declarations
    re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([a-zA-Z_$][\w$]*)\s*\(", re.MULTILINE),
    # JS/TS arrow / const
    re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([a-zA-Z_$][\w$]*)\s*=\s*(?:async\s*)?\(", re.MULTILINE),
    # Java/C# method (return type + name)
    re.compile(r"^\s*(?:public|private|protected|static|override|virtual|async)[\w\s<>\[\]]*\s+([a-zA-Z_]\w*)\s*\(", re.MULTILINE),
    # C++ function
    re.compile(r"^\s*(?:[\w:*&<>]+\s+)+([a-zA-Z_]\w*)\s*\([^;]*\)\s*(?:const\s*)?(?:noexcept\s*)?\{", re.MULTILINE),
]

_NOISE_SYMBOLS: Set[str] = {
    "if", "for", "while", "with", "try", "except", "else", "elif",
    "import", "from", "return", "yield", "raise", "print", "len",
    "main", "init", "new", "get", "set", "run", "start", "stop",
    "true", "false", "null", "undefined", "none", "self", "cls",
}


def _extract_symbols(text: str) -> list[str]:
    """Extract unique code symbol names from OCR screen text."""
    seen: Set[str] = set()
    results = []
    for pat in _SYMBOL_PATTERNS:
        for m in pat.finditer(text):
            name = m.group(1).strip()
            if (
                len(name) >= 3
                and name.lower() not in _NOISE_SYMBOLS
                and name not in seen
            ):
                seen.add(name)
                results.append(name)
    return results


def _is_ide_window(window_title: str) -> bool:
    """Return True if the active window looks like a code editor."""
    title_lower = (window_title or "").lower()
    return any(frag in title_lower for frag in _IDE_TITLE_FRAGMENTS)


# ---------------------------------------------------------------------------
# PredictivePrefetcher
# ---------------------------------------------------------------------------

class PredictivePrefetcher:
    """
    P3.1: Predictive Prefetch Engine.

    Watches each screen OCR update. When the active window is an IDE and
    a new function/class name is detected, it fires a background RAG
    prefetch for "explain <symbol_name>", so docs are cached before the
    user asks.
    """

    def __init__(self, config, prefetch_fn: Callable):
        """
        Args:
            config: App config object.
            prefetch_fn: Async coroutine factory: prefetch_fn(screen_text, audio_text)
                         This is engine.prefetch_rag.
        """
        self._config = config
        self._prefetch_fn = prefetch_fn
        self._enabled: bool = bool(config.get("ai.prefetch.enabled", True))
        self._debounce_s: float = float(config.get("ai.prefetch.debounce_s", 3.0))
        self._max_symbols: int = int(config.get("ai.prefetch.max_symbols", 3))

        # State
        self._last_symbols: Set[str] = set()
        self._last_fire_ts: Dict[str, float] = {}   # symbol -> last prefetch time
        self._last_analyze_ts: float = 0.0
        self._lock = threading.Lock()
        self._loop = None  # set by wire()

        if self._enabled:
            logger.info(
                f"[P3.1 Prefetcher] Initialised — debounce={self._debounce_s}s, "
                f"max_symbols={self._max_symbols}"
            )
        else:
            logger.info("[P3.1 Prefetcher] Disabled via config")

    def wire(self, loop) -> None:
        """Call once after the asyncio loop is running."""
        self._loop = loop

    def reset(self) -> None:
        """Call on new session."""
        with self._lock:
            self._last_symbols.clear()
            self._last_fire_ts.clear()
            self._last_analyze_ts = 0.0
        logger.info("[P3.1 Prefetcher] Reset for new session")

    def analyze(self, screen_text: str, window_title: str = "") -> None:
        """
        Called from _on_screen_text on every OCR update.
        Runs synchronously — analysis is cheap (regex only).
        Prefetch is dispatched async and never blocks.
        """
        if not self._enabled or not self._loop:
            return

        now = time.time()
        # Overall debounce — don't re-analyze more often than every N seconds
        if now - self._last_analyze_ts < self._debounce_s:
            return
        self._last_analyze_ts = now

        # Only run in IDE windows
        if not _is_ide_window(window_title):
            logger.debug(
                f"[P3.1 Prefetcher] Non-IDE window ('{window_title[:40]}'), skipping"
            )
            return

        symbols = _extract_symbols(screen_text)
        if not symbols:
            logger.debug("[P3.1 Prefetcher] No code symbols detected in screen text")
            return

        with self._lock:
            new_symbols = [
                s for s in symbols
                if s not in self._last_symbols
                or (now - self._last_fire_ts.get(s, 0)) > 120  # re-prefetch after 2 min
            ]
            if not new_symbols:
                logger.debug(
                    f"[P3.1 Prefetcher] All {len(symbols)} symbols already cached, skipping"
                )
                return

            # Cap to avoid spamming the RAG engine
            targets = new_symbols[: self._max_symbols]
            for s in targets:
                self._last_fire_ts[s] = now
            self._last_symbols.update(targets)

        for symbol in targets:
            query = f"explain {symbol}"
            logger.info(
                f"[P3.1 Prefetcher] 🔮 Prefetching docs for new symbol: '{symbol}' "
                f"→ RAG query: '{query}'"
            )
            # Fire-and-forget into the existing async loop
            import asyncio
            asyncio.run_coroutine_threadsafe(
                self._prefetch_fn(screen_text=query, audio_text=""),
                self._loop,
            )
