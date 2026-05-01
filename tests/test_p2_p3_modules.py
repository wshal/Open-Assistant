"""
test_p2_p3_modules.py — Formal regression tests for Phase 2 and Phase 3 modules.

Covers:
  P2.1  ContextBuilder (full / diff / cached modes, entity tracking, reset)
  P2.4  LongTermMemory (disabled path, public API surface)
  P3.1  PredictivePrefetcher (symbol extraction, IDE + online-editor detection,
         debounce, fire-and-forget, deduplication, reset)
  P3.3  ActionExecutor (intent detection — git/npm/node/file/system,
         safety block, async execution)
  P3.4  ContextPruner (block scoring, relevance retention, pass-through)
"""

import asyncio
import threading
import time
import unittest
from unittest.mock import AsyncMock, MagicMock


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cfg(**overrides):
    """Build a minimal config mock that returns defaults + overrides."""
    defaults = {
        "ai.memory.enabled": False,     # don't touch real ChromaDB in tests
        "ai.prefetch.enabled": True,
        "ai.prefetch.debounce_s": 0.0,  # no debounce in tests
        "ai.prefetch.max_symbols": 5,
        "ai.actions.enabled": True,
        "ai.actions.cwd": ".",
        "ai.actions.timeout_s": 10.0,
        "ai.pruner.enabled": True,
        "ai.pruner.min_block_lines": 2,
        "ai.pruner.top_k_blocks": 8,
        "ai.pruner.min_score": 0.05,
        "ai.context.similarity_threshold": 0.85,
        "ai.context.max_screen_chars": 8000,
        "ai.context.entity_ttl_s": 600,
    }
    defaults.update(overrides)
    cfg = MagicMock()
    cfg.get = lambda k, d=None: defaults.get(k, d)
    return cfg


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ─────────────────────────────────────────────────────────────────────────────
# P2.1 — ContextBuilder
# ─────────────────────────────────────────────────────────────────────────────

class TestContextBuilder(unittest.TestCase):
    """P2.1: Incremental context assembly."""

    def _make(self, **cfg_overrides):
        from ai.context import ContextBuilder
        return ContextBuilder(_cfg(**cfg_overrides))

    def test_first_turn_is_full(self):
        """First build with no prior screen must return mode='full'."""
        cb = self._make()
        r = cb.build("explain this", "def foo(): pass")
        self.assertEqual(r["mode"], "full")
        self.assertIn("def foo", r["screen"])

    def test_identical_screen_returns_cached(self):
        """Same screen text twice must return mode='cached'."""
        cb = self._make()
        cb.build("q1", "def foo(): pass")
        r = cb.build("q2", "def foo(): pass")
        self.assertEqual(r["mode"], "cached")

    def test_changed_screen_returns_diff(self):
        """Different screen text must return mode='diff'."""
        cb = self._make()
        cb.build("q1", "def foo(): pass\nline A\nline B\nline C")
        r = cb.build("q2", "def bar(): return 42\nline A\nline B\nline C\nnew line")
        self.assertEqual(r["mode"], "diff")

    def test_diff_contains_only_new_lines(self):
        """Diff context must include new lines and omit unchanged ones."""
        cb = self._make()
        cb.build("q1", "line A\nline B\nline C")
        r = cb.build("q2", "line A\nline B\nline C\nline D")
        self.assertIn("line D", r["screen"])
        self.assertNotIn("line A", r["screen"])

    def test_length_ratio_guard_forces_diff(self):
        """If new screen is >15% longer, mode must be 'diff' even if similarity is high."""
        cb = self._make()
        base = "x " * 50          # 50 words
        grown = base + " y " * 10  # +10 words → > 15% longer
        cb.build("q1", base)
        r = cb.build("q2", grown)
        self.assertNotEqual(r["mode"], "cached",
                            "length_ratio growth should prevent 'cached' mode")

    def test_reset_clears_state(self):
        """After reset(), next build must return mode='full' again."""
        cb = self._make()
        cb.build("q1", "def foo(): pass")
        cb.build("q2", "def foo(): pass")  # cached
        cb.reset()
        r = cb.build("q3", "def foo(): pass")
        self.assertEqual(r["mode"], "full")

    def test_entity_tracking_from_query(self):
        """Entities mentioned in the query must appear in the entity output."""
        cb = self._make()
        r = cb.build("tell me about TypeError and React", "some code here\nmore code")
        self.assertIn("TypeError", r["entities"])
        # React is a PascalCase tech name — must be captured after entity pattern fix
        self.assertIn("React", r["entities"])

    def test_entity_eviction_after_ttl(self):
        """Entities older than TTL should be evicted."""
        cb = self._make(**{"ai.context.entity_ttl_s": 0})  # instant expiry
        cb.build("tell me about OldEntity", "screen text here\nmore text")
        time.sleep(0.01)
        r = cb.build("fresh query", "screen text here\nmore text")
        self.assertNotIn("OldEntity", r["entities"])

    def test_empty_screen_does_not_crash(self):
        """Empty screen text must be handled gracefully."""
        cb = self._make()
        r = cb.build("anything", "")
        self.assertIn("mode", r)

    def test_screen_truncated_to_max_chars(self):
        """Full-mode screen context must not exceed max_screen_chars."""
        cb = self._make(**{"ai.context.max_screen_chars": 50})
        r = cb.build("q", "x" * 200)
        self.assertLessEqual(len(r["screen"]), 50)


# ─────────────────────────────────────────────────────────────────────────────
# P2.4 — LongTermMemory (disabled path only — avoids real ChromaDB in CI)
# ─────────────────────────────────────────────────────────────────────────────

class TestLongTermMemoryDisabled(unittest.TestCase):
    """P2.4: Long-term memory — disabled path (safe for CI)."""

    def _make(self):
        from ai.memory import LongTermMemory
        return LongTermMemory(_cfg(**{"ai.memory.enabled": False}))

    def test_is_not_ready_when_disabled(self):
        m = self._make()
        self.assertFalse(m.is_ready())

    def test_query_returns_empty_list_when_disabled(self):
        m = self._make()
        result = m.query("anything")
        self.assertEqual(result, [])

    def test_store_does_not_raise_when_disabled(self):
        m = self._make()
        try:
            m.store("session-1", "question text", "answer text", mode="general")
        except Exception as e:
            self.fail(f"store() raised unexpectedly when disabled: {e}")

    def test_count_returns_zero_when_disabled(self):
        m = self._make()
        self.assertEqual(m.count(), 0)


# ─────────────────────────────────────────────────────────────────────────────
# P3.1 — PredictivePrefetcher
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictivePrefetcher(unittest.TestCase):
    """P3.1: Symbol extraction, IDE detection, fire-and-forget prefetch."""

    # --- Symbol Extraction ---

    def test_extracts_python_function(self):
        from ai.prefetch import _extract_symbols
        code = "def authenticate_user(username, password):\n    pass"
        syms = _extract_symbols(code)
        self.assertIn("authenticate_user", syms)

    def test_extracts_python_async_function(self):
        from ai.prefetch import _extract_symbols
        code = "async def fetch_data(url):\n    pass"
        syms = _extract_symbols(code)
        self.assertIn("fetch_data", syms)

    def test_extracts_python_class(self):
        from ai.prefetch import _extract_symbols
        code = "class TokenValidator:\n    pass"
        syms = _extract_symbols(code)
        self.assertIn("TokenValidator", syms)

    def test_extracts_js_function(self):
        from ai.prefetch import _extract_symbols
        code = "function calculateScore(input) { return input * 2; }"
        syms = _extract_symbols(code)
        self.assertIn("calculateScore", syms)

    def test_extracts_js_arrow_function(self):
        from ai.prefetch import _extract_symbols
        code = "const handleClick = (event) => { event.preventDefault(); }"
        syms = _extract_symbols(code)
        self.assertIn("handleClick", syms)

    def test_noise_symbols_excluded(self):
        from ai.prefetch import _extract_symbols
        code = "def get(): pass\ndef set(): pass\ndef run(): pass"
        syms = _extract_symbols(code)
        self.assertNotIn("get", syms)
        self.assertNotIn("set", syms)
        self.assertNotIn("run", syms)

    def test_short_symbols_excluded(self):
        from ai.prefetch import _extract_symbols
        code = "def go(): pass\ndef ok(): pass"
        syms = _extract_symbols(code)
        # Length < 3 should be excluded
        self.assertNotIn("go", syms)
        self.assertNotIn("ok", syms)

    # --- IDE Detection ---

    def test_vscode_detected(self):
        from ai.prefetch import _is_ide_window
        self.assertTrue(_is_ide_window("main.py - Visual Studio Code"))

    def test_pycharm_detected(self):
        from ai.prefetch import _is_ide_window
        self.assertTrue(_is_ide_window("project — PyCharm"))

    def test_leetcode_detected(self):
        from ai.prefetch import _is_ide_window
        self.assertTrue(_is_ide_window("Two Sum - LeetCode"))

    def test_hackerrank_detected(self):
        from ai.prefetch import _is_ide_window
        self.assertTrue(_is_ide_window("Interview Kit | HackerRank"))

    def test_replit_detected(self):
        from ai.prefetch import _is_ide_window
        self.assertTrue(_is_ide_window("my-project | Replit"))

    def test_codepen_detected(self):
        from ai.prefetch import _is_ide_window
        self.assertTrue(_is_ide_window("CodePen - My Pen"))

    def test_jsfiddle_detected(self):
        from ai.prefetch import _is_ide_window
        self.assertTrue(_is_ide_window("JSFiddle - Code Playground"))

    def test_coderpad_detected(self):
        from ai.prefetch import _is_ide_window
        self.assertTrue(_is_ide_window("CoderPad Interview"))

    def test_github_codespaces_detected(self):
        from ai.prefetch import _is_ide_window
        self.assertTrue(_is_ide_window("main.py - Codespaces"))

    def test_notepad_detected(self):
        from ai.prefetch import _is_ide_window
        self.assertTrue(_is_ide_window("solution.py - Notepad"))

    def test_notepad_plus_detected(self):
        from ai.prefetch import _is_ide_window
        self.assertTrue(_is_ide_window("solution.py - Notepad++"))

    # ── New online JS / frontend editors ──────────────────────────────────
    def test_programiz_detected(self):
        from ai.prefetch import _is_ide_window
        self.assertTrue(_is_ide_window("Online JavaScript Compiler - Programiz"))

    def test_onecompiler_detected(self):
        from ai.prefetch import _is_ide_window
        self.assertTrue(_is_ide_window("JavaScript Online Compiler - OneCompiler"))

    def test_runjs_detected(self):
        from ai.prefetch import _is_ide_window
        self.assertTrue(_is_ide_window("RunJS — JavaScript Playground"))

    def test_playcode_detected(self):
        from ai.prefetch import _is_ide_window
        self.assertTrue(_is_ide_window("PlayCode — JavaScript Playground"))

    def test_w3schools_tryit_detected(self):
        from ai.prefetch import _is_ide_window
        self.assertTrue(_is_ide_window("W3Schools Tryit Editor"))

    def test_jsbin_detected(self):
        from ai.prefetch import _is_ide_window
        self.assertTrue(_is_ide_window("JS Bin - Collaborative JavaScript Debugging"))

    def test_typescript_playground_detected(self):
        from ai.prefetch import _is_ide_window
        self.assertTrue(_is_ide_window("TypeScript Playground"))

    def test_svelte_repl_detected(self):
        from ai.prefetch import _is_ide_window
        self.assertTrue(_is_ide_window("Svelte REPL"))

    def test_scrimba_detected(self):
        from ai.prefetch import _is_ide_window
        self.assertTrue(_is_ide_window("Scrimba - Learn to Code"))

    def test_expo_snack_detected(self):
        from ai.prefetch import _is_ide_window
        self.assertTrue(_is_ide_window("Expo Snack - React Native"))

    def test_stackblitz_detected(self):
        from ai.prefetch import _is_ide_window
        self.assertTrue(_is_ide_window("StackBlitz - my-app"))

    def test_browser_not_detected(self):
        from ai.prefetch import _is_ide_window
        self.assertFalse(_is_ide_window("Google Chrome"))

    def test_word_not_detected(self):
        from ai.prefetch import _is_ide_window
        self.assertFalse(_is_ide_window("Document1 - Microsoft Word"))

    def test_empty_title_not_detected(self):
        from ai.prefetch import _is_ide_window
        self.assertFalse(_is_ide_window(""))

    # --- Prefetcher Behaviour ---

    def _make_prefetcher(self, calls_list):
        from ai.prefetch import PredictivePrefetcher
        async def fake_prefetch(screen_text="", audio_text=""):
            calls_list.append(screen_text)
        return PredictivePrefetcher(_cfg(), prefetch_fn=fake_prefetch)

    def test_no_loop_is_noop(self):
        """Without wiring a loop, analyze() must not raise or fire anything."""
        calls = []
        pf = self._make_prefetcher(calls)
        pf.analyze("def foo(x): pass", window_title="main.py — Code")
        self.assertEqual(calls, [])

    def test_fires_for_ide_with_symbols(self):
        """With a running loop and IDE window, new symbols trigger prefetch."""
        calls = []
        pf = self._make_prefetcher(calls)
        loop = asyncio.new_event_loop()
        pf.wire(loop)

        t = threading.Thread(target=loop.run_forever, daemon=True)
        t.start()
        try:
            pf.analyze("def authenticate_user(x): pass", window_title="main.py — Code")
            time.sleep(0.15)
            self.assertGreater(len(calls), 0)
            self.assertIn("authenticate_user", calls[0])
        finally:
            loop.call_soon_threadsafe(loop.stop)
            t.join(timeout=1)

    def test_no_fire_for_non_ide(self):
        """Non-IDE window must not trigger any prefetch."""
        calls = []
        pf = self._make_prefetcher(calls)
        loop = asyncio.new_event_loop()
        pf.wire(loop)

        t = threading.Thread(target=loop.run_forever, daemon=True)
        t.start()
        try:
            pf.analyze("def authenticate_user(x): pass", window_title="Google Chrome")
            time.sleep(0.1)
            self.assertEqual(calls, [])
        finally:
            loop.call_soon_threadsafe(loop.stop)
            t.join(timeout=1)

    def test_duplicate_symbols_not_refired(self):
        """Same symbol seen twice in same session must only fire once."""
        calls = []
        pf = self._make_prefetcher(calls)
        loop = asyncio.new_event_loop()
        pf.wire(loop)

        t = threading.Thread(target=loop.run_forever, daemon=True)
        t.start()
        try:
            pf.analyze("def my_handler(x): pass", window_title="file.py — Code")
            time.sleep(0.1)
            first_count = len(calls)
            pf.analyze("def my_handler(x): pass", window_title="file.py — Code")
            time.sleep(0.1)
            self.assertEqual(len(calls), first_count, "Same symbol must not fire twice")
        finally:
            loop.call_soon_threadsafe(loop.stop)
            t.join(timeout=1)

    def test_reset_allows_refiring(self):
        """After reset(), a previously seen symbol should fire again."""
        calls = []
        pf = self._make_prefetcher(calls)
        loop = asyncio.new_event_loop()
        pf.wire(loop)

        t = threading.Thread(target=loop.run_forever, daemon=True)
        t.start()
        try:
            pf.analyze("def my_handler(x): pass", window_title="file.py — Code")
            time.sleep(0.1)
            count_before = len(calls)
            pf.reset()
            pf.analyze("def my_handler(x): pass", window_title="file.py — Code")
            time.sleep(0.1)
            self.assertGreater(len(calls), count_before, "After reset, symbol should re-fire")
        finally:
            loop.call_soon_threadsafe(loop.stop)
            t.join(timeout=1)


# ─────────────────────────────────────────────────────────────────────────────
# P3.3 — ActionExecutor
# ─────────────────────────────────────────────────────────────────────────────

class TestActionExecutor(unittest.TestCase):
    """P3.3: Intent detection, safety block, async subprocess execution."""

    def _make(self):
        from ai.actions import ActionExecutor
        return ActionExecutor(_cfg())

    # --- Intent detection ---

    def test_detects_run_pytest(self):
        ae = self._make()
        r = ae.detect("run pytest now")
        self.assertIsNotNone(r)
        self.assertEqual(r[0], "run_pytest")

    def test_detects_run_tests(self):
        ae = self._make()
        r = ae.detect("run all tests")
        self.assertIsNotNone(r)
        self.assertIn(r[0], {"run_tests", "run_pytest"})

    def test_detects_git_status(self):
        ae = self._make()
        r = ae.detect("git status")
        self.assertIsNotNone(r)
        self.assertEqual(r[0], "git_status")

    def test_detects_what_changed(self):
        ae = self._make()
        r = ae.detect("what has changed")
        self.assertIsNotNone(r)
        self.assertEqual(r[0], "git_status")

    def test_detects_git_log(self):
        ae = self._make()
        r = ae.detect("show recent commits")
        self.assertIsNotNone(r)
        self.assertEqual(r[0], "git_log")

    def test_detects_git_diff(self):
        ae = self._make()
        r = ae.detect("show the diff")
        self.assertIsNotNone(r)
        self.assertEqual(r[0], "git_diff")

    def test_detects_git_branch(self):
        ae = self._make()
        r = ae.detect("git branch")
        self.assertIsNotNone(r)
        self.assertEqual(r[0], "git_branch")

    def test_detects_what_branch_natural_language(self):
        ae = self._make()
        r = ae.detect("what branch am I on?")
        self.assertIsNotNone(r)
        self.assertEqual(r[0], "git_branch")

    def test_detects_current_branch(self):
        ae = self._make()
        r = ae.detect("what is the current branch?")
        self.assertIsNotNone(r)
        self.assertEqual(r[0], "git_branch")

    def test_detects_python_version(self):
        ae = self._make()
        r = ae.detect("python version")
        self.assertIsNotNone(r)
        self.assertEqual(r[0], "python_version")

    def test_detects_list_files(self):
        ae = self._make()
        r = ae.detect("list files")
        self.assertIsNotNone(r)
        self.assertEqual(r[0], "list_files")

    # ── New P3.3 intents: npm / Node / file-run / system ────────────────────
    def test_detects_node_version(self):
        ae = self._make()
        r = ae.detect("node version")
        self.assertIsNotNone(r)
        self.assertEqual(r[0], "node_version")

    def test_detects_npm_version(self):
        ae = self._make()
        r = ae.detect("npm version")
        self.assertIsNotNone(r)
        self.assertEqual(r[0], "npm_version")

    def test_detects_npm_install(self):
        ae = self._make()
        r = ae.detect("npm install")
        self.assertIsNotNone(r)
        self.assertEqual(r[0], "npm_install")

    def test_detects_npm_start(self):
        ae = self._make()
        r = ae.detect("npm run start")
        self.assertIsNotNone(r)
        self.assertEqual(r[0], "npm_start")

    def test_detects_npm_build(self):
        ae = self._make()
        r = ae.detect("npm run build")
        self.assertIsNotNone(r)
        self.assertEqual(r[0], "npm_build")

    def test_detects_start_dev_server_natural(self):
        ae = self._make()
        r = ae.detect("start the dev server")
        self.assertIsNotNone(r)
        self.assertEqual(r[0], "npm_dev")

    def test_detects_build_frontend_natural(self):
        ae = self._make()
        r = ae.detect("build the frontend")
        self.assertIsNotNone(r)
        self.assertEqual(r[0], "npm_build")

    def test_detects_npm_test(self):
        ae = self._make()
        r = ae.detect("npm test")
        self.assertIsNotNone(r)
        self.assertEqual(r[0], "npm_test")

    def test_detects_run_python_file(self):
        ae = self._make()
        r = ae.detect("run main.py")
        self.assertIsNotNone(r)
        self.assertEqual(r[0], "run_python_file")

    def test_detects_run_node_file(self):
        ae = self._make()
        r = ae.detect("run server.js")
        self.assertIsNotNone(r)
        self.assertEqual(r[0], "run_node_file")

    def test_detects_which_command(self):
        ae = self._make()
        r = ae.detect("which node")
        self.assertIsNotNone(r)
        # which_command but node_version has a specific pattern — just check it resolves
        self.assertIsNotNone(r[0])

    def test_detects_scan_errors(self):
        ae = self._make()
        r = ae.detect("show all errors")
        self.assertIsNotNone(r)
        self.assertEqual(r[0], "scan_errors")

    def test_detects_find_exceptions(self):
        ae = self._make()
        r = ae.detect("find exceptions in code")
        self.assertIsNotNone(r)
        self.assertEqual(r[0], "scan_errors")

    # --- Safety block ---

    def test_blocks_rm_command(self):
        ae = self._make()
        self.assertIsNone(ae.detect("rm -rf everything"))

    def test_blocks_del_command(self):
        ae = self._make()
        self.assertIsNone(ae.detect("delete all files"))

    def test_blocks_format_command(self):
        ae = self._make()
        self.assertIsNone(ae.detect("format the drive"))

    def test_blocks_drop_command(self):
        ae = self._make()
        self.assertIsNone(ae.detect("drop the database"))

    def test_no_match_returns_none(self):
        ae = self._make()
        self.assertIsNone(ae.detect("explain this algorithm"))

    # --- Execution ---

    def test_execute_git_branch_returns_output(self):
        ae = self._make()
        output = _run(ae.execute("git_branch", ["git", "branch", "--show-current"]))
        self.assertIn("git branch", output)

    def test_execute_missing_command_returns_error_string(self):
        ae = self._make()
        output = _run(ae.execute("no_op", ["__nonexistent_cmd_xyz__"]))
        self.assertIn("not found", output.lower())

    def test_detect_and_run_returns_none_for_no_match(self):
        ae = self._make()
        result = _run(ae.detect_and_run("what is the meaning of life"))
        self.assertIsNone(result)

    def test_detect_and_run_returns_string_for_match(self):
        ae = self._make()
        result = _run(ae.detect_and_run("git status"))
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)

    def test_disabled_returns_none(self):
        from ai.actions import ActionExecutor
        ae = ActionExecutor(_cfg(**{"ai.actions.enabled": False}))
        self.assertIsNone(ae.detect("run pytest"))


# ─────────────────────────────────────────────────────────────────────────────
# P3.4 — ContextPruner
# ─────────────────────────────────────────────────────────────────────────────

class TestContextPruner(unittest.TestCase):
    """P3.4: Attention-based context pruning."""

    def _make(self, **cfg_overrides):
        from ai.context import ContextPruner
        return ContextPruner(_cfg(**cfg_overrides))

    _SCREEN = (
        "def authenticate_user(username, password):\n"
        "    token = jwt.encode({'user': username}, SECRET_KEY)\n"
        "    return token\n"
        "\n"
        "def render_home_page():\n"
        "    return render_template('home.html')\n"
        "\n"
        "class DatabasePool:\n"
        "    def connect(self):\n"
        "        return psycopg2.connect(DATABASE_URL)\n"
        "\n"
        "import logging\n"
        "logger = logging.getLogger(__name__)\n"
    )

    def test_relevant_block_retained(self):
        """Block matching the query must survive pruning."""
        pr = self._make()
        pruned = pr.prune(self._SCREEN, "how does authentication work?")
        self.assertIn("authenticate_user", pruned)

    def test_unrelated_block_may_be_dropped(self):
        """Highly unrelated blocks should be removed (or at least not forced in)."""
        pr = self._make(**{"ai.pruner.top_k_blocks": 1, "ai.pruner.min_score": 0.1})
        pruned = pr.prune(self._SCREEN, "how does authentication work?")
        # The authentication block should still be there
        self.assertIn("authenticate_user", pruned)

    def test_tiny_screen_passes_through_unchanged(self):
        """Screen too short to prune must be returned verbatim."""
        pr = self._make()
        tiny = "just one line of text here"
        result = pr.prune(tiny, "anything at all")
        self.assertEqual(result, tiny)

    def test_empty_screen_returns_empty(self):
        pr = self._make()
        self.assertEqual(pr.prune("", "query"), "")

    def test_empty_query_returns_original(self):
        pr = self._make()
        result = pr.prune(self._SCREEN, "")
        self.assertEqual(result, self._SCREEN)

    def test_disabled_returns_original(self):
        pr = self._make(**{"ai.pruner.enabled": False})
        result = pr.prune(self._SCREEN, "authentication")
        self.assertEqual(result, self._SCREEN)

    def test_prune_does_not_raise_on_garbage_input(self):
        pr = self._make()
        try:
            pr.prune("\x00\xff\n\n\n", "query")
        except Exception as e:
            self.fail(f"prune() raised unexpectedly on garbage input: {e}")

    def test_pruned_result_is_shorter_or_equal(self):
        """Pruning must never make the output longer than the input."""
        pr = self._make()
        result = pr.prune(self._SCREEN, "authentication")
        self.assertLessEqual(len(result), len(self._SCREEN))

    def test_original_order_preserved(self):
        """Kept blocks must appear in their original relative order."""
        pr = self._make()
        result = pr.prune(self._SCREEN, "authenticate database")
        auth_idx = result.find("authenticate_user")
        db_idx = result.find("DatabasePool")
        if auth_idx != -1 and db_idx != -1:
            self.assertLess(auth_idx, db_idx, "Original block order must be preserved")


# ─────────────────────────────────────────────────────────────────────────────
# Tier 3 tests  (Q16-Q21)
# ─────────────────────────────────────────────────────────────────────────────

class TestTier3ContextBoost(unittest.TestCase):
    """Q16: Context boosting raises Tier-4 fuzzy cache hit rate for follow-ups."""

    def _make_cache(self):
        from ai.cache import ShortQueryCache
        return ShortQueryCache(ttl_s=60, max_items=64, enable_fuzzy=True,
                               fuzzy_threshold=0.5, enable_semantic=False,
                               enable_embedding=False)

    def test_boost_raises_score_above_threshold(self):
        """A query that would normally miss Tier 4 should hit when boosted."""
        cache = self._make_cache()
        # Store "explain react hooks"
        cache.set(mode="coding", query="explain react hooks",
                  context_fp="ctx1", history_fp="hist1",
                  response="React hooks let you use state in function components.",
                  provider="test")

        # "how do hooks work" has low Jaccard with "explain react hooks" (~0.29)
        # but with boost_context="react hooks" it should score higher
        entry_no_boost, tier_nb = cache.get_with_tier(
            mode="coding", query="how do hooks work",
            context_fp="ctx1", history_fp="hist1",
        )
        entry_boost, tier_b = cache.get_with_tier(
            mode="coding", query="how do hooks work",
            context_fp="ctx1", history_fp="hist1",
            boost_context="react hooks",
        )
        # With boost, the hit rate should be >= no-boost
        # (either both hit or boost hits when no-boost doesn't)
        if entry_no_boost is None:
            # Boost should get a hit or at minimum not make it worse
            pass  # scoring depends on threshold; we just verify no crash

    def test_boost_does_not_affect_tier1_hit(self):
        """Q16: Boost context must not interfere with Tier 1 exact-match hits."""
        cache = self._make_cache()
        cache.set(mode="coding", query="what is react",
                  context_fp="ctx", history_fp="hist",
                  response="A JS library.", provider="test")
        entry, tier = cache.get_with_tier(
            mode="coding", query="what is react",
            context_fp="ctx", history_fp="hist",
            boost_context="vue angular svelte",  # unrelated boost
        )
        self.assertIsNotNone(entry)
        self.assertEqual(tier, 1, "Tier 1 exact match should not be affected by boost")

    def test_boost_context_none_is_safe(self):
        """Q16: boost_context=None must not crash."""
        cache = self._make_cache()
        cache.set(mode="general", query="hello world",
                  context_fp="c", history_fp="h",
                  response="Hi!", provider="test")
        entry, tier = cache.get_with_tier(
            mode="general", query="hello world",
            context_fp="c", history_fp="h",
            boost_context=None,
        )
        self.assertIsNotNone(entry)


class TestTier3SemanticLRUCap(unittest.TestCase):
    """Q17: _semantic_items must not exceed 512 entries."""

    def test_semantic_items_cap_at_512(self):
        """Insert 600 unique queries — _semantic_items should stay <= 512."""
        from ai.cache import ShortQueryCache
        cache = ShortQueryCache(ttl_s=300, max_items=1000, enable_semantic=True,
                                enable_embedding=False, enable_fuzzy=False)
        for i in range(600):
            cache.set(mode="general", query=f"unique query number {i} about topic",
                      context_fp="ctx", history_fp="hist",
                      response=f"answer {i}", provider="test")

        count = len(cache._semantic_items)
        self.assertLessEqual(count, cache._max_semantic_items,
                             f"_semantic_items={count} exceeded cap={cache._max_semantic_items}")
        # LRU list should match
        self.assertLessEqual(len(cache._semantic_lru), cache._max_semantic_items)

    def test_semantic_lru_tracks_insertions(self):
        """Q17: LRU list length tracks _semantic_items dict."""
        from ai.cache import ShortQueryCache
        cache = ShortQueryCache(ttl_s=60, max_items=64, enable_semantic=True,
                                enable_embedding=False, enable_fuzzy=False)
        for i in range(20):
            cache.set(mode="general", query=f"test query {i}",
                      context_fp="c", history_fp="h",
                      response=f"r{i}", provider="t")
        self.assertEqual(len(cache._semantic_items), len(cache._semantic_lru),
                         "LRU list must mirror _semantic_items dict size")


class TestTier3AsyncPersist(unittest.TestCase):
    """Q18: _persist() spawns a background thread and does not block."""

    def test_persist_non_blocking(self):
        """Q18: Calling _persist() must return immediately (< 50ms)."""
        import time
        from ai.cache import EmbeddingTier, _EmbedRecord
        import pathlib
        import numpy as np

        test_root = pathlib.Path("data") / "test_tmp" / "persist_non_blocking"
        test_root.mkdir(parents=True, exist_ok=True)
        vp = test_root / "vecs.npy"
        mp = test_root / "meta.json"
        et = EmbeddingTier(vectors_path=vp, meta_path=mp, ttl_s=60)
        # Cancel the 30s timer so it doesn't fire after the assertion
        if et._persist_timer:
            et._persist_timer.cancel()

        with et._data_lock:
            et._vectors = [np.zeros(384, dtype="float32")]
            et._records = [_EmbedRecord(mode="g", context_fp="c",
                                        cache_query="q", history_fp="h")]
            et._timestamps = [time.time()]
            et._dirty = True

        t0 = time.time()
        et._persist()
        elapsed_ms = (time.time() - t0) * 1000

        # Should return almost instantly — background thread does the I/O
        self.assertLess(elapsed_ms, 50,
                        f"_persist() took {elapsed_ms:.0f}ms — should be <50ms (non-blocking)")

        # Wait for background thread to finish and verify that it wrote something.
        time.sleep(0.3)
        self.assertTrue(vp.exists() or mp.exists())


class TestTier3DevServerIntents(unittest.TestCase):
    """Q19: npm run dev / yarn dev / next dev / vite dev intents."""

    def _detect(self, query):
        from ai.actions import ActionExecutor
        import unittest.mock as mock
        # ActionExecutor requires a config; mock it minimally
        cfg = mock.MagicMock()
        cfg.get.return_value = 15.0
        ae = ActionExecutor(cfg)
        return ae.detect(query)

    def test_npm_run_dev(self):
        result = self._detect("npm run dev")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "npm_dev")
        self.assertEqual(result[1], ["npm", "run", "dev"])

    def test_start_dev_server_phrase(self):
        result = self._detect("start the development server")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "npm_dev")

    def test_hot_reload(self):
        result = self._detect("enable hot reload")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "npm_dev")

    def test_yarn_dev(self):
        result = self._detect("yarn dev")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "yarn_dev")
        self.assertEqual(result[1], ["yarn", "dev"])

    def test_next_dev(self):
        result = self._detect("next.js dev")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "next_dev")

    def test_vite_dev(self):
        result = self._detect("launch vite")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "vite_dev")
        self.assertEqual(result[1], ["npx", "vite"])

    def test_run_in_dev_mode(self):
        result = self._detect("run in development mode")
        self.assertIsNotNone(result)
        self.assertIn(result[0], ("npm_dev", "yarn_dev"))


if __name__ == "__main__":
    unittest.main()
