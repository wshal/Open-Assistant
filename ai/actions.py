"""
Actionable Queries Engine — P3.3
Detects action-intent queries (run tests, git status, open file, etc.)
and executes them directly, streaming output back to the overlay.

Intent → Command mapping is conservative and explicit for safety.
Only a curated whitelist of read-only / low-risk commands is auto-executed.
Anything destructive (rm, del, format, drop) is blocked.
"""

import re
import asyncio
import subprocess
import os
import shutil
import time
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path

from utils.logger import setup_logger

logger = setup_logger(__name__)

_WINDOWS_SHELL_BUILTINS = {"dir", "set", "start"}


# ---------------------------------------------------------------------------
# Intent Patterns
# ---------------------------------------------------------------------------

# Each entry: (intent_label, compiled_pattern, command_template_fn)
# command_template_fn(match, cwd) -> Optional[List[str]]
# Returns None if the action should NOT execute (safety block).

# Issue #1: Restrict auto-action surface to read-only, low-risk commands only.
# Removed: npm install/start/dev/build/test, yarn dev, next dev, vite, run_python_file,
#          run_node_file, open_file, show_env, scan_errors. These could install packages,
#          execute project scripts, leak environment variables (API keys), or open
#          untrusted files in OS apps — all triggerable from spoken/OCR-derived prompts.
# Kept: git read-only commands, version checks, pip list, list files, show tree, which.
_BLOCKED_PATTERNS = [
    re.compile(
        r"\b(rm|del|rmdir|rd|format|drop|truncate|shred|wipe|destroy|"
        r"env|environment|secret|api[_ -]?key|token|password|credential)\b",
        re.I,
    ),
]

_INTENT_RULES: List[Tuple[str, re.Pattern, Any]] = [
    # --- Git (read-only) ---
    (
        "git_status",
        re.compile(r"\bgit\s+status\b|\bwhat.{0,15}changed\b|\bwhat.{0,15}modified\b", re.I),
        lambda m, cwd: ["git", "status", "--short"],
    ),
    (
        "git_log",
        re.compile(r"\bgit\s+log\b|\brecent\s+commits?\b|\blast\s+\d+\s+commits?\b", re.I),
        lambda m, cwd: ["git", "log", "--oneline", "-10"],
    ),
    (
        "git_diff",
        re.compile(r"\bgit\s+diff\b|\bwhat.{0,15}diff\b|\bshow\s+(?:the\s+)?diff\b", re.I),
        lambda m, cwd: ["git", "diff", "--stat"],
    ),
    (
        "git_branch",
        re.compile(
            r"\bgit\s+branch\b"
            r"|\bcurrent\s+branch\b"
            r"|\bwhich\s+branch\b"
            r"|\bwhat\s+branch\b"
            r"|\bbranch\b.{0,20}\b(on|active|now|using|am i)\b"
            r"|\bam\s+i\s+on\b.{0,30}\bbranch\b",
            re.I,
        ),
        lambda m, cwd: ["git", "branch", "--show-current"],
    ),
    # --- Version / which (read-only metadata) ---
    (
        "node_version",
        re.compile(r"\bnode\s*(?:\.js)?\s*version\b|\bwhich\s+node\b|\bnode\s+-v\b", re.I),
        lambda m, cwd: ["node", "--version"],
    ),
    (
        "npm_version",
        re.compile(r"\bnpm\s+version\b|\bwhich\s+npm\b|\bnpm\s+-v\b", re.I),
        lambda m, cwd: ["npm", "--version"],
    ),
    (
        "npm_list_packages",
        re.compile(r"\bnpm\s+list\b|\bnpm\s+ls\b|\blist\s+npm\s+packages?\b", re.I),
        lambda m, cwd: ["npm", "list", "--depth=0"],
    ),
    (
        "python_version",
        re.compile(r"\bpython\s+version\b|\bwhich\s+python\b", re.I),
        lambda m, cwd: ["python", "--version"],
    ),
    (
        "pip_list",
        re.compile(r"\b(list|show)\s+(?:installed\s+)?packages?\b|\bpip\s+list\b", re.I),
        lambda m, cwd: ["pip", "list", "--format=columns"],
    ),
    (
        "which_command",
        re.compile(r"\bwhich\s+(\w+)\b", re.I),
        lambda m, cwd: (
            ["where", m.group(1)] if os.name == "nt" else ["which", m.group(1)]
        ),
    ),
    # --- Directory listing (read-only) ---
    (
        "list_files",
        re.compile(r"\b(list|show)\s+(?:all\s+)?files?\b|\bwhat\s+files\b", re.I),
        lambda m, cwd: ["dir", "/B"] if os.name == "nt" else ["ls", "-la"],
    ),
    (
        "show_tree",
        re.compile(r"\bfolder\s+structure\b|\bdirectory\s+tree\b|\bshow\s+tree\b", re.I),
        lambda m, cwd: ["tree", "/F", "/A"] if os.name == "nt" else ["find", ".", "-maxdepth", "3"],
    ),
]


# ---------------------------------------------------------------------------
# ActionExecutor
# ---------------------------------------------------------------------------

class ActionExecutor:
    """
    P3.3: Actionable Queries.

    Detects action-intent in a user query, runs the corresponding
    command, and returns output as a formatted string for the AI to
    incorporate into its response — or surfaces directly in the overlay.
    """

    def __init__(self, config):
        self._config = config
        # Issue #1: Default-disabled. Auto-actions execute local commands derived
        # from spoken/OCR prompts; require explicit opt-in.
        self._enabled: bool = bool(config.get("ai.actions.enabled", False))
        # Issue #2: Clamp timeout to a sensible bounded range.
        self._timeout_s: float = max(1.0, min(float(config.get("ai.actions.timeout_s", 10.0)), 30.0))
        self._cwd: str = str(Path(config.get("ai.actions.cwd", ".")).resolve())

        if self._enabled:
            logger.info(
                f"[P3.3 Actions] Initialised — timeout={self._timeout_s}s, cwd={self._cwd}"
            )
        else:
            logger.info("[P3.3 Actions] Disabled via config")

    def detect(self, query: str) -> Optional[Tuple[str, List[str]]]:
        """
        Check if a query matches any action intent.

        Returns (intent_label, command_list) or None if no match.
        """
        if not self._enabled:
            return None

        q = (query or "").strip()

        # Safety: never execute if query contains destructive patterns
        for blocked in _BLOCKED_PATTERNS:
            if blocked.search(q):
                logger.warning(
                    f"[P3.3 Actions] BLOCKED — destructive pattern in query: '{q[:60]}'"
                )
                return None

        for intent_label, pattern, cmd_fn in _INTENT_RULES:
            m = pattern.search(q)
            if m:
                try:
                    cmd = cmd_fn(m, self._cwd)
                    if cmd:
                        logger.info(
                            f"[P3.3 Actions] Intent matched: '{intent_label}', "
                            f"cmd={cmd}"
                        )
                        return intent_label, cmd
                except Exception as e:
                    logger.warning(f"[P3.3 Actions] Command builder error: {e}")
        return None

    async def execute(self, intent_label: str, command: List[str]) -> str:
        """
        Execute the command and return its stdout/stderr as a string.
        Always runs with a timeout to prevent hangs.
        """
        logger.info(
            f"[P3.3 Actions] Executing '{intent_label}': {' '.join(command)}"
        )
        t0 = time.perf_counter()
        try:
            if not command:
                return "[Action execution failed: empty command]"

            if os.name == "nt" and command[0].lower() in _WINDOWS_SHELL_BUILTINS:
                run_cmd = ["cmd", "/c", *command]
            else:
                resolved = shutil.which(command[0])
                if not resolved:
                    logger.warning(
                        f"[P3.3 Actions] Command not found: '{command[0]}'"
                    )
                    return f"[Command not found: {command[0]}]"
                run_cmd = [resolved, *command[1:]]

            # Issue #2: subprocess.run() inside asyncio.wait_for() only cancels the
            # Python awaitable on timeout — the child process keeps running. Use
            # Popen + communicate(timeout) so we can explicitly kill the child.
            timeout_s = self._timeout_s

            def _run_command():
                proc = subprocess.Popen(
                    run_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    cwd=self._cwd,
                    creationflags=(
                        subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0  # type: ignore[attr-defined]
                    ),
                )
                try:
                    stdout, _ = proc.communicate(timeout=timeout_s)
                    return subprocess.CompletedProcess(run_cmd, proc.returncode, stdout)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    try:
                        stdout, _ = proc.communicate(timeout=2)
                    except Exception:
                        stdout = b""
                    raise TimeoutError((stdout or b"").decode("utf-8", errors="replace"))

            try:
                completed = await asyncio.to_thread(_run_command)
            except TimeoutError as exc:
                logger.warning(
                    f"[P3.3 Actions] '{intent_label}' timed out after {self._timeout_s}s — process killed"
                )
                partial = str(exc).strip()
                suffix = f"\n{partial[:1000]}" if partial else ""
                return f"[Action timed out after {self._timeout_s:.0f}s and was terminated]{suffix}"

            elapsed = (time.perf_counter() - t0) * 1000
            output = (completed.stdout or b"").decode("utf-8", errors="replace").strip()
            rc = completed.returncode

            logger.info(
                f"[P3.3 Actions] '{intent_label}' finished in {elapsed:.0f}ms, "
                f"rc={rc}, output_len={len(output)}"
            )

            if not output:
                return f"[Command completed with exit code {rc}, no output]"

            # Cap output sent to AI to avoid blowing the context window
            if len(output) > 3000:
                output = output[:3000] + "\n... [truncated]"

            header = f"$ {' '.join(command)}\n"
            return header + output

        except Exception as e:
            logger.error(f"[P3.3 Actions] Execution error: {e}")
            return f"[Action execution failed: {e}]"

    async def detect_and_run(self, query: str) -> Optional[str]:
        """
        Convenience: detect intent + run if matched.
        Returns the command output string, or None if no action matched.
        """
        result = self.detect(query)
        if result is None:
            return None
        intent_label, command = result
        return await self.execute(intent_label, command)
