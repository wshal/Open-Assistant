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
import time
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path

from utils.logger import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Intent Patterns
# ---------------------------------------------------------------------------

# Each entry: (intent_label, compiled_pattern, command_template_fn)
# command_template_fn(match, cwd) -> Optional[List[str]]
# Returns None if the action should NOT execute (safety block).

_BLOCKED_PATTERNS = [
    re.compile(r"\b(rm|del|rmdir|rd|format|drop|truncate|shred|wipe|destroy)\b", re.I),
]

_INTENT_RULES: List[Tuple[str, re.Pattern, Any]] = [
    # --- Testing ---
    (
        "run_pytest",
        re.compile(r"\b(run|execute)\b.{0,30}\bpytest\b|\bpytest\b.{0,20}\b(run|now)\b", re.I),
        lambda m, cwd: ["python", "-m", "pytest", "--tb=short", "-q"],
    ),
    (
        "run_tests",
        re.compile(r"\b(run|execute)\s+(?:all\s+)?tests?\b", re.I),
        lambda m, cwd: (
            ["python", "-m", "pytest", "--tb=short", "-q"]
            if (Path(cwd) / "pytest.ini").exists() or (Path(cwd) / "pyproject.toml").exists()
            else ["python", "-m", "unittest", "discover"]
        ),
    ),
    # --- Run specific file ---
    (
        "run_python_file",
        re.compile(r"\b(run|execute)\s+([\w./\\]+\.py)\b", re.I),
        lambda m, cwd: ["python", m.group(2)],
    ),
    (
        "run_node_file",
        re.compile(r"\b(run|execute|node)\s+([\w./\\]+\.[jt]s)\b", re.I),
        lambda m, cwd: ["node", m.group(2)],
    ),
    # --- npm / Node ecosystem ---
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
        "npm_install",
        re.compile(r"\bnpm\s+install\b|\binstall\s+(?:the\s+)?(?:npm\s+)?packages?\b|\bnpm\s+i\b", re.I),
        lambda m, cwd: ["npm", "install"],
    ),
    (
        "npm_start",
        # Note: 'start dev server' is handled by npm_dev above — npm_start covers npm run start only
        re.compile(r"\bnpm\s+(?:run\s+)?start\b|\bstart\s+(?:the\s+)?server\b(?!.*dev)", re.I),
        lambda m, cwd: ["npm", "run", "start"],
    ),
    (
        "npm_build",
        re.compile(r"\bnpm\s+(?:run\s+)?build\b|\bbuild\s+(?:the\s+)?(?:project|app|frontend)\b", re.I),
        lambda m, cwd: ["npm", "run", "build"],
    ),
    (
        "npm_test",
        re.compile(r"\bnpm\s+(?:run\s+)?test\b|\brun\s+(?:the\s+)?npm\s+tests?\b", re.I),
        lambda m, cwd: ["npm", "test"],
    ),
    (
        "npm_list_packages",
        re.compile(r"\bnpm\s+list\b|\bnpm\s+ls\b|\blist\s+npm\s+packages?\b", re.I),
        lambda m, cwd: ["npm", "list", "--depth=0"],
    ),
    # --- Dev server / Hot-reload (Q19) --- placed BEFORE npm_start to take priority
    (
        "npm_dev",
        re.compile(
            r"\bnpm\s+run\s+dev\b"
            r"|\bstart\s+(?:the\s+)?dev(?:elopment)?\s+(?:server|mode)\b"
            r"|\brun\s+(?:in\s+)?dev(?:elopment)?\s+mode\b"
            r"|\bhot[\s-]?reload\b",
            re.I,
        ),
        lambda m, cwd: ["npm", "run", "dev"],
    ),
    (
        "yarn_dev",
        re.compile(r"\byarn\s+dev\b|\byarn\s+run\s+dev\b", re.I),
        lambda m, cwd: ["yarn", "dev"],
    ),
    (
        "next_dev",
        re.compile(
            r"\bnext(?:\.js)?\s+dev\b|\bnpx\s+next\s+dev\b"
            r"|\bstart\s+next(?:\.js)?\b|\blaunch\s+next(?:\.js)?\b",
            re.I,
        ),
        lambda m, cwd: ["npx", "next", "dev"],
    ),
    (
        "vite_dev",
        re.compile(
            r"\bvite\b|\bnpx\s+vite\b|\brun\s+vite\b|\blaunch\s+vite\b"
            r"|\bstart\s+vite\b",
            re.I,
        ),
        lambda m, cwd: ["npx", "vite"],
    ),
    # --- Git ---
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
    # --- Python ---
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
    # --- Directory & Files ---
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
    # --- Open file in OS default app ---
    (
        "open_file",
        re.compile(r"\bopen\s+([\w./\\]+\.\w{1,6})\b", re.I),
        lambda m, cwd: (
            ["start", "", m.group(1)] if os.name == "nt" else ["open", m.group(1)]
        ),
    ),
    # --- Environment & system info ---
    (
        "show_env",
        re.compile(r"\bshow\s+(?:environment\s+)?variables?\b|\benv\s+vars?\b|\bprint\s+env\b", re.I),
        lambda m, cwd: ["set"] if os.name == "nt" else ["env"],
    ),
    (
        "which_command",
        re.compile(r"\bwhich\s+(\w+)\b", re.I),
        lambda m, cwd: (
            ["where", m.group(1)] if os.name == "nt" else ["which", m.group(1)]
        ),
    ),
    # --- Error scanning ---
    (
        "scan_errors",
        re.compile(
            r"\b(find|show|list|grep)\s+(?:all\s+)?(?:errors?|exceptions?|tracebacks?)\b"
            r"|\bwhere\s+(?:are\s+)?(?:the\s+)?errors?\b",
            re.I,
        ),
        lambda m, cwd: (
            ["findstr", "/S", "/I", "Error", "*.py", "*.js", "*.ts"]
            if os.name == "nt"
            else ["grep", "-r", "--include=*.py", "--include=*.js", "--include=*.ts", "-l", "Error", "."]
        ),
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
        self._enabled: bool = bool(config.get("ai.actions.enabled", True))
        self._timeout_s: float = float(config.get("ai.actions.timeout_s", 15.0))
        self._cwd: str = str(config.get("ai.actions.cwd", "."))

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
            proc = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=self._cwd,
            )
            try:
                stdout_bytes, _ = await asyncio.wait_for(
                    proc.communicate(), timeout=self._timeout_s
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                logger.warning(
                    f"[P3.3 Actions] '{intent_label}' timed out after {self._timeout_s}s"
                )
                return f"[Action timed out after {self._timeout_s:.0f}s]"

            elapsed = (time.perf_counter() - t0) * 1000
            output = (stdout_bytes or b"").decode("utf-8", errors="replace").strip()
            rc = proc.returncode

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

        except FileNotFoundError:
            logger.warning(
                f"[P3.3 Actions] Command not found: '{command[0]}'"
            )
            return f"[Command not found: {command[0]}]"
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
