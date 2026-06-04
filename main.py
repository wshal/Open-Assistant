#!/usr/bin/env python3
"""OpenAssist AI v1.0.0 — Ultimate Free AI Assistant"""

import sys
import os
import signal
import argparse
import importlib.util
import traceback
import warnings
from pathlib import Path
from dotenv import load_dotenv
from PyQt6.QtWidgets import QApplication

_APP_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_APP_ROOT))
load_dotenv(dotenv_path=_APP_ROOT / ".env")

# Patch SSL on Windows to fallback to certifi if access to system certificate store is denied
if sys.platform == "win32":
    try:
        import ssl
        import certifi
        orig_load = ssl.SSLContext.load_default_certs
        def patched_load(self, purpose=ssl.Purpose.SERVER_AUTH):
            try:
                orig_load(self, purpose)
            except PermissionError:
                self.load_verify_locations(cafile=certifi.where())
        ssl.SSLContext.load_default_certs = patched_load
    except Exception:
        pass

def _block_pyinstaller_namespace_stub(mod_name: str) -> None:
    """Stop empty namespace stubs from shadowing real ImportError paths."""
    spec = importlib.util.find_spec(mod_name)
    if spec is None:
        return
    if spec.origin is None and spec.submodule_search_locations is not None:
        sys.modules[mod_name] = None


if getattr(sys, "frozen", False):
    # PyInstaller can leave empty namespace stubs behind for excluded modules.
    # Mark them as unavailable so downstream imports fail cleanly instead of
    # half-importing and crashing deep inside dependency code.
    for mod_name in ("torch", "cv2", "sentence_transformers", "easyocr"):
        _block_pyinstaller_namespace_stub(mod_name)

from core.config import Config
from core.app import OpenAssistApp
from core.constants import CONFIG_FILE
from utils.logger import setup_logger


def parse_args():
    p = argparse.ArgumentParser(description="OpenAssist AI v1.0.0")
    p.add_argument("--config", default=CONFIG_FILE)
    p.add_argument(
        "--mode",
        choices=["general", "interview", "meeting", "coding", "writing", "exam"],
    )
    p.add_argument("--provider", help="Force provider: groq, gemini, cerebras, etc.")
    p.add_argument("--local-only", action="store_true")
    p.add_argument("--stealth", action="store_true")
    p.add_argument("--no-audio", action="store_true")
    p.add_argument("--no-screen", action="store_true")
    p.add_argument("--parallel", action="store_true")
    p.add_argument("--mini", action="store_true", help="Start in mini overlay mode")
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--add-docs", type=str, help="Add directory to knowledge base")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def global_exception_handler(exctype, value, tb):
    """Crash hook to capture and log ALL unhandled exceptions."""
    logger = setup_logger("openassist")
    try:
        err_str = "".join(traceback.format_exception(exctype, value, tb))
    except Exception:
        err_str = f"Exception: {exctype.__name__}: {value} (Traceback unavailable)"
    logger.critical("❌ FATAL CRASH DETECTED!")
    logger.critical(err_str)
    # Also print to stderr just in case (safe-guarded if sys.stderr is None)
    if sys.stderr is not None:
        try:
            print(err_str, file=sys.stderr)
        except Exception:
            pass
    sys.exit(1)


def cleanup_old_instances():
    """Ensure no zombie instances of OpenAssist are running from previous sessions."""
    try:
        import psutil
        import os
        current_pid = os.getpid()
        # Find absolute path of THIS main.py being launched right now
        main_path = os.path.abspath(sys.argv[0])
        main_dir = os.path.dirname(main_path).lower()

        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                # Do not terminate ourselves or our parent wrapper process (such as the venv redirector)
                if proc.info['pid'] == current_pid or proc.info['pid'] == os.getppid():
                    continue
                cmdline = proc.info['cmdline']
                if not cmdline or len(cmdline) < 2:
                    continue

                proc_name = (proc.info.get('name') or '').lower()
                if 'python' not in proc_name:
                    continue

                is_same_app = False

                # Only consider the SCRIPT argument (argv[0] equivalent, which is cmdline[1])
                # Skip processes invoked with -c (inline code), -m (module), etc.
                # A real "python main.py" invocation has the script as the first non-flag argument.
                script_arg = None
                skip_next = False
                for i, arg in enumerate(cmdline[1:], start=1):
                    if skip_next:
                        skip_next = False
                        continue
                    # -c and -m mean what follows is code/module, not a file.
                    if arg in ('-c', '-m'):
                        break  # not a script invocation we care about
                    # Only flags that actually consume a following value should
                    # skip that next token. Pure switches like -q/-s/-v do not.
                    if arg in ('-W', '-X'):
                        skip_next = True
                        continue
                    if arg.startswith('-'):
                        continue
                    # First positional non-flag argument is the script
                    script_arg = arg
                    break

                if script_arg and 'main.py' in script_arg:
                    arg_abs = os.path.abspath(script_arg)
                    if os.path.dirname(arg_abs).lower() == main_dir:
                        is_same_app = True

                # Fallback: match compiled/frozen OpenAssist executables by name
                if not is_same_app and 'openassist' in proc_name:
                    is_same_app = True

                if is_same_app:
                    try:
                        # M-9: Try graceful terminate first so the prior
                        # instance can flush history, release audio, and write
                        # its on-disk state. On Windows ``terminate()`` maps
                        # to TerminateProcess (process-scoped, no CTRL_BREAK
                        # group propagation), so it is safe here. Fall through
                        # to kill() only if the prior instance refuses to exit
                        # within 1.5s.
                        try:
                            proc.terminate()
                            proc.wait(timeout=1.5)
                        except psutil.TimeoutExpired:
                            proc.kill()
                            proc.wait(timeout=1.0)
                    except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                        pass
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        # Brief pause after killing old instances so the OS can release audio
        # devices, file handles, and sockets before we try to acquire them.
        import time as _time
        _time.sleep(0.5)
    except Exception:
        pass


def main():
    if sys.platform == "win32":
        cleanup_old_instances()
        import asyncio
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except Exception:
            pass
    args = parse_args()
    logger = setup_logger("openassist", "DEBUG" if args.debug else "INFO")

    # Set global exception handler
    sys.excepthook = global_exception_handler

    logger.info("🚀 OpenAssist AI starting...")

    # Suppress upstream noise
    warnings.filterwarnings(
        "ignore",
        message=r".*pin_memory.*no accelerator is found.*",
        category=UserWarning,
        module=r"torch\.utils\.data\.dataloader",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*SOCKS support in urllib3 requires.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*unauthenticated requests to the HF Hub.*",
    )

    try:
        # Initialize Qt Application FIRST
        qt_app = QApplication(sys.argv)

        config = Config(args.config)

        # CLI overrides
        if args.local_only:
            config.set("ai.strategy", "fixed")
            config.set("ai.fixed_provider", "ollama")
        if args.mode:
            config.set("ai.mode", args.mode)
        if args.stealth:
            config.set("stealth.enabled", True)
        if args.no_audio:
            config.set("capture.audio.enabled", False)
        if args.no_screen:
            config.set("capture.screen.enabled", False)
        if args.parallel:
            config.set("ai.parallel.enabled", True)
        if args.provider:
            config.set("ai.strategy", "fixed")
            config.set("ai.fixed_provider", args.provider)

        if args.benchmark:
            import asyncio
            from tests.test_benchmark import run_benchmark

            asyncio.run(run_benchmark(config))
            return

        if args.add_docs:
            from ai.rag import RAGEngine

            rag = RAGEngine(config)
            rag.add_directory(args.add_docs)
            logger.info(f"✅ Added documents from {args.add_docs}")
            return

        # M39 FIX: sys.exit() in a signal handler deadlocks Qt. Use
        # QApplication.quit() which schedules a clean event-loop exit.
        _term_handler = lambda *_: qt_app.quit()
        signal.signal(signal.SIGINT, _term_handler)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, _term_handler)
        if sys.platform != "win32" and hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, _term_handler)

        app = OpenAssistApp(config, mini_mode=args.mini)
        try:
            exit_code = app.run()
        finally:
            app.shutdown()
        sys.exit(exit_code)

    except Exception as e:
        logger.critical(f"Main Entry Point Failure: {e}")
        logger.critical(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
