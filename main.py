#!/usr/bin/env python3
"""OpenAssist AI v4.1 — Ultimate Free AI Assistant"""

import sys
import os
import signal
import argparse
import traceback
from pathlib import Path
from dotenv import load_dotenv
from PyQt6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

from core.config import Config
from core.app import OpenAssistApp
from utils.logger import setup_logger


def parse_args():
    p = argparse.ArgumentParser(description="OpenAssist AI v4.0")
    p.add_argument("--config", default="config.yaml")
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
    err_str = "".join(traceback.format_exception(exctype, value, tb))
    logger.critical("❌ FATAL CRASH DETECTED!")
    logger.critical(err_str)
    # Also print to stderr just in case
    print(err_str, file=sys.stderr)
    sys.exit(1)


def main():
    args = parse_args()
    logger = setup_logger("openassist", "DEBUG" if args.debug else "INFO")

    # Set global exception handler
    sys.excepthook = global_exception_handler

    logger.info("🚀 OpenAssist AI starting...")

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

        signal.signal(signal.SIGINT, lambda *_,: sys.exit(0))

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
