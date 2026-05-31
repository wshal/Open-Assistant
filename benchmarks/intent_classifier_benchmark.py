#!/usr/bin/env python3
"""Benchmark Auto Mode intent classification on transcript-like text.

This is intentionally much cheaper than the full audio benchmark: it skips VAD,
STT, UI, and providers, then tests the decision layer that decides whether a
transcript is setup context, an actionable question, a greeting, or a follow-up.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as stats
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.auto_query_utils import looks_like_setup_statement
from ai.intent_classifier import IntentClassifier
from utils.text_utils import looks_like_actionable_auto_query, sanitize_query_label


Case = dict[str, str]


CASES: list[Case] = [
    {
        "id": "css_setup_1",
        "expected": "setup",
        "text": "Let's pivot to some CSS basics.",
    },
    {
        "id": "css_setup_2",
        "expected": "setup",
        "text": "A lot of developers get confused between CSS Grid and Flexbox.",
    },
    {
        "id": "css_question",
        "expected": "question",
        "text": (
            "Can you explain the primary differences between the two, and give an "
            "example of a layout where you would definitely choose grid over Flexbox?"
        ),
    },
    {
        "id": "react_setup",
        "expected": "setup",
        "text": "I was looking at your resume, and I see you've used React quite a bit.",
    },
    {
        "id": "react_question",
        "expected": "question",
        "text": (
            "Could you walk me through how you would decide between using a custom hook "
            "versus a higher-order component for sharing stateful logic?"
        ),
    },
    {
        "id": "api_setup",
        "expected": "setup",
        "text": "Imagine we are designing a new public-facing API for our mobile app.",
    },
    {
        "id": "api_question_versionable",
        "expected": "question",
        "text": (
            "What are some of the key principles you would follow to ensure the API is "
            "robust, versionable, and provides a good developer experience for the front-end team?"
        ),
    },
    {
        "id": "jwt_setup",
        "expected": "setup",
        "text": "When building a secure REST API, authentication is critical.",
    },
    {
        "id": "jwt_question",
        "expected": "question",
        "text": (
            "Could you explain how JWT tokens work, and what the potential security risks "
            "are if you store a JWT in the browser's local storage instead of an HTTP-only cookie?"
        ),
    },
    {
        "id": "db_scaling_setup",
        "expected": "setup",
        "text": (
            "Imagine we have a monolithic application backed by a single relational "
            "database that's starting to slow down under heavy read traffic."
        ),
    },
    {
        "id": "db_scaling_question",
        "expected": "question",
        "text": "What are some strategies you would consider to alleviate that bottleneck?",
    },
    {
        "id": "agile_open_ended",
        "expected": "question",
        "text": (
            "Tell me about a time when you disagreed with a senior engineer or product "
            "manager about a technical decision."
        ),
    },
    {
        "id": "agile_followup",
        "expected": "followup",
        "text": "How did you approach the situation and what was the outcome?",
    },
    {
        "id": "code_review_setup",
        "expected": "setup",
        "text": "Code reviews are a big part of our engineering culture here.",
    },
    {
        "id": "code_review_question",
        "expected": "question",
        "text": (
            "What do you consider to be the most important aspects to look for when "
            "reviewing a colleague's pull request?"
        ),
    },
    {
        "id": "greeting_help",
        "expected": "question",
        "text": "Hello there, can you help me with some coding?",
    },
    {
        "id": "ack",
        "expected": "greeting",
        "text": "Okay, got it. That makes sense.",
    },
    {
        "id": "noisy_filler",
        "expected": "greeting",
        "text": "you know i mean sort of kind of you know i mean",
    },
    {
        "id": "ui_noise",
        "expected": "other",
        "text": "Click-through enabled. Press Ctrl+M to restore interaction.",
    },
    {
        "id": "clickthrough_real_question",
        "expected": "question",
        "text": "How would you implement click-through analytics for a notification UI?",
    },
    {
        "id": "autoscaling_question",
        "expected": "question",
        "text": "Should auto scaling be enabled for the service?",
    },
    {
        "id": "microphone_question",
        "expected": "question",
        "text": "How does a microphone array use beam forming?",
    },
    {
        "id": "architecture_context",
        "expected": "setup",
        "text": "Before we dive in, let me give you some background on the architecture.",
    },
    {
        "id": "future_question_event_sourcing",
        "expected": "question",
        "text": "What are the tradeoffs between event sourcing and CQRS?",
    },
]


QUESTION_TEMPLATES = [
    "What are the tradeoffs between {a} and {b}?",
    "How would you design {a} for {b}?",
    "Can you explain how {a} works in {b}?",
    "When would you choose {a} instead of {b}?",
    "Should we enable {a} for {b}?",
    "Tell me how you would debug {a} in {b}.",
    "Could you walk me through {a} and the risks around {b}?",
    "Why does {a} matter when building {b}?",
    "What are common failure modes for {a} in {b}?",
    "How do I implement {a} without breaking {b}?",
]

SETUP_TEMPLATES = [
    "Let's talk about {a} for a moment.",
    "Before we get into the question, here is some context about {a}.",
    "Imagine we are building {a} for {b}.",
    "Suppose the team is migrating from {a} to {b}.",
    "I was looking at your background and noticed experience with {a}.",
    "The next topic is {a} and how it affects {b}.",
    "In our current system, {a} is becoming a bottleneck for {b}.",
    "Consider a scenario where {a} has to support {b}.",
    "A lot of teams struggle with {a} when they scale {b}.",
    "Here is a short setup: {a} is already deployed in {b}.",
]

FOLLOWUP_TEMPLATES = [
    "Can you give a specific example?",
    "What happened after that?",
    "How did the team react?",
    "What would you do differently next time?",
    "Could you elaborate on the tradeoffs?",
    "And what was the result?",
    "Why did you choose that approach?",
    "How did you measure whether it worked?",
]

GREETING_TEMPLATES = [
    "Okay, got it.",
    "Sure, that makes sense.",
    "Thanks, let's continue.",
    "Great, I understand.",
    "Sounds good to me.",
    "Yes, go ahead.",
]

OTHER_TEMPLATES = [
    "The deployment finished successfully.",
    "I opened the dashboard on the second monitor.",
    "The pull request has three changed files.",
    "Click-through enabled. Press Ctrl+M to restore interaction.",
    "The browser tab is already on the settings page.",
    "The terminal is showing the latest logs.",
]

TOPICS = [
    "Redis caching",
    "database indexing",
    "React suspense",
    "JWT refresh tokens",
    "Kubernetes autoscaling",
    "message queues",
    "event sourcing",
    "CQRS",
    "OAuth consent",
    "rate limiting",
    "API pagination",
    "WebSocket fanout",
    "feature flags",
    "microphone beam forming",
    "click-through analytics",
    "schema migrations",
    "observability",
    "load shedding",
]

CONTEXTS = [
    "a mobile app",
    "a high-traffic checkout flow",
    "a multi-tenant SaaS product",
    "an internal developer platform",
    "a real-time collaboration editor",
    "a video meeting client",
    "a legacy monolith",
    "a distributed system",
    "a public API",
    "a data ingestion pipeline",
]


def build_random_cases(count: int, *, seed: int = 7) -> list[Case]:
    rng = random.Random(seed)
    groups: list[tuple[str, list[str]]] = [
        ("question", QUESTION_TEMPLATES),
        ("setup", SETUP_TEMPLATES),
        ("followup", FOLLOWUP_TEMPLATES),
        ("greeting", GREETING_TEMPLATES),
        ("other", OTHER_TEMPLATES),
    ]
    cases: list[Case] = []
    for idx in range(count):
        expected, templates = groups[idx % len(groups)]
        template = rng.choice(templates)
        a = rng.choice(TOPICS)
        b = rng.choice([topic for topic in TOPICS if topic != a] + CONTEXTS)
        text = template.format(a=a, b=b)
        if rng.random() < 0.20 and expected in {"question", "setup"}:
            text = rng.choice(["So, ", "Actually, ", "For context, "]) + text
        cases.append({"id": f"random_{idx + 1:02d}_{expected}", "expected": expected, "text": text})
    return cases


def _expected_from_auto_report_result(item: dict) -> str:
    responses = item.get("responses") or []
    events = item.get("events") or []
    if responses or any(e.get("type") == "response_dispatched" for e in events if isinstance(e, dict)):
        return "question"
    return "setup"


def load_cases_from_auto_report(path: Path) -> list[Case]:
    """Build extra cases from an Auto Mode benchmark JSON report."""
    data = json.loads(path.read_text(encoding="utf-8"))
    cases: list[Case] = []
    for idx, fixture in enumerate(data.get("fixtures", []) or []):
        expected = _expected_from_auto_report_result(fixture)
        for t_idx, item in enumerate(fixture.get("transcripts", []) or []):
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            state = str(item.get("state") or "")
            if not text or state in {"processing", "error", "idle", "interim"}:
                continue
            if text.startswith("Processing") or text.startswith("Click-through"):
                continue
            cases.append(
                {
                    "id": f"{fixture.get('file', 'fixture')}_{idx}_{t_idx}",
                    "expected": expected,
                    "text": text,
                }
            )
    return cases


def regex_intent(text: str) -> str:
    cleaned = sanitize_query_label(text) or text
    if looks_like_actionable_auto_query(cleaned) or "?" in cleaned:
        return "question"
    if looks_like_setup_statement(text):
        return "setup"
    return "other"


def combined_intent(classifier: IntentClassifier, text: str) -> tuple[str, dict]:
    regex = regex_intent(text)
    scores = classifier.classify(text)
    if scores is None:
        return regex, {"regex": regex, "scores": None}

    if scores.best_intent == "followup" and scores.is_confident:
        verdict = "followup"
    elif scores.best_intent == "greeting" and scores.is_confident and regex != "question":
        verdict = "greeting"
    else:
        is_question = classifier.is_likely_question(text, regex_says_question=(regex == "question"))
        is_setup = classifier.is_likely_setup(text, regex_says_setup=(regex == "setup"))
        if is_question:
            verdict = "question"
        elif is_setup:
            verdict = "setup"
        else:
            verdict = "other"

    return verdict, {
        "regex": regex,
        "scores": {
            "question": scores.question,
            "setup": scores.setup,
            "greeting": scores.greeting,
            "followup": scores.followup,
            "best_intent": scores.best_intent,
            "best_score": scores.best_score,
            "is_confident": scores.is_confident,
        },
    }


def is_match(expected: str, actual: str) -> bool:
    if expected == actual:
        return True
    # Follow-ups are often actionable questions in isolation. Treat either as
    # acceptable for the standalone classifier audit.
    if expected == "followup" and actual in {"followup", "question"}:
        return True
    return False


def run(cases: list[Case]) -> dict:
    classifier = IntentClassifier()

    t0 = time.perf_counter()
    classifier.classify("What is the warmup question?")
    warmup_ms = (time.perf_counter() - t0) * 1000.0

    rows = []
    classify_times = []
    for case in cases:
        start = time.perf_counter()
        actual, details = combined_intent(classifier, case["text"])
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        classify_times.append(elapsed_ms)
        rows.append(
            {
                "id": case["id"],
                "expected": case["expected"],
                "actual": actual,
                "ok": is_match(case["expected"], actual),
                "elapsed_ms": elapsed_ms,
                "text": case["text"],
                **details,
            }
        )

    failures = [row for row in rows if not row["ok"]]
    return {
        "summary": {
            "cases": len(rows),
            "passed": len(rows) - len(failures),
            "failed": len(failures),
            "accuracy": (len(rows) - len(failures)) / max(len(rows), 1),
            "warmup_ms": warmup_ms,
            "median_classify_ms": stats.median(classify_times) if classify_times else 0,
            "max_classify_ms": max(classify_times) if classify_times else 0,
        },
        "failures": failures,
        "cases": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="benchmarks/intent_classifier_report.json")
    parser.add_argument("--random", type=int, default=0, help="Add N seeded synthetic stress cases.")
    parser.add_argument("--seed", type=int, default=7, help="Seed for --random cases.")
    parser.add_argument(
        "--auto-report",
        help="Optional Auto Mode benchmark JSON to add real transcript rows from.",
    )
    args = parser.parse_args()

    cases = list(CASES)
    if args.random:
        cases.extend(build_random_cases(args.random, seed=args.seed))
    if args.auto_report:
        cases.extend(load_cases_from_auto_report(Path(args.auto_report)))

    report = run(cases)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary = report["summary"]
    print(
        "Intent benchmark: "
        f"{summary['passed']}/{summary['cases']} passed "
        f"({summary['accuracy']:.1%}) | warmup={summary['warmup_ms']:.1f}ms | "
        f"median={summary['median_classify_ms']:.1f}ms | max={summary['max_classify_ms']:.1f}ms"
    )
    if report["failures"]:
        print("\nFailures:")
        for row in report["failures"]:
            print(f"- {row['id']}: expected={row['expected']} actual={row['actual']} text={row['text'][:100]}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
