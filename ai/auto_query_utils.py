import re


def looks_like_setup_statement(text: str) -> bool:
    """Return True when *text* is a conversational setup / preamble that should
    be accumulated as context rather than dispatched as a query.

    Covers all patterns seen in real-world interview benchmarks:
      • Transition phrases   "Let's pivot to CSS basics"
      • Imagination prompts  "Imagine we're designing an API"
      • Scene-setters        "Alright, let's talk about scaling"
      • Topic bridges        "So, moving on to the next topic"
      • Observation starters "I was looking at your resume and I see..."
      • When-building frames "When building a secure REST API..."
    """
    cleaned = " ".join((text or "").lower().split()).strip(" .,!?:;-")
    if not cleaned:
        return False
    cleaned = re.sub(
        r"^(actually|basically|honestly|so|okay|ok|right),?\s+",
        "",
        cleaned,
    ).strip(" .,!?:;-")
    # Any explicit question mark → definitely a query, not setup
    if "?" in text:
        return False

    # Strip punctuation from every word so "so," == "so", "alright," == "alright"
    words = cleaned.split()
    stripped_words = [w.strip(".,!?;:-_'\"") for w in words]
    first = stripped_words[0] if stripped_words else ""
    first2 = " ".join(stripped_words[:2]) if len(stripped_words) >= 2 else first

    # ── Transition / topic-shift starters ──────────────────────────────────
    TRANSITION_STARTERS = {
        "let's", "lets", "alright", "alright let's",
        "so", "so let's", "so now", "so moving",
        "okay", "ok", "right", "great", "perfect", "moving on",
        "now", "next", "next up", "first", "second", "third",
    }
    if first in TRANSITION_STARTERS or first2 in TRANSITION_STARTERS:
        # Still might be a question if it contains strong question cues after the starter
        question_words = {
            "what", "why", "where", "who", "which",
            "can", "could", "would", "should", "do", "does", "did",
            "is", "are", "tell", "explain", "compare", "define", "walk",
        }
        word_set = set(stripped_words)
        if not (word_set & question_words):
            return True
        # "Let's talk about X" is setup even if it starts with a question word like "how"
        # e.g. "Alright, let's talk about how React works" is still a topic intro
        TALK_ABOUT_VERBS = {"talk", "discuss", "cover", "go over", "look at", "explore", "pivot"}
        if any(v in cleaned for v in TALK_ABOUT_VERBS):
            return True

    # ── Imagination / hypothetical framers ─────────────────────────────────
    IMAGINATION_STARTERS = {
        "imagine", "suppose", "consider", "picture", "pretend",
        "let's say", "let's suppose", "let's imagine",
        "assume", "assuming",
    }
    if first in IMAGINATION_STARTERS or first2 in IMAGINATION_STARTERS:
        return True

    # ── Observation / resume-reading starters ──────────────────────────────
    OBSERVATION_STARTERS = (
        "i was looking",
        "i see you",
        "i notice",
        "i noticed",
        "looking at your",
        "looking at the",
        "i can see",
        "based on your",
        "from your",
        "according to",
    )
    if any(cleaned.startswith(s) for s in OBSERVATION_STARTERS):
        return True

    # ── Context / scene-setting starters ───────────────────────────────────
    CONTEXT_STARTERS = (
        "in this scenario",
        "for this exercise",
        "for context",
        "before we get into",
        "before we dive in",
        "before we start",
        "here is some context",
        "here's some context",
        "here is a short setup",
        "here's a short setup",
        "here is the setup",
        "here's the setup",
        "let me give you some background",
        "as some background",
        "some context",
        "a bit of context",
        "a little context",
        "to give you some context",
        "to set the scene",
        "the scenario is",
        "in our system",
        "in our codebase",
        "at our company",
        "for this problem",
    )
    if any(cleaned.startswith(s) for s in CONTEXT_STARTERS):
        return True

    # ── "When building / When designing / When working on" ─────────────────
    if re.match(r"^when\s+\w+ing\b", cleaned):
        question_cues = (
            " what ", " how ", " why ", " could ",
            " can ", " should ", " would ", " do ", " does ",
        )
        return not any(cue in f" {cleaned} " for cue in question_cues)

    # ── Declarative observations about confusion / state-of-the-art ────────
    # e.g. "A lot of developers get confused between CSS Grid and Flexbox"
    # e.g. "Most teams struggle with state management in large React apps"
    DECLARATIVE_STARTERS = (
        "a lot of ", "many developers", "most developers",
        "most teams", "many engineers", "a common", "it's common",
        "often times", "typically ", "generally ", "one thing that",
        "one common", "one issue", "one challenge",
    )
    if any(cleaned.startswith(s) for s in DECLARATIVE_STARTERS):
        return True

    return False


def looks_like_clipped_query_fragment(text: str) -> bool:
    """Return True when *text* looks like an incomplete audio fragment that should
    be buffered rather than dispatched.

    Conservative: only clips text that is clearly not a standalone question.
    A false-negative (dispatching a fragment) is far less harmful than a
    false-positive (suppressing a valid query like "Write a quicksort in Python").
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return True
    words = cleaned.split()
    if not words:
        return True

    # Explicit question mark + enough words → never a fragment
    if cleaned.endswith("?") and len(words) >= 3:
        return False

    # Anything with 5+ words is almost certainly not a bare fragment
    if len(words) >= 5:
        return False

    first_lower = words[0].strip(".,!?;:-_'\"").lower()

    # Setup/transition starters without a question component → fragment/setup
    setup_starters = {
        "let's", "lets", "so", "alright", "okay", "ok", "now",
        "moving", "great", "good", "perfect", "right", "sure",
        "imagine", "suppose", "consider",
    }
    if first_lower in setup_starters and not cleaned.endswith("?"):
        question_words = {
            "what", "why", "how", "when", "where", "who", "which",
            "can", "could", "would", "should", "do", "does", "did",
            "is", "are", "tell", "explain", "compare", "define",
        }
        all_lower = {w.strip(".,!?;:-_'\"").lower() for w in words}
        if not (all_lower & question_words):
            return True

    # Safe question/imperative starters → never a fragment
    SAFE_STARTERS = {
        # WH-questions
        "what", "why", "how", "when", "where", "who", "which",
        # Modal/auxiliary openers
        "can", "could", "would", "should", "do", "does", "did",
        "is", "are", "was", "were", "will", "shall", "has", "have",
        # Explicit imperative starters for coding / tech queries
        "tell", "explain", "compare", "define", "describe", "walk",
        "write", "build", "create", "implement", "show", "give",
        "list", "name", "discuss", "outline", "summarize", "design",
        "calculate", "analyze", "find", "identify", "evaluate",
    }
    if first_lower in SAFE_STARTERS:
        return False

    # Single-word or two-word texts that aren't in the safe list → fragment
    if len(words) <= 2:
        return True

    # 3-4 word text not starting with a safe starter — check if it has any
    # question signal at all before clipping
    question_words = {
        "what", "why", "how", "when", "where", "who", "which",
        "can", "could", "would", "should", "do", "does", "did",
        "is", "are", "tell", "explain", "compare", "define",
    }
    all_lower = {w.strip(".,!?;:-_'\"").lower() for w in words}
    if all_lower & question_words:
        return False

    return True



def normalized_query_words(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[A-Za-z0-9_'-]+", (text or "").lower())
        if token
    ]


def auto_query_label_score(
    text: str,
    *,
    looks_question_like_transcript,
    looks_like_acknowledgement,
) -> tuple[int, int, int, int]:
    from utils.text_utils import looks_like_actionable_auto_query

    cleaned = (text or "").strip()
    if not cleaned:
        return (-1000, 0, 0, 0)
    words = cleaned.split()
    question_like = int(looks_question_like_transcript(cleaned))
    actionable = int(looks_like_actionable_auto_query(cleaned))
    clipped_penalty = -2 if looks_like_clipped_query_fragment(cleaned) else 0
    non_question_penalty = -2 if not question_like and not actionable and len(words) < 8 else 0
    punctuation_bonus = 1 if cleaned.endswith(("?", ".")) else 0
    assembled_question_bonus = 0
    if cleaned.endswith("?") and len(words) >= 6 and any(mark in cleaned for mark in (".", ":", ";")):
        assembled_question_bonus = 2
    repetition_penalty = 0
    normalized_words = normalized_query_words(cleaned)
    if len(normalized_words) >= 10:
        unique_ratio = len(set(normalized_words)) / max(len(normalized_words), 1)
        if unique_ratio < 0.7:
            repetition_penalty -= 3
        windows = [
            tuple(normalized_words[idx : idx + 4])
            for idx in range(len(normalized_words) - 3)
        ]
        if len(windows) != len(set(windows)):
            repetition_penalty -= 4
    return (
        question_like * 100
        + actionable * 80
        + clipped_penalty * 20
        + non_question_penalty * 20
        + punctuation_bonus * 5
        + assembled_question_bonus * 15
        + repetition_penalty * 20
        + min(len(words), 16),
        len(cleaned),
        len(words),
        1 if not looks_like_acknowledgement(cleaned) else 0,
    )


def select_best_auto_query_label(
    *candidates: str,
    looks_question_like_transcript,
    looks_like_acknowledgement,
) -> str:
    best = ""
    best_score = (-1000, 0, 0, 0)
    for candidate in candidates:
        cleaned = (candidate or "").strip()
        if not cleaned:
            continue
        score = auto_query_label_score(
            cleaned,
            looks_question_like_transcript=looks_question_like_transcript,
            looks_like_acknowledgement=looks_like_acknowledgement,
        )
        if score > best_score:
            best = cleaned
            best_score = score
    return best
