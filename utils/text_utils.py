"""Text processing utilities for ultra-low latency transcription repair and gating."""

import re

QUESTION_WORDS = {
    "what", "who", "where", "when", "why", "how", "which",
}
HELPER_WORDS = {
    "can", "could", "would", "should", "will", "do", "does", "did",
    "is", "are", "am", "was", "were", "have", "has", "had",
}
PRONOUN_WORDS = {
    "you", "we", "i", "me", "us", "they", "them", "he", "she", "it",
    "this", "that", "these", "those",
}
ACTION_WORDS = {
    "tell", "explain", "define", "describe", "show", "give", "use", "see",
    "mean", "means", "work", "works", "help", "helps", "fix", "build",
    "create", "answer", "expand", "compare", "solve", "understand",
}

# ── Compiled regex constants for is_question_complete ─────────────────────────
# Compiled once at import; reusing compiled objects avoids repeated re.compile
# overhead on every transcription event.
_RE_ACTION_STARTERS = re.compile(
    r'^(explain|define|describe|tell|show|list|write|create|give|compare|contrast|'
    r'what is|what are|what was|what were|what does|what do|'
    r'how does|how do|how is|how are|how to|'
    r'why does|why do|why is|why are|'
    r'who is|who are|where is|where are|when is|when does)\b',
    re.IGNORECASE,
)
_RE_QUESTION_STARTERS = re.compile(
    r'^(what|who|where|when|why|how|which|can|could|would|should|tell|explain|'
    r'describe|give|show|list|write|create|define|difference|advantages?|'
    r'disadvantages?|benefits?|pros|cons)\b',
    re.IGNORECASE,
)
_RE_WEAK_QUESTION_WORDS = re.compile(
    r'^(is|are|do|does|did|have|has|will|shall|meaning)\b', re.IGNORECASE
)
_RE_ENDS_INCOMPLETE = re.compile(
    r' (me|you|him|her|them|it|this|that|what|how|why|and|or|is|are|the|a|an|to|with|for|in|on|at|by|from)$',
    re.IGNORECASE,
)
_RE_INTENT_PHRASES = re.compile(
    r'^(tell|explain|describe|define|difference|advantage|disadvantage)', re.IGNORECASE
)
CONNECTOR_WORDS = {
    "and", "or", "but", "if", "then", "also", "because", "about", "with",
    "without", "from", "into", "onto", "over", "under", "between", "through",
    "around", "after", "before",
}
PREPOSITION_WORDS = {
    "in", "on", "at", "to", "of", "for", "by", "as",
}
TECH_WORDS = {
    "react", "hook", "hooks", "frontend", "backend", "javascript",
    "typescript", "python", "api", "ocr", "component", "state", "props",
}
PRESERVE_WORDS = {
    "frontend", "backend", "javascript", "typescript",
}
COMMON_WORDS = (
    QUESTION_WORDS
    | HELPER_WORDS
    | PRONOUN_WORDS
    | ACTION_WORDS
    | CONNECTOR_WORDS
    | PREPOSITION_WORDS
    | TECH_WORDS
    | {"about", "please", "care", "question", "screen", "react", "hook", "hooks"}
)
CONTINUATION_STARTERS = {
    "and", "or", "but", "so", "because", "then", "also",
    "for", "to", "with", "without", "about", "from", "in", "on", "at", "by",
}


def clean_question_text(text: str) -> str:
    """Remove double spaces and normalize duplicate stutter phrases."""
    if not text:
        return ""
    
    # Replace multiple spaces
    cleaned = re.sub(r'\s+', ' ', text).strip()
    
    # Remove duplicate phrases like "get started. get started." or "Okay. Okay."
    phrase_pattern = re.compile(
        r'(?P<phrase>\b\S+.*?\b)\s+(?P=phrase)(?=\s|$|[.?!,;:])',
        re.IGNORECASE,
    )
    prev = ""
    while cleaned != prev:
        prev = cleaned
        cleaned = phrase_pattern.sub(r'\1', cleaned)
        
    # Handle "word word" (no punctuation between)
    word_repeat_pattern = re.compile(r'\b(\w+)(?:\s+\1)+\b', re.IGNORECASE)
    prev = ""
    while cleaned != prev:
        prev = cleaned
        cleaned = word_repeat_pattern.sub(r'\1', cleaned)
        
    return cleaned.strip()


def is_stop_word(word: str) -> bool:
    stops = {
        "the", "a", "an", "is", "are", "do", "does", "did", "was", "were",
        "to", "of", "in", "for", "on", "with", "as", "at", "by", "from",
        "it", "this", "that", "these", "those", "and", "or", "but", "if",
        "i", "you", "we", "they", "he", "she", "me", "us", "them",
    }
    return word.lower() in stops


def glue_fragments(text: str) -> str:
    """Repair broken word fragments by joining syllables (e.g., 'Ex pla in' -> 'Explain')."""
    if not text:
        return text
        
    # 1. Join simple char-space-char chains
    words = re.split(r'\s+', text)
    glued = []
    current = ""
    
    for word in words:
        single_char_fragment = len(word) == 1 and word.lower() != "i"
        alpha_fragment = word.startswith('α') and len(word) <= 4
        if single_char_fragment or alpha_fragment:
            current += word
            continue
        if current:
            glued.append(current)
        glued.append(word)
        current = ""
            
    if current:
        glued.append(current)
        
    result = " ".join(glued)
    
    # 2. Syllable joiner for common fragments
    common_suffixes = {'ing', 'ion', 'ed', 'ly', 'ment', 'able', 'ness', 'ity', 'er', 'est', 's', 't', 'd', 'p', 'b', 'k', 'g', 'in'}
    
    words2 = re.split(r'\s+', result)
    if not words2:
        return ""
        
    final = []
    buffer = words2[0]
    
    for i in range(1, len(words2)):
        word = words2[i]
        prev = buffer
        low_word = word.lower()
        
        is_suffix = low_word in common_suffixes
        is_fragment = len(word) <= 3 and (not is_stop_word(low_word) or is_suffix)
        
        if (is_suffix or is_fragment) and len(prev) <= 5 and not is_stop_word(prev.lower()):
            if len(prev) + len(word) < 12:
                buffer += word
                continue
                
        final.append(buffer)
        buffer = word
        
    final.append(buffer)
    return " ".join(final)


def _split_punctuation(token: str):
    match = re.match(r"^([^A-Za-z0-9]*)([A-Za-z][A-Za-z'-]*)([^A-Za-z0-9]*)$", token)
    if not match:
        return "", token, ""
    return match.group(1), match.group(2), match.group(3)


def _merged_split_score(left: str, right: str) -> int:
    score = 0
    if left in QUESTION_WORDS and right in (HELPER_WORDS | PRONOUN_WORDS | ACTION_WORDS):
        score += 10
    if left in HELPER_WORDS and right in (PRONOUN_WORDS | ACTION_WORDS | QUESTION_WORDS):
        score += 9
    if left in PRONOUN_WORDS and right in (HELPER_WORDS | ACTION_WORDS):
        score += 8
    if left in ACTION_WORDS and right in (PRONOUN_WORDS | PREPOSITION_WORDS | CONNECTOR_WORDS):
        score += 7
    if left in COMMON_WORDS and right in COMMON_WORDS:
        score += 6
    if right in (PREPOSITION_WORDS | CONNECTOR_WORDS) and len(left) >= 4:
        score += 5
    if left in TECH_WORDS and right in COMMON_WORDS:
        score += 4
    if left.endswith("s") and right in PREPOSITION_WORDS and len(left) >= 5:
        score += 3
    return score


def repair_merged_words(text: str) -> str:
    """Split likely merged tokens such as 'couldyou' -> 'could you'."""
    if not text:
        return ""

    repaired = []
    for token in re.split(r"\s+", text.strip()):
        prefix, core, suffix = _split_punctuation(token)
        lower = core.lower()
        if not core or not lower.isalpha() or len(lower) < 5 or lower in PRESERVE_WORDS:
            repaired.append(token)
            continue

        best = None
        best_score = 0
        for idx in range(2, len(lower) - 1):
            left = lower[:idx]
            right = lower[idx:]
            score = _merged_split_score(left, right)
            if score > best_score:
                best = idx
                best_score = score

        if best is None or best_score < 6:
            repaired.append(token)
            continue

        repaired.append(f"{prefix}{core[:best]} {core[best:]}{suffix}")

    return " ".join(repaired)


def normalize_transcript(text: str) -> str:
    """Apply layered ASR cleanup without overfitting to exact phrases."""
    if not text:
        return ""
    cleaned = clean_question_text(glue_fragments(text))
    cleaned = repair_merged_words(cleaned)
    cleaned = clean_question_text(cleaned)
    return cleaned.strip()


def is_likely_fragment(text: str) -> bool:
    """Heuristic filter for low-information trailing scraps from speech ASR.

    Returns True only for scraps that are definitely NOT actionable questions:
    - Continuations starting with connectors ("and", "or", "but"…)
    - Very short utterances with no question signal AND no action verb
    - Preposition-led phrases under 5 words
    """
    if not text:
        return True

    trimmed = text.strip()
    lower = trimmed.lower()
    words = [w for w in re.split(r"\s+", re.sub(r"[.?!,;:]+", " ", lower)) if w]
    if not words:
        return True

    # Explicit question mark — never a fragment
    if trimmed.endswith("?"):
        return False

    # Starts with a discourse-level connector — definitely trailing scrap
    if words[0] in CONTINUATION_STARTERS:
        return True

    # Action / question word at the front gives intent — not a fragment
    if words[0] in QUESTION_WORDS or words[0] in ACTION_WORDS:
        return False

    # Very short utterances with no semantic signal
    if len(words) <= 3 and not any(
        w in QUESTION_WORDS or w in ACTION_WORDS for w in words
    ):
        return True

    # Preposition-led short phrases ("about closures", "with async"…)
    if len(words) <= 5 and words[0] in {"for", "to", "with", "about"}:
        return True

    return False


def is_question_complete(text: str) -> bool:
    """Check if question is complete with strict heuristics to gate LLM."""
    if not text:
        return False
        
    trimmed = text.strip()
    lower = trimmed.lower()
    # Strip trailing punctuation noise (commas, semicolons) before word-counting
    lower_clean = lower.rstrip(",;: ")
    trimmed_clean = trimmed.rstrip(",;: ")
    word_count = len(re.split(r'\s+', trimmed_clean.strip()))
    
    # 1. Explicit question mark - most reliable
    if trimmed.endswith('?'):
        return True

    # 2a. Action/question word + at least one content word = complete command.
    #     e.g. "explain memoization", "define closure", "describe promises"
    #     Rule 3 already rejects bare verb stubs ("explain" alone).
    if _RE_ACTION_STARTERS.match(lower_clean) and word_count >= 2:
        return True

    # 2. Minimum length check for non-action queries
    if word_count < 5:
        return False

    # 3. Ignore common fragments under 3 words
    if re.match(r'^(what is|tell me|explain|why do|how to|can you)$', lower_clean):
        return False

    # 4. Multiple complete sentences (2+)
    sentences = len(re.findall(r'[.!?]+', trimmed))
    if sentences >= 2:
        return True

    # 5. Single exclamation - only if followed by question intent
    if sentences == 1 and '?' in lower:
        return True

    # 6. Key starter word + minimum word count
    if _RE_QUESTION_STARTERS.match(lower_clean) and word_count >= 5:
        return True

    # 7. Ends with weak word but has question marks elsewhere
    if _RE_WEAK_QUESTION_WORDS.match(lower_clean) and word_count >= 5 and '?' in lower:
        return True

    # 8. Avoid false positives: ends with common nouns/verbs without punctuation
    if _RE_ENDS_INCOMPLETE.search(lower_clean):
        return False

    # 9. Contains question intent phrases anywhere
    if _RE_INTENT_PHRASES.match(lower_clean) and word_count >= 5:
        return True

    return False
