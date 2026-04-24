"""Text processing utilities for ultra-low latency transcription repair and gating."""

import re

def clean_question_text(text: str) -> str:
    """Remove double spaces and normalize duplicate stutter phrases."""
    if not text:
        return ""
    
    # Replace multiple spaces
    cleaned = re.sub(r'\s+', ' ', text).strip()
    
    # Remove duplicate phrases like "get started. get started." or "Okay. Okay."
    phrase_pattern = re.compile(r'(\b\S+.*?\b)\s+\1', re.IGNORECASE)
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
        "it", "this", "that", "these", "those", "and", "or", "but", "if"
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
        if len(word) == 1 or (word.startswith('α') and len(word) <= 4):
            current += word
        else:
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


def is_question_complete(text: str) -> bool:
    """Check if question is complete with strict heuristics to gate LLM."""
    if not text:
        return False
        
    trimmed = text.strip()
    lower = trimmed.lower()
    word_count = len(re.split(r'\s+', trimmed))
    
    # 1. Explicit question mark - most reliable
    if trimmed.endswith('?'):
        return True
        
    # 2. Minimum length check
    if word_count < 5:
        return False
        
    # 3. Ignore common fragments under 3 words
    if re.match(r'^(what is|tell me|explain|why do|how to|can you)$', lower):
        return False
        
    # 4. Multiple complete sentences (2+)
    sentences = len(re.findall(r'[.!?]+', trimmed))
    if sentences >= 2:
        return True
        
    # 5. Single exclamation - only if followed by question intent
    if sentences == 1 and '?' in lower:
        return True
        
    # 6. Key starter word + minimum word count
    question_starters = re.compile(r'^(what|who|where|when|why|how|which|can|could|would|should|tell|explain|describe|give|show|list|write|create|define|difference|advantages?|disadvantages?|benefits?|pros|cons)\b', re.IGNORECASE)
    if question_starters.match(lower) and word_count >= 5:
        return True
        
    # 7. Ends with weak word but has question marks elsewhere
    weak_question_words = re.compile(r'^(is|are|do|does|did|have|has|will|shall|meaning)\b', re.IGNORECASE)
    if weak_question_words.match(lower) and word_count >= 5 and '?' in lower:
        return True
        
    # 8. Avoid false positives: ends with common nouns/verbs without punctuation
    ends_with_incomplete = re.compile(r' (me|you|him|her|them|it|this|that|what|how|why|and|or|is|are|the|a|an|to|with|for|in|on|at|by|from)$', re.IGNORECASE)
    if ends_with_incomplete.search(lower):
        return False
        
    # 9. Contains question intent phrases anywhere
    intent_phrases = re.compile(r'^(tell|explain|describe|define|difference|advantage|disadvantage)', re.IGNORECASE)
    if intent_phrases.match(lower) and word_count >= 5:
        return True
        
    return False
