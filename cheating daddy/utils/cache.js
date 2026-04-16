// Simple Question Cache - Hash-based exact match caching
// Uses Map with time-to-live for simple caching

// Lazy-load fs to avoid issues in renderer
let fs = null;
let getCachePath = null;

function getFs() {
    if (!fs) {
        const storage = require('../storage');
        fs = require('fs');
        getCachePath = () => require('path').join(storage.getConfigDir(), 'cache.json');
    }
    return fs;
}

class SimpleCache {
    constructor(maxSize = 500, ttlMs = 3600000) {
        this.cache = new Map();
        this.maxSize = maxSize;
        this.ttlMs = ttlMs;
        this.hits = 0;
        this.misses = 0;
        this.loaded = false;
    }

    hashQuestion(text) {
        // Clean text before hashing to ensure consistent cache keys
        const cleaned = text
            .toLowerCase()
            .replace(/[^\w\s]/g, ' ') // Replace non-word chars with space
            .replace(/\s+/g, ' ') // Collapse multiple spaces
            .replace(/\s([a-z])\s/g, ' $1 ') // Ensure single spaces between lowercase
            .trim();

        let hash = 0;
        for (let i = 0; i < cleaned.length; i++) {
            const char = cleaned.charCodeAt(i);
            hash = (hash << 5) - hash + char;
            hash = hash & hash;
        }
        return hash.toString(36);
    }

    // Common words to ignore for fuzzy matching
    _stopWords = new Set([
        'the',
        'is',
        'a',
        'an',
        'what',
        'how',
        'why',
        'when',
        'where',
        'who',
        'can',
        'could',
        'would',
        'should',
        'do',
        'does',
        'did',
        'have',
        'has',
        'will',
        'shall',
        'to',
        'of',
        'in',
        'for',
        'on',
        'with',
        'at',
        'by',
        'from',
        'as',
        'or',
        'and',
        'it',
        'me',
        'you',
        'your',
        'i',
        'my',
        'tell',
        'explain',
        'describe',
        'give',
        'show',
        'about',
        'difference',
        'between',
        'explain',
        'tell',
        'describe',
        'describe',
        'give',
        'show',
        'can',
        'you',
        'please',
    ]);

    get(question) {
        const key = this.hashQuestion(question);
        const entry = this.cache.get(key);

        if (!entry) {
            // Try fuzzy match before giving up
            const fuzzyResult = this._findFuzzyMatch(question);
            if (fuzzyResult) {
                this.hits++;
                return fuzzyResult.response;
            }
            this.misses++;
            return null;
        }

        // Check if expired
        if (Date.now() - entry.timestamp > this.ttlMs) {
            this.cache.delete(key);
            this.misses++;
            return null;
        }

        this.hits++;
        return entry.response;
    }

    // Fuzzy matching: find cached question with sufficient matching words
    _findFuzzyMatch(input) {
        const inputWords = this._getContentWords(input);
        if (inputWords.length < 4) return null; // Require at least 4 technical/content words

        // Skip fuzzy for non-Latin scripts (likely non-English)
        if (/[^\u0000-\u007F]/.test(input)) return null;

        let bestMatch = null;
        let bestScore = 0;
        const minMatchWords = 3; // Require at least 3 matching content words
        const minOverlapRatio = 0.8; // Require 80% overlap for fuzzy hits

        for (const [key, entry] of this.cache) {
            const cachedWords = this._getContentWords(entry.question);
            const overlap = inputWords.filter(w => cachedWords.includes(w)).length;
            const overlapRatio = overlap / inputWords.length;

            // Must meet BOTH thresholds
            if (overlap >= minMatchWords && overlapRatio >= minOverlapRatio && overlap > bestScore) {
                bestScore = overlap;
                bestMatch = entry;
            }
        }

        if (bestMatch) {
            console.log(`[Cache] FUZZY HIT: "${bestMatch.question.substring(0, 30)}..." (${bestScore} words)`);
        }
        return bestMatch;
    }

    _getContentWords(text) {
        return text
            .toLowerCase()
            .replace(/[^\w\s]/g, ' ')
            .split(/\s+/)
            .filter(w => w.length > 2 && !this._stopWords.has(w));
    }

    // Check if question is "garbage" (stuttering, low quality, or encoding artifacts)
    isGarbage(text) {
        const trimmed = text.trim();

        // Much more lenient: only reject truly broken text
        // Don't block valid questions with speech artifacts
        if (trimmed.length < 3) return true;
        if (trimmed.split(/\s+/).length < 2) return true;

        // 1. Check for VERY obvious stuttering - same word repeated 3+ times (e.g., "the the the the")
        const words = trimmed.toLowerCase().split(/\s+/);
        let consecutiveRepeats = 1;
        let maxConsecutive = 1;
        for (let i = 1; i < words.length; i++) {
            if (words[i] === words[i - 1] && words[i].length >= 3) {
                consecutiveRepeats++;
                maxConsecutive = Math.max(maxConsecutive, consecutiveRepeats);
            } else {
                consecutiveRepeats = 1;
            }
        }
        if (maxConsecutive >= 3) {
            console.log(`[Cache Guard] Rejected garbage: repeating word 3+ times`);
            return true;
        }

        // 2. Check for character entropy (Artifact detection) - very lenient
        const totalChars = trimmed.length;
        const standardChars = (trimmed.match(/[a-zA-Z0-9\s.,?!'"()-]/g) || []).length;
        if (totalChars > 10 && standardChars / totalChars < 0.4) {
            console.log(`[Cache Guard] Rejected garbage: high character entropy`);
            return true;
        }

        // 3. Skip all other checks - they're too aggressive for speech recognition
        return false;
    }

    set(question, response) {
        // Skip garbage
        if (this.isGarbage(question)) {
            return;
        }
        // Remove oldest if at capacity
        if (this.cache.size >= this.maxSize) {
            const firstKey = this.cache.keys().next().value;
            this.cache.delete(firstKey);
        }

        const key = this.hashQuestion(question);
        this.cache.set(key, {
            question: question,
            response: response,
            timestamp: Date.now(),
        });

        // Save to disk (debounced)
        this._scheduleSave();
        console.log(`[Cache] SET: "${question.substring(0, 50)}..."`);
    }

    deleteQuestion(question) {
        const key = this.hashQuestion(question);
        if (this.cache.has(key)) {
            this.cache.delete(key);
            console.log(`[Cache] Deleted entry for: "${question.substring(0, 30)}..."`);
            this._scheduleSave();
            return true;
        }
        return false;
    }

    clear() {
        this.cache.clear();
        this.hits = 0;
        this.misses = 0;
        console.log('[Cache] Cleared all entries');
        this._scheduleSave();
    }

    deleteQuestion(question) {
        const key = this.hashQuestion(question);
        if (this.cache.has(key)) {
            this.cache.delete(key);
            console.log(`[Cache] Deleted entry for: "${question.substring(0, 30)}..."`);
            this._scheduleSave();
            return true;
        }
        return false;
    }

    clear() {
        this.cache.clear();
        console.log('[Cache] Cleared all entries');
        this._scheduleSave();
    }

    _saveTimer = null;
    _scheduleSave() {
        if (this._saveTimer) clearTimeout(this._saveTimer);
        this._saveTimer = setTimeout(() => this._saveToDisk(), 5000);
    }

    _saveToDisk() {
        try {
            const f = getFs();
            const path = getCachePath();
            const data = Array.from(this.cache.entries());
            f.writeFileSync(path, JSON.stringify(data), 'utf8');
            console.log(`[Cache] Saved ${data.length} entries to disk`);
        } catch (e) {
            // Silent fail - disk write not critical
        }
    }

    loadFromDisk() {
        if (this.loaded) return;
        try {
            const f = getFs();
            const path = getCachePath();
            if (f.existsSync(path)) {
                const data = JSON.parse(f.readFileSync(path, 'utf8'));
                this.cache = new Map(data);
                console.log(`[Cache] Loaded ${this.cache.size} entries from disk`);
            }
        } catch (e) {
            console.log('[Cache] No saved cache found');
        }
        this.loaded = true;
    }

    getStats() {
        const total = this.hits + this.misses;
        return {
            size: this.cache.size,
            hits: this.hits,
            misses: this.misses,
            hitRate: total > 0 ? (this.hits / total) * 100 : 0,
        };
    }

    // Get all cache entries for inspection
    getAllEntries() {
        const entries = [];
        for (const [key, entry] of this.cache) {
            entries.push({
                question: entry.question,
                response: entry.response.substring(0, 100) + '...',
                age: Math.round((Date.now() - entry.timestamp) / 60000) + ' min ago',
            });
        }
        return entries;
    }

    // Print cache contents to console
    printCache() {
        console.log('=== CACHE CONTENTS ===');
        for (const [key, entry] of this.cache) {
            console.log(`Q: ${entry.question.substring(0, 50)}...`);
            console.log(`A: ${entry.response.substring(0, 80)}...`);
            console.log('---');
        }
    }

    clear() {
        this.cache.clear();
        this.hits = 0;
        this.misses = 0;
    }
}

const cache = new SimpleCache(500, 3600000);

// Timestamped log helper
const t = () => new Date().toISOString().slice(11, 19);

function getCachedResponse(question) {
    const response = cache.get(question);
    if (response) {
        console.log(`[${t()}]`, '[Cache] HIT:', question.substring(0, 50) + '...');
        return { response };
    }
    return null;
}

function setCachedResponse(question, response) {
    // Logging now handled inside SimpleCache.set to ensure accuracy
    cache.set(question, response);
}

function getCacheStats() {
    return cache.getStats();
}

function clearCache() {
    cache.clear();
    console.log('[Cache] Cleared');
}

function loadCacheFromDisk() {
    cache.loadFromDisk();
}

function deleteCachedResponse(question) {
    return cache.deleteQuestion(question);
}

module.exports = {
    getCachedResponse,
    setCachedResponse,
    getCacheStats,
    clearCache,
    loadCacheFromDisk,
    deleteCachedResponse,
    isGarbage: text => cache.isGarbage(text),
    isStopWord: word => {
        const stopWords = new Set([
            'is',
            'am',
            'are',
            'was',
            'were',
            'be',
            'of',
            'in',
            'on',
            'at',
            'to',
            'for',
            'by',
            'as',
            'but',
            'or',
            'so',
            'if',
            'a',
            'an',
            'the',
            'what',
            'which',
            'who',
            'whom',
            'this',
            'that',
            'these',
            'those',
            'it',
            'its',
            'they',
            'them',
            'their',
            'our',
            'us',
            'we',
            'you',
            'your',
            'my',
            'me',
            'mine',
            'and',
            'with',
            'from',
            'between',
            'into',
        ]);
        return stopWords.has(word.toLowerCase());
    },
};
