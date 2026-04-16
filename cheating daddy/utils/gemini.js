const { GoogleGenAI, Modality } = require('@google/genai');
const { BrowserWindow, ipcMain } = require('electron');
const { spawn } = require('child_process');
const { saveDebugAudio } = require('../audioUtils');
const { getSystemPrompt } = require('./prompts');
const { getAvailableModel, incrementLimitCount, getApiKey, getGroqApiKey, incrementCharUsage, getModelForToday } = require('../storage');
const { connectCloud, sendCloudAudio, sendCloudText, sendCloudImage, closeCloud, isCloudActive, setOnTurnComplete } = require('./cloud');
const {
    getCachedResponse,
    setCachedResponse,
    getCacheStats,
    clearCache,
    loadCacheFromDisk,
    deleteCachedResponse,
    isGarbage,
    isStopWord,
} = require('./cache');
const { shouldUseFastModel, analyzeQueryComplexity } = require('./router');

// Timestamped log helper
const log = {
    q: (...args) => console.log(`[Q ${new Date().toISOString().slice(11, 19)}]`, ...args),
    a: (...args) => console.log(`[A ${new Date().toISOString().slice(11, 19)}]`, ...args),
    info: (...args) => console.log(`[${new Date().toISOString().slice(11, 19)}]`, ...args),
};

// Lazy-loaded to avoid circular dependency (localai.js imports from gemini.js)
let _localai = null;
function getLocalAi() {
    if (!_localai) _localai = require('./localai');
    return _localai;
}

// Provider mode: 'byok', 'cloud', or 'local'
let currentProviderMode = 'byok';

// Groq conversation history for context
let groqConversationHistory = [];

// Conversation tracking variables
let currentSessionId = null;
let currentTranscription = '';
let conversationHistory = [];
let screenAnalysisHistory = [];
let currentProfile = null;
let currentCustomPrompt = null;
let isInitializingSession = false;
let currentSystemPrompt = null;

// Fast response tracking
let isProcessingResponse = false;
let lastProcessedTranscription = '';
let pendingTranscription = '';
let pendingTranscriptionTimestamp = 0;
let pauseTimer = null;
let responseProcessedForCurrentTurn = false;

// Rate limiting: minimum 2s between questions
const MIN_QUESTION_INTERVAL = 2000;
let lastQuestionTime = 0;
let lastQuestionText = '';
let queuedTranscription = null;
let questionQueueTimer = null;
let groqCooldownUntil = 0; // After rate limit, wait before retrying
let groqFailedForSession = false; // If Groq fails (rate limit), skip for entire session
let speechStartTime = 0; // When user started speaking
let transcriptionStartTime = 0; // When current question started
let processingStartTime = 0; // When we started processing the question

// Debounce for intent logging
let lastIntentLogTime = 0;
const INTENT_LOG_DEBOUNCE = 1000;

// Sequential Task Queue for questions
const questionQueue = [];
let isProcessingQueue = false;

// Question indicators - strongest signals of a question
const _questionStarters =
    /^(what|who|where|when|why|how|which|can|could|would|should|tell|explain|describe|give|show|list|write|create|define|difference|advantages?|disadvantages?|benefits?|pros|cons)\b/i;

// Weak question words - can appear in statements, require more context
const _weakQuestionWords = /^(is|are|do|does|did|have|has|will|shall|meaning)\b/i;

// Check if question is complete with improved heuristics
function isQuestionComplete(text) {
    const trimmed = text.trim();
    const lower = trimmed.toLowerCase();
    const wordCount = trimmed.split(/\s+/).length;

    // 1. Explicit question mark - most reliable
    if (trimmed.endsWith('?')) return true;

    // 2. Minimum length check - 5 words minimum (more reasonable)
    if (wordCount < 5) return false;

    // 3. Ignore common fragments under 3 words
    if (/^(what is|tell me|explain|why do|how to|can you)$/i.test(lower)) return false;
    if (wordCount < 3) return false;

    // 4. Multiple complete sentences (2+)
    const sentences = (trimmed.match(/[.!?]+/g) || []).length;
    if (sentences >= 2) return true;

    // 5. Single exclamation - only if followed by question intent
    if (sentences === 1 && lower.includes('?')) return true;

    // 6. Key starter word + minimum word count = question
    if (_questionStarters.test(lower) && wordCount >= 5) return true;

    // 7. Ends with weak word but has question marks elsewhere
    if (_weakQuestionWords.test(lower) && wordCount >= 5 && lower.includes('?')) return true;

    // 8. Avoid false positives: ends with common nouns/verbs without punctuation (likely incomplete)
    const endsWithIncomplete = / (me|you|him|her|them|it|this|that|what|how|why|and|or|is|are|the|a|an|to|with|for|in|on|at|by|from)$/i;
    if (endsWithIncomplete.test(lower)) return false;

    // 9. Contains question intent phrases anywhere
    const intentPhrases = /^(tell|explain|describe|define|difference|advantage|disadvantage)/i;
    if (intentPhrases.test(lower) && wordCount >= 5) return true;

    return false;
}

// Smart string merging to avoid duplicates like "What is is routing"
function mergeTranscripts(current, next) {
    if (!current) return next;
    const currentLower = current.toLowerCase().trim();
    const nextLower = next.toLowerCase().trim();

    // If next is already contained in current, return current
    if (currentLower.includes(nextLower)) return current;

    // If current is already contained in next, return next
    if (nextLower.startsWith(currentLower)) return next;

    // Find overlapping words
    const currentWords = current.split(/\s+/);
    const nextWords = next.split(/\s+/);

    // Check if the end of current matches the start of next (up to 5 words)
    for (let i = Math.min(5, currentWords.length); i > 0; i--) {
        const currentTail = currentWords.slice(-i).join(' ').toLowerCase();
        const nextHead = nextWords.slice(0, i).join(' ').toLowerCase();

        if (currentTail === nextHead) {
            // Merge: current + [the rest of next]
            return currentWords.concat(nextWords.slice(i)).join(' ');
        }
    }

    // No overlap found, just append
    return current + ' ' + next;
}

// Check if new input looks like continuation of previous (not a new question)
function isQuestionContinuation(newText, previousText) {
    if (!previousText || previousText.trim().length === 0) return false;

    const newTrimmed = newText.trim();
    const prevTrimmed = previousText.trim();

    // If new text is short fragment (< 15 chars), likely continuation
    if (newTrimmed.length < 15 && newTrimmed.length > 0) {
        return true;
    }

    // If new text starts with lowercase and previous doesn't end with punctuation, likely continuation
    if (/^[a-z]/.test(newTrimmed) && /[a-zA-Z]$/.test(prevTrimmed)) {
        return true;
    }

    return false;
}

// Stricter duplicate check to avoid blocking valid follow-ups
function isDuplicateQuestion(text) {
    if (!lastProcessedTranscription) return false;

    const normalizedNew = text.toLowerCase().trim();
    const normalizedLast = lastProcessedTranscription.toLowerCase().trim();

    // Exactly the same
    if (normalizedNew === normalizedLast) return true;

    // Check if the difference is very small (likely minor transcription update)
    if (normalizedNew.length > normalizedLast.length) {
        // If it just added a few chars at the end, it might be the same turn
        const diff = normalizedNew.substring(0, normalizedLast.length);
        if (diff === normalizedLast && normalizedNew.length - normalizedLast.length < 5) {
            return true;
        }
    }

    return false;
}

// Clean up question text - remove double spaces, normalize, and handle duplicate phrases
function cleanQuestionText(text) {
    if (!text) return '';
    // Replace multiple spaces with single space
    let cleaned = text.replace(/\s+/g, ' ').trim();

    // Remove duplicate phrases like "get started. get started." or "Okay. Okay."
    // Pattern: capture phrase, then match same phrase again (with optional punctuation between)
    const phrasePattern = /(\b\S+.*?\b)\s+\1/gi;
    let prev;
    do {
        prev = cleaned;
        cleaned = cleaned.replace(phrasePattern, '$1');
    } while (cleaned !== prev);

    // Also handle "word word" (no punctuation between) - must be same word 2+ times
    const wordRepeatPattern = /\b(\w+)(?:\s+\1)+/gi;
    do {
        prev = cleaned;
        cleaned = cleaned.replace(wordRepeatPattern, '$1');
    } while (cleaned !== prev);

    return cleaned.trim();
}

// Atomic reset of all transcription-related state
function clearTranscriptionBuffers() {
    currentTranscription = '';
    pendingTranscription = '';
    pendingTranscriptionTimestamp = 0;
    // Note: We intentionally do NOT clear lastProcessedTranscription here
    // because it's used for duplicate detection across questions
    if (pauseTimer) {
        clearTimeout(pauseTimer);
        pauseTimer = null;
    }
    // Also clear audio buffer state
    responseProcessedForCurrentTurn = false;
}

// Repair broken word fragments by joining single characters or syllables (e.g., "Ex pla in" -> "Explain")
function glueFragments(text) {
    if (!text) return text;

    // 1. Join simple char-space-char chains (Handles "J u s t" and artifacts)
    let words = text.split(/\s+/);
    let glued = [];
    let current = '';

    for (let word of words) {
        // If word is special artifact or single char
        if (word.length === 1 || (/^α/.test(word) && word.length <= 4)) {
            current += word;
        } else {
            if (current) glued.push(current);
            glued.push(word);
            current = '';
        }
    }
    if (current) glued.push(current);

    let result = glued.join(' ');

    // 2. Syllable joiner for common fragments (e.g. "Expla in", "get ting")
    // Note: includes 'in' as a suffix even though it is a stopword
    const commonSuffixes = new Set(['ing', 'ion', 'ed', 'ly', 'ment', 'able', 'ness', 'ity', 'er', 'est', 's', 't', 'd', 'p', 'b', 'k', 'g', 'in']);

    const words2 = result.split(/\s+/);
    const final = [];
    let buffer = words2[0];

    for (let i = 1; i < words2.length; i++) {
        const word = words2[i];
        const prev = buffer;
        const lowWord = word.toLowerCase();

        const isSuffix = commonSuffixes.has(lowWord);
        // A fragment is a short word (1-3 chars) that isn't a stop word
        // BUT if it's a known suffix, we join even if it's a stop word (like "s" or "t")
        const isFragment = word.length <= 3 && (!isStopWord(lowWord) || isSuffix);

        if ((isSuffix || isFragment) && prev.length <= 5 && !isStopWord(prev)) {
            // Check if joining doesn't create something huge
            if (prev.length + word.length < 12) {
                buffer += word;
                continue;
            }
        }
        final.push(buffer);
        buffer = word;
    }
    final.push(buffer);

    return final.join(' ');
}

// Smartly merge new transcription chunk with buffer by detecting overlaps
function mergeTranscripts(buffer, chunk) {
    const b = (buffer || '').trim();
    const c = (chunk || '').trim();

    if (!b) return glueFragments(c);
    if (!c) return glueFragments(b);

    // Glue both independently first to normalize
    const gBuffer = glueFragments(b);
    const gChunk = glueFragments(c);

    const cleanedBuffer = gBuffer.trim();
    const cleanedChunk = gChunk.trim();

    // Strategy 1: Substring/Incremental match
    if (cleanedChunk.toLowerCase().startsWith(cleanedBuffer.toLowerCase())) {
        return cleanedChunk;
    }
    if (cleanedBuffer.toLowerCase().endsWith(cleanedChunk.toLowerCase())) {
        return cleanedBuffer;
    }

    const bWords = cleanedBuffer.split(/\s+/);
    const cWords = cleanedChunk.split(/\s+/);

    // Strategy 2: Improved Overlap with fuzzy boundary
    const normalize = w => w.toLowerCase().replace(/[.,?!]$/, '');

    for (let overlap = Math.min(20, bWords.length, cWords.length); overlap > 0; overlap--) {
        const bSuffixWords = bWords.slice(-overlap).map(normalize);
        const cPrefixWords = cWords.slice(0, overlap).map(normalize);

        let match = true;
        for (let j = 0; j < overlap; j++) {
            const bw = bSuffixWords[j];
            const cw = cPrefixWords[j];
            // If it's the start of the overlap, allow partial match (suffix)
            // This handles cases like "elaborate" (buffer) and "borate" (chunk start)
            if (j === 0) {
                if (bw !== cw && !bw.endsWith(cw)) {
                    match = false;
                    break;
                }
            } else if (bw !== cw) {
                match = false;
                break;
            }
        }

        if (match) {
            return bWords.join(' ') + ' ' + cWords.slice(overlap).join(' ');
        }
    }

    // Strategy 3: Boundary Heal (Character level) with Stop-word protection
    const lastWord = bWords[bWords.length - 1].toLowerCase();
    const firstWord = cWords[0].toLowerCase();

    if (!isStopWord(lastWord) && !isStopWord(firstWord)) {
        if (lastWord.length <= 3 && firstWord.length <= 3) {
            const head = cleanedBuffer;
            const tail = cleanedChunk.slice(cWords[0].length);
            return head + cWords[0] + (tail ? ' ' + tail.trim() : '');
        }
    }

    return cleanedBuffer + ' ' + cleanedChunk;
}

// Unified entry point for all transcription AI processing
async function processTranscription(text) {
    if (!text || text.trim().length < 5) return;

    // Clean the text before processing to remove double spaces and duplicates
    const cleaned = cleanQuestionText(text);
    if (!cleaned || cleaned.length < 5) return;

    const trimmed = cleaned;

    // Check for duplicates with refined logic - ignore if exact same as last processed OR last queued
    const lastQueued = questionQueue.length > 0 ? questionQueue[questionQueue.length - 1] : null;
    const lower = trimmed.toLowerCase();

    if (isDuplicateQuestion(trimmed) || (lastQueued && lower === lastQueued.toLowerCase())) {
        log.info('Skipping duplicate question (queued or processed)');
        return;
    }

    // Subset check: if this question is just a shorter version of the last one, or vice-versa
    if (lastQueued && (lastQueued.toLowerCase().includes(lower) || lower.includes(lastQueued.toLowerCase()))) {
        log.info('Skipping subset question (already in queue)');
        // Update the queue with the longer version if the new one is more complete
        if (trimmed.length > lastQueued.length) {
            questionQueue[questionQueue.length - 1] = trimmed;
        }
        return;
    }

    // Garbage/Stutter check: Skip before queueing
    if (isGarbage(trimmed)) {
        log.info(`[Pipeline] Blocked garbage question: "${trimmed.substring(0, 30)}..."`);
        return;
    }

    // Capture the question and CLEAR BUFFERS IMMEDIATELY
    // This allows the user to start the next question without bleed-through
    log.info(`Queueing question: "${trimmed.substring(0, 40)}${trimmed.length > 40 ? '...' : ''}"`);
    questionQueue.push(trimmed);
    clearTranscriptionBuffers();

    // Trigger the queue processor
    processQueue();
}

// Background worker to process questions one by one
async function processQueue() {
    if (isProcessingQueue || questionQueue.length === 0 || isProcessingResponse) {
        return;
    }

    isProcessingQueue = true;

    try {
        while (questionQueue.length > 0) {
            // Check if we are already processing (double safety)
            if (isProcessingResponse) break;

            const textToProcess = questionQueue.shift();
            log.info(`[Queue] Processing next question: "${textToProcess.substring(0, 30)}..."`);

            // Set global lock
            isProcessingResponse = true;
            lastProcessedTranscription = textToProcess;
            sendToRenderer('update-status', 'Processing...');

            // Use Gemini if Groq is disabled for session
            if (hasGroqKey() && !groqFailedForSession && groqCooldownUntil <= Date.now()) {
                await sendToGroq(textToProcess, false);
            } else {
                await sendToGemma(textToProcess, groqFailedForSession);
            }
        }
    } catch (error) {
        log.error('[Queue] Error in worker loop:', error);
    } finally {
        isProcessingQueue = false;
        isProcessingResponse = false; // Emergency reset
        sendToRenderer('update-status', 'Listening...');
    }
}

function formatSpeakerResults(results) {
    let text = '';
    for (const result of results) {
        if (result.transcript && result.speakerId) {
            const speakerLabel = result.speakerId === 1 ? 'Interviewer' : 'Candidate';
            text += `[${speakerLabel}]: ${result.transcript}\n`;
        }
    }
    return text;
}

module.exports.formatSpeakerResults = formatSpeakerResults;

// Audio capture variables
let systemAudioProc = null;
let messageBuffer = '';

// Reconnection variables
let isUserClosing = false;
let sessionParams = null;
let reconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 3;
const RECONNECT_DELAY = 2000;

function sendToRenderer(channel, data) {
    const windows = BrowserWindow.getAllWindows();
    if (windows.length > 0) {
        windows[0].webContents.send(channel, data);
    }
}

// Build context message for session restoration
function buildContextMessage() {
    const lastTurns = conversationHistory.slice(-20);
    const validTurns = lastTurns.filter(turn => turn.transcription?.trim() && turn.ai_response?.trim());

    if (validTurns.length === 0) return null;

    const contextLines = validTurns.map(turn => `[Interviewer]: ${turn.transcription.trim()}\n[Your answer]: ${turn.ai_response.trim()}`);

    return `Session reconnected. Here's the conversation so far:\n\n${contextLines.join('\n\n')}\n\nContinue from here.`;
}

// Conversation management functions
function initializeNewSession(profile = null, customPrompt = null) {
    currentSessionId = Date.now().toString();
    currentTranscription = '';
    conversationHistory = [];
    screenAnalysisHistory = [];
    groqConversationHistory = [];
    currentProfile = profile;
    currentCustomPrompt = customPrompt;
    console.log('New conversation session started:', currentSessionId, 'profile:', profile);

    // Load cached responses from disk
    loadCacheFromDisk();

    // Save initial session with profile context
    if (profile) {
        sendToRenderer('save-session-context', {
            sessionId: currentSessionId,
            profile: profile,
            customPrompt: customPrompt || '',
        });
    }
}

function saveConversationTurn(transcription, aiResponse) {
    if (!currentSessionId) {
        initializeNewSession();
    }

    const conversationTurn = {
        timestamp: Date.now(),
        transcription: transcription.trim(),
        ai_response: aiResponse.trim(),
    };

    conversationHistory.push(conversationTurn);
    console.log('Saved conversation turn:', conversationTurn);

    // Send to renderer to save in IndexedDB
    sendToRenderer('save-conversation-turn', {
        sessionId: currentSessionId,
        turn: conversationTurn,
        fullHistory: conversationHistory,
    });
}

function saveScreenAnalysis(prompt, response, model) {
    if (!currentSessionId) {
        initializeNewSession();
    }

    const analysisEntry = {
        timestamp: Date.now(),
        prompt: prompt,
        response: response.trim(),
        model: model,
    };

    screenAnalysisHistory.push(analysisEntry);
    console.log('Saved screen analysis:', analysisEntry);

    // Send to renderer to save
    sendToRenderer('save-screen-analysis', {
        sessionId: currentSessionId,
        analysis: analysisEntry,
        fullHistory: screenAnalysisHistory,
        profile: currentProfile,
        customPrompt: currentCustomPrompt,
    });
}

function getCurrentSessionData() {
    return {
        sessionId: currentSessionId,
        history: conversationHistory,
    };
}

async function getEnabledTools() {
    const tools = [];

    // Check if Google Search is enabled (default: true)
    const googleSearchEnabled = await getStoredSetting('googleSearchEnabled', 'true');
    console.log('Google Search enabled:', googleSearchEnabled);

    if (googleSearchEnabled === 'true') {
        tools.push({ googleSearch: {} });
        console.log('Added Google Search tool');
    } else {
        console.log('Google Search tool disabled');
    }

    return tools;
}

async function getStoredSetting(key, defaultValue) {
    try {
        const preferences = require('../storage').getPreferences();
        const value = preferences[key];
        return value === undefined || value === null ? defaultValue : String(value);
    } catch (error) {
        console.error('Error getting stored setting for', key, ':', error.message);
        return defaultValue;
    }
}

function describeCloseReason(closeEvent) {
    if (!closeEvent) {
        return 'unknown';
    }

    if (typeof closeEvent.reason === 'string' && closeEvent.reason.trim() !== '') {
        return closeEvent.reason;
    }

    if (typeof closeEvent.code === 'number') {
        return `code ${closeEvent.code}`;
    }

    return 'no reason provided';
}

function summarizeLiveServerMessage(message) {
    const serverContent = message?.serverContent;
    if (!serverContent) {
        return null;
    }

    if (serverContent.setupComplete) {
        return 'setupComplete';
    }

    if (serverContent.generationComplete) {
        return 'generationComplete';
    }

    // TurnComplete handling is in onmessage, don't duplicate the summary
    // if (serverContent.turnComplete) {
    //     return 'turnComplete';
    // }

    if (typeof serverContent.inputTranscription?.text === 'string') {
        return `inputTranscription: ${serverContent.inputTranscription.text}`;
    }

    if (typeof serverContent.outputTranscription?.text === 'string') {
        // DISABLED - WebSocket voice output causes duplicate log noise with Fast Mode
        // The voice output is only for audio feedback, not displayed
        return null;
    }

    if (serverContent.modelTurn?.parts) {
        // Log periodically - there are many chunks
    }

    return null;
}

// helper to check if groq has been configured
function hasGroqKey() {
    const key = getGroqApiKey();
    return key && key.trim() != '';
}

function trimConversationHistoryForGemma(history, maxChars = 42000) {
    if (!history || history.length === 0) return [];
    let totalChars = 0;
    const trimmed = [];

    for (let i = history.length - 1; i >= 0; i--) {
        const turn = history[i];
        const turnChars = (turn.content || '').length;

        if (totalChars + turnChars > maxChars) break;
        totalChars += turnChars;
        trimmed.unshift(turn);
    }
    return trimmed;
}

function stripThinkingTags(text) {
    if (!text) return '';
    let result = text;
    // Remove ALL instances of thinking tags (using global flag)
    result = result.replace(/<think>[\s\S]*?<\/think>/gi, '');
    // Also handle cases where tags might be split across streamed chunks (incomplete)
    result = result.replace(/<think>/gi, '');
    result = result.replace(/<\/think>/gi, '');
    return result.trim();
}

async function sendToGroq(transcription, isPartial = false, preferFastModel = null) {
    const groqApiKey = getGroqApiKey();
    if (!groqApiKey) {
        console.log('No Groq API key configured, skipping Groq response');
        return;
    }

    // Clean up the transcription text
    const cleanedText = cleanQuestionText(transcription);

    if (!cleanedText || cleanedText.length < 3) {
        console.log('Empty transcription, skipping Groq');
        return;
    }

    // Rate limiting: skip if too soon or duplicate
    const now = Date.now();
    const normalized = cleanedText.toLowerCase();
    if (now < groqCooldownUntil || groqFailedForSession) {
        log.info('Groq disabled for session, falling back to Gemini');
        const apiKey = getApiKey();
        if (apiKey) sendToGemma(cleanedText, true); // true = isFallback
        return;
    }
    if (normalized === lastQuestionText) {
        log.info('Duplicate question, skipping');
        return;
    }
    if (now - lastQuestionTime < MIN_QUESTION_INTERVAL) {
        // Queue the transcription to process after cooldown
        const remainingWait = MIN_QUESTION_INTERVAL - (now - lastQuestionTime);
        log.info(`Rate limited, queueing for ${Math.round(remainingWait / 100) / 10}s`);

        // Clear any existing queued question (keep the newer one)
        queuedTranscription = { transcription, timestamp: now };

        if (questionQueueTimer) clearTimeout(questionQueueTimer);
        questionQueueTimer = setTimeout(() => {
            if (queuedTranscription) {
                const q = queuedTranscription;
                queuedTranscription = null;
                log.info('Processing queued question after rate limit cooldown');
                sendToGroq(q.transcription, isPartial, preferFastModel);
            }
        }, remainingWait);

        return;
    }

    // Check cache for exact or similar questions (skip for partial transcriptions)
    if (!isPartial) {
        const cachedResponse = getCachedResponse(transcription);
        if (cachedResponse) {
            const { response } = cachedResponse;
            console.log('[Cache] Using cached response');
            sendToRenderer('new-response', response);

            groqConversationHistory.push({
                role: 'user',
                content: transcription.trim(),
            });
            groqConversationHistory.push({
                role: 'assistant',
                content: response,
            });

            saveConversationTurn(transcription, response);
            isProcessingResponse = false;
            sendToRenderer('update-status', 'Listening... (cached)');
            return;
        }
    }

    // Determine which model to use based on query complexity
    // preferFastModel can be: true (force fast), false (force complex), null (auto-detect)
    let useFastModel = preferFastModel;
    if (useFastModel === null) {
        useFastModel = shouldUseFastModel(transcription);
    }

    // Select model based on complexity
    // Fast models: llama-3.3-70b-versatile (default), llama-3.1-8b-instant, mixtral-8x7b-32768
    // Complex models: llama-3.1-405b-reasoning-ultra, deepseek-r1-distill-llama-70b

    // Check if we have the model available
    const availableModel = getModelForToday();
    if (!availableModel) {
        console.log('All Groq daily limits exhausted');
        sendToRenderer('update-status', 'Groq limits reached for today');
        return;
    }

    let modelToUse;
    if (useFastModel) {
        modelToUse = availableModel;
    } else {
        // Use a more capable model for complex queries
        modelToUse = 'llama-3.1-405b-reasoning-ultra';
    }

    // If the complex model isn't available in today's pool, fall back to available model
    if (!useFastModel && availableModel !== modelToUse) {
        console.log('[Router] Complex model not in daily pool, using available:', availableModel);
        modelToUse = availableModel;
    }

    log.q(`Groq Q:`, transcription.substring(0, 80) + (transcription.length > 80 ? '...' : ''));
    console.log(`Sending to Groq (${modelToUse}, fast=${useFastModel}):`, transcription.substring(0, 100) + '...');
    log.info(`Groq TPM: ~10933/12000 (rate limited at this limit)`);

    // Mark as processing to prevent turnComplete from clearing state prematurely
    isProcessingResponse = true;
    processingStartTime = Date.now(); // Track when we started processing

    // Clear previous response and show processing status - protect against bad timestamps
    const transcribeTime = transcriptionStartTime > 0 ? processingStartTime - transcriptionStartTime : 0;
    if (transcribeTime >= 0 && transcribeTime < 60000) {
        log.info(`Transcription took ${Math.round(transcribeTime / 100) / 10}s, starting Groq...`);
    }
    sendToRenderer('new-response', 'Processing...');

    groqConversationHistory.push({
        role: 'user',
        content: transcription.trim(),
    });

    if (groqConversationHistory.length > 20) {
        groqConversationHistory = groqConversationHistory.slice(-20);
    }

    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 30000); // 30s safeguard

        const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
            method: 'POST',
            signal: controller.signal,
            headers: {
                Authorization: `Bearer ${groqApiKey}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model: modelToUse,
                messages: [{ role: 'system', content: currentSystemPrompt || 'You are a helpful assistant.' }, ...groqConversationHistory],
                stream: true,
                temperature: useFastModel ? 0.7 : 0.3, // Lower temp for complex reasoning
                max_tokens: useFastModel ? 1024 : 2048, // More tokens for complex
            }),
        });
        clearTimeout(timeoutId);

        if (!response.ok) {
            const errorText = await response.text();
            console.error('Groq API error:', response.status, errorText);

            // Fallback to Gemini on rate limit (429) or model decommissioned (400)
            if (response.status === 429 || errorText.includes('model_decommissioned')) {
                groqCooldownUntil = Date.now() + 30000; // 30s cooldown
                groqFailedForSession = true; // Skip Groq for rest of session
                lastQuestionTime = 0; // Reset so next question isn't blocked
                console.log('[Fast Mode] Groq rate limited, switching to Gemini...');
                console.log('[Fast Mode] Groq disabled for this session');
                const fallbackTime = Date.now() - processingStartTime;
                log.info(`Groq failed after ${Math.round(fallbackTime / 100) / 10}s, switching to Gemini...`);
                sendToRenderer('update-status', 'Using Gemini...');
                const apiKey = getApiKey();
                if (apiKey) {
                    sendToGemma(transcription, true); // true = isFallback
                    return;
                }
            }

            sendToRenderer('update-status', `Groq error: ${response.status}`);
            return;
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullText = '';
        // false because "Processing..." already sent as new-response
        let isFirst = false;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\n').filter(line => line.trim() !== '');

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = line.slice(6);
                    if (data === '[DONE]') continue;

                    try {
                        const json = JSON.parse(data);
                        const token = json.choices?.[0]?.delta?.content || '';
                        if (token) {
                            fullText += token;
                            const displayText = stripThinkingTags(fullText);
                            if (displayText) {
                                // Only log occasionally to avoid spam
                                if (displayText.length % 200 === 0) {
                                    console.log('[Fast Mode Debug] Response chunk:', displayText.substring(0, 20) + '...');
                                }
                                sendToRenderer(isFirst ? 'new-response' : 'update-response', displayText);
                                isFirst = false;
                            }
                        }
                    } catch (parseError) {
                        // Skip invalid JSON chunks
                    }
                }
            }
        }

        const cleanedResponse = stripThinkingTags(fullText);
        const modelKey = modelToUse.split('/').pop();

        const systemPromptChars = (currentSystemPrompt || 'You are a helpful assistant.').length;
        const historyChars = groqConversationHistory.reduce((sum, msg) => sum + (msg.content || '').length, 0);
        const inputChars = systemPromptChars + historyChars;
        const outputChars = cleanedResponse.length;

        incrementCharUsage('groq', modelKey, inputChars + outputChars);

        if (cleanedResponse) {
            groqConversationHistory.push({
                role: 'assistant',
                content: cleanedResponse,
            });

            // Cache the response (skip partial transcriptions)
            if (!isPartial) {
                setCachedResponse(transcription, cleanedResponse);
            }

            saveConversationTurn(transcription, cleanedResponse);
        }

        log.a(`Groq done (${modelToUse})`);
        console.log(`Groq response completed (${modelToUse})`);
        // Track successful question
        lastQuestionTime = Date.now();
        lastQuestionText = transcription.toLowerCase().trim();
        // Reset timers for next question
        const totalTime = Date.now() - transcriptionStartTime;
        log.info(`Total time: ${Math.round(totalTime / 100) / 10}s (${Math.round(totalTime)}ms)`);
        speechStartTime = 0;
        transcriptionStartTime = 0;
        // Allow next question to be processed immediately
        isProcessingResponse = false;
        sendToRenderer('update-status', 'Listening...');
        // Process next item in queue if any
        processQueue();
    } catch (error) {
        console.error('Error calling Groq API:', error);
        isProcessingResponse = false;
        sendToRenderer('update-status', 'Groq error: ' + error.message);
        processQueue(); // Still trigger next in queue
    }
}

async function sendToGemma(transcription, isFallback = false, retries = 2) {
    const apiKey = getApiKey();
    if (!apiKey) {
        console.log('No Gemini API key configured');
        return;
    }

    // Clean up the transcription text
    const cleanedText = cleanQuestionText(transcription);

    if (!cleanedText || cleanedText.length < 3) {
        console.log('Empty transcription, skipping Gemma');
        return;
    }

    // Rate limiting: skip if too soon (queue instead of skip)
    const now = Date.now();
    const normalized = transcription.toLowerCase().trim();
    if (normalized === lastQuestionText) {
        log.info('Gemini skipping dup');
        return;
    }
    if (now - lastQuestionTime < MIN_QUESTION_INTERVAL) {
        // Queue the transcription to process after cooldown
        const remainingWait = MIN_QUESTION_INTERVAL - (now - lastQuestionTime);
        log.info(`Rate limited (Gemini), queueing for ${Math.round(remainingWait / 100) / 10}s`);

        // Clear any existing queued question (keep the newer one)
        queuedTranscription = { transcription, isFallback: true };

        if (questionQueueTimer) clearTimeout(questionQueueTimer);
        questionQueueTimer = setTimeout(() => {
            if (queuedTranscription) {
                const q = queuedTranscription;
                queuedTranscription = null;
                log.info('Processing queued question after Gemini cooldown');
                sendToGemma(q.transcription, true);
            }
        }, remainingWait);

        return;
    }

    // Check cache first for fast response (using cleaned text for better cache hits)
    const cachedResponse = getCachedResponse(cleanedText);
    if (cachedResponse) {
        const { response } = cachedResponse;
        console.log('[Cache] Using cached response (Gemini)');
        sendToRenderer('new-response', response);

        groqConversationHistory.push({
            role: 'user',
            content: cleanedText,
        });
        groqConversationHistory.push({
            role: 'assistant',
            content: response,
        });

        saveConversationTurn(cleanedText, response);
        lastQuestionTime = Date.now();
        lastQuestionText = cleanedText.toLowerCase();
        speechStartTime = 0;
        transcriptionStartTime = 0;
        processingStartTime = 0;
        isProcessingResponse = false;
        sendToRenderer('update-status', 'Listening... (cached)');
        return;
    }

    log.q(`Gemini Q:`, cleanedText.substring(0, 80) + (cleanedText.length > 80 ? '...' : ''));
    console.log('Sending to Gemma:', cleanedText.substring(0, 100) + '...');

    // Mark as processing to prevent turnComplete from clearing state prematurely
    isProcessingResponse = true;

    // Only show "Processing..." if not a fallback (Groq already showed it)
    if (!isFallback) {
        const transcribeTime = processingStartTime > 0 ? Date.now() - processingStartTime : 0;
        log.info(`Starting Gemini after ${Math.round(transcribeTime / 100) / 10}s since question detected`);
        sendToRenderer('new-response', 'Processing...');
    }

    groqConversationHistory.push({
        role: 'user',
        content: transcription.trim(),
    });

    const trimmedHistory = trimConversationHistoryForGemma(groqConversationHistory, 42000);

    try {
        const ai = new GoogleGenAI({ apiKey: apiKey });

        const messages = trimmedHistory.map(msg => ({
            role: msg.role === 'assistant' ? 'model' : 'user',
            parts: [{ text: msg.content }],
        }));

        const systemPrompt = currentSystemPrompt || 'You are a helpful assistant.';
        const messagesWithSystem = [
            { role: 'user', parts: [{ text: systemPrompt }] },
            { role: 'model', parts: [{ text: 'Understood. I will follow these instructions.' }] },
            ...messages,
        ];

        // Use a timeout to avoid infinite hangs
        const timeoutPromise = new Promise((_, reject) => {
            setTimeout(() => reject(new Error('Gemini request timed out (30s)')), 30000);
        });

        const responsePromise = ai.models.generateContentStream({
            model: 'gemma-3-27b-it',
            contents: messagesWithSystem,
        });

        const response = await Promise.race([responsePromise, timeoutPromise]);

        let fullText = '';
        let isFirst = true;

        for await (const chunk of response) {
            const chunkText = chunk.text;
            if (chunkText) {
                fullText += chunkText;
                sendToRenderer(isFirst ? 'new-response' : 'update-response', fullText);
                isFirst = false;
            }
        }

        const systemPromptChars = (currentSystemPrompt || 'You are a helpful assistant.').length;
        const historyChars = trimmedHistory.reduce((sum, msg) => sum + (msg.content || '').length, 0);
        const inputChars = systemPromptChars + historyChars;
        const outputChars = fullText.length;

        incrementCharUsage('gemini', 'gemma-3-27b-it', inputChars + outputChars);

        if (fullText.trim()) {
            groqConversationHistory.push({
                role: 'assistant',
                content: fullText.trim(),
            });

            if (groqConversationHistory.length > 40) {
                groqConversationHistory = groqConversationHistory.slice(-40);
            }

            saveConversationTurn(transcription, fullText);

            // Cache the response for future questions
            setCachedResponse(transcription, fullText);
        }

        log.a('Gemini done');
        console.log('Gemma response completed');
        // Track successful question
        lastQuestionTime = Date.now();
        lastQuestionText = transcription.toLowerCase().trim();
        // Reset timers for next question
        const totalTime = processingStartTime > 0 ? Date.now() - processingStartTime : 0;
        log.info(`Gemini total time: ${Math.round(totalTime / 100) / 10}s`);
        speechStartTime = 0;
        transcriptionStartTime = 0;
        processingStartTime = 0;
        isProcessingResponse = false;
        sendToRenderer('update-status', 'Listening...');
    } catch (error) {
        console.error('Error calling Gemma API:', error);
        isProcessingResponse = false;

        // Retry on 503 UNAVAILABLE errors
        if (retries > 0 && error.message && error.message.includes('UNAVAILABLE')) {
            const waitTime = (3 - retries) * 2000; // 2s, 4s backoff
            console.log(`[Gemini] 503 error, retrying in ${waitTime / 1000}s (${retries} retries left)`);
            await new Promise(r => setTimeout(r, waitTime));
            return sendToGemma(cleanedText, isFallback, retries - 1);
        }

        sendToRenderer('update-status', 'Gemma error: ' + error.message);
    } finally {
        isProcessingResponse = false;
        // Ensure the queue continues processing regardless of success or failure
        setTimeout(() => processQueue(), 100);
        // Only clear buffers if no retries left (final attempt)
        if (retries === 0 || retries === undefined) {
            clearTranscriptionBuffers();
        }
    }
}

async function initializeGeminiSession(apiKey, customPrompt = '', profile = 'interview', language = 'en-US', isReconnect = false) {
    if (isInitializingSession) {
        console.log('Session initialization already in progress');
        return false;
    }

    isInitializingSession = true;
    if (!isReconnect) {
        sendToRenderer('session-initializing', true);
    }

    // Store params for reconnection
    if (!isReconnect) {
        sessionParams = { apiKey, customPrompt, profile, language };
        reconnectAttempts = 0;
    }

    const client = new GoogleGenAI({
        vertexai: false,
        apiKey: apiKey,
        httpOptions: { apiVersion: 'v1alpha' },
    });

    // Get enabled tools first to determine Google Search status
    const enabledTools = await getEnabledTools();
    const googleSearchEnabled = enabledTools.some(tool => tool.googleSearch);

    const systemPrompt = getSystemPrompt(profile, customPrompt, googleSearchEnabled);
    currentSystemPrompt = systemPrompt; // Store for Groq

    // Initialize new conversation session only on first connect
    if (!isReconnect) {
        initializeNewSession(profile, customPrompt);
    }

    try {
        const session = await client.live.connect({
            model: 'gemini-2.5-flash-native-audio-preview-09-2025',
            callbacks: {
                onopen: function () {
                    sendToRenderer('update-status', 'Live session connected');
                    // Reset state for new session
                    responseProcessedForCurrentTurn = false;
                    lastProcessedTranscription = '';
                    pendingTranscription = '';
                    pendingTranscriptionTimestamp = 0;
                    isProcessingResponse = false;
                    if (pauseTimer) {
                        clearTimeout(pauseTimer);
                        pauseTimer = null;
                    }
                },
                onmessage: function (message) {
                    const serverContent = message?.serverContent;
                    if (!serverContent) return;

                    const summary = summarizeLiveServerMessage(message);
                    if (summary) {
                        console.log('[Gemini]', summary);
                    }
                    // Handle input transcription (what was spoken)
                    let newInputText = '';
                    if (serverContent.inputTranscription?.results) {
                        const transcript = formatSpeakerResults(serverContent.inputTranscription.results);
                        currentTranscription = mergeTranscripts(currentTranscription, transcript);
                        newInputText = transcript;
                    } else if (serverContent.inputTranscription?.text) {
                        const text = serverContent.inputTranscription.text;
                        if (text.trim() !== '') {
                            currentTranscription = mergeTranscripts(currentTranscription, text);
                            newInputText = text;

                            if (speechStartTime === 0) {
                                speechStartTime = Date.now();
                                transcriptionStartTime = speechStartTime;
                            }

                            // REAL-TIME HEARTBEAT - Terminal Feedback
                            // Use a newline-aware format to avoid overwriting complexity/intent logs
                            console.log(`[HEARING] "${text.trim().substring(0, 60)}${text.trim().length > 60 ? '...' : ''}"`);

                            // Debounced Intent Logging
                            const now = Date.now();
                            if (now - lastIntentLogTime > INTENT_LOG_DEBOUNCE) {
                                const intent = analyzeQueryComplexity(text);
                                const isComplex = intent.complexScore > intent.simpleScore;
                                console.log(`[Intent Analysis] Text: "${text.substring(0, 30)}..." | ${isComplex ? 'COMPLEX' : 'SIMPLE'}`);
                                lastIntentLogTime = now;
                            }
                        }
                    }

                    // Handle turn completion - final word in a sequence
                    if (serverContent.turnComplete) {
                        const textToProcess = pendingTranscription || currentTranscription;
                        if (textToProcess.trim().length > 10) {
                            processTranscription(textToProcess);
                        } else {
                            // Turn ended without enough content, reset
                            isProcessingResponse = false;
                            clearTranscriptionBuffers();
                            speechStartTime = 0;
                        }
                        return;
                    }

                    // Fast Mode Logic: wait for complete question before processing
                    const inputText = serverContent.inputTranscription?.text;
                    if (!inputText || inputText.trim().length === 0) return;

                    // DEBUG: Log state
                    if (inputText.length > 5) {
                        console.log('[Fast Mode Debug]', {
                            input: inputText.substring(0, 20),
                            processing: isProcessingResponse,
                        });
                    }

                    // Check if question is complete via heuristic
                    const accumulatedText = (currentTranscription + ' ' + inputText).trim();
                    const isComplete = isQuestionComplete(accumulatedText);
                    const wordCount = accumulatedText.split(/\s+/).length;

                    if (isComplete || wordCount >= 10) {
                        processTranscription(accumulatedText);
                    } else {
                        // Not complete yet, set/update silence timer
                        pendingTranscription = accumulatedText;
                        pendingTranscriptionTimestamp = Date.now();

                        if (pauseTimer) clearTimeout(pauseTimer);
                        pauseTimer = setTimeout(() => {
                            if (!isProcessingResponse && pendingTranscription.trim().length > 15) {
                                // Slightly longer wait for potential continuation (2s total)
                                processTranscription(pendingTranscription);
                            }
                        }, 2000);
                    }
                },
                onerror: function (e) {
                    console.log('Session error:', e.message);
                    sendToRenderer('update-status', 'Error: ' + e.message);
                },
                onclose: function (e) {
                    console.log('Session closed:', describeCloseReason(e));

                    // Don't reconnect if user intentionally closed
                    if (isUserClosing) {
                        isUserClosing = false;
                        sendToRenderer('update-status', 'Session closed');
                        return;
                    }

                    // Attempt reconnection
                    if (sessionParams && reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
                        attemptReconnect();
                    } else {
                        sendToRenderer('update-status', 'Session closed');
                    }
                },
            },
            config: {
                responseModalities: [Modality.AUDIO],
                proactivity: { proactiveAudio: true },
                outputAudioTranscription: {},
                tools: enabledTools,
                // Enable speaker diarization
                inputAudioTranscription: {
                    enableSpeakerDiarization: true,
                    minSpeakerCount: 2,
                    maxSpeakerCount: 2,
                },
                contextWindowCompression: { slidingWindow: {} },
                speechConfig: { languageCode: language },
                systemInstruction: {
                    parts: [{ text: systemPrompt }],
                },
            },
        });

        isInitializingSession = false;
        if (!isReconnect) {
            sendToRenderer('session-initializing', false);
        }
        return session;
    } catch (error) {
        console.error('Failed to initialize Gemini session:', error);
        isInitializingSession = false;
        if (!isReconnect) {
            sendToRenderer('session-initializing', false);
        }
        return null;
    }
}

async function attemptReconnect() {
    reconnectAttempts++;
    console.log(`Reconnection attempt ${reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS}`);

    // Clear stale buffers
    messageBuffer = '';
    currentTranscription = '';
    // Don't reset groqConversationHistory to preserve context across reconnects

    sendToRenderer('update-status', `Reconnecting... (${reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS})`);

    // Wait before attempting
    await new Promise(resolve => setTimeout(resolve, RECONNECT_DELAY));

    try {
        const session = await initializeGeminiSession(
            sessionParams.apiKey,
            sessionParams.customPrompt,
            sessionParams.profile,
            sessionParams.language,
            true // isReconnect
        );

        if (session && global.geminiSessionRef) {
            global.geminiSessionRef.current = session;

            // Restore context from conversation history via text message
            const contextMessage = buildContextMessage();
            if (contextMessage) {
                try {
                    console.log('Restoring conversation context...');
                    await session.sendRealtimeInput({ text: contextMessage });
                } catch (contextError) {
                    console.error('Failed to restore context:', contextError);
                    // Continue without context - better than failing
                }
            }

            // Don't reset reconnectAttempts here - let it reset on next fresh session
            sendToRenderer('update-status', 'Reconnected! Listening...');
            console.log('Session reconnected successfully');
            return true;
        }
    } catch (error) {
        console.error(`Reconnection attempt ${reconnectAttempts} failed:`, error);
    }

    // If we still have attempts left, try again
    if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
        return attemptReconnect();
    }

    // Max attempts reached - notify frontend
    console.log('Max reconnection attempts reached');
    sendToRenderer('reconnect-failed', {
        message: 'Tried 3 times to reconnect. Must be upstream/network issues. Try restarting or download updated app from site.',
    });
    sessionParams = null;
    return false;
}

function killExistingSystemAudioDump() {
    return new Promise(resolve => {
        console.log('Checking for existing SystemAudioDump processes...');

        // Kill any existing SystemAudioDump processes
        const killProc = spawn('pkill', ['-f', 'SystemAudioDump'], {
            stdio: 'ignore',
        });

        killProc.on('close', code => {
            if (code === 0) {
                console.log('Killed existing SystemAudioDump processes');
            } else {
                console.log('No existing SystemAudioDump processes found');
            }
            resolve();
        });

        killProc.on('error', err => {
            console.log('Error checking for existing processes (this is normal):', err.message);
            resolve();
        });

        // Timeout after 2 seconds
        setTimeout(() => {
            killProc.kill();
            resolve();
        }, 2000);
    });
}

async function startMacOSAudioCapture(geminiSessionRef) {
    if (process.platform !== 'darwin') return false;

    // Kill any existing SystemAudioDump processes first
    await killExistingSystemAudioDump();

    console.log('Starting macOS audio capture with SystemAudioDump...');

    const { app } = require('electron');
    const path = require('path');

    let systemAudioPath;
    if (app.isPackaged) {
        systemAudioPath = path.join(process.resourcesPath, 'SystemAudioDump');
    } else {
        systemAudioPath = path.join(__dirname, '../assets', 'SystemAudioDump');
    }

    console.log('SystemAudioDump path:', systemAudioPath);

    const spawnOptions = {
        stdio: ['ignore', 'pipe', 'pipe'],
        env: {
            ...process.env,
        },
    };

    systemAudioProc = spawn(systemAudioPath, [], spawnOptions);

    if (!systemAudioProc.pid) {
        console.error('Failed to start SystemAudioDump');
        return false;
    }

    console.log('SystemAudioDump started with PID:', systemAudioProc.pid);

    const CHUNK_DURATION = 0.1;
    const SAMPLE_RATE = 24000;
    const BYTES_PER_SAMPLE = 2;
    const CHANNELS = 2;
    const CHUNK_SIZE = SAMPLE_RATE * BYTES_PER_SAMPLE * CHANNELS * CHUNK_DURATION;

    let audioBuffer = Buffer.alloc(0);

    systemAudioProc.stdout.on('data', data => {
        audioBuffer = Buffer.concat([audioBuffer, data]);

        while (audioBuffer.length >= CHUNK_SIZE) {
            const chunk = audioBuffer.slice(0, CHUNK_SIZE);
            audioBuffer = audioBuffer.slice(CHUNK_SIZE);

            const monoChunk = CHANNELS === 2 ? convertStereoToMono(chunk) : chunk;

            if (currentProviderMode === 'cloud') {
                sendCloudAudio(monoChunk);
            } else if (currentProviderMode === 'local') {
                getLocalAi().processLocalAudio(monoChunk);
            } else {
                const base64Data = monoChunk.toString('base64');
                sendAudioToGemini(base64Data, geminiSessionRef);
            }

            if (process.env.DEBUG_AUDIO) {
                console.log(`Processed audio chunk: ${chunk.length} bytes`);
                saveDebugAudio(monoChunk, 'system_audio');
            }
        }

        const maxBufferSize = SAMPLE_RATE * BYTES_PER_SAMPLE * 1;
        if (audioBuffer.length > maxBufferSize) {
            audioBuffer = audioBuffer.slice(-maxBufferSize);
        }
    });

    systemAudioProc.stderr.on('data', data => {
        console.error('SystemAudioDump stderr:', data.toString());
    });

    systemAudioProc.on('close', code => {
        console.log('SystemAudioDump process closed with code:', code);
        systemAudioProc = null;
    });

    systemAudioProc.on('error', err => {
        console.error('SystemAudioDump process error:', err);
        systemAudioProc = null;
    });

    return true;
}

function convertStereoToMono(stereoBuffer) {
    const samples = stereoBuffer.length / 4;
    const monoBuffer = Buffer.alloc(samples * 2);

    for (let i = 0; i < samples; i++) {
        const leftSample = stereoBuffer.readInt16LE(i * 4);
        monoBuffer.writeInt16LE(leftSample, i * 2);
    }

    return monoBuffer;
}

function stopMacOSAudioCapture() {
    if (systemAudioProc) {
        console.log('Stopping SystemAudioDump...');
        systemAudioProc.kill('SIGTERM');
        systemAudioProc = null;
    }
}

async function sendAudioToGemini(base64Data, geminiSessionRef) {
    if (!geminiSessionRef.current) return;

    try {
        await geminiSessionRef.current.sendRealtimeInput({
            audio: {
                data: base64Data,
                mimeType: 'audio/pcm;rate=24000',
            },
        });
    } catch (error) {
        console.error('Error sending audio to Gemini:', error);
    }
}

async function sendImageToGeminiHttp(base64Data, prompt) {
    // Get available model based on rate limits
    const model = getAvailableModel();

    const apiKey = getApiKey();
    if (!apiKey) {
        return { success: false, error: 'No API key configured' };
    }

    try {
        const ai = new GoogleGenAI({ apiKey: apiKey });

        const contents = [
            {
                inlineData: {
                    mimeType: 'image/jpeg',
                    data: base64Data,
                },
            },
            { text: prompt },
        ];

        console.log(`Sending image to ${model} (streaming)...`);
        const response = await ai.models.generateContentStream({
            model: model,
            contents: contents,
        });

        // Increment count after successful call
        incrementLimitCount(model);

        // Stream the response
        let fullText = '';
        let isFirst = true;
        for await (const chunk of response) {
            const chunkText = chunk.text;
            if (chunkText) {
                fullText += chunkText;
                // Send to renderer - new response for first chunk, update for subsequent
                sendToRenderer(isFirst ? 'new-response' : 'update-response', fullText);
                isFirst = false;
            }
        }

        console.log(`Image response completed from ${model}`);

        // Save screen analysis to history
        saveScreenAnalysis(prompt, fullText, model);

        return { success: true, text: fullText, model: model };
    } catch (error) {
        console.error('Error sending image to Gemini HTTP:', error);
        return { success: false, error: error.message };
    }
}

function setupGeminiIpcHandlers(geminiSessionRef) {
    // Store the geminiSessionRef globally for reconnection access
    global.geminiSessionRef = geminiSessionRef;

    ipcMain.handle('initialize-cloud', async (event, token, profile, userContext) => {
        try {
            currentProviderMode = 'cloud';
            initializeNewSession(profile);
            setOnTurnComplete((transcription, response) => {
                saveConversationTurn(transcription, response);
            });
            sendToRenderer('session-initializing', true);
            await connectCloud(token, profile, userContext);
            sendToRenderer('session-initializing', false);
            return true;
        } catch (err) {
            console.error('[Cloud] Init error:', err);
            currentProviderMode = 'byok';
            sendToRenderer('session-initializing', false);
            return false;
        }
    });

    ipcMain.handle('initialize-gemini', async (event, apiKey, customPrompt, profile = 'interview', language = 'en-US') => {
        currentProviderMode = 'byok';
        const session = await initializeGeminiSession(apiKey, customPrompt, profile, language);
        if (session) {
            geminiSessionRef.current = session;
            return true;
        }
        return false;
    });

    ipcMain.handle('initialize-local', async (event, ollamaHost, ollamaModel, whisperModel, profile, customPrompt) => {
        currentProviderMode = 'local';
        const success = await getLocalAi().initializeLocalSession(ollamaHost, ollamaModel, whisperModel, profile, customPrompt);
        if (!success) {
            currentProviderMode = 'byok';
        }
        return success;
    });

    ipcMain.handle('send-audio-content', async (event, { data, mimeType }) => {
        if (currentProviderMode === 'cloud') {
            try {
                const pcmBuffer = Buffer.from(data, 'base64');
                sendCloudAudio(pcmBuffer);
                return { success: true };
            } catch (error) {
                console.error('Error sending cloud audio:', error);
                return { success: false, error: error.message };
            }
        }
        if (currentProviderMode === 'local') {
            try {
                const pcmBuffer = Buffer.from(data, 'base64');
                getLocalAi().processLocalAudio(pcmBuffer);
                return { success: true };
            } catch (error) {
                console.error('Error sending local audio:', error);
                return { success: false, error: error.message };
            }
        }
        if (!geminiSessionRef.current) return { success: false, error: 'No active Gemini session' };
        try {
            await geminiSessionRef.current.sendRealtimeInput({
                audio: { data: data, mimeType: mimeType },
            });
            return { success: true };
        } catch (error) {
            console.error('Error sending system audio:', error);
            return { success: false, error: error.message };
        }
    });

    // Handle microphone audio on a separate channel
    ipcMain.handle('send-mic-audio-content', async (event, { data, mimeType }) => {
        if (currentProviderMode === 'cloud') {
            try {
                const pcmBuffer = Buffer.from(data, 'base64');
                sendCloudAudio(pcmBuffer);
                return { success: true };
            } catch (error) {
                console.error('Error sending cloud mic audio:', error);
                return { success: false, error: error.message };
            }
        }
        if (currentProviderMode === 'local') {
            try {
                const pcmBuffer = Buffer.from(data, 'base64');
                getLocalAi().processLocalAudio(pcmBuffer);
                return { success: true };
            } catch (error) {
                console.error('Error sending local mic audio:', error);
                return { success: false, error: error.message };
            }
        }
        if (!geminiSessionRef.current) return { success: false, error: 'No active Gemini session' };
        try {
            await geminiSessionRef.current.sendRealtimeInput({
                audio: { data: data, mimeType: mimeType },
            });
            return { success: true };
        } catch (error) {
            console.error('Error sending mic audio:', error);
            return { success: false, error: error.message };
        }
    });

    ipcMain.handle('send-image-content', async (event, { data, prompt }) => {
        try {
            if (!data || typeof data !== 'string') {
                console.error('Invalid image data received');
                return { success: false, error: 'Invalid image data' };
            }

            const buffer = Buffer.from(data, 'base64');

            if (buffer.length < 1000) {
                console.error(`Image buffer too small: ${buffer.length} bytes`);
                return { success: false, error: 'Image buffer too small' };
            }

            if (currentProviderMode === 'cloud') {
                const sent = sendCloudImage(data);
                if (!sent) {
                    return { success: false, error: 'Cloud connection not active' };
                }
                return { success: true, model: 'cloud' };
            }

            if (currentProviderMode === 'local') {
                const result = await getLocalAi().sendLocalImage(data, prompt);
                return result;
            }

            // Use HTTP API instead of realtime session
            const result = await sendImageToGeminiHttp(data, prompt);
            return result;
        } catch (error) {
            console.error('Error sending image:', error);
            return { success: false, error: error.message };
        }
    });

    ipcMain.handle('send-text-message', async (event, text) => {
        if (!text || typeof text !== 'string' || text.trim().length === 0) {
            return { success: false, error: 'Invalid text message' };
        }

        if (currentProviderMode === 'cloud') {
            try {
                console.log('Sending text to cloud:', text);
                sendCloudText(text.trim());
                return { success: true };
            } catch (error) {
                console.error('Error sending cloud text:', error);
                return { success: false, error: error.message };
            }
        }

        if (currentProviderMode === 'local') {
            try {
                console.log('Sending text to local Ollama:', text);
                return await getLocalAi().sendLocalText(text.trim());
            } catch (error) {
                console.error('Error sending local text:', error);
                return { success: false, error: error.message };
            }
        }

        if (!geminiSessionRef.current) return { success: false, error: 'No active Gemini session' };

        try {
            console.log('Sending text message:', text);

            if (hasGroqKey()) {
                sendToGroq(text.trim());
            } else {
                sendToGemma(text.trim());
            }

            await geminiSessionRef.current.sendRealtimeInput({ text: text.trim() });
            return { success: true };
        } catch (error) {
            console.error('Error sending text:', error);
            return { success: false, error: error.message };
        }
    });

    ipcMain.handle('start-macos-audio', async event => {
        if (process.platform !== 'darwin') {
            return {
                success: false,
                error: 'macOS audio capture only available on macOS',
            };
        }

        try {
            const success = await startMacOSAudioCapture(geminiSessionRef);
            return { success };
        } catch (error) {
            console.error('Error starting macOS audio capture:', error);
            return { success: false, error: error.message };
        }
    });

    ipcMain.handle('stop-macos-audio', async event => {
        try {
            stopMacOSAudioCapture();
            return { success: true };
        } catch (error) {
            console.error('Error stopping macOS audio capture:', error);
            return { success: false, error: error.message };
        }
    });

    ipcMain.handle('close-session', async event => {
        try {
            stopMacOSAudioCapture();

            if (currentProviderMode === 'cloud') {
                closeCloud();
                currentProviderMode = 'byok';
                return { success: true };
            }

            if (currentProviderMode === 'local') {
                getLocalAi().closeLocalSession();
                currentProviderMode = 'byok';
                return { success: true };
            }

            // Set flag to prevent reconnection attempts
            isUserClosing = true;
            sessionParams = null;

            // Cleanup session
            if (geminiSessionRef.current) {
                await geminiSessionRef.current.close();
                geminiSessionRef.current = null;
            }

            return { success: true };
        } catch (error) {
            console.error('Error closing session:', error);
            return { success: false, error: error.message };
        }
    });

    // Conversation history IPC handlers
    ipcMain.handle('get-current-session', async event => {
        try {
            return { success: true, data: getCurrentSessionData() };
        } catch (error) {
            console.error('Error getting current session:', error);
            return { success: false, error: error.message };
        }
    });

    ipcMain.handle('start-new-session', async event => {
        try {
            initializeNewSession();
            return { success: true, sessionId: currentSessionId };
        } catch (error) {
            console.error('Error starting new session:', error);
            return { success: false, error: error.message };
        }
    });

    ipcMain.handle('update-google-search-setting', async (event, enabled) => {
        try {
            console.log('Google Search setting updated to:', enabled);
            // The setting is already saved in localStorage by the renderer
            // This is just for logging/confirmation
            return { success: true };
        } catch (error) {
            console.error('Error updating Google Search setting:', error);
            return { success: false, error: error.message };
        }
    });

    ipcMain.handle('get-available-tools', async () => {
        return await getEnabledTools();
    });

    ipcMain.on('clear-conversation', () => {
        groqConversationHistory = [];
        lastQuestionText = '';
        lastQuestionTime = 0;
        console.log('Conversation history cleared');
    });

    // Fast Response Mode - directly process text without Gemini Live
    ipcMain.handle('fast-response', async (event, text) => {
        try {
            if (!text || typeof text !== 'string' || text.trim().length === 0) {
                return { success: false, error: 'Invalid text' };
            }

            sendToRenderer('update-status', 'Processing...');

            // Check cache first
            const cachedResponse = getCachedResponse(text);
            if (cachedResponse) {
                const { response } = cachedResponse;
                sendToRenderer('new-response', response);
                sendToRenderer('update-status', 'Listening... (cached)');
                return { success: true, cached: true };
            }

            // Send to Groq directly
            if (hasGroqKey()) {
                await sendToGroq(text.trim(), false);
                return { success: true, cached: false };
            } else {
                return { success: false, error: 'No Groq API key configured' };
            }
        } catch (error) {
            console.error('Error in fast response:', error);
            return { success: false, error: error.message };
        }
    });
}

// Emergency removal of the last processed question from cache
function emergencyEraseLastCache() {
    if (lastProcessedTranscription) {
        const { deleteCachedResponse } = require('./cache');
        deleteCachedResponse(lastProcessedTranscription);
        console.log(`[Emergency] Erased last question from cache: "${lastProcessedTranscription.substring(0, 30)}..."`);
        lastProcessedTranscription = '';
    }
}

module.exports = {
    initializeGeminiSession,
    getEnabledTools,
    getStoredSetting,
    sendToRenderer,
    initializeNewSession,
    saveConversationTurn,
    getCurrentSessionData,
    killExistingSystemAudioDump,
    startMacOSAudioCapture,
    convertStereoToMono,
    stopMacOSAudioCapture,
    sendAudioToGemini,
    sendImageToGeminiHttp,
    setupGeminiIpcHandlers,
    formatSpeakerResults,
    emergencyEraseLastCache,
};
