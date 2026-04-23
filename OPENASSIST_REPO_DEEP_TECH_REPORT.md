# OpenAssist — Deep Technical Report & Low‑Latency Roadmap (2026-04-22)

Scope: review of the Python app under `core/`, `ai/`, `capture/`, `ui/`, `modes/`, `stealth/`, `utils/` plus the Electron reference implementation under `cheating/`.

This document is written to be actionable for a **low-latency, “never feels stuck”** assistant: fast first token, predictable fallback, minimal UI drift, and clear provenance.

Method (honesty note): this report is based on repo-wide greps, targeted reading of the core hot paths, and UI wiring extraction. It intentionally excludes `venv/` and does not claim literal manual review of every dependency line.

---

## 1) Repo Snapshot (What’s here)

### Entrypoints
- `main.py` → constructs `Config` + `OpenAssistApp` and starts Qt + background warmups.

### Primary runtime graph (Python app)
- `core/app.py` (`OpenAssistApp`) orchestrates:
  - UI: `ui/overlay.py` + `ui/mini_overlay.py`
  - Context: `core/nexus.py` (`ContextNexus`)
  - Capture: `capture/audio.py`, `capture/screen.py` (+ `capture/ocr.py`)
  - AI: `ai/engine.py` (`AIEngine`) + provider registry `ai/providers/*`
  - Modes: `modes/*` via `modes/ModeManager`
  - Hotkeys: `core/hotkeys.py`
  - Stealth: `stealth/*`
  - History: `ai/history.py` (encrypted persistence)

### Electron reference app (`cheating/`)
An Electron/Lit app with:
- a **router** for “fast vs complex” (`cheating/utils/router.js`)
- a **question cache** with fuzzy matching (`cheating/utils/cache.js`)
- prompt packs (`cheating/utils/prompts.js`)
- aggressive hot-path caching of preferences (renderer caches)
- screen/audio capture via `getDisplayMedia` + offscreen canvas (`cheating/utils/renderer.js`)

---

## 2) What We’ve Built (Current State)

### Session lifecycle + orchestration
- Background asyncio loop thread for AI work and non-Qt tasks (`core/app.py`).
- Session start/stop, history archiving, and “emergency erase / factory reset” flow.

### Context model (“Nexus”)
- `core/nexus.py` buffers window title, OCR text, and audio transcript events; builds snapshots with:
  - `latest_ocr`, `ocr_context`
  - `recent_audio`, `full_audio_history`
  - `active_window`, timestamp, TTL cleanup

### Screen capture + OCR pipeline (latency aware)
- Debounced periodic capture (`capture/screen.py:capture`) with:
  - Smart-crop focus region, hysteresis, change detection.
  - Separate “analysis capture” quality knobs (`capture.screen.analysis_quality` and JPEG downscale/compress).
- Manual “Analyze Screen” uses:
  - JPEG bytes capture with smart crop (`capture_image_bytes(for_analysis=True)`).
  - OCR extraction on same screenshot in parallel.

### Audio pipeline
- VAD loop with transcription offloaded to a single-worker pool (`capture/audio.py`).
- Faster-Whisper with GPU auto-detect (CUDA float16) and language hint.

### AI engine
- Provider init + routing (`ai/engine.py` + `ai/router.py`).
- Provider cooldown on transient failures (429/503/timeouts) using `BaseProvider.disable()` and `check_rate()` gates.
- Streaming, fallback chain, and race modes:
  - Text “race mode” for non-streaming fastest-success.
  - Vision “race mode” with **staggered launch** and clean failure bubble-up.
- Vision tier compatibility:
  - Best-effort `tier="vision"` kwarg with fallback for older providers/test doubles.

### Provider hot path work
- Ollama: persistent `aiohttp.ClientSession` reuse (`ai/providers/ollama_provider.py`).
- Ollama: auto-downgrade for RAM/model-load failures in vision calls (tries locally available smaller vision models and persists the working one).

### UX improvements already integrated
- Markdown wobble mitigation (stop periodic markdown rerender on code fences) in overlays.
- Status/tooltips show stage timings and provenance in overlay.
- History feed includes provenance strip + fallback/race labeling.
- Settings hot-apply now re-applies mode profile to the live runtime (ModeManager, VAD sensitivity, etc.).
- Hardware tab layout improvements: no horizontal scroll, wrapped checkbox labels.

---

## 3) Where We Still Lack (High‑Signal Gaps)

### A) “Never stuck” invariants (user-perceived correctness)
Target behavior:
1) Every user-visible action either completes or shows a clear failure within a bounded time.
2) When failing, the UI must reflect the final outcome (no “Analyzing…” forever).

Current risks:
- Multiple async flows can emit final UI signals out-of-order (especially with fallbacks). We’ve fixed a key screen-analysis pending bug, but more edge cases remain (see §6).

### B) Latency: screen analysis still pays a 3s OCR gate
Manual Analyze Screen currently awaits OCR (`await wait_for(ocr_task, timeout=3.0)`) to extract “task” text before calling vision.
- This improves prompt quality but increases perceived latency.
- When cloud vision is overloaded (503), the *extra 3s* makes it feel worse.

### C) No “fast-path cache” for repeated questions / repeated screen states
The Electron app uses a question cache with fuzzy matching and TTL (`cheating/utils/cache.js`).
Python app currently lacks a comparable cache for:
- repeated quick answers
- repeated “Analyze Screen” on same content
- repeated audio “same question” patterns

### D) QSS parse warnings hide real issues
Logs show `Could not parse stylesheet of object QPushButton(...)`.
Root cause is usually unsupported Qt CSS properties (notably `letter-spacing` appears throughout UI style strings).
These warnings create log noise and make it harder to debug actual faults.

### E) Prompt system drift / duplication
`ai/prompts.py` has both `SYSTEMS` and `PROMPT_PACKS` plus context ranking logic; the engine also has mode-aware quick-answer logic and screen-analysis origin logic.
This increases “prompt drift” risk (small changes affect behavior inconsistently across origins).

### F) Dead/duplicate UI code
- `ui/setup_wizard.py` exists alongside the active `ui/onboarding_wizard.py`. The overlay uses onboarding, not setup wizard.
This is a maintenance hazard: “fix in one place, bug persists in the other.”

---

## 4) Latency & Hot Path Analysis

### 4.1 Text query path (manual/speech)
Path:
`Overlay/MiniOverlay` → `OpenAssistApp.generate_response()` → `_process_ai()` → `AIEngine.generate_response()`

Latency contributors:
- ASR refinement (optional)
- RAG refine/prefetch (if enabled)
- Provider TTFT + streaming

Current wins:
- Immediate UI “QUERY + Thinking…” rendering before first token.
- Stage timing tracking plumbed into status tooltips.

Remaining low-latency ideas:
- Add a tiny **response cache** (exact+fuzzy) for short questions in “quick” and “speech” origins.
- Add “first-token timeout” and pre-emptive fallback when streaming doesn’t begin quickly (cloud stalls).

### 4.2 Vision path (“Analyze Screen”)
Path:
`Overlay.AnalyzeScreen` → `OpenAssistApp.analyze_current_screen()` →
`capture_image_bytes(for_analysis=True)` + `ocr_task` → `AIEngine.analyze_image_response()`

Current wins:
- JPEG downscale and quality tier for analysis.
- Smart crop reduces pixels/bytes.
- Vision provider cooldown + fallback chain.
- Staggered race reduces median latency while avoiding hammering.
- On full failure: explicit toast + transcript + OCR fallback (no “pretend success”).

Remaining low-latency ideas (highest ROI):
1) **Start vision immediately**, don’t await OCR for 3 seconds.
   - Use OCR-derived “task” only if it’s ready fast (e.g. 250–500ms), otherwise proceed with generic “identify task from screenshot and complete it”.
   - Keep OCR running in parallel; if vision fails, reuse OCR result.
2) **Budgeted vision**: set an overall wall-clock budget (e.g. 8s) then force OCR fallback.
3) **Image retry policy**:
   - If provider returns “too large / timeout”, auto-downscale and retry once.

### 4.3 Provider health monitoring overhead
`AIEngine.poll_provider_health()` can call `check_availability()` on providers periodically.
- This can be noisy/expensive for network providers.
- The current logic avoids overwriting cooldown/rate-limited state (recent fix), but a more conservative approach is recommended:
  - poll health only for providers that advertise a lightweight health check
  - or poll less frequently when a session is idle

---

## 5) Robustness / Correctness Risk Register (Bugs & Smells)

### Exception swallowing / “best effort” blocks
There are many broad `except Exception` blocks across `ai/engine.py`, `ai/rag.py`, `ai/history.py`, and UI modules.
Tradeoff: stability vs debuggability.

Actionable improvement:
- Standardize a rule: **swallow only when we also log at debug** (and attach `exc_info=True` when useful).
- For user-impacting failures (e.g. vision exhausted), always emit an explicit UI signal.

### Encoding drift / mojibake
Multiple files contain mojibake sequences (e.g. “—”, “ðŸ…”).
This impacts:
- logs readability
- QSS parsing (if style strings get corrupted)

Actionable improvement:
- Run a one-time repo encoding normalization (UTF-8) and add CI/lint guard.

### API key handling / secrets drift
`Config` stores keys in encrypted storage but `config.yaml` can still end up with plaintext keys (depending on user flows).
Actionable:
- ensure `config.yaml` never persists real keys (write placeholders only) and keep keys only in secure storage.
- add `.gitignore` guard / pre-commit check (if this repo is committed anywhere).

---

## 6) UI Wiring & Clickables Audit (Python UI)

This is a “sheet style” mapping of user actions → handler → downstream behavior.

### Overlay header
| UI element | File | Handler | Notes |
|---|---|---|---|
| Audio icon button | `ui/overlay.py` | `app.toggle_audio` | Updates `AppState.is_muted` and audio pipeline. |
| End session | `ui/overlay.py` | `app.end_session` | Stops session and resets UI state. |
| Timeline | `ui/overlay.py` | `_show_timeline` | Shows Nexus timeline view; Clear action handled inside timeline view. |
| History | `ui/overlay.py` | `_show_history` | Shows history feed. |
| Settings | `ui/overlay.py` | `_show_settings` | Shows settings. |
| Close | `ui/overlay.py` | `hide()` | Hides overlay. |

### Overlay chat input bar
| UI element | File | Handler | Notes |
|---|---|---|---|
| Enter in input | `ui/overlay.py` | `_send` → `OpenAssistApp.generate_response` | Immediate “QUERY + Thinking…” HTML injected. |
| Analyze Screen button | `ui/overlay.py` | `_analyze_screen` → `OpenAssistApp.analyze_current_screen` | Manual screenshot flow; now explicitly reports vision failure and falls back. |

### Standby view
| UI element | File | Handler | Notes |
|---|---|---|---|
| Mode buttons | `ui/standby_view.py` | emits `mode_selected` → `OpenAssistApp.switch_mode` | Highlights mode; mode change now also applied on Settings hot-apply. |
| Audio source buttons | `ui/standby_view.py` | emits `audio_source_changed` → `_on_audio_source_ui_change` | Restarts audio pipeline. |
| Start session | `ui/standby_view.py` | emits `start_clicked` → `OpenAssistApp.start_new_session` | Warmup gating handled by warmup latch. |
| Provider badges | `ui/standby_view.py` | `set_provider_statuses()` | Shows active/cooldown provider list. |

### Mini overlay
| UI element | File | Handler | Notes |
|---|---|---|---|
| Enter in input | `ui/mini_overlay.py` | `_send` → `generate_response` | Similar “Thinking…” UX; includes race hint. |
| Expand/collapse | `ui/mini_overlay.py` | `_toggle_expand` | Adapts height based on rendered markdown. |
| Type response | `ui/mini_overlay.py` | `_type_response` | Injects last response into focused window (stealth risk area). |
| Mode icon double click | `ui/mini_overlay.py` | `_toggle_nano_mode` | UI-only. |

### Settings view
| UI element | File | Handler | Notes |
|---|---|---|---|
| Apply settings | `ui/settings_view.py` | `_save_all` → `OpenAssistApp._apply_settings` | Now also re-applies mode profile via `switch_mode(state.mode)` on hot-apply. |
| Provider test buttons | `ui/settings_view.py` | `_test_pid` | Uses aiohttp per-test; OK. |
| Factory reset | `ui/settings_view.py` | `_factory_reset` → `OpenAssistApp.factory_reset` | Only reset entrypoint remaining (setup-wizard reset removed). |

### History / timeline views
| UI element | File | Handler | Notes |
|---|---|---|---|
| Export MD | `ui/history_feed.py` | `_export_current_session` | Currently swallows write errors silently. |
| Timeline clear | `ui/nexus_timeline.py` | `_clear_log` | Calls `nexus.clear()` and deletes cards idempotently. |

### UI issues observed
- QSS parse warnings likely caused by `letter-spacing` usage in style strings across multiple UI modules.
  - Recommendation: remove `letter-spacing` from QSS and emulate spacing via font choice / padding, or accept warnings but route them to debug logs only.

---

## 7) Cheating App (Electron) — Smart Moves Worth Porting

### A) Smart routing based on complexity
`cheating/utils/router.js` uses keyword heuristics + code-ish regex patterns.
Python has similar heuristics in `ai/engine.py`, but:
- it’s split across complexity detection, mode routing, and provider routing.

Port suggestion:
- Create a single “query classifier” module in Python, returning:
  - `complexity`, `is_coding`, `is_followup`, `confidence`
  - and use it consistently for both provider selection and prompt pack selection.

### B) A real cache for repeated asks
`cheating/utils/cache.js`:
- exact hash + TTL
- fuzzy match on “content words” with stopwords removal
- garbage detection

Port suggestion (P0 latency win):
- Add `ai/cache.py` for short queries:
  - cache on `(mode, normalized_query, maybe last_window_title)` with TTL
  - use fuzzy match only for ASCII short queries
  - bypass cache when screen/audio context changed materially (fingerprint)

### C) Preference caching for hot paths
Electron caches preferences to avoid repeated async IPC.
Python analogue:
- `Config.get()` is cheap, but avoid repeated heavy recompute in tight loops.
- Cache “derived configs” (provider lists, thresholds) in `AIEngine` and invalidate on hot-apply.

---

## 8) Low‑Latency Roadmap (Prioritized)

### Action Plan (Sorted: P0 → P3)

Use this as the execution checklist. Each item is phrased as a “done means…” outcome.

#### P0 — Stability / “Never Stuck”
- [ ] **Vision wall-clock budget + OCR fallback**: screen analysis never waits unbounded; after N seconds, show failure + fall back to OCR/text automatically.
  - Primary files: `ai/engine.py`, `core/app.py`
- [ ] **Start vision immediately (remove 3s OCR gate)**: manual Analyze Screen kicks off vision right away; OCR is best-effort enrichment (≤300–500ms) not a blocker.
  - Primary files: `core/app.py`, `capture/screen.py`
- [ ] **Eliminate QSS parse spam**: remove/replace unsupported Qt CSS (start with `letter-spacing`) so logs are clean and real issues are visible.
  - Primary files: `ui/overlay.py`, `ui/settings_view.py`, `ui/standby_view.py`, `ui/onboarding_wizard.py`
- [ ] **Provider “overload storm” hardening**: on 503/429 bursts, immediately cool down + skip provider for a window; avoid retry storms across requests.
  - Primary files: `ai/engine.py`, `ai/providers/base.py`

#### P1 — Latency (TTFT + median)
- [ ] **Short-query cache (exact + conservative fuzzy)**: repeated small questions return instantly when context fingerprint hasn’t changed.
  - Primary files: new `ai/cache.py` + integration in `ai/engine.py`
- [ ] **First-token timeout for streaming**: if no first token within X ms, automatically fail over to the next provider (without waiting for a long timeout).
  - Primary files: `ai/engine.py`
- [ ] **Adaptive provider-health polling**: poll less when idle; don’t call expensive availability checks every cycle.
  - Primary files: `ai/engine.py`, `ai/router.py`
- [ ] **Vision byte-size control**: enforce max resolution/quality per provider; auto-downscale on failure (payload too big / slow).
  - Primary files: `capture/screen.py`, `ai/engine.py`

#### P2 — Product / UX (Signals, Provenance, Controls)
- [ ] **Provenance everywhere**: history + overlay always show which context sources contributed (including OCR-only fallback and quick answers).
  - Primary files: `ai/engine.py`, `core/app.py`, `ui/overlay.py`, `ui/history_feed.py`
- [ ] **Local-only visibility**: when local-only is enabled, show a clear UI indicator (and disable conflicting controls) so users understand routing.
  - Primary files: `ui/settings_view.py`, `ui/standby_view.py`
- [ ] **Cooldown reason surfacing**: show *why* a provider is in cooldown (503 vs 429 vs timeout) in UI badges/tooltips.
  - Primary files: `ai/engine.py`, `ai/providers/base.py`, `ui/standby_view.py`

#### P3 — Maintainability / Cleanup
- [ ] **Remove dead/duplicate UI module(s)**: confirm `ui/setup_wizard.py` is unused and delete it (or wire it intentionally, but only one wizard should exist).
  - Primary files: `ui/overlay.py`, `ui/setup_wizard.py`, `ui/onboarding_wizard.py`
- [ ] **UTF‑8 normalization + guard**: remove mojibake and add a simple check (pre-commit or CI script) to prevent regressions.
  - Primary files: repo-wide (script under `utils/` or `scripts/`)
- [ ] **Prompt pack consolidation**: reduce duplication between `SYSTEMS` and `PROMPT_PACKS`; enforce a single mode prompt path.
  - Primary files: `ai/prompts.py`

---

## 9) Suggested Prompt Improvements (Low‑Latency Oriented)

Goals:
- Minimize tokens, maximize first-token speed.
- Make responses deterministic and “output-first”.

Recommendations:
1) Split prompts into two tiers:
   - “fast response” system prompt (ultra short)
   - “detailed response” system prompt (only when complexity requires it)
2) For screen analysis, default to:
   - “Identify the *single* task from the screenshot and complete it”
   - avoid long “what I see” summaries
3) Add a strict output contract for interview mode:
   - 3–5 bullets max, no paragraphs

---

## Appendix A — Current Known Noisy Logs
- QSS parse warnings (likely `letter-spacing`).
- Torch DataLoader `pin_memory` warning on CPU (suppressed at startup in `main.py`).
