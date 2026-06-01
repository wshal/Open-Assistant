# Repository Audit - Open Assist

Audit date: 2026-06-01

Scope reviewed: application source, UI, capture, AI providers/router/cache, benchmarks, scripts, tests, packaging helpers, and repository configuration. Excluded from code-quality findings: checked-in `venv/`, generated caches, binary fixtures, and image/audio fixture payloads.

Verification performed:

- `python -m compileall -q ai benchmarks capture core knowledge modes scripts stealth tests ui utils main.py build.py` passed.
- `python -m pytest -q` passed: `500 passed` in 20.03s.
- Remaining test-environment warning: Pytest still cannot write `.pytest_cache\v\cache\nodeids` under the checkout, so cache persistence is disabled/noisy even though the suite passes.

Cross-check performed:

- Reviewed `docs/comprehensive_audit_report.md` and verified its claims against the current checkout before merging findings.
- Accepted and added only claims that match current source. Several claims in that report are stale or refer to older/different code: examples include YAML using unsafe `yaml.load` (current code uses `yaml.safe_load`), `build.py` using `shell=True` (current code passes an argument list), missing `@abstractmethod` decorators (current provider base has them), Ollama lacking timeouts (current code has `aiohttp.ClientTimeout`), Mistral missing streaming (current provider implements `generate_stream`), and `Mode.context_weights` using a mutable default (current code uses `default_factory`).

## 1. Auto actions can exfiltrate secrets and execute high-impact commands

Location: `ai/actions.py`, `_INTENT_RULES` lines 78-201, `ActionExecutor.__init__` lines 240-244, `detect()` lines 253-285.

Problem:

`ActionExecutor` defaults to enabled and maps natural language directly to commands including `npm install`, `npm start`, `npm run dev`, `npx vite`, `open file`, and `show environment variables`. This creates a local command-execution and secret-exfiltration surface. A spoken or OCR-derived prompt such as "show environment variables" returns process environment variables to the AI context, including API keys injected by `Config._inject_secrets()`. The blocked-pattern list only rejects a small set of destructive words; it does not address data disclosure, dependency lifecycle scripts, package-manager execution, or malicious project scripts.

Risk:

High. A prompt injection from screen/audio context can trigger local commands, leak credentials, or execute dependency lifecycle scripts inside the configured working directory.

Refactored code:

```python
# ai/actions.py
_SAFE_INTENT_RULES: List[Tuple[str, re.Pattern, Any]] = [
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
        "git_diff_stat",
        re.compile(r"\bgit\s+diff\b|\bwhat.{0,15}diff\b|\bshow\s+(?:the\s+)?diff\b", re.I),
        lambda m, cwd: ["git", "diff", "--stat"],
    ),
    (
        "python_version",
        re.compile(r"\bpython\s+version\b|\bwhich\s+python\b", re.I),
        lambda m, cwd: ["python", "--version"],
    ),
]

class ActionExecutor:
    def __init__(self, config):
        self._config = config
        self._enabled = bool(config.get("ai.actions.enabled", False))
        self._timeout_s = max(1.0, min(float(config.get("ai.actions.timeout_s", 10.0)), 30.0))
        self._cwd = str(Path(config.get("ai.actions.cwd", ".")).resolve())
        self._allow_write_actions = bool(config.get("ai.actions.allow_write_actions", False))

    def detect(self, query: str) -> Optional[Tuple[str, List[str]]]:
        if not self._enabled:
            return None

        q = (query or "").strip()
        if not q:
            return None

        blocked = re.compile(
            r"\b(rm|del|rmdir|rd|format|drop|truncate|shred|wipe|destroy|env|environment|secret|api[_ -]?key|token)\b",
            re.I,
        )
        if blocked.search(q):
            logger.warning("[Actions] blocked unsafe query: %r", q[:120])
            return None

        for intent_label, pattern, cmd_fn in _SAFE_INTENT_RULES:
            match = pattern.search(q)
            if not match:
                continue
            cmd = cmd_fn(match, self._cwd)
            if cmd:
                return intent_label, cmd
        return None
```

## 2. Action timeout does not stop the child process

Location: `ai/actions.py`, `execute()` lines 311-328.

Problem:

`execute()` runs `subprocess.run()` inside `asyncio.to_thread()` and wraps the thread await in `asyncio.wait_for()`. When the timeout fires, the thread keeps running and the child process remains alive because `wait_for()` only cancels the Python awaitable, not the underlying process. Long-running commands such as dev servers or package scripts can leak processes after the UI reports a timeout.

Risk:

High for availability and local resource exhaustion. Combined with Issue 1, this can leave persistent unobserved processes.

Refactored code:

```python
# ai/actions.py
async def execute(self, intent_label: str, command: List[str]) -> str:
    if not command:
        return "[Action execution failed: empty command]"

    if os.name == "nt" and command[0].lower() in _WINDOWS_SHELL_BUILTINS:
        run_cmd = ["cmd", "/c", *command]
    else:
        resolved = shutil.which(command[0])
        if not resolved:
            return f"[Command not found: {command[0]}]"
        run_cmd = [resolved, *command[1:]]

    def _run_command() -> subprocess.CompletedProcess:
        proc = subprocess.Popen(
            run_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=self._cwd,
            text=False,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
        try:
            stdout, _ = proc.communicate(timeout=self._timeout_s)
            return subprocess.CompletedProcess(run_cmd, proc.returncode, stdout)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, _ = proc.communicate(timeout=2)
            raise TimeoutError((stdout or b"").decode("utf-8", errors="replace"))

    try:
        completed = await asyncio.to_thread(_run_command)
    except TimeoutError as exc:
        partial = str(exc).strip()
        suffix = f"\n{partial[:1000]}" if partial else ""
        return f"[Action timed out after {self._timeout_s:.0f}s and was terminated]{suffix}"

    output = (completed.stdout or b"").decode("utf-8", errors="replace").strip()
    if len(output) > 3000:
        output = output[:3000] + "\n... [truncated]"
    return f"$ {' '.join(command)}\n{output or f'[exit code {completed.returncode}, no output]'}"
```

## 3. User stealth preference is overwritten on every config load

Location: `core/config.py`, `_load()` line 141, `set("stealth.enabled", True)` call.

Problem:

`_load()` calls `self.set("stealth.enabled", True)` after applying defaults. This overwrites a user-saved `stealth.enabled: false` every time the app starts.

Risk:

Medium. User privacy/visibility preferences cannot persist, and code review cannot trust the YAML state to reflect runtime behavior.

Refactored code:

```python
# core/config.py
def _load(self):
    if self._path.exists():
        try:
            with open(self._path, encoding="utf-8") as f:
                self._data = yaml.safe_load(f) or {}
        except Exception as e:
            logger.error("Config: Malformed YAML in %s: %s", self._path, e)
            self._data = {}

    migrated = self._migrate_plaintext_keys_from_yaml()
    self._apply_defaults()
    self._resolve_env(self._data)
    self._inject_secrets()
    self._disable_providers_without_keys()
    if migrated:
        self.save()
    self._log_key_validation()
```

## 4. Clearing an API key does not fully revoke runtime credentials

Location: `core/config.py`, `set_api_key()` lines 657-670, `get_api_key()` lines 672-673.

Problem:

When `set_api_key(provider, "")` is called, the empty string is stored and line 670 writes `os.environ[env_key] = clean` instead of removing the environment variable. That leaves the variable present with an empty value, which some SDKs treat differently from an unset credential and which can cause confusing authentication failures instead of clean provider fallback. Separately, any already-created provider clients may still hold the previous key in memory until providers are reinitialized. `get_api_key()` also ignores environment fallback, while `_get_raw_key()` does not, which makes runtime behavior inconsistent.

Risk:

High. Secret revocation is incomplete and can leave provider/client state inconsistent after a key is removed.

Refactored code:

```python
# core/config.py
def set_api_key(self, provider: str, key: str):
    clean = (key or "").strip().strip('"').strip("'").strip()
    self.secrets.set_api_key(provider, clean)

    ai_cfg = self._data.setdefault("ai", {})
    provs = ai_cfg.setdefault("providers", {})
    prov_cfg = provs.setdefault(provider, {})
    prov_cfg["api_key"] = clean
    prov_cfg["enabled"] = bool(clean) if provider != "ollama" else True

    from core.constants import PROVIDERS
    env_key = PROVIDERS.get(provider, {}).get("env_key", "")
    if env_key:
        if clean:
            os.environ[env_key] = clean
        else:
            os.environ.pop(env_key, None)

def get_api_key(self, provider: str) -> str:
    stored = self.secrets.get_api_key(provider)
    if stored:
        return stored
    from core.constants import PROVIDERS
    env_key = PROVIDERS.get(provider, {}).get("env_key", "")
    return os.environ.get(env_key, "") if env_key else ""
```

## 5. SecureStorage encryption is deterministic and weakly machine-bound

Location: `utils/crypto.py`, `_create_fernet()` lines 59-72, `_get_machine_id()` lines 74-82.

Problem:

The encryption key is derived from public-ish machine identifiers plus a constant password (`b"openassist-ai-v4"`). Anyone with local file access and repository knowledge can reproduce the key for that machine. The fallback `"default-machine-id-openassist"` makes encryption globally reproducible if machine-id retrieval fails.

Risk:

High. API keys are encrypted at rest, but the effective secret is not secret.

Refactored code:

```python
# utils/crypto.py
MASTER_KEY_FILE = "master.key"

def _create_fernet(self) -> Fernet:
    key_path = self.filepath.with_name(MASTER_KEY_FILE)
    if key_path.exists():
        key = key_path.read_bytes().strip()
    else:
        key = Fernet.generate_key()
        key_path.write_bytes(key)
        try:
            if os.name == "nt":
                import ctypes
                FILE_ATTRIBUTE_HIDDEN = 0x02
                ctypes.windll.kernel32.SetFileAttributesW(str(key_path), FILE_ATTRIBUTE_HIDDEN)
            else:
                os.chmod(key_path, 0o600)
        except Exception as exc:
            logger.warning("SecureStorage: could not harden key file permissions: %s", exc)

    try:
        return Fernet(key)
    except Exception as exc:
        raise RuntimeError(f"SecureStorage master key is invalid: {key_path}") from exc
```

Better production hardening would use Windows DPAPI, macOS Keychain, or libsecret/keyring rather than a local key file. The snippet above is still materially safer than deterministic derivation and preserves the current file-based architecture.

## 6. Corrupt encrypted settings are silently discarded and can be overwritten

Location: `utils/crypto.py`, `_load()` lines 84-93, `_save()` lines 95-101.

Problem:

If decryption or JSON parsing fails, `_load()` logs a warning and returns `{}`. Any later `set()`/`set_api_key()` saves that empty state, permanently overwriting previously recoverable secrets. This is data loss from a transient key mismatch, partial write, or file corruption.

Risk:

High for credential persistence and migration reliability.

Refactored code:

```python
# utils/crypto.py
def _load(self) -> dict:
    self._load_failed = False
    if not self.filepath.exists():
        return {}
    try:
        encrypted = self.filepath.read_bytes()
        decrypted = self._fernet.decrypt(encrypted)
        data = json.loads(decrypted.decode("utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        self._load_failed = True
        logger.error("SecureStorage: could not decrypt %s: %s", self.filepath, exc)
        return {}

def _save(self):
    if getattr(self, "_load_failed", False) and self.filepath.exists():
        raise RuntimeError(
            f"Refusing to overwrite unreadable secure settings file: {self.filepath}"
        )
    data = json.dumps(self._data, separators=(",", ":")).encode("utf-8")
    encrypted = self._fernet.encrypt(data)
    tmp = self.filepath.with_suffix(self.filepath.suffix + ".tmp")
    tmp.write_bytes(encrypted)
    tmp.replace(self.filepath)
```

## 7. Gemini API key is placed in the URL

Location: `ui/settings_view.py`, provider test endpoint lines 188-192 and auth handling lines 245-247.

Problem:

The Gemini test request embeds the API key in the query string. URLs are more likely to appear in proxy logs, exception traces, telemetry, and debugging output than headers.

Risk:

Medium to high secret exposure, especially when API test failures are reported or copied.

Refactored code:

```python
# ui/settings_view.py
endpoints = {
    "gemini": (
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
        {"contents": [{"parts": [{"text": "Hi"}]}]},
    ),
    # other providers unchanged
}

headers = {"Content-Type": "application/json"}
if self.provider_id == "gemini":
    headers["x-goog-api-key"] = key
elif self.provider_id == "anthropic":
    headers["x-api-key"] = key
    headers["anthropic-version"] = "2023-06-01"
else:
    headers["Authorization"] = f"Bearer {key}"
```

## 8. Ollama endpoint validation allows unintended remote prompt/screenshot egress

Location: `ai/providers/ollama_provider.py`, `__init__()` lines 38-48.

Problem:

Any string beginning with `"http"` is accepted as an Ollama endpoint. Because Ollama is treated as local/offline, users may not realize that a remote endpoint receives prompts, screen OCR context, and screenshot images.

Risk:

High privacy risk. Local-only routing can silently become remote exfiltration.

Refactored code:

```python
# ai/providers/ollama_provider.py
from urllib.parse import urlparse

def _validated_endpoint(self, endpoint: str) -> str:
    raw = (endpoint or "http://localhost:11434").strip().rstrip("/")
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return "http://localhost:11434"

    host = (parsed.hostname or "").lower()
    local_hosts = {"localhost", "127.0.0.1", "::1"}
    allow_remote = bool(self.config.get("ai.providers.ollama.allow_remote_endpoint", False))
    if host not in local_hosts and not allow_remote:
        logger.warning("Ollama remote endpoint blocked: %s", raw)
        return "http://localhost:11434"
    return raw

def __init__(self, config):
    super().__init__("ollama", config)
    endpoint = self.pcfg.get("endpoint") or self.config.get_api_key("ollama") or "http://localhost:11434"
    self.endpoint = self._validated_endpoint(endpoint)
    # remaining initialization unchanged
```

## 9. Updater downloads and opens release assets without verification

Location: `ui/standby_view.py`, `_download_update()` lines 441-473; `core/updater.py`, `select_best_asset()` lines 62-107.

Problem:

The updater selects a release asset by filename, downloads it with `urlretrieve()`, saves it under the asset-provided name, and opens it. There is no digest/signature verification, no host allowlist at download time, and no filename sanitization. A compromised release, malicious repo setting, or unexpected asset metadata can lead to untrusted executable launch or path manipulation.

Risk:

High supply-chain risk.

Refactored code:

```python
# ui/standby_view.py
def _safe_update_destination(dest_dir: Path, asset_name: str) -> Path:
    safe_name = Path(asset_name).name
    if safe_name != asset_name or not safe_name:
        raise ValueError("Unsafe update asset name")
    dest = (dest_dir / safe_name).resolve()
    if dest_dir.resolve() not in dest.parents:
        raise ValueError("Update destination escaped update directory")
    return dest

def _download_verified_asset(url: str, dest: Path, expected_sha256: str = "") -> None:
    from urllib.parse import urlparse
    import hashlib
    import urllib.request

    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in {"github.com", "objects.githubusercontent.com"}:
        raise ValueError(f"Untrusted update host: {parsed.hostname}")

    tmp = dest.with_suffix(dest.suffix + ".tmp")
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=30) as response, open(tmp, "wb") as fh:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            fh.write(chunk)
    if expected_sha256 and digest.hexdigest().lower() != expected_sha256.lower():
        tmp.unlink(missing_ok=True)
        raise ValueError("Downloaded update checksum mismatch")
    tmp.replace(dest)
```

## 10. Cache worker can call `task_done()` without a matching queue item

Location: `ai/cache.py`, `_embed_worker_loop()` lines 474-483.

Problem:

`task_done()` is called in `finally` even if `self._embed_queue.get()` raises before returning an item. That violates Queue's unfinished-task accounting and can raise `ValueError: task_done() called too many times`, killing the worker thread.

Risk:

Medium. Embedding cache indexing can silently stop, reducing correctness/performance.

Refactored code:

```python
# ai/cache.py
def _embed_worker_loop(self) -> None:
    while True:
        item = self._embed_queue.get()
        try:
            if item is None:
                return
            query, rec = item
            if self._embed:
                self._embed.add(query, rec)
        except Exception as e:
            logger.debug("Embed indexing error: %s", e)
        finally:
            self._embed_queue.task_done()
```

## 11. Cache background timer has no shutdown path

Location: `ai/cache.py`, `EmbeddingTier.__init__()` lines 245-248, `_schedule_persist_timer()` lines 377-386, `ShortQueryCache.__init__()` lines 465-472.

Problem:

`EmbeddingTier` starts a self-rescheduling daemon `Timer` and `ShortQueryCache` starts a daemon worker, but there is no `close()`/`shutdown()` path. In tests, repeated engine creation accumulates background timers and threads. In a long-running app, these objects cannot be deterministically stopped.

Risk:

Medium. Resource leak and nondeterministic test/runtime behavior.

Refactored code:

```python
# ai/cache.py
class EmbeddingTier:
    def __init__(self, *args, **kwargs):
        self._closed = False
        # existing init...
        self._schedule_persist_timer()

    def _schedule_persist_timer(self, interval: float = 30.0) -> None:
        if self._closed:
            return
        def _tick():
            try:
                self._persist()
            finally:
                self._schedule_persist_timer(interval)
        self._persist_timer = threading.Timer(interval, _tick)
        self._persist_timer.daemon = True
        self._persist_timer.start()

    def close(self) -> None:
        self._closed = True
        if self._persist_timer:
            self._persist_timer.cancel()
        self._persist()

class ShortQueryCache:
    def close(self) -> None:
        if getattr(self, "_embed_queue", None):
            try:
                self._embed_queue.put_nowait(None)
            except Exception:
                pass
        if self._embed:
            self._embed.close()
```

## 12. Round-robin router skips the first provider on the first selection

Location: `ai/router.py`, `select()` lines 68-72.

Problem:

`self._rr` starts at `0`, then the first call increments it before indexing, so the first provider used is `names[1]`. That is an off-by-one state-management bug.

Risk:

Low to medium. Provider ordering is not honored, and load distribution is skewed.

Refactored code:

```python
# ai/router.py
if self.strategy == "roundrobin":
    names = list(avail.keys())
    provider = avail[names[self._rr % len(names)]]
    self._rr = (self._rr + 1) % len(names)
    return provider, self._selected_tier(provider, tier or "balanced")
```

## 13. Provider rate limiting is not thread-safe

Location: `ai/providers/base.py`, `check_rate()` lines 111-114, `_pre_request()` lines 116-121, `Stats.record()` lines 22-28.

Problem:

Provider request timestamps and statistics are mutated without a lock. The engine can race providers concurrently (`ai/parallel.py`, `ai/engine.py` race/fallback paths), so two requests can both pass `check_rate()` before either appends to `_req_times`.

Risk:

Medium. RPM limits can be exceeded, and stats can be inaccurate under parallel inference.

Refactored code:

```python
# ai/providers/base.py
import threading

@dataclass
class Stats:
    name: str
    requests: int = 0
    errors: int = 0
    total_time: float = 0.0
    total_tokens: int = 0
    last_latency: float = 0.0
    tps: float = 0.0

    def __post_init__(self):
        self._lock = threading.Lock()

    def record(self, tokens: int, latency: float):
        with self._lock:
            self.requests += 1
            self.total_tokens += int(tokens or 0)
            self.total_time += float(latency or 0.0)
            self.last_latency = float(latency or 0.0)
            if latency > 0:
                self.tps = self.tps * 0.7 + (tokens / latency) * 0.3

class BaseProvider(ABC):
    def __init__(self, name: str, config):
        # existing fields...
        self._rate_lock = threading.Lock()

    def _pre_request(self):
        now = time.time()
        with self._rate_lock:
            self._req_times = [t for t in self._req_times if now - t < 60]
            if len(self._req_times) >= self.rpm or self.is_disabled():
                raise Exception(f"{self.name}: rate limit ({self.rpm} RPM) or temporary cooldown")
            self._req_times.append(now)
```

## 14. Screen text change detection misses short-string changes

Location: `capture/screen.py`, `_has_changed()` lines 544–562.

Problem:

`get_grams()` returns an empty set for strings shorter than 3 characters. If both old and new text are short, `union` is empty and `_has_changed()` returns `False`, even when the text differs.

Risk:

Low to medium. Small but meaningful OCR changes can be ignored, especially in compact UI states or code fragments.

Refactored code:

```python
# capture/screen.py
def _has_changed(self, new_text: str) -> bool:
    old = (self._last_text or "").strip()
    new = (new_text or "").strip()
    if not old:
        return bool(new)
    if not new:
        return False
    if len(old) < 3 or len(new) < 3:
        return old != new

    def get_grams(text: str) -> set[str]:
        lowered = text.lower()
        return {lowered[i : i + 3] for i in range(len(lowered) - 2)}

    old_grams = get_grams(old)
    new_grams = get_grams(new)
    union = old_grams | new_grams
    if not union:
        return old != new
    similarity = len(old_grams & new_grams) / len(union)
    return similarity < (1.0 - self._threshold)
```

## 15. Cloud STT audio duration is computed with the wrong sample rate after resampling

Location: `capture/audio.py`, `_transcribe_groq()` helper `_build_wav()` lines 2352-2369 and duration calculation line 2489.

Problem:

`_build_wav()` resamples audio to `TARGET_SR = 16_000` and returns `len(audio)` after resampling. The caller computes `audio_duration_ms = (n_samples / self.sr) * 1000`, which is wrong whenever `self.sr != 16_000`.

Risk:

Low to medium. Telemetry and VAD/STT timing analysis become inaccurate on non-16 kHz inputs.

Refactored code:

```python
# capture/audio.py
def _build_wav(buf) -> tuple[bytes, int, int]:
    audio = np.concatenate(buf, axis=0).flatten()
    audio = self._apply_gain_normalization(audio)
    target_sr = 16_000
    if self.sr != target_sr:
        target_len = max(1, int(round(len(audio) * target_sr / self.sr)))
        src_x = np.linspace(0.0, 1.0, len(audio), endpoint=False)
        dst_x = np.linspace(0.0, 1.0, target_len, endpoint=False)
        audio = np.interp(dst_x, src_x, audio).astype(np.float32)
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(target_sr)
        pcm = np.clip(audio, -1.0, 1.0)
        wf.writeframes((pcm * 32767).astype(np.int16).tobytes())
    return wav_buf.getvalue(), len(audio), target_sr

wav_bytes, n_samples, encoded_sr = _build_wav(buffer)
audio_duration_ms = (n_samples / float(max(encoded_sr, 1))) * 1000.0
```

## 16. PDF ingestion can leak document handles on extraction exceptions

Location: `knowledge/ingest.py`, `extract_text_from_pdf()` lines 38-50.

Problem:

The code manually calls `doc.close()` after iterating pages. If `page.get_text()` raises, `doc.close()` is skipped.

Risk:

Low to medium. Repeated ingestion of malformed PDFs can leak file handles.

Refactored code:

```python
# knowledge/ingest.py
def extract_text_from_pdf(pdf_path: Path) -> str:
    try:
        import fitz
        pages: list[str] = []
        with fitz.open(str(pdf_path)) as doc:
            for page in doc:
                pages.append(page.get_text())
        return "\n\n".join(pages)
    except Exception as e:
        logger.warning("[Ingest] PDF extraction failed for %s: %s", pdf_path.name, e)
        return ""
```

## 17. RAG ingest manifest uses only mtime and never removes stale chunks

Location: `knowledge/ingest.py`, manifest helpers lines 145-155, ingestion loop lines 159-206.

Problem:

Files are considered unchanged solely by `st_mtime`. If content changes while mtime is preserved, ingestion is skipped. Conversely, if a file is deleted or heavily edited, old chunks remain in Chroma because the manifest/index does not remove chunks for missing or changed sources.

Risk:

Medium. Retrieval can answer from stale or deleted knowledge base content.

Refactored code:

```python
# knowledge/ingest.py
def _fingerprint(p: Path) -> str:
    import hashlib
    try:
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""

def _unchanged(p: Path) -> bool:
    return _manifest.get(str(p), {}).get("sha256") == _fingerprint(p)

def _mark(p: Path) -> None:
    _manifest[str(p)] = {"sha256": _fingerprint(p), "mtime": str(p.stat().st_mtime)}

def _remove_deleted_sources(rag_engine, manifest: dict[str, dict]) -> None:
    if rag_engine.collection is None:
        return
    for source in list(manifest.keys()):
        if not Path(source).exists():
            try:
                rag_engine.collection.delete(where={"source": source})
            finally:
                manifest.pop(source, None)
```

## 18. Benchmark helper scripts contain hard-coded absolute local paths

Location: `benchmarks/analyze_sweep.py` line 4, `benchmarks/compare_fixtures.py` line 3, `benchmarks/label_audit.py` lines 12 and 31.

Problem:

These scripts read `C:\Users\Vishal\Desktop\Open Assist\...` directly. They fail on any other checkout path, CI runner, or user account.

Risk:

Low to medium. Benchmark tooling is not reproducible and can accidentally read stale files outside the current checkout.

Refactored code:

```python
# benchmarks/analyze_sweep.py, benchmarks/compare_fixtures.py, benchmarks/label_audit.py
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SWEEP = ROOT / "benchmarks" / "audio_asr_matrix_sweep.json"

def load_json(path: Path = DEFAULT_SWEEP) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
```

## 19. Benchmark monkey-patches app methods without restoration

Location: `benchmarks/auto_mode_benchmark.py`, `AutoModeTester.__init__()` lines 94-100, `finish_suite()` lines 1189-1200.

Problem:

The benchmark replaces `overlay.update_transcript`, `app.generate_response`, and `history.add`, but `finish_suite()` does not restore the originals. If the benchmark is embedded in a longer process or fails mid-suite, application behavior stays patched.

Risk:

Medium for benchmark correctness and test isolation.

Refactored code:

```python
# benchmarks/auto_mode_benchmark.py
def restore_hooks(self) -> None:
    self.app.overlay.update_transcript = self.original_on_transcription
    self.app.generate_response = self.original_generate_response
    self.app.history.add = self.original_history_add

def finish_suite(self):
    safe_print("Finishing test suite and saving report...")
    try:
        if getattr(self.app, "session_active", False):
            self.app.end_session()
        self._save_report()
        safe_print(f"Report saved to {self.out_path}")
    finally:
        self.restore_hooks()
        sys.stdout.flush()
        sys.stderr.flush()
        try:
            self.app.shutdown()
        finally:
            self.app.qt_app.quit()
```

## 20. Pytest cache remains permission-denied even though temp setup is fixed

Location: repository `.pytest_cache\v\cache\nodeids`; verification command `python -m pytest -q`.

Problem:

The earlier `tmp_path` setup blocker is resolved and the suite now passes (`500 passed`). However, Pytest still warns that it cannot create/write `.pytest_cache\v\cache\nodeids` inside the checkout. This does not fail tests, but it means local cache behavior remains noisy and non-reproducible.

Risk:

Low to medium for developer experience and CI signal clarity.

Refactored code / invocation:

```powershell
# Repair cache ownership/permissions, or delete it and let Pytest recreate it.
Remove-Item -LiteralPath ".pytest_cache\v\cache\nodeids" -Force -ErrorAction SilentlyContinue
python -m pytest -q --cache-clear
```

Optional `pytest.ini` hardening:

```ini
[pytest]
addopts = --cache-clear
```

## 21. AppState mutates QObject-backed state without thread affinity enforcement

Location: `core/state.py`, `AppState` setters lines 42-123.

Problem:

The comprehensive audit's thread-safety concern is valid for the current code. `AppState` is a `QObject`, emits Qt signals, and mutates fields such as `_mode`, `_audio_source`, `_is_stealth`, and `_session_context` without a lock or main-thread dispatch. The app has audio, AI, hotkey, warmup, and UI threads; writing these properties from a non-GUI thread can race state reads and can emit Qt signals from an unexpected thread.

Risk:

Medium. Race conditions can produce stale UI state, missed mode updates, or hard-to-reproduce Qt threading issues.

Refactored code:

```python
# core/state.py
import threading
from PyQt6.QtCore import QObject, pyqtSignal, QTimer, QThread, QCoreApplication

class AppState(QObject):
    state_changed = pyqtSignal(str, object)
    mode_changed = pyqtSignal(str)
    # existing signals...

    def __init__(self, config=None):
        super().__init__()
        self._lock = threading.RLock()
        self._config = config
        # existing fields...

    def _on_owner_thread(self) -> bool:
        app = QCoreApplication.instance()
        return bool(app) and QThread.currentThread() == self.thread()

    def _dispatch(self, fn):
        if self._on_owner_thread():
            fn()
        else:
            QTimer.singleShot(0, fn)

    @property
    def mode(self):
        with self._lock:
            return self._mode

    @mode.setter
    def mode(self, val):
        def apply():
            with self._lock:
                if self._mode == val:
                    return
                self._mode = val
                if self._config:
                    self._config.set("ai.mode", val)
            self.mode_changed.emit(val)
            self.state_changed.emit("mode", val)
        self._dispatch(apply)
```

## 22. ContextStore singleton initialization is not thread-safe

Location: `utils/context_store.py`, module singleton `_store` and `get_store()` lines 157-165.

Problem:

`get_store()` checks and assigns `_store` without a module-level lock. Two concurrent callers can both see `_store is None`, construct separate `ContextStore` instances, and race their loads/writes.

Risk:

Medium. Presets or last-context writes can be lost when UI and background startup code access the store concurrently.

Refactored code:

```python
# utils/context_store.py
_store: ContextStore | None = None
_store_lock = threading.Lock()

def get_store() -> ContextStore:
    global _store
    if _store is not None:
        return _store
    with _store_lock:
        if _store is None:
            _store = ContextStore()
        return _store
```

## 23. ContextStore writes are not atomic

Location: `utils/context_store.py`, `_persist()` lines 147-154.

Problem:

`_persist()` writes directly to `data/context_presets.json`. A crash, power loss, or process termination during write can leave a truncated JSON file. `_load()` then catches the parse failure and silently reverts to default state, losing user presets.

Risk:

Medium. User-authored session-context presets can be lost.

Refactored code:

```python
# utils/context_store.py
def _persist(self):
    try:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self._data, indent=2, ensure_ascii=False)
        tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp_path.write_text(payload, encoding="utf-8")
        tmp_path.replace(self._path)
    except Exception as e:
        logger.error("ContextStore save failed: %s", e)
```

## 24. Shared RateLimiter is not thread-safe

Location: `utils/rate_limiter.py`, `can_request()` lines 19-31, `record()` lines 33-34, `wait_if_needed()` lines 36-38.

Problem:

The rate limiter stores timestamps in a shared `defaultdict(list)` without a lock. `wait_if_needed()` calls `can_request()` and `record()` as separate operations, so concurrent callers can all pass the check before any records, exceeding RPM/RPD limits.

Risk:

Medium. Parallel AI calls can exceed provider quotas or cause avoidable 429 failures.

Refactored code:

```python
# utils/rate_limiter.py
import threading

class RateLimiter:
    def __init__(self):
        self._windows = defaultdict(list)
        self._limits = {}
        self._lock = threading.RLock()

    def configure(self, provider: str, rpm: int = 60, rpd: int = 10000):
        with self._lock:
            self._limits[provider] = (int(rpm), int(rpd))

    def try_record(self, provider: str) -> bool:
        now = time.time()
        with self._lock:
            rpm, rpd = self._limits.get(provider, (60, 10000))
            window = [t for t in self._windows[provider] if now - t < 86400]
            minute_count = sum(1 for t in window if now - t < 60)
            if minute_count >= rpm or len(window) >= rpd:
                self._windows[provider] = window
                return False
            window.append(now)
            self._windows[provider] = window
            return True

    async def wait_if_needed(self, provider: str):
        while not self.try_record(provider):
            await asyncio.sleep(1)
```

## 25. MarkdownRenderer allows unsafe link schemes

Location: `ui/markdown_renderer.py`, `_render_inline()` lines 256-260.

Problem:

The renderer escapes HTML before inline rendering, which mitigates direct tag injection. However, Markdown links are converted to raw `<a href="\2">` without validating URL schemes. A model-generated link such as `[x](javascript:...)`, `file:///...`, or an app-specific deep link can become clickable in `QTextEdit`.

Risk:

Medium. Unsafe links can trigger local file/app actions or script-like URL handling depending on the Qt platform backend.

Refactored code:

```python
# ui/markdown_renderer.py
from urllib.parse import urlparse

def _safe_link(self, label: str, url: str) -> str:
    parsed = urlparse(html.unescape(url).strip())
    if parsed.scheme and parsed.scheme.lower() not in {"http", "https", "mailto"}:
        return label
    safe_url = html.escape(parsed.geturl(), quote=True)
    return f'<a style="color: #6699cc; text-decoration: underline;" href="{safe_url}">{label}</a>'

def _render_inline(self, text: str) -> str:
    # existing escaping and inline rendering...
    text = re.sub(
        r"\[(.+?)\]\((.+?)\)",
        lambda m: self._safe_link(m.group(1), m.group(2)),
        text,
    )
    return text
```

## 26. Config `save()` is not atomic

Location: `core/config.py`, `save()` lines 645–647.

Problem:

`save()` opens the config YAML and writes directly to it via `yaml.dump()`. If the process is killed or the machine loses power during the write, the file will be truncated or empty. On next startup, `_load()` will parse an empty/corrupt file and fall back to default state — losing all user configuration. This is the same class of bug as Issues 6 and 23, repeated for the main config path.

Risk:

Medium. User settings (hotkeys, mode, all capture/AI preferences) are lost on crash-during-save.

Refactored code:

```python
# core/config.py
import tempfile

def save(self):
    config_dir = str(self._path.parent)
    fd, tmp_path = tempfile.mkstemp(dir=config_dir, suffix=".yaml.tmp")
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            yaml.dump(self._data_for_save(), f, default_flow_style=False, allow_unicode=True)
        os.replace(tmp_path, str(self._path))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
```

## 27. `config.py` `_load()` opens file without explicit encoding

Location: `core/config.py`, `_load()` line 132.

Problem:

`open(self._path)` relies on the platform default encoding. On some Windows systems this is `cp1252`, not `utf-8`. A config file containing Unicode characters (e.g. in a custom prompt or mode name) would be silently mis-decoded or raise `UnicodeDecodeError`, triggering the fallback to empty config.

Risk:

Low-medium. Affects users with non-ASCII config content on specific Windows locale settings.

Refactored code:

```python
# core/config.py
with open(self._path, encoding="utf-8") as f:
    self._data = yaml.safe_load(f) or {}
```

## Architecture Notes

- The core architecture is understandable: `core.app` orchestrates capture, AI routing, UI, history, and warmup; providers are isolated behind `BaseProvider`; and benchmark/test coverage is broad.
- The largest architectural risk is boundary confusion between "assistant actions" and "local automation." Action execution should be opt-in, confirmation-gated for anything beyond read-only inspection, and prevented from returning secrets into model context.
- The second largest risk is secret handling. Secrets are moved out of YAML, which is good, but runtime environment injection and deterministic encryption weaken the design.
- The third largest risk is lifecycle management. Several timers, daemon threads, provider sessions, and monkey-patches lack explicit shutdown/restoration paths.
- Non-atomic file writes are a systemic pattern. Config (Issue 26), SecureStorage (Issue 6), and ContextStore (Issue 23) all use direct writes. A single `atomic_write()` utility could be extracted and applied to all three.
- The `Stats.record()` and `BaseProvider._pre_request()` / `check_rate()` methods (Issue 13) share a pattern with the standalone `RateLimiter` (Issue 24) — both track timestamps without locks. Consolidating into one thread-safe rate-tracking primitive would reduce surface area for concurrency bugs.
