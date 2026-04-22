"""Config with env var resolution, secrets, and KEY VALIDATION."""

import os
import re
import yaml
from pathlib import Path
from typing import Any, Dict, List, Tuple
from utils.crypto import SecureStorage
from utils.logger import setup_logger

logger = setup_logger(__name__)


# Known API key patterns for validation
KEY_PATTERNS: Dict[str, Dict] = {
    "groq": {
        "prefix": "gsk_",
        "min_length": 20,
        "max_length": 200,
        "pattern": r'^gsk_[a-zA-Z0-9]{20,}$',
    },
    "gemini": {
        "prefix": "AIza",
        "min_length": 30,
        "max_length": 60,
        "pattern": r'^AIza[a-zA-Z0-9_-]{30,}$',
    },
    "anthropic": {
        "prefix": "sk-ant-",
        "min_length": 40,
        "max_length": 200,
        "pattern": r'^sk-ant-[a-zA-Z0-9_-]{20,}$',
    },
    "openai": {
        "prefix": "sk-",
        "min_length": 20,
        "max_length": 200,
        "pattern": r'^sk-[a-zA-Z0-9_-]{20,}$',
    },
    "openrouter": {
        "prefix": "sk-or-",
        "min_length": 20,
        "max_length": 200,
        "pattern": r'^sk-or-[a-zA-Z0-9_-]{20,}$',
    },
    "together": {
        "prefix": "",
        "min_length": 20,
        "max_length": 200,
        "pattern": r'^[a-f0-9]{40,}$',
    },
    "mistral": {
        "prefix": "",
        "min_length": 20,
        "max_length": 200,
        "pattern": None,  # Variable format
    },
    "cohere": {
        "prefix": "",
        "min_length": 20,
        "max_length": 200,
        "pattern": None,
    },
    "cerebras": {
        "prefix": "csk-",
        "min_length": 20,
        "max_length": 200,
        "pattern": r'^csk-[a-zA-Z0-9_-]{20,}$',
    },
    "sambanova": {
        "prefix": "",
        "min_length": 20,
        "max_length": 200,
        "pattern": None,
    },
    "hyperbolic": {
        "prefix": "",
        "min_length": 20,
        "max_length": 200,
        "pattern": None,
    },
    "ollama": {
        "prefix": "http",
        "min_length": 10,
        "max_length": 200,
        "pattern": r'^https?://[a-zA-Z0-9.:/_-]+$',
    },
}


class Config:
    def __init__(self, path: str = "config.yaml"):
        self._path = Path(path)
        self._data = {}
        self.secrets = SecureStorage()
        self._load()

    def _load(self):
        if self._path.exists():
            try:
                with open(self._path) as f:
                    self._data = yaml.safe_load(f) or {}
            except Exception as e:
                logger.error(f"Config: Malformed YAML in {self._path}: {e}")
                logger.warning("Config: Reverting to empty/default state.")
                self._data = {}
        self._apply_defaults()
        self.set("stealth.enabled", True)
        self._resolve_env(self._data)
        self._inject_secrets()

        # P1 FIX: Validate keys on load
        validation = self.validate_api_keys()
        for msg in validation["warnings"]:
            logger.warning(f"Config: {msg}")
        for msg in validation["errors"]:
            logger.error(f"Config: {msg}")
        if validation["valid"]:
            logger.info(f"Config loaded ({len(validation['valid'])} valid keys)")
        else:
            logger.info("Config loaded (no API keys configured)")

    def _apply_defaults(self):
        self._data.setdefault("ai", {})
        self._data["ai"].setdefault("text", {})
        # Text routing: provider priority + optional "race" for lowest latency.
        self._data["ai"]["text"].setdefault(
            "preferred_providers",
            ["groq", "gemini", "cerebras", "together", "ollama"],
        )
        self._data["ai"]["text"].setdefault("race_enabled", False)
        # P2: Local-only mode to force Ollama usage without YAML edits.
        self._data["ai"]["text"].setdefault("local_only", False)
        self._data["ai"].setdefault("vision", {})
        self._data["ai"]["vision"].setdefault("allow_paid_fallback", False)
        # Vision routing: provider priority + optional "race" for lowest latency.
        self._data["ai"]["vision"].setdefault("preferred_providers", ["gemini", "ollama"])
        self._data["ai"]["vision"].setdefault("race_enabled", False)
        self._data["ai"]["vision"].setdefault("local_only", False)
        self._data.setdefault("app", {})
        self._data["app"].setdefault("focus_on_show", False)
        self._data.setdefault("stealth", {})
        self._data["stealth"].setdefault("enabled", True)
        self._data["stealth"].setdefault("auto_hide_on_share", True)
        self._data["stealth"].setdefault("low_opacity", 0.75)
        self._data.setdefault("hotkeys", {})
        hotkeys = self._data["hotkeys"]
        hotkeys.setdefault("toggle", "ctrl+\\")
        hotkeys.setdefault("toggle_click_through", "ctrl+m")
        hotkeys.setdefault("analyze_screen", "ctrl+enter")
        hotkeys.setdefault("quick_answer", "ctrl+shift+q")
        hotkeys.setdefault("switch_mode", "ctrl+shift+m")
        hotkeys.setdefault("move_up", "ctrl+up")
        hotkeys.setdefault("move_down", "ctrl+down")
        hotkeys.setdefault("move_left", "ctrl+left")
        hotkeys.setdefault("move_right", "ctrl+right")
        hotkeys.setdefault("scroll_up", "ctrl+shift+up")
        hotkeys.setdefault("scroll_down", "ctrl+shift+down")
        hotkeys.setdefault("history_prev", "ctrl+[")
        hotkeys.setdefault("history_next", "ctrl+]")
        hotkeys.setdefault("toggle_audio", "ctrl+shift+a")
        hotkeys.setdefault("stealth", "ctrl+shift+z")
        hotkeys.setdefault("mini_mode", "ctrl+alt+n")
        hotkeys.setdefault("emergency_erase", "ctrl+shift+e")

    def _resolve_env(self, d):
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, (dict, list)):
                    self._resolve_env(v)
                elif isinstance(v, str) and "${" in v:
                    d[k] = re.sub(
                        r'\$\{([^}]+)\}',
                        lambda m: self._env(m.group(1)), v
                    )
        elif isinstance(d, list):
            for i, v in enumerate(d):
                if isinstance(v, (dict, list)):
                    self._resolve_env(v)
                elif isinstance(v, str) and "${" in v:
                    d[i] = re.sub(
                        r'\$\{([^}]+)\}',
                        lambda m: self._env(m.group(1)), v
                    )

    @staticmethod
    def _env(expr):
        if ":-" in expr:
            var, default = expr.split(":-", 1)
            return os.environ.get(var.strip(), default.strip())
        return os.environ.get(expr.strip(), "")

    def _inject_secrets(self):
        from core.constants import PROVIDERS
        for pid, meta in PROVIDERS.items():
            stored_key = self.secrets.get_api_key(pid)
            if stored_key:
                # Ensure provider config exists so providers can read api_key from config.
                ai_cfg = self._data.setdefault("ai", {})
                provs = ai_cfg.setdefault("providers", {})
                prov_cfg = provs.setdefault(pid, {})
                prov_cfg["api_key"] = stored_key
                prov_cfg["enabled"] = True
                os.environ[meta["env_key"]] = stored_key

    # P1 FIX #6: API Key Validation

    def validate_api_keys(self) -> Dict[str, List]:
        """
        Validate all configured API keys.
        
        Returns:
            {
                "valid": ["groq", "gemini"],
                "warnings": ["openai: key looks like Anthropic key (sk-ant-)"],
                "errors": ["cerebras: key too short (expected 20+ chars)"],
                "cleaned": {"groq": "gsk_abc..."} # Keys with whitespace/quotes stripped
            }
        """
        result = {"valid": [], "warnings": [], "errors": [], "cleaned": {}}

        from core.constants import PROVIDERS

        for pid in PROVIDERS:
            key = self._get_raw_key(pid)
            if not key:
                continue

            issues = self._validate_single_key(pid, key)

            if issues["cleaned"] != key:
                result["cleaned"][pid] = issues["cleaned"]
                # Auto-fix: save cleaned key
                self.set_api_key(pid, issues["cleaned"])
                result["warnings"].append(
                    f"{pid}: Key had whitespace/quotes - auto-cleaned"
                )

            result["errors"].extend(issues["errors"])
            result["warnings"].extend(issues["warnings"])

            if not issues["errors"]:
                result["valid"].append(pid)

        return result

    def _get_raw_key(self, provider: str) -> str:
        """Get raw key from all sources."""
        # Check encrypted storage first
        key = self.secrets.get_api_key(provider)
        if key:
            return key

        # Check config file
        key = self.get(f"ai.providers.{provider}.api_key", "")
        if key:
            return key

        # Check environment
        from core.constants import PROVIDERS
        env_var = PROVIDERS.get(provider, {}).get("env_key", "")
        if env_var:
            return os.environ.get(env_var, "")

        return ""

    @staticmethod
    def _validate_single_key(provider: str, key: str) -> Dict:
        """
        Validate a single API key.
        
        Checks:
          1. Strip whitespace and surrounding quotes
          2. Check minimum/maximum length
          3. Check expected prefix
          4. Check for common copy-paste errors
          5. Warn if key looks like it belongs to a different provider
        """
        issues = {"errors": [], "warnings": [], "cleaned": key}

        # Step 1: Clean the key
        cleaned = key.strip().strip('"').strip("'").strip()
        issues["cleaned"] = cleaned

        if not cleaned:
            return issues  # Empty key, nothing to validate

        rules = KEY_PATTERNS.get(provider, {})
        if not rules:
            return issues  # Unknown provider, skip validation

        # Step 2: Length check
        min_len = rules.get("min_length", 10)
        max_len = rules.get("max_length", 500)

        if len(cleaned) < min_len:
            issues["errors"].append(
                f"{provider}: Key too short ({len(cleaned)} chars, expected {min_len}+)"
            )
            return issues

        if len(cleaned) > max_len:
            issues["errors"].append(
                f"{provider}: Key too long ({len(cleaned)} chars, expected max {max_len})"
            )
            return issues

        # Step 3: Prefix check
        expected_prefix = rules.get("prefix", "")
        if expected_prefix and not cleaned.startswith(expected_prefix):
            # Check if it's a key for a DIFFERENT provider
            wrong_provider = None
            for other_pid, other_rules in KEY_PATTERNS.items():
                other_prefix = other_rules.get("prefix", "")
                if other_prefix and cleaned.startswith(other_prefix) and other_pid != provider:
                    wrong_provider = other_pid
                    break

            if wrong_provider:
                issues["errors"].append(
                    f"{provider}: This looks like a {wrong_provider} key "
                    f"(starts with '{cleaned[:8]}...'). "
                    f"Expected prefix: '{expected_prefix}'"
                )
            else:
                issues["warnings"].append(
                    f"{provider}: Key doesn't start with expected prefix '{expected_prefix}'"
                )

        # Step 4: Pattern check (if available)
        pattern = rules.get("pattern")
        if pattern and not re.match(pattern, cleaned):
            issues["warnings"].append(
                f"{provider}: Key format looks unusual (may still work)"
            )

        # Step 5: Common errors
        if " " in cleaned:
            issues["errors"].append(
                f"{provider}: Key contains spaces (copy-paste error?)"
            )
        if "\n" in cleaned or "\r" in cleaned:
            issues["errors"].append(
                f"{provider}: Key contains newlines (copy-paste error?)"
            )

        return issues

    def validate_key_for_ui(self, provider: str, key: str) -> Tuple[bool, str]:
        """
        Quick validation for UI feedback.
        Returns (is_valid, message).
        """
        if not key or not key.strip():
            return False, "No key entered"

        issues = self._validate_single_key(provider, key)

        if issues["errors"]:
            return False, issues["errors"][0]
        if issues["warnings"]:
            return True, issues["warnings"][0]
        return True, "Key format looks valid"

    # Existing methods (unchanged)

    def get(self, path: str, default: Any = None) -> Any:
        keys = path.split(".")
        val = self._data
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                return default
            if val is None:
                return default
        return val

    def set(self, path: str, value: Any):
        keys = path.split(".")
        d = self._data
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    def save(self):
        with open(self._path, 'w') as f:
            yaml.dump(self._data, f, default_flow_style=False)

    def reset_all(self):
        """Restore config to first-run state and wipe encrypted secrets."""
        self.secrets.clear_all()
        self._data = {}
        self._apply_defaults()
        self.set("onboarding.completed", False)
        self.save()

    def set_api_key(self, provider: str, key: str):
        # Clean the key before saving
        clean = key.strip().strip('"').strip("'").strip()

        self.secrets.set_api_key(provider, clean)
        ai_cfg = self._data.setdefault("ai", {})
        provs = ai_cfg.setdefault("providers", {})
        prov_cfg = provs.setdefault(provider, {})
        prov_cfg["api_key"] = clean
        prov_cfg["enabled"] = bool(clean)
        from core.constants import PROVIDERS
        env_key = PROVIDERS.get(provider, {}).get("env_key", "")
        if env_key:
            os.environ[env_key] = clean

    def get_api_key(self, provider: str) -> str:
        return self.secrets.get_api_key(provider)

    @property
    def data(self):
        return self._data
