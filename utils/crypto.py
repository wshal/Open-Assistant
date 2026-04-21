"""Encrypt/decrypt API keys stored on disk."""

import os
import json
import base64
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from utils.logger import setup_logger

logger = setup_logger(__name__)


class SecureStorage:
    """Encrypt API keys at rest using machine-specific key."""

    def __init__(self, filepath: str = "data/settings.enc"):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._fernet = self._create_fernet()
        self._data = self._load()

    def _create_fernet(self) -> Fernet:
        """Derive encryption key from machine-specific data."""
        # Use machine ID as salt (unique per machine)
        machine_id = self._get_machine_id()
        salt = machine_id.encode()[:16].ljust(16, b'\0')

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100_000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(b"openassist-ai-v4"))
        return Fernet(key)

    @staticmethod
    def _get_machine_id() -> str:
        """Get a machine-specific identifier."""
        import platform
        import uuid
        try:
            return str(uuid.getnode()) + platform.node()
        except Exception:
            return "default-machine-id-openassist"

    def _load(self) -> dict:
        if not self.filepath.exists():
            return {}
        try:
            encrypted = self.filepath.read_bytes()
            decrypted = self._fernet.decrypt(encrypted)
            return json.loads(decrypted)
        except Exception:
            logger.warning("Could not decrypt settings, starting fresh")
            return {}

    def _save(self):
        try:
            data = json.dumps(self._data).encode()
            encrypted = self._fernet.encrypt(data)
            self.filepath.write_bytes(encrypted)
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def set(self, key: str, value):
        self._data[key] = value
        self._save()

    def delete(self, key: str):
        self._data.pop(key, None)
        self._save()

    def get_api_key(self, provider: str) -> str:
        """Get API key for a provider."""
        return self._data.get(f"api_key_{provider}", "")

    def set_api_key(self, provider: str, key: str):
        """Set API key for a provider."""
        self._data[f"api_key_{provider}"] = key
        self._save()

    def get_all_keys(self) -> dict:
        """Get all stored API keys."""
        return {
            k.replace("api_key_", ""): v
            for k, v in self._data.items()
            if k.startswith("api_key_") and v
        }

    def clear_all(self):
        self._data.clear()
        self._save()

    @property
    def all_settings(self):
        return dict(self._data)