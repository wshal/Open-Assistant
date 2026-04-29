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

    def export_keys(self) -> bytes:
        """Export all API keys as an encrypted blob (machine-locked, for local backup)."""
        keys = self.get_all_keys()
        plaintext = json.dumps(keys).encode("utf-8")
        return self._fernet.encrypt(plaintext)

    def import_keys(self, data: bytes) -> int:
        """Import API keys from a machine-locked encrypted blob. Returns count imported."""
        try:
            plaintext = self._fernet.decrypt(data)
            keys = json.loads(plaintext.decode("utf-8"))
        except Exception as e:
            logger.error("Failed to import keys: %s", e)
            return 0
        if not isinstance(keys, dict):
            return 0
        count = 0
        for pid, key in keys.items():
            if isinstance(key, str) and key.strip():
                self.set_api_key(pid, key)
                count += 1
        return count

    # ── Portable (password-based) export/import ─────────────────────────────
    # These produce .enc files that can be transferred between machines or
    # installations (Python script or compiled .exe) as long as the user
    # supplies the same password. The file format is:
    #   [16-byte random salt] + [Fernet-AES128 encrypted JSON payload]

    EXPORT_MAGIC = b"OAKEYS1"
    EXPORT_ITERATIONS = 480_000  # OWASP 2023 PBKDF2-SHA256 minimum

    @classmethod
    def _fernet_from_password(cls, password: str, salt: bytes) -> "Fernet":
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=cls.EXPORT_ITERATIONS,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode("utf-8")))
        return Fernet(key)

    def export_keys_portable(self, password: str) -> bytes:
        """Export all API keys as a password-encrypted blob portable across machines.

        Format: magic(7) + salt(16) + fernet_ciphertext
        """
        import os as _os
        keys = self.get_all_keys()
        plaintext = json.dumps(keys).encode("utf-8")
        salt = _os.urandom(16)
        fernet = self._fernet_from_password(password, salt)
        ciphertext = fernet.encrypt(plaintext)
        return self.EXPORT_MAGIC + salt + ciphertext

    def import_keys_portable(self, data: bytes, password: str) -> int:
        """Import API keys from a portable password-encrypted blob.

        Returns the number of keys successfully imported, or raises ValueError
        on bad magic/password.
        """
        if not data.startswith(self.EXPORT_MAGIC):
            raise ValueError("Not a valid OpenAssist key backup (wrong file format).")
        payload = data[len(self.EXPORT_MAGIC):]
        if len(payload) < 17:
            raise ValueError("Backup file is too short / corrupt.")
        salt, ciphertext = payload[:16], payload[16:]
        fernet = self._fernet_from_password(password, salt)
        try:
            plaintext = fernet.decrypt(ciphertext)
        except Exception:
            raise ValueError("Wrong password or corrupted backup file.")
        try:
            keys = json.loads(plaintext.decode("utf-8"))
        except Exception:
            raise ValueError("Backup payload is not valid JSON.")
        if not isinstance(keys, dict):
            raise ValueError("Invalid backup format (expected a JSON object).")
        count = 0
        for pid, key in keys.items():
            if isinstance(key, str) and key.strip():
                self.set_api_key(pid, key)
                count += 1
        return count

    def clear_all(self):
        self._data.clear()
        self._save()

    @property
    def all_settings(self):
        return dict(self._data)