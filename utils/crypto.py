"""Encrypt/decrypt API keys stored on disk."""

import os
import json
import base64
import shutil
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from utils.logger import setup_logger
from core.constants import SETTINGS_FILE

logger = setup_logger(__name__)


class SecureStorage:
    """Encrypt API keys at rest using machine-specific key."""

    # Issue #5: Filename for the randomly-generated master key.
    MASTER_KEY_FILE = "master.key"

    def __init__(self, filepath: str = SETTINGS_FILE):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._maybe_import_legacy_storage()
        # Issue #6: Track whether the last load failed (decrypt/parse) so _save()
        # can refuse to overwrite an unreadable-but-existing settings file.
        self._load_failed = False
        self._fernet = self._create_fernet()
        self._data = self._load()
        logger.info("SecureStorage: using file %s", self.filepath)

    def _maybe_import_legacy_storage(self) -> None:
        """Migrate legacy side-by-side encrypted settings into persistent storage."""
        if self.filepath.name != "settings.enc":
            return
        if self.filepath.exists():
            return

        candidate_paths = [
            Path("data") / "settings.enc",
            Path("settings.enc"),
        ]
        legacy_path = None
        for candidate in candidate_paths:
            try:
                if candidate.resolve() == self.filepath.resolve():
                    continue
            except Exception:
                pass
            if candidate.exists():
                legacy_path = candidate
                break

        if legacy_path is None:
            return

        try:
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(legacy_path, self.filepath)
            logger.info(
                "SecureStorage: imported legacy settings from %s -> %s",
                legacy_path,
                self.filepath,
            )
            # Also copy the master key so the new instance can decrypt the
            # migrated ciphertext.  Without this, _create_fernet() would
            # generate a new random key that cannot unlock the copied bytes.
            legacy_key_path = legacy_path.with_name(self.MASTER_KEY_FILE)
            if legacy_key_path.exists():
                target_key_path = self.filepath.with_name(self.MASTER_KEY_FILE)
                if not target_key_path.exists():
                    shutil.copy2(legacy_key_path, target_key_path)
                    self._harden_key_file(target_key_path)
                    logger.info(
                        "SecureStorage: copied master key from %s -> %s",
                        legacy_key_path,
                        target_key_path,
                    )
        except Exception as e:
            logger.warning(
                "SecureStorage: could not import legacy settings from %s: %s",
                legacy_path,
                e,
            )


    def _create_fernet(self) -> Fernet:
        """Return a Fernet instance backed by a per-installation random key.

        Issue #5: The previous implementation derived the key from public
        machine identifiers (uuid.getnode() + platform.node()) plus a constant
        password, which is reproducible by anyone with local file access. We
        now generate a random Fernet key on first run and persist it next to
        the settings file with restrictive permissions. The deterministic
        legacy key is still used as a one-time fallback to migrate existing
        installs (see ``_legacy_fernet``).
        """
        key_path = self.filepath.with_name(self.MASTER_KEY_FILE)
        if key_path.exists():
            try:
                key = key_path.read_bytes().strip()
                return Fernet(key)
            except Exception as exc:
                # Don't silently regenerate: that would orphan whatever is in
                # settings.enc and look like a data-loss bug.
                raise RuntimeError(
                    f"SecureStorage master key is unreadable: {key_path}"
                ) from exc

        key = Fernet.generate_key()
        try:
            key_path.write_bytes(key)
            self._harden_key_file(key_path)
        except Exception as exc:
            logger.warning(
                "SecureStorage: could not persist master key at %s: %s",
                key_path,
                exc,
            )
        return Fernet(key)

    @staticmethod
    def _harden_key_file(key_path: Path) -> None:
        try:
            if os.name == "nt":
                import ctypes  # type: ignore[import-not-found]
                import subprocess

                FILE_ATTRIBUTE_HIDDEN = 0x02
                ctypes.windll.kernel32.SetFileAttributesW(
                    str(key_path), FILE_ATTRIBUTE_HIDDEN
                )
                try:
                    username = os.environ.get("USERNAME") or os.getlogin()
                    subprocess.run(
                        ["icacls", str(key_path), "/inheritance:r", "/grant:r", f"{username}:F"],
                        capture_output=True,
                        check=False,
                    )
                except Exception as acl_exc:
                    logger.warning("SecureStorage: Windows ACL harden failed: %s", acl_exc)
            else:
                os.chmod(key_path, 0o600)
        except Exception as exc:
            logger.warning(
                "SecureStorage: could not harden key file permissions on %s: %s",
                key_path,
                exc,
            )

    def _legacy_fernet(self) -> Fernet:
        """Recreate the v1 deterministic key for one-shot migration only."""
        machine_id = self._get_machine_id()
        salt = machine_id.encode()[:16].ljust(16, b"\0")
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
        """Get a machine-specific identifier (legacy migration only)."""
        import platform
        import uuid
        try:
            return str(uuid.getnode()) + platform.node()
        except Exception:
            return "default-machine-id-openassist"

    def _load(self) -> dict:
        self._load_failed = False
        if not self.filepath.exists():
            return {}
        try:
            encrypted = self.filepath.read_bytes()
        except Exception as exc:
            self._load_failed = True
            logger.error(
                "SecureStorage: could not read %s: %s", self.filepath, exc
            )
            return {}

        try:
            decrypted = self._fernet.decrypt(encrypted)
        except Exception:
            # Issue #5 migration: existing installs encrypted with the legacy
            # deterministic key need to be readable and rewritten under the new
            # random master key.
            try:
                decrypted = self._legacy_fernet().decrypt(encrypted)
            except Exception as exc:
                # Issue #6: Do NOT silently revert to {} and clobber the file.
                # Mark the load as failed so subsequent saves refuse to overwrite.
                self._load_failed = True
                logger.error(
                    "SecureStorage: could not decrypt %s (file kept intact): %s",
                    self.filepath,
                    exc,
                )
                return {}
            else:
                logger.info(
                    "SecureStorage: migrating legacy-encrypted settings to master key"
                )
                try:
                    data = json.loads(decrypted.decode("utf-8"))
                except Exception as exc:
                    self._load_failed = True
                    logger.error(
                        "SecureStorage: legacy payload is not valid JSON (file kept intact): %s",
                        exc,
                    )
                    return {}
                # Re-encrypt with the new key atomically.
                self._data = data if isinstance(data, dict) else {}
                self._save()
                return self._data

        try:
            data = json.loads(decrypted.decode("utf-8"))
        except Exception as exc:
            self._load_failed = True
            logger.error(
                "SecureStorage: decrypted payload is not valid JSON (file kept intact): %s",
                exc,
            )
            return {}
        return data if isinstance(data, dict) else {}

    def _save(self):
        # Issue #6: Refuse to overwrite a settings file we could not decrypt;
        # otherwise a transient mismatch silently destroys recoverable data.
        if self._load_failed and self.filepath.exists():
            logger.error(
                "SecureStorage: refusing to overwrite unreadable secure settings %s",
                self.filepath,
            )
            return
        try:
            data = json.dumps(self._data, separators=(",", ":")).encode("utf-8")
            encrypted = self._fernet.encrypt(data)
            # Issue #6: Atomic write so a crash mid-save can't truncate the file.
            tmp = self.filepath.with_suffix(self.filepath.suffix + ".tmp")
            tmp.write_bytes(encrypted)
            os.replace(str(tmp), str(self.filepath))
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
                self._data[f"api_key_{pid}"] = key
                count += 1
        if count > 0:
            self._save()
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
                self._data[f"api_key_{pid}"] = key
                count += 1
        if count > 0:
            self._save()
        return count

    def clear_all(self):
        self._data.clear()
        self._save()

    @property
    def all_settings(self):
        return dict(self._data)
