import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import Config
from utils.crypto import SecureStorage


def test_config_imports_legacy_config_when_new_path_missing(tmp_path, monkeypatch):
    legacy_path = tmp_path / "config.yaml"
    legacy_path.write_text("ai:\n  mode: coding\n", encoding="utf-8")

    target_path = tmp_path / "appdata" / "config.yaml"
    monkeypatch.chdir(tmp_path)

    cfg = Config(str(target_path))

    assert target_path.exists()
    assert cfg.get("ai.mode") == "coding"


def test_secure_storage_imports_legacy_file_when_new_path_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    legacy_dir = tmp_path / "data"
    legacy_dir.mkdir()
    legacy_storage = SecureStorage(str(legacy_dir / "settings.enc"))
    legacy_storage.set_api_key("groq", "gsk_test_12345678901234567890")

    target_path = tmp_path / "appdata" / "settings.enc"
    migrated = SecureStorage(str(target_path))

    assert target_path.exists()
    assert migrated.get_api_key("groq") == "gsk_test_12345678901234567890"
