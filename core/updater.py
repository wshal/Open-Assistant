"""Auto-updater - checks GitHub for new releases."""

import asyncio
from typing import Optional, Tuple

import aiohttp
from packaging import version as pkg_version

from core.constants import APP_VERSION
from utils.logger import setup_logger

logger = setup_logger(__name__)

GITHUB_REPO = "openassist-ai/openassist"
RELEASES_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"


class Updater:
    """Check for and download updates from GitHub."""

    def __init__(self, config):
        self.config = config
        self.current_version = APP_VERSION
        self.latest_version = None
        self.download_url = None
        self.release_notes = None

    async def check_for_update(self) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Check GitHub for newer version.
        Returns: (update_available, latest_version, release_notes)
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    RELEASES_URL,
                    timeout=aiohttp.ClientTimeout(total=10),
                    headers={"Accept": "application/vnd.github.v3+json"},
                ) as resp:
                    if resp.status != 200:
                        return False, None, None

                    data = await resp.json()
                    tag = data.get("tag_name", "").lstrip("v")
                    notes = data.get("body", "")
                    assets = data.get("assets", [])

                    self.latest_version = tag
                    self.release_notes = notes

                    # Find download URL for current platform
                    import platform

                    os_name = platform.system().lower()
                    for asset in assets:
                        name = asset.get("name", "").lower()
                        if os_name in name or "universal" in name:
                            self.download_url = asset.get("browser_download_url")
                            break

                    # Compare versions
                    try:
                        current = pkg_version.parse(self.current_version)
                        latest = pkg_version.parse(tag)
                        update_available = latest > current
                    except Exception:
                        update_available = tag != self.current_version

                    if update_available:
                        logger.info(
                            f"Update available: v{self.current_version} -> v{tag}"
                        )
                    else:
                        logger.debug(f"Up to date (v{self.current_version})")

                    return update_available, tag, notes

        except Exception as exc:
            logger.debug(f"Update check failed: {exc}")
            return False, None, None

    def get_download_url(self) -> Optional[str]:
        return self.download_url

    async def download_update(self, save_path: str) -> bool:
        """Download the update file."""
        if not self.download_url:
            return False

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.download_url) as resp:
                    if resp.status == 200:
                        with open(save_path, "wb") as file_obj:
                            async for chunk in resp.content.iter_chunked(8192):
                                file_obj.write(chunk)
                        logger.info(f"Update downloaded to {save_path}")
                        return True
        except Exception as exc:
            logger.error(f"Download failed: {exc}")
        return False
