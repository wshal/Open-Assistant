"""
Update helper (best-effort).

This is NOT a full self-updater (replacing a running EXE is non-trivial),
but it provides:
- GitHub releases check
- Optional asset selection
- Optional asset download path
"""

from __future__ import annotations

import json
import platform
import re
import urllib.request
from dataclasses import dataclass
from typing import Optional


_VER_RE = re.compile(r"[^0-9.]")


def version_tuple(v: str) -> tuple[int, int, int]:
    v = (v or "").strip().lstrip("v")
    v = _VER_RE.sub("", v)
    parts = [p for p in v.split(".") if p]
    nums = []
    for p in parts[:3]:
        try:
            nums.append(int(p))
        except Exception:
            nums.append(0)
    while len(nums) < 3:
        nums.append(0)
    return tuple(nums[:3])


def is_newer(latest: str, current: str) -> bool:
    return version_tuple(latest) > version_tuple(current)


@dataclass
class ReleaseInfo:
    tag: str
    html_url: str
    asset_url: str = ""
    asset_name: str = ""


def _user_agent() -> str:
    return f"OpenAssist-Updater/1.0 ({platform.system()}; {platform.machine()})"


def fetch_latest_release(repo: str, timeout_s: float = 5.0) -> Optional[dict]:
    url = f"https://api.github.com/repos/{repo}/releases/latest"
    req = urllib.request.Request(url, headers={"User-Agent": _user_agent()})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return json.loads(r.read())


def select_best_asset(release: dict) -> tuple[str, str]:
    assets = release.get("assets", []) if isinstance(release, dict) else []
    if not isinstance(assets, list) or not assets:
        return "", ""

    sysname = (platform.system() or "").lower()
    arch = (platform.machine() or "").lower()

    preferred_ext = []
    if sysname.startswith("win"):
        preferred_ext = [".exe", ".msi", ".zip"]
    elif sysname.startswith("darwin") or sysname.startswith("mac"):
        preferred_ext = [".dmg", ".zip"]
    else:
        preferred_ext = [".appimage", ".tar.gz", ".zip"]

    def score(name: str) -> int:
        n = (name or "").lower()
        s = 0
        for i, ext in enumerate(preferred_ext):
            if n.endswith(ext):
                s += 100 - i * 10
                break
        if arch and arch in n:
            s += 15
        if "portable" in n:
            s -= 5
        return s

    best = None
    best_score = -1
    for a in assets:
        if not isinstance(a, dict):
            continue
        name = a.get("name", "")
        url = a.get("browser_download_url", "")
        if not url:
            continue
        sc = score(name)
        if sc > best_score:
            best_score = sc
            best = a

    if not best:
        return "", ""
    return best.get("browser_download_url", ""), best.get("name", "")


def get_latest_release_info(repo: str, timeout_s: float = 5.0) -> Optional[ReleaseInfo]:
    data = fetch_latest_release(repo, timeout_s=timeout_s)
    if not data:
        return None
    tag = str(data.get("tag_name", "") or "").lstrip("v")
    html_url = str(data.get("html_url", "") or "")
    asset_url, asset_name = select_best_asset(data)
    return ReleaseInfo(tag=tag, html_url=html_url, asset_url=asset_url, asset_name=asset_name)

