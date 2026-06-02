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
    asset_sha256: str = ""


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


# C-1: SHA-256 extraction. Release publishers can record per-asset digests either
# in the release body (common: a `SHA256` table or `sha256: <hex>  <name>` lines)
# or by uploading a sibling `*.sha256` / `SHA256SUMS` asset. Both are read here.

_SHA256_RE = re.compile(r"\b([0-9a-fA-F]{64})\b")


def _extract_sha256_for_asset(release: dict, asset_name: str) -> str:
    """Best-effort hex digest extraction for ``asset_name`` from a release dict.

    Looks in (in order):
      1. The release body text (line containing the asset name + 64-hex token).
      2. Any sibling release asset named ``<asset_name>.sha256`` or ``SHA256SUMS``.

    Returns the lowercase hex digest or an empty string. Network errors fetching
    sibling assets are swallowed; the caller treats an empty result as "no
    digest available, do not verify" (but the download path still refuses to
    proceed when verification is required by config).
    """
    if not asset_name:
        return ""
    name_l = asset_name.lower()

    body = str(release.get("body", "") or "") if isinstance(release, dict) else ""
    if body:
        for raw_line in body.splitlines():
            line_l = raw_line.lower()
            if name_l in line_l:
                m = _SHA256_RE.search(raw_line)
                if m:
                    return m.group(1).lower()

    assets = release.get("assets", []) if isinstance(release, dict) else []
    if not isinstance(assets, list):
        return ""

    sibling_targets = {f"{name_l}.sha256", f"{name_l}.sha256sum", "sha256sums", "sha256sums.txt"}
    for a in assets:
        if not isinstance(a, dict):
            continue
        a_name = str(a.get("name", "") or "").lower()
        if a_name not in sibling_targets:
            continue
        url = a.get("browser_download_url", "")
        if not url:
            continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _user_agent()})
            with urllib.request.urlopen(req, timeout=5.0) as r:
                data = r.read(8192).decode("utf-8", errors="replace")
        except Exception:
            continue
        for raw_line in data.splitlines():
            m = _SHA256_RE.search(raw_line)
            if not m:
                continue
            if a_name == f"{name_l}.sha256" or a_name == f"{name_l}.sha256sum" or name_l in raw_line.lower():
                return m.group(1).lower()
    return ""


def get_latest_release_info(repo: str, timeout_s: float = 5.0) -> Optional[ReleaseInfo]:
    data = fetch_latest_release(repo, timeout_s=timeout_s)
    if not data:
        return None
    tag = str(data.get("tag_name", "") or "").lstrip("v")
    html_url = str(data.get("html_url", "") or "")
    asset_url, asset_name = select_best_asset(data)
    asset_sha256 = _extract_sha256_for_asset(data, asset_name) if asset_name else ""
    return ReleaseInfo(
        tag=tag,
        html_url=html_url,
        asset_url=asset_url,
        asset_name=asset_name,
        asset_sha256=asset_sha256,
    )

