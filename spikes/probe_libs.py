"""Spike probe: activity snapshot of agentic-RL library candidates from the GitHub API.

Unauthenticated (60 requests/hour). Prints facts only.

Usage:
    .venv/bin/python spikes/probe_libs.py
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import UTC, datetime

CANDIDATES = (
    "volcengine/verl",
    "PeterGriffinJin/Search-R1",
    "OpenPipe/ART",
    "PrimeIntellect-ai/verifiers",
    "NovaSky-AI/SkyRL",
)
API = "https://api.github.com/repos/"
TIMEOUT_S = 15


def _get(path: str) -> dict[str, object] | None:
    request = urllib.request.Request(API + path, headers={"Accept": "application/vnd.github+json"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            data: dict[str, object] = json.load(response)
            return data
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            print(f"  {path}: HTTP {exc.code}")
        return None
    except urllib.error.URLError as exc:
        print(f"  {path}: {exc.reason}")
        return None


def _days_ago(timestamp: object) -> str:
    if not isinstance(timestamp, str):
        return "-"
    then = datetime.fromisoformat(timestamp)
    return f"{(datetime.now(UTC) - then).days}d ago"


def main() -> None:
    header = (
        f"{'repo':<30}{'stars':>7}  {'pushed':>8}  {'issues':>6}  {'archived':<8}  latest release"
    )
    print(header)
    print("-" * len(header))
    for repo in CANDIDATES:
        info = _get(repo)
        if info is None:
            print(f"{repo:<30}not found")
            continue
        release = _get(f"{repo}/releases/latest")
        tag = (
            f"{release.get('tag_name')} ({_days_ago(release.get('published_at'))})"
            if release
            else "none"
        )
        print(
            f"{repo:<30}{info.get('stargazers_count', '-')!s:>7}  "
            f"{_days_ago(info.get('pushed_at')):>8}  {info.get('open_issues_count', '-')!s:>6}  "
            f"{info.get('archived')!s:<8}  {tag}"
        )


if __name__ == "__main__":
    main()
