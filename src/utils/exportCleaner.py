"""
exportCleaner.py — Cleanup old files in exports/ subdirectories.

Two modes:
  - auto_cleanup():    called silently from VerboseCapture.__exit__().
  - collect_expired(): returns lists of files to delete; called from
                       devs/cleanup.py for interactive review.

Retention policy (hybrid):
  1. Always keep the N newest files per logical group key.
  2. Delete everything else older than M days.
  Both thresholds are configurable via ExportCleanupConfig in config.py.
"""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import NamedTuple, Optional


# ── Filename pattern helpers ──────────────────────────────────────────────────

# verbose_logs: {base}_{var_key}_step{N}_{YYYYMMDD}_{HHMMSS}.txt
_VERBOSE_LOG_RE = re.compile(
    r"^(?P<group>.+_step\d+)_\d{8}_\d{6}\.txt$"
)

# cluster_results: {type}_{version}_{dataset}_{var}_{sample}_{YYYYMMDD}.txt
_CLUSTER_RESULT_RE = re.compile(
    r"^(?P<group>.+)_\d{8}\.txt$"
)


def _group_key_verbose(name: str) -> Optional[str]:
    """Extract group key from a verbose log filename."""
    m = _VERBOSE_LOG_RE.match(name)
    return m.group("group") if m else None


def _group_key_cluster(name: str) -> Optional[str]:
    """Extract group key from a cluster result filename."""
    m = _CLUSTER_RESULT_RE.match(name)
    return m.group("group") if m else None


# ── Core collection logic ─────────────────────────────────────────────────────

class ExpiredFile(NamedTuple):
    path: Path
    age_days: int
    size: int
    reason: str  # "age"


def collect_expired(
    exports_dir: Path,
    max_age_days: int = 30,
    keep_latest_n: int = 3,
) -> list[ExpiredFile]:
    """
    Return all files under exports/ that should be deleted.

    Rules per subdirectory:
      verbose_logs/    — grouped by (base_name + var_key + step).
                         Keep the newest keep_latest_n per group;
                         delete the rest if older than max_age_days.
      cluster_results/ — grouped by everything except trailing _YYYYMMDD.
                         Same keep/age policy.
      prompts/         — never auto-deleted (no timestamp, small count).
    """
    expired: list[ExpiredFile] = []
    now = datetime.now()
    cutoff = now - timedelta(days=max_age_days)

    subdir_configs = [
        (exports_dir / "verbose_logs", _group_key_verbose),
        (exports_dir / "cluster_results", _group_key_cluster),
    ]

    for subdir, keyfn in subdir_configs:
        if not subdir.exists():
            continue

        # Group files by their logical key
        groups: dict[str, list[Path]] = defaultdict(list)
        ungrouped: list[Path] = []

        for f in subdir.iterdir():
            if not f.is_file():
                continue
            key = keyfn(f.name)
            if key:
                groups[key].append(f)
            else:
                ungrouped.append(f)

        for key, files in groups.items():
            # Sort newest first (timestamp suffix is lexically sortable)
            files.sort(key=lambda p: p.name, reverse=True)

            for rank, f in enumerate(files):
                stat = f.stat()
                mtime = datetime.fromtimestamp(stat.st_mtime)
                age_days = (now - mtime).days

                if rank < keep_latest_n:
                    continue  # Always keep the N newest per group

                if mtime < cutoff:
                    expired.append(ExpiredFile(
                        path=f,
                        age_days=age_days,
                        size=stat.st_size,
                        reason="age",
                    ))

        # Files that don't match any pattern: delete if old
        for f in ungrouped:
            stat = f.stat()
            mtime = datetime.fromtimestamp(stat.st_mtime)
            if mtime < cutoff:
                expired.append(ExpiredFile(
                    path=f,
                    age_days=(now - mtime).days,
                    size=stat.st_size,
                    reason="age",
                ))

    return expired


def auto_cleanup(
    exports_dir: Path,
    max_age_days: int = 30,
    keep_latest_n: int = 3,
    silent: bool = True,
) -> int:
    """
    Delete expired files and return the count deleted.

    Called from VerboseCapture.__exit__(). Failures are swallowed
    so they never interrupt the pipeline.
    """
    try:
        expired = collect_expired(exports_dir, max_age_days, keep_latest_n)
        for ef in expired:
            try:
                ef.path.unlink()
            except OSError:
                pass
        if not silent and expired:
            print(f"[exportCleaner] Deleted {len(expired)} old export files.")
        return len(expired)
    except Exception:
        return 0
