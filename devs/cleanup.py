#!/usr/bin/env python3
"""
Weekly Maintenance Assistant for CoderingsTool
===============================================
Interactive script that walks you through backup hygiene:
  1. Git status — unstaged/uncommitted changes, offer to commit + tag
  2. Full src snapshots — large backup/ folders, suggest cleanup
  3. Old backup files — individual files in src/backup/ and src/utils/backup/

Run:  python devs/cleanup.py
      python devs/cleanup.py --dry-run   (preview only, no deletions)
"""

import os
import sys
import shutil
import subprocess
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKUP_DIRS_FULL_SNAPSHOTS = [PROJECT_ROOT / "backup"]
BACKUP_DIRS_INDIVIDUAL = [
    PROJECT_ROOT / "src" / "backup",
    PROJECT_ROOT / "src" / "utils" / "backup",
]
FILE_AGE_THRESHOLD_DAYS = 30

# ─── Terminal Colors ──────────────────────────────────────────────────────────

class C:
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    RESET = "\033[0m"


def header(title: str):
    width = 60
    print(f"\n{C.BOLD}{C.BLUE}{'─' * width}{C.RESET}")
    print(f"{C.BOLD}{C.BLUE}  {title}{C.RESET}")
    print(f"{C.BOLD}{C.BLUE}{'─' * width}{C.RESET}\n")


def fmt_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def fmt_age(days: int) -> str:
    if days == 0:
        return "today"
    elif days == 1:
        return "1 day ago"
    elif days < 30:
        return f"{days} days ago"
    elif days < 60:
        return f"{days // 30} month ago"
    else:
        return f"{days // 30} months ago"


def dir_size(path: Path) -> int:
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def dir_file_count(path: Path) -> int:
    return sum(1 for f in path.rglob("*") if f.is_file())


def ask_yn(prompt: str, default: bool = False) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        answer = input(f"{C.YELLOW}  {prompt} {suffix}: {C.RESET}").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    if not answer:
        return default
    return answer in ("y", "yes")


def ask_choice(prompt: str, options: list[str]) -> list[int]:
    """Ask user to select from numbered options. Returns list of selected indices."""
    print(f"{C.YELLOW}  {prompt}{C.RESET}")
    print(f"{C.DIM}  Enter numbers separated by commas, 'all', or 'none':{C.RESET}")
    try:
        answer = input(f"{C.YELLOW}  > {C.RESET}").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return []
    if answer in ("none", "n", ""):
        return []
    if answer == "all":
        return list(range(len(options)))
    selected = []
    for part in answer.split(","):
        part = part.strip()
        if part.isdigit():
            idx = int(part) - 1  # 1-indexed for user
            if 0 <= idx < len(options):
                selected.append(idx)
    return selected


def run_git(*args) -> tuple[int, str]:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout.strip()


def get_git_tags_with_dates() -> dict[str, datetime]:
    """Get all git tags mapped to their dates."""
    code, output = run_git("tag", "-l", "--format=%(refname:short) %(creatordate:iso)")
    tags = {}
    if code == 0 and output:
        for line in output.splitlines():
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                tag_name = parts[0]
                try:
                    date_str = parts[1].strip()
                    # Parse ISO format: 2026-01-31 12:00:00 +0100
                    tag_date = datetime.strptime(date_str[:19], "%Y-%m-%d %H:%M:%S")
                    tags[tag_name] = tag_date
                except (ValueError, IndexError):
                    pass
    return tags


# ─── Stage 1: Git Status ─────────────────────────────────────────────────────

def stage_git_status(dry_run: bool):
    header("Stage 1: Git Status")

    # Check for uncommitted changes
    code, status = run_git("status", "--porcelain")
    if code != 0:
        print(f"  {C.RED}Not a git repository or git error.{C.RESET}")
        return

    if not status:
        print(f"  {C.GREEN}Working tree is clean — nothing to commit.{C.RESET}")
        return

    lines = status.strip().splitlines()
    staged = [l for l in lines if l[0] != " " and l[0] != "?"]
    unstaged = [l for l in lines if l[0] == " " and l[1] != " "]
    untracked = [l for l in lines if l.startswith("??")]

    # Calculate sizes of changed files
    total_size = 0
    for line in lines:
        filepath = line[3:].strip().strip('"')
        full_path = PROJECT_ROOT / filepath
        if full_path.is_file():
            total_size += full_path.stat().st_size

    print(f"  {C.CYAN}Changes detected:{C.RESET} ({fmt_size(total_size)} total)")
    if staged:
        print(f"    {C.GREEN}Staged:    {len(staged)} files{C.RESET}")
    if unstaged:
        print(f"    {C.YELLOW}Modified:  {len(unstaged)} files{C.RESET}")
    if untracked:
        print(f"    {C.RED}Untracked: {len(untracked)} files{C.RESET}")

    # Show recent commits for context
    _, log = run_git("log", "--oneline", "-5")
    if log:
        print(f"\n  {C.DIM}Recent commits:{C.RESET}")
        for line in log.splitlines():
            print(f"    {C.DIM}{line}{C.RESET}")

    # Check if remote is ahead/behind
    run_git("fetch", "--quiet")
    _, ahead_behind = run_git("rev-list", "--left-right", "--count", "HEAD...@{upstream}")
    if ahead_behind:
        parts = ahead_behind.split()
        if len(parts) == 2:
            ahead, behind = int(parts[0]), int(parts[1])
            if ahead > 0:
                print(f"\n  {C.YELLOW}Local is {ahead} commit(s) ahead of remote.{C.RESET}")
            if behind > 0:
                print(f"  {C.YELLOW}Local is {behind} commit(s) behind remote.{C.RESET}")

    print()
    if dry_run:
        print(f"  {C.DIM}[dry-run] Would ask to commit + tag as milestone.{C.RESET}")
        return

    if ask_yn("Create a milestone commit + tag and push to GitHub?"):
        # Stage all changes
        tag_name = f"milestone-{datetime.now().strftime('%Y-%m-%d')}"
        msg = input(f"{C.YELLOW}  Commit message (or Enter for '{tag_name}'): {C.RESET}").strip()
        if not msg:
            msg = f"Milestone backup: {tag_name}"

        run_git("add", "-A")
        code, _ = run_git("commit", "-m", msg)
        if code != 0:
            print(f"  {C.RED}Commit failed.{C.RESET}")
            return

        # Check if tag already exists, append suffix if so
        actual_tag = tag_name
        _, existing = run_git("tag", "-l", tag_name)
        if existing:
            suffix = 1
            while True:
                actual_tag = f"{tag_name}-{suffix}"
                _, check = run_git("tag", "-l", actual_tag)
                if not check:
                    break
                suffix += 1

        run_git("tag", actual_tag)
        code, _ = run_git("push", "--follow-tags")
        if code == 0:
            print(f"  {C.GREEN}Committed, tagged as '{actual_tag}', and pushed.{C.RESET}")
        else:
            print(f"  {C.YELLOW}Committed and tagged locally. Push failed (check remote).{C.RESET}")


# ─── Stage 2: Full Src Snapshots ─────────────────────────────────────────────

def stage_full_snapshots(dry_run: bool):
    header("Stage 2: Full src/ Snapshots")

    snapshots = []
    for backup_dir in BACKUP_DIRS_FULL_SNAPSHOTS:
        if not backup_dir.exists():
            continue
        for entry in sorted(backup_dir.iterdir()):
            if entry.is_dir():
                size = dir_size(entry)
                file_count = dir_file_count(entry)
                mtime = datetime.fromtimestamp(entry.stat().st_mtime)
                age_days = (datetime.now() - mtime).days
                snapshots.append({
                    "path": entry,
                    "name": entry.name,
                    "size": size,
                    "files": file_count,
                    "mtime": mtime,
                    "age_days": age_days,
                })

    if not snapshots:
        print(f"  {C.GREEN}No full snapshots found. Nothing to do.{C.RESET}")
        return

    # Sort oldest first
    snapshots.sort(key=lambda s: s["mtime"])

    total_size = sum(s["size"] for s in snapshots)
    print(f"  Found {C.BOLD}{len(snapshots)}{C.RESET} snapshots totaling {C.BOLD}{fmt_size(total_size)}{C.RESET}:\n")

    # Get git tags for cross-reference
    tags = get_git_tags_with_dates()

    for i, snap in enumerate(snapshots, 1):
        # Check if a git tag exists within 2 days of this snapshot
        has_git = False
        for tag_name, tag_date in tags.items():
            if abs((tag_date - snap["mtime"]).days) <= 2:
                has_git = True
                break

        git_indicator = f"{C.GREEN}[has git tag]{C.RESET}" if has_git else f"{C.DIM}[no git tag]{C.RESET}"
        age_color = C.RED if snap["age_days"] > 60 else C.YELLOW if snap["age_days"] > 30 else C.RESET

        print(f"  {C.BOLD}{i:>3}.{C.RESET} {snap['name']}")
        print(f"       {fmt_size(snap['size']):>8}  |  {snap['files']:>4} files  |  {age_color}{fmt_age(snap['age_days'])}{C.RESET}  |  {git_indicator}")

    print()
    if dry_run:
        print(f"  {C.DIM}[dry-run] Would ask which snapshots to delete.{C.RESET}")
        return

    selected = ask_choice(
        "Which snapshots should we delete?",
        [s["name"] for s in snapshots],
    )

    if not selected:
        print(f"  {C.DIM}Skipped — no snapshots deleted.{C.RESET}")
        return

    freed = 0
    for idx in selected:
        snap = snapshots[idx]
        print(f"  Deleting {snap['name']}...", end=" ")
        shutil.rmtree(snap["path"])
        freed += snap["size"]
        print(f"{C.GREEN}done{C.RESET}")

    print(f"\n  {C.GREEN}Freed {fmt_size(freed)} from {len(selected)} snapshot(s).{C.RESET}")


# ─── Stage 3: Old Individual Backup Files ────────────────────────────────────

def stage_old_files(dry_run: bool, age_days: int = FILE_AGE_THRESHOLD_DAYS):
    header(f"Stage 3: Backup Files Older Than {age_days} Days")

    cutoff = datetime.now() - timedelta(days=age_days)
    old_files = []

    for backup_dir in BACKUP_DIRS_INDIVIDUAL:
        if not backup_dir.exists():
            continue
        for f in sorted(backup_dir.rglob("*")):
            if not f.is_file():
                continue
            mtime = datetime.fromtimestamp(f.stat().st_mtime)
            if mtime < cutoff:
                rel = f.relative_to(PROJECT_ROOT)
                old_files.append({
                    "path": f,
                    "rel": str(rel),
                    "size": f.stat().st_size,
                    "mtime": mtime,
                    "age_days": (datetime.now() - mtime).days,
                })

    if not old_files:
        print(f"  {C.GREEN}No backup files older than {age_days} days. Clean!{C.RESET}")
        return

    # Sort by directory then age
    old_files.sort(key=lambda f: (f["path"].parent, -f["age_days"]))

    total_size = sum(f["size"] for f in old_files)
    print(f"  Found {C.BOLD}{len(old_files)}{C.RESET} files older than {age_days} days, totaling {C.BOLD}{fmt_size(total_size)}{C.RESET}:\n")

    current_dir = None
    for i, f in enumerate(old_files, 1):
        parent = str(f["path"].parent.relative_to(PROJECT_ROOT))
        if parent != current_dir:
            current_dir = parent
            print(f"\n  {C.CYAN}{current_dir}/{C.RESET}")

        age_color = C.RED if f["age_days"] > 90 else C.YELLOW
        print(f"    {i:>3}. {f['path'].name}")
        print(f"         {fmt_size(f['size']):>8}  |  {age_color}{fmt_age(f['age_days'])}{C.RESET}")

    print()
    if dry_run:
        print(f"  {C.DIM}[dry-run] Would ask to delete these files.{C.RESET}")
        return

    if ask_yn(f"Delete all {len(old_files)} files older than {age_days} days ({fmt_size(total_size)})?"):
        freed = 0
        for f in old_files:
            f["path"].unlink()
            freed += f["size"]
        # Clean up empty directories
        for backup_dir in BACKUP_DIRS_INDIVIDUAL:
            if backup_dir.exists():
                for d in sorted(backup_dir.rglob("*"), reverse=True):
                    if d.is_dir() and not any(d.iterdir()):
                        d.rmdir()
        print(f"  {C.GREEN}Deleted {len(old_files)} files, freed {fmt_size(freed)}.{C.RESET}")
    else:
        # Offer individual selection
        if ask_yn("Select individual files to delete instead?"):
            selected = ask_choice(
                "Which files should we delete?",
                [f["rel"] for f in old_files],
            )
            if selected:
                freed = 0
                for idx in selected:
                    f = old_files[idx]
                    f["path"].unlink()
                    freed += f["size"]
                print(f"  {C.GREEN}Deleted {len(selected)} files, freed {fmt_size(freed)}.{C.RESET}")
            else:
                print(f"  {C.DIM}Skipped — no files deleted.{C.RESET}")


# ─── Summary ─────────────────────────────────────────────────────────────────

def show_summary():
    header("Summary")
    total = 0
    for d in BACKUP_DIRS_FULL_SNAPSHOTS + BACKUP_DIRS_INDIVIDUAL:
        if d.exists():
            size = dir_size(d)
            total += size
            rel = d.relative_to(PROJECT_ROOT)
            print(f"  {str(rel) + '/':.<40} {fmt_size(size):>10}")
    print(f"  {'':─<40} {'':─>10}")
    print(f"  {C.BOLD}{'Total':.<40} {fmt_size(total):>10}{C.RESET}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CoderingsTool Maintenance Assistant")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, no deletions")
    parser.add_argument("--skip-git", action="store_true", help="Skip git status stage")
    parser.add_argument("--age", type=int, default=FILE_AGE_THRESHOLD_DAYS,
                        help=f"Age threshold in days (default: {FILE_AGE_THRESHOLD_DAYS})")
    args = parser.parse_args()

    age_threshold = args.age

    print(f"\n{C.BOLD}{C.CYAN}  CoderingsTool — Maintenance Assistant{C.RESET}")
    print(f"{C.DIM}  {datetime.now().strftime('%A %d %B %Y, %H:%M')}{C.RESET}")
    if args.dry_run:
        print(f"  {C.YELLOW}[DRY RUN MODE — no changes will be made]{C.RESET}")

    try:
        if not args.skip_git:
            stage_git_status(args.dry_run)
        stage_full_snapshots(args.dry_run)
        stage_old_files(args.dry_run, age_threshold)
        show_summary()
    except KeyboardInterrupt:
        print(f"\n\n  {C.DIM}Interrupted. No further changes.{C.RESET}")
        sys.exit(0)

    # Record last cleanup time (used by shell login reminder)
    if not args.dry_run:
        marker = PROJECT_ROOT / "devs" / ".last_cleanup"
        marker.touch()

    print(f"\n{C.DIM}  Done. Run again anytime: python devs/cleanup.py{C.RESET}\n")


if __name__ == "__main__":
    main()