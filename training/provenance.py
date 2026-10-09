"""Git provenance for run logs.

Each entry point (training.train, evaluation.eval_dtu, scripts/costvol_diag.py,
scripts/costvol_profile_probe.py, and pytest via tests/conftest.py) calls
log_git_commit() first, so every queue step's log records the code it ran with.
"""
from __future__ import annotations

import os
import subprocess

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def git_commit_line() -> str:
    """'git commit <HEAD> (clean)', or with the modified tracked files listed. Never raises."""
    try:
        def git(*args):
            return subprocess.run(
                ["git", "-C", REPO_ROOT, *args], capture_output=True, text=True, timeout=30, check=True
            ).stdout

        head = git("rev-parse", "HEAD").strip()
        dirty = [line[3:] for line in git("status", "--porcelain", "--untracked-files=no").splitlines() if line.strip()]
        state = f"uncommitted changes: {', '.join(dirty)}" if dirty else "clean"
        return f"git commit {head} ({state})"
    except Exception as e:  # logging must never stop a run
        return f"git commit unknown ({type(e).__name__}: {e})"


def log_git_commit() -> None:
    print(f"[provenance] {git_commit_line()}", flush=True)
