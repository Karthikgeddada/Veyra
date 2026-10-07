"""Command safety policy for Forge (DevTeam AI).

Classifies a shell command into one of three risk levels before it is ever
executed:

    SAFE             - read-only / low-risk; may auto-run (see auto_approve_safe)
    APPROVAL_REQUIRED - modifies the project/environment; needs explicit user approval
    BLOCKED           - destructive, exfiltrating, or workspace-escaping; never runs

Classification is *structured*: the command is parsed with ``shlex`` and the
program + subcommand are inspected, so it does not rely on naive substring
matching. Working-directory validation reuses the same path-traversal rules
as :class:`services.tools.ProjectTools`.

The allowlists/patterns are module-level constants so they can be tuned
without touching call sites.
"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple


class CommandRisk(Enum):
    SAFE = "safe"
    APPROVAL_REQUIRED = "approval"
    BLOCKED = "blocked"


@dataclass
class CommandDecision:
    risk: CommandRisk
    reason: str
    argv: List[str]


# ---------------------------------------------------------------------------
# Configurable policy tables
# ---------------------------------------------------------------------------
# Read-only programs that are safe to run without approval.
SAFE_PROGRAMS = {
    "pytest", "ruff", "flake8", "pylint", "mypy", "pyflakes",
    "node", "npm", "npx", "python", "python3", "git",
}

# git subcommands that are read-only -> safe
GIT_SAFE_SUB = {"status", "diff", "log", "show", "branch", "blame", "ls-files", "rev-parse"}
# git subcommands that mutate state -> approval
GIT_APPROVAL_SUB = {"commit", "push", "pull", "checkout", "merge", "reset", "add", "rm", "stash", "rebase", "cherry-pick", "clone", "init", "tag"}

# npm subcommands that are read-only -> safe
NPM_SAFE_SUB = {"test", "run", "run-script"}  # run is safe for build/test scripts
NPM_SAFE_RUN_SCRIPTS = {"build", "test", "lint", "check"}

# Programs that install/modify the environment -> approval
APPROVAL_PROGRAMS = {"pip", "pip3", "uv", "poetry", "npm", "npx", "yarn", "pnpm", "git", "docker", "uvicorn", "gunicorn"}

# Blocked patterns (compiled, case-insensitive) matched against the raw command.
BLOCKED_PATTERNS = [
    (r"\brm\s+(-[a-z]*r[a-z]*f|--force)\b", "recursive/forced delete (rm -rf)"),
    (r"\bsudo\b", "privilege escalation (sudo)"),
    (r"\bmkfs\b", "filesystem format (mkfs)"),
    (r"\bdd\b.*\bof=", "low-level disk write (dd)"),
    (r":\(\)\s*\{", "fork bomb"),
    (r"\b(shutdown|reboot|halt|poweroff)\b", "system power control"),
    (r"\b(kill|pkill|killall)\b", "process kill"),
    (r"\bsystemctl\b", "service control (systemctl)"),
    (r"\b(chmod|chown)\b.*(\b777\b|/etc|/usr|/bin|/var|/root)", "unsafe permission change on system path"),
    (r"\b(curl|wget)\b.*\|\s*(sh|bash|zsh)", "remote script execution (curl|sh)"),
    (r"\b(curl|wget)\b", "network fetch (curl/wget) blocked by default"),
    (r"\beval\b", "eval blocked"),
    # secret / credential access
    (r"\b(printenv|env)\b(\s|$)", "environment variable disclosure"),
    (r"\b(set|export)\b.*(API_KEY|TOKEN|SECRET|PASSWORD)", "secret disclosure"),
    (r"(\$\{?OPENROUTER|OPENROUTER_API_KEY)", "API key access"),
    (r"(\~\/\.ssh|\.ssh\/|\~\/\.aws|\.aws\/)", "credential directory access"),
    (r"(\~\/\.git-credentials|\.npmrc|\.pypirc|\.netrc)", "credential file access"),
    (r"\/etc\/(passwd|shadow|gshadow|sudoers)", "system file access"),
    (r"\bhistory\b", "shell history access"),
    (r"\bcmd\.exe\b|\bpowershell\b", "foreign shell"),
]
_BLOCKED_RE = [(re.compile(p, re.IGNORECASE), msg) for p, msg in BLOCKED_PATTERNS]

# Path tokens that indicate workspace escape / system access inside command args.
ESCAPE_PATH_RE = re.compile(r"(\.\./|\.\.|^/etc/|^/root/|^/var/|^/usr/|^/bin/|^/sbin/|^/boot/|^/proc/|^/sys/|^[A-Za-z]:\\)")


def _parse(command: str) -> Tuple[List[str], str]:
    """Return (argv, normalized_lower_string). Robust to shlex errors."""
    try:
        argv = shlex.split(command)
    except ValueError:
        argv = command.split()
    return argv, " ".join(argv).strip().lower()


def command_signature(argv: List[str]) -> str:
    """A stable signature used for session-scoped approval caching."""
    if not argv:
        return ""
    prog = os.path.basename(argv[0])
    # include a non-flag subcommand when present (e.g. 'git commit', 'pip install')
    sub = ""
    for a in argv[1:]:
        if not a.startswith("-"):
            sub = a
            break
    return f"{prog} {sub}".strip()


def _working_dir_ok(working_directory: str, project_root: Path) -> bool:
    """True if working_directory resolves inside project_root (no escape)."""
    if not working_directory or working_directory in {".", "./"}:
        return True
    try:
        root = Path(project_root).resolve()
        candidate = (root / working_directory).resolve()
        return candidate == root or root in candidate.parents
    except Exception:
        return False


def classify_command(command: str, working_directory: str, project_root) -> CommandDecision:
    """Classify ``command`` for the Forge command policy.

    Returns a :class:`CommandDecision`. Never executes anything.
    """
    root = Path(project_root).resolve()
    if not command or not command.strip():
        return CommandDecision(CommandRisk.BLOCKED, "empty command", [])

    argv, norm = _parse(command)
    if not argv:
        return CommandDecision(CommandRisk.BLOCKED, "unparseable command", [])

    # 1. Working-directory escape check first.
    if not _working_dir_ok(working_directory, root):
        return CommandDecision(CommandRisk.BLOCKED, "working directory escapes the project workspace", argv)

    raw = command.strip()

    # 2. Blocked regex patterns.
    for rx, msg in _BLOCKED_RE:
        if rx.search(raw):
            return CommandDecision(CommandRisk.BLOCKED, f"blocked: {msg}", argv)

    # 3. Blocked: path-traversal / system paths in arguments.
    for a in argv:
        if ESCAPE_PATH_RE.search(a):
            return CommandDecision(CommandRisk.BLOCKED, "blocked: command argument references a path outside the workspace", argv)

    # 4. Block any attempt to read the project's own .env / secrets.
    if any(a in {".env", "../.env", "~/.env"} or a.endswith("/.env") for a in argv):
        return CommandDecision(CommandRisk.BLOCKED, "blocked: secret file (.env) access", argv)

    prog = os.path.basename(argv[0])
    sub = next((a for a in argv[1:] if not a.startswith("-")), "")

    # 5. SAFE classification (read-only).
    if prog == "pytest":
        return CommandDecision(CommandRisk.SAFE, "test runner (pytest)", argv)
    if prog in {"python", "python3"}:
        if "--version" in argv and len(argv) == 2:
            return CommandDecision(CommandRisk.SAFE, "version check", argv)
        if "-m" in argv:
            mi = argv.index("-m")
            mod = argv[mi + 1] if mi + 1 < len(argv) else ""
            if mod in {"pytest", "py_compile"}:
                return CommandDecision(CommandRisk.SAFE, f"python -m {mod}", argv)
        # running arbitrary python scripts -> approval (could do anything)
        return CommandDecision(CommandRisk.APPROVAL_REQUIRED, "executing a python script", argv)
    if prog == "git":
        if sub in GIT_SAFE_SUB:
            return CommandDecision(CommandRisk.SAFE, f"git {sub} (read-only)", argv)
        if sub in GIT_APPROVAL_SUB:
            return CommandDecision(CommandRisk.APPROVAL_REQUIRED, f"git {sub} mutates repository state", argv)
        return CommandDecision(CommandRisk.APPROVAL_REQUIRED, "git subcommand not in safe list", argv)
    if prog in {"ruff", "flake8", "pylint", "mypy", "pyflakes"}:
        return CommandDecision(CommandRisk.SAFE, f"linter ({prog})", argv)
    if prog == "node":
        if "--version" in argv or "-v" in argv:
            return CommandDecision(CommandRisk.SAFE, "version check", argv)
        return CommandDecision(CommandRisk.APPROVAL_REQUIRED, "executing node script", argv)
    if prog in {"npm", "npx", "yarn", "pnpm"}:
        if sub in {"--version", "-v"}:
            return CommandDecision(CommandRisk.SAFE, "version check", argv)
        if prog == "npm" and sub in NPM_SAFE_SUB:
            if sub == "run":
                script = next((a for a in argv[2:] if not a.startswith("-")), "")
                if script in NPM_SAFE_RUN_SCRIPTS:
                    return CommandDecision(CommandRisk.SAFE, f"npm run {script}", argv)
            elif sub in {"test"}:
                return CommandDecision(CommandRisk.SAFE, "npm test", argv)
        if sub in {"install", "i", "add", "uninstall", "remove", "update"}:
            return CommandDecision(CommandRisk.APPROVAL_REQUIRED, f"{prog} {sub} modifies dependencies", argv)
        return CommandDecision(CommandRisk.APPROVAL_REQUIRED, f"{prog} subcommand not in safe list", argv)
    if prog in {"pip", "pip3", "uv", "poetry"}:
        return CommandDecision(CommandRisk.APPROVAL_REQUIRED, f"{prog} modifies the environment", argv)

    # 6. Unknown program -> require approval by default (defense in depth).
    return CommandDecision(CommandRisk.APPROVAL_REQUIRED, f"unknown program '{prog}' requires approval", argv)


def needs_approval(decision: CommandDecision, auto_approve_safe: bool) -> bool:
    """True when the command should pause for human approval."""
    if decision.risk == CommandRisk.BLOCKED:
        return False  # blocked commands never run; no approval prompt
    if decision.risk == CommandRisk.SAFE and auto_approve_safe:
        return False
    return decision.risk == CommandRisk.APPROVAL_REQUIRED or decision.risk == CommandRisk.SAFE