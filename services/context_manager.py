"""Context manager for Forge LLM requests.

Ensures that only *relevant* project context is sent to the model, never
the entire repository. Excludes secrets, caches, virtual environments,
binaries, and generated artifacts. Redacts secret-like patterns from any
text that will be included in an LLM prompt.

Deterministic — no LLM calls, no I/O beyond reading files from the
project root.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("devteam_ai.context")

# ---------------------------------------------------------------------------
# Exclusion patterns — files and directories that must NEVER be sent to the LLM
# ---------------------------------------------------------------------------
EXCLUDE_DIRS: Set[str] = {
    ".git", ".venv", "venv", "env", "node_modules", "__pycache__",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".eggs",
    ".idea", ".vscode", ".sass-cache", "dist", "build",
    ".next", ".nuxt", ".cache", ".turbo",
}

EXCLUDE_FILES: Set[str] = {
    ".env", ".env.local", ".env.production", ".env.development",
    ".env.staging", ".env.test",
    ".gitignore", ".gitattributes",
    ".npmrc", ".pypirc", ".netrc",
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
}

EXCLUDE_EXTENSIONS: Set[str] = {
    ".pyc", ".pyo", ".pyd", ".so", ".dll", ".dylib",
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
    ".egg", ".whl",
    ".bin", ".exe", ".obj", ".o", ".a", ".lib",
    ".jpg", ".jpeg", ".png", ".gif", ".ico", ".svg", ".bmp",
    ".mp3", ".mp4", ".avi", ".mov", ".wav",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".sqlite", ".db", ".sqlite3",
    ".pyc",
}

# Files that may contain secrets — never send full content to the LLM.
SECRET_FILE_PATTERNS: List[str] = [
    ".env*", "*credentials*", "*secret*", "*token*", "*api_key*",
    "*.pem", "*.key", "*.crt", "id_rsa*", "id_ed25519*",
]

# Secret-like patterns to redact from file contents.
_SECRET_RE = re.compile(
    r'('
    r'(?:sk-or-v1-|nvapi-)[A-Za-z0-9\-_]+'
    r'|(?:ghp_|gho_|ghu_|ghs_|ghr_)[A-Za-z0-9]+'
    r'|AIza[0-9A-Za-z\-_]+'
    r'|(?:(?:API_KEY|TOKEN|SECRET|PASSWORD|PRIVATE_KEY)\s*=\s*)["\']?[^\s"\']+'
    r')',
    re.IGNORECASE,
)

# Maximum file size (bytes) to include in context.
MAX_FILE_SIZE = 50_000  # 50 KB

# Maximum number of files to include in a single context build.
MAX_CONTEXT_FILES = 20


def is_excluded(path: Path) -> bool:
    """Return True if *path* should be excluded from LLM context."""
    parts = path.parts
    # Check directory components.
    for part in parts:
        if part in EXCLUDE_DIRS:
            return True
    # Check file name.
    name = path.name
    if name in EXCLUDE_FILES:
        return True
    # Check extension.
    ext = path.suffix.lower()
    if ext in EXCLUDE_EXTENSIONS:
        return True
    # Check secret file patterns.
    for pattern in SECRET_FILE_PATTERNS:
        if fnmatch.fnmatch(name, pattern):
            return True
    return False


def filter_files(file_paths: List[str]) -> List[str]:
    """Filter out excluded paths from a list of relative file paths."""
    result: List[str] = []
    for fp in file_paths:
        p = Path(fp)
        if not is_excluded(p):
            result.append(fp)
    return result


def redact_secrets(text: str) -> str:
    """Replace secret-like patterns in *text* with [REDACTED]."""
    if not text:
        return text
    return _SECRET_RE.sub('[REDACTED]', text)


def read_file_safe(path: Path, max_size: int = MAX_FILE_SIZE) -> Optional[str]:
    """Read a file's text content, returning None if too large or binary."""
    try:
        size = path.stat().st_size
        if size > max_size:
            return f"(file too large: {size} bytes)"
        content = path.read_text(encoding='utf-8', errors='ignore')
        # Skip files that are mostly binary (null bytes).
        if '\x00' in content[:1000]:
            return None
        return redact_secrets(content)
    except Exception:
        return None


def detect_stack(project_root: Path) -> Dict[str, Any]:
    """Detect the technology stack of the project at *project_root*.

    Returns a dict with language, framework, database, package_manager,
    test_framework, and entry_points.
    """
    info: Dict[str, Any] = {
        "language": "unknown",
        "framework": "unknown",
        "backend": "unknown",
        "frontend": "unknown",
        "database": "unknown",
        "package_manager": "unknown",
        "test_framework": "unknown",
        "entry_points": [],
        "config_files": [],
    }

    root = Path(project_root)

    # --- Python ---
    py_files = list(root.rglob("*.py"))
    py_files = [f for f in py_files if not is_excluded(f)]
    if py_files or (root / "requirements.txt").exists() or (root / "pyproject.toml").exists() or (root / "setup.py").exists():
        info["language"] = "python"
        if (root / "requirements.txt").exists():
            info["config_files"].append("requirements.txt")
        if (root / "pyproject.toml").exists():
            info["config_files"].append("pyproject.toml")
        if (root / "setup.py").exists():
            info["config_files"].append("setup.py")
        # Detect framework.
        req_content = ""
        for req_name in ("requirements.txt", "pyproject.toml"):
            req_path = root / req_name
            if req_path.exists():
                req_content += req_path.read_text(encoding="utf-8", errors="ignore").lower()
        if "fastapi" in req_content:
            info["framework"] = "fastapi"
            info["backend"] = "fastapi"
        elif "flask" in req_content:
            info["framework"] = "flask"
            info["backend"] = "flask"
        elif "django" in req_content:
            info["framework"] = "django"
            info["backend"] = "django"
        # Test framework.
        if any("pytest" in f.name or "test_" in f.name for f in py_files):
            info["test_framework"] = "pytest"
        elif (root / "pytest.ini").exists() or (root / "pyproject.toml").exists():
            info["test_framework"] = "pytest"
        # Package manager.
        if (root / "pyproject.toml").exists() and (root / "poetry.lock").exists():
            info["package_manager"] = "poetry"
        elif (root / "uv.lock").exists():
            info["package_manager"] = "uv"
        else:
            info["package_manager"] = "pip"
        # Entry points.
        for candidate in ("main.py", "app.py", "run.py", "manage.py"):
            if (root / candidate).exists():
                info["entry_points"].append(candidate)
        if (root / "app" / "main.py").exists():
            info["entry_points"].append("app/main.py")
        # Database.
        if "sqlalchemy" in req_content or "sqlite" in req_content:
            info["database"] = "sqlite/sqlalchemy"
        elif "psycopg" in req_content or "asyncpg" in req_content:
            info["database"] = "postgresql"

    # --- JavaScript / TypeScript ---
    pkg_path = root / "package.json"
    if pkg_path.exists():
        info["config_files"].append("package.json")
        pkg_content = pkg_path.read_text(encoding="utf-8", errors="ignore")
        try:
            pkg = json.loads(pkg_content)
        except Exception:
            pkg = {}
        deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
        if "typescript" in deps or any(root.rglob("*.ts")):
            info["language"] = "typescript"
        else:
            info["language"] = "javascript"
        if "react" in deps:
            info["framework"] = "react"
            info["frontend"] = "react"
        elif "vue" in deps:
            info["framework"] = "vue"
            info["frontend"] = "vue"
        elif "next" in deps:
            info["framework"] = "next"
            info["frontend"] = "next"
        elif "express" in deps:
            info["framework"] = "express"
            info["backend"] = "express"
        if "jest" in deps or "vitest" in deps:
            info["test_framework"] = "jest" if "jest" in deps else "vitest"
        info["package_manager"] = "npm"
        if (root / "yarn.lock").exists():
            info["package_manager"] = "yarn"
        elif (root / "pnpm-lock.yaml").exists():
            info["package_manager"] = "pnpm"
        for candidate in ("index.js", "index.ts", "server.js", "server.ts", "app.js", "app.ts"):
            if (root / candidate).exists():
                info["entry_points"].append(candidate)

    # --- SQL / database files ---
    sql_files = list(root.rglob("*.sql"))
    sql_files = [f for f in sql_files if not is_excluded(f)]
    if sql_files and info["database"] == "unknown":
        info["database"] = "sql"

    return info


def build_context(
    project_root: str,
    prompt: str,
    task_type: Optional[str] = None,
    relevant_files: Optional[List[str]] = None,
    recent_errors: Optional[str] = None,
    test_results: Optional[str] = None,
    agent_decisions: Optional[List[str]] = None,
    max_files: int = MAX_CONTEXT_FILES,
) -> str:
    """Build a compact, relevant context string for an LLM prompt.

    Includes only:
        - User requirement (prompt)
        - Current task type
        - Project structure (file tree, filtered)
        - Relevant file contents (with secrets redacted)
        - Recent test results / errors
        - Previous agent decisions

    Excludes secrets, caches, virtual environments, binaries, etc.
    """
    root = Path(project_root)
    if not root.exists():
        return f"Project root does not exist: {project_root}"

    sections: List[str] = []

    # 1. User requirement.
    sections.append(f"## User Request\n{prompt}")

    # 2. Task type.
    if task_type:
        sections.append(f"## Current Task\n{task_type}")

    # 3. Project structure (filtered file tree).
    all_files: List[str] = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(root))
            if not is_excluded(Path(rel)):
                all_files.append(rel)
    if all_files:
        tree_text = "\n".join(all_files[:100])  # cap at 100 entries
        sections.append(f"## Project Structure\n{tree_text}")

    # 4. Stack detection.
    stack = detect_stack(root)
    stack_lines = []
    for k, v in stack.items():
        if v and v != "unknown":
            if isinstance(v, list):
                stack_lines.append(f"- {k}: {', '.join(str(x) for x in v)}")
            else:
                stack_lines.append(f"- {k}: {v}")
    if stack_lines:
        sections.append("## Detected Stack\n" + "\n".join(stack_lines))

    # 5. Relevant file contents.
    files_to_read = relevant_files or []
    if not files_to_read and all_files:
        # Heuristic: prioritize files matching the task type.
        priority_exts: Dict[str, List[str]] = {
            "coding": [".py", ".js", ".ts", ".html", ".css"],
            "debugging": [".py", ".js", ".ts"],
            "testing": ["test_*.py", "*_test.py", ".spec.js", ".spec.ts"],
            "review": [".py", ".js", ".ts"],
            "documentation": [".py", ".js", ".ts", ".md"],
            "architecture": [".py", ".js", ".ts", ".yaml", ".yml", ".json"],
        }
        exts = priority_exts.get(task_type or "", [".py", ".js", ".ts"])
        for fp in all_files:
            if any(fp.endswith(e) or fnmatch.fnmatch(Path(fp).name, e) for e in exts):
                files_to_read.append(fp)
            if len(files_to_read) >= max_files:
                break

    file_contents: List[str] = []
    for fp in files_to_read[:max_files]:
        full_path = root / fp
        content = read_file_safe(full_path)
        if content is not None:
            file_contents.append(f"### {fp}\n```\n{content}\n```")
    if file_contents:
        sections.append("## Relevant Files\n" + "\n\n".join(file_contents))

    # 6. Recent errors.
    if recent_errors:
        sections.append(f"## Recent Errors\n{redact_secrets(recent_errors)}")

    # 7. Test results.
    if test_results:
        sections.append(f"## Test Results\n{redact_secrets(test_results[:3000])}")

    # 8. Previous agent decisions.
    if agent_decisions:
        sections.append("## Previous Decisions\n" + "\n".join(f"- {d}" for d in agent_decisions[-10:]))

    return "\n\n".join(sections)
