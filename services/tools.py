"""Tool layer for DevTeam AI agents.

A small, controlled set of file-system and validation tools that the
workflow (and, where practical, agents) use to inspect and operate on a
generated project. Every operation is sandboxed to a single project root
directory and rejects path traversal outside it, so agents cannot perform
arbitrary destructive commands.

These tools are intentionally synchronous and deterministic -- the LLM
agents produce code/fixes while these tools provide the *verification*
loop (compile checks, test execution) that keeps cost down and avoids
sending the whole repository back to the model.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("devteam_ai.tools")


class ProjectTools:
    """Sandboxed file/test operations bound to one project root."""

    def __init__(self, project_root: str):
        self.root = Path(project_root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Path safety
    # ------------------------------------------------------------------
    def _safe_path(self, relative_path: str) -> Path:
        """Resolve ``relative_path`` inside the project root, refusing escapes."""
        candidate = (self.root / relative_path).resolve()
        if self.root != candidate and self.root not in candidate.parents:
            raise PermissionError(
                f"Path '{relative_path}' escapes the project root and is not allowed."
            )
        return candidate

    # ------------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------------
    def read_file(self, relative_path: str) -> str:
        """Return the contents of a file inside the project."""
        path = self._safe_path(relative_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {relative_path}")
        return path.read_text(encoding="utf-8")

    def write_file(self, relative_path: str, content: str) -> str:
        """Write ``content`` to a file inside the project, creating dirs."""
        path = self._safe_path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"wrote {relative_path} ({len(content)} bytes)"

    def edit_file(self, relative_path: str, old_text: str, new_text: str, replace_all: bool = False) -> str:
        """Replace ``old_text`` with ``new_text`` in a project file."""
        content = self.read_file(relative_path)
        if old_text not in content:
            raise ValueError(f"edit_file: snippet not found in {relative_path}")
        if replace_all:
            content = content.replace(old_text, new_text)
        else:
            content = content.replace(old_text, new_text, 1)
        self.write_file(relative_path, content)
        return f"edited {relative_path}"

    def list_files(self, relative_dir: str = ".") -> List[str]:
        """List all files under ``relative_dir`` recursively."""
        base = self._safe_path(relative_dir)
        if not base.exists():
            return []
        results: List[str] = []
        for p in sorted(base.rglob("*")):
            if p.is_file():
                results.append(str(p.relative_to(self.root)))
        return results

    def search_files(self, pattern: str, relative_dir: str = ".") -> List[Dict[str, Any]]:
        """Grep for ``pattern`` across project files, returning matches."""
        base = self._safe_path(relative_dir)
        regex = re.compile(pattern)
        hits: List[Dict[str, Any]] = []
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            try:
                for i, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                    if regex.search(line):
                        hits.append({"file": str(p.relative_to(self.root)), "line": i, "text": line.strip()})
            except Exception:
                continue
        return hits

    # ------------------------------------------------------------------
    # Validation & test execution
    # ------------------------------------------------------------------
    def run_tests(self, timeout: int = 60) -> Dict[str, Any]:
        """Materialize the project into a temp dir and run available checks.

        Runs ``python -m py_compile`` on every ``.py`` file (fast, deterministic)
        and, if pytest is importable, ``pytest -q``. Both run with a timeout
        inside a copy of the project so the original tree is untouched.
        """
        with tempfile.TemporaryDirectory(prefix="devteam_validate_") as tmp:
            tmp_root = Path(tmp) / "project"
            _copy_tree(self.root, tmp_root)
            compile_result = _compile_python_files(tmp_root)
            pytest_result = _run_pytest(tmp_root, timeout=timeout)
        passed = compile_result["passed"] and pytest_result["passed"]
        combined_output = "\n".join(["=== py_compile ===", compile_result["output"], "", "=== pytest ===", pytest_result["output"]])
        return {"passed": passed, "compile_ok": compile_result["passed"], "pytest_ok": pytest_result["passed"], "output": combined_output.strip(), "errors": compile_result["errors"] + pytest_result["errors"]}

    def inspect_errors(self, run_result: Dict[str, Any]) -> str:
        """Format a run_tests result into a concise error report for the debugger."""
        if run_result.get("passed"):
            return "No errors detected."
        lines = ["Validation/test failures:"]
        for err in run_result.get("errors", []):
            lines.append(f"- {err}")
        out = run_result.get("output", "")
        if out:
            lines.append("")
            lines.append("Captured output (last 2000 chars):")
            lines.append(out[-2000:])
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Controlled command execution (Forge)
    # ------------------------------------------------------------------
    def run_command(self, command: str, working_directory: str = ".", timeout: int = 60, reason: str = "", approved: bool = False) -> Dict[str, Any]:
        """Classify then (optionally) execute a command inside the workspace.

        Never uses a shell (``subprocess.run`` with ``shell=False`` and
        ``shlex``-split argv), which prevents shell injection. Approval is
        *not* decided here -- the LangGraph node decides whether to ask the
        user via ``interrupt()`` and then calls this with ``approved=True``.
        BLOCKED commands are never executed.
        """
        from services.command_policy import CommandRisk, classify_command

        decision = classify_command(command, working_directory, self.root)
        card: Dict[str, Any] = {
            "command": command,
            "working_directory": working_directory or ".",
            "reason": reason,
            "risk": decision.risk.value,
            "risk_reason": decision.reason,
        }
        if decision.risk == CommandRisk.BLOCKED:
            return {**card, "executed": False, "blocked": True, "needs_approval": False, "stdout": "", "stderr": decision.reason, "exit_code": -1, "duration_ms": 0, "timeout": False}
        if decision.risk == CommandRisk.APPROVAL_REQUIRED and not approved:
            return {**card, "executed": False, "blocked": False, "needs_approval": True}
        result = self._execute_command(command, working_directory, timeout)
        return {**card, "executed": True, "blocked": False, "needs_approval": False, **result}

    def _execute_command(self, command: str, working_directory: str, timeout: int) -> Dict[str, Any]:
        """Run an already-validated command without a shell, capturing output."""
        wd = self._safe_path(working_directory) if working_directory and working_directory not in {".", "./"} else self.root
        try:
            argv = shlex.split(command)
        except ValueError as exc:
            return {"stdout": "", "stderr": f"invalid command: {exc}", "exit_code": -1, "duration_ms": 0, "timeout": False}
        if not argv:
            return {"stdout": "", "stderr": "empty command", "exit_code": -1, "duration_ms": 0, "timeout": False}
        start = time.time()
        try:
            proc = subprocess.run(argv, cwd=str(wd), capture_output=True, text=True, timeout=timeout)
            return {"stdout": proc.stdout or "", "stderr": proc.stderr or "", "exit_code": proc.returncode, "duration_ms": int((time.time() - start) * 1000), "timeout": False}
        except subprocess.TimeoutExpired as exc:
            out = exc.stdout if isinstance(exc.stdout, str) else ""
            err = (exc.stderr if isinstance(exc.stderr, str) else "") + "\n[command timed out]"
            return {"stdout": out, "stderr": err, "exit_code": -1, "duration_ms": int((time.time() - start) * 1000), "timeout": True}
        except FileNotFoundError:
            return {"stdout": "", "stderr": f"command not found: {argv[0]}", "exit_code": -1, "duration_ms": int((time.time() - start) * 1000), "timeout": False}
        except Exception as exc:  # noqa: BLE001
            return {"stdout": "", "stderr": str(exc), "exit_code": -1, "duration_ms": int((time.time() - start) * 1000), "timeout": False}

    def git_status(self) -> Dict[str, Any]:
        """Read-only ``git status --porcelain``. Safe; no approval needed."""
        res = self.run_command("git status --porcelain", reason="Inspect repository status", approved=True)
        res["tool"] = "git_status"
        return res

    def git_diff(self) -> Dict[str, Any]:
        """Read-only ``git diff``. Safe; no approval needed."""
        res = self.run_command("git diff", reason="Inspect uncommitted changes", approved=True)
        res["tool"] = "git_diff"
        return res

    # ------------------------------------------------------------------
    # Project understanding (Forge)
    # ------------------------------------------------------------------
    def analyze_project(self) -> Dict[str, Any]:
        """Analyze the project structure and return a summary.

        Deterministic — no LLM call. Returns file tree, file count by
        extension, and the detected stack.
        """
        from services.context_manager import detect_stack, filter_files, is_excluded

        files = self.list_files()
        filtered = filter_files(files)
        # Count by extension.
        ext_counts: Dict[str, int] = {}
        for fp in filtered:
            ext = os.path.splitext(fp)[1].lower() or "(no ext)"
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
        stack = detect_stack(self.root)
        return {
            "total_files": len(files),
            "filtered_files": len(filtered),
            "excluded_files": len(files) - len(filtered),
            "files": filtered,
            "extensions": ext_counts,
            "stack": stack,
        }

    def detect_stack(self) -> Dict[str, Any]:
        """Detect and return the technology stack of the project.

        Convenience wrapper around :func:`context_manager.detect_stack`.
        Returns language, framework, database, package_manager,
        test_framework, entry_points, and config_files.
        """
        from services.context_manager import detect_stack

        return detect_stack(self.root)

    # ------------------------------------------------------------------
    # Bulk helpers used by the workflow
    # ------------------------------------------------------------------
    def write_files(self, files: List[Dict[str, str]]) -> List[str]:
        """Write a list of {path, content} dicts into the project."""
        written: List[str] = []
        for f in files:
            path = f.get("path")
            content = f.get("content", "")
            if not path:
                continue
            self.write_file(path, content)
            written.append(path)
        return written


# ---------------------------------------------------------------------------
# Module-level helpers (also usable without an instance)
# ---------------------------------------------------------------------------
def validate_files(code_files: List[Dict[str, str]]) -> Tuple[bool, str]:
    """Lightweight structural validation of generated files.

    Returns ``(ok, message)``. Checks that files are non-empty and that
    Python files at least parse. Deterministic; costs no LLM tokens.
    """
    if not code_files:
        return False, "No code files were generated."
    errors: List[str] = []
    for f in code_files:
        path = f.get("path", "")
        content = f.get("content", "")
        if not content.strip():
            errors.append(f"{path}: empty file")
            continue
        if path.endswith(".py"):
            try:
                compile(content, path, "exec")
            except SyntaxError as e:
                errors.append(f"{path}: syntax error: {e.msg} (line {e.lineno})")
    if errors:
        return False, "\n".join(errors)
    return True, "All files structurally valid."


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _copy_tree(src: Path, dst: Path) -> None:
    import shutil

    dst.mkdir(parents=True, exist_ok=True)
    for item in src.rglob("*"):
        rel = item.relative_to(src)
        target = dst / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif item.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def _compile_python_files(root: Path) -> Dict[str, Any]:
    errors: List[str] = []
    output_lines: List[str] = []
    py_files = sorted(root.rglob("*.py"))
    if not py_files:
        return {"passed": True, "output": "(no python files)", "errors": []}
    for pf in py_files:
        try:
            compile(pf.read_text(encoding="utf-8", errors="ignore"), str(pf), "exec")
        except SyntaxError as e:
            errors.append(f"{pf.relative_to(root)}: {e.msg} (line {e.lineno})")
            output_lines.append(f"FAIL {pf.relative_to(root)}: {e.msg} (line {e.lineno})")
        else:
            output_lines.append(f"ok   {pf.relative_to(root)}")
    return {"passed": not errors, "output": "\n".join(output_lines), "errors": errors}


def _run_pytest(root: Path, timeout: int = 60) -> Dict[str, Any]:
    test_files = list(root.rglob("test_*.py")) + list(root.rglob("*_test.py"))
    if not test_files:
        return {"passed": True, "output": "(no tests found)", "errors": []}
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--no-header", "--tb=short", str(root)],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        passed = proc.returncode == 0
        errors: List[str] = []
        if not passed:
            for line in out.splitlines():
                if line.startswith("E ") or "Error" in line or "error" in line:
                    errors.append(line.strip())
            if not errors:
                errors.append("pytest reported failures (see captured output)")
        return {"passed": passed, "output": out[-3000:], "errors": errors}
    except FileNotFoundError:
        return {"passed": True, "output": "(pytest not installed; skipped)", "errors": []}
    except subprocess.TimeoutExpired:
        return {"passed": False, "output": "pytest timed out", "errors": ["pytest timed out"]}
    except Exception as exc:  # noqa: BLE001
        return {"passed": False, "output": str(exc), "errors": [str(exc)]}