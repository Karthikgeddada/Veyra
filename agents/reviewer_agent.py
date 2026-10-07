"""Reviewer agent.

Reviews generated code files for best practices, structure, naming
conventions and logic issues. Returns "LGTM" when the code is acceptable,
otherwise a concise list of issues. Uses a coding/review model via the
central LLM factory.
"""

import json

from config.llm_config import generate_response


async def reviewer_agent(code_files: list) -> str:
    """Return a code review as text ("LGTM" if no issues)."""
    # Send a compact summary rather than full file bodies to save tokens
    # when there are many files; include full content only for small projects.
    files_str = json.dumps(code_files, indent=2)
    system_prompt = f"""You are the Reviewer Agent.
Review the provided code files for best practices, structure, naming conventions, and logic issues.
List the potential issues or suggestions for improvement. If the code is good, say "LGTM".

Code Files: {files_str}
"""
    return await generate_response(system_prompt, task_type="review")
