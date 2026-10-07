"""Debugger agent.

Takes generated code files plus a feedback string (reviewer comments or
captured test/validation errors) and produces corrected source files.
Returns a list of {{"path", "content"}} dicts. If the feedback indicates
no problems ("LGTM" / no errors), the original files are returned unchanged
to avoid an unnecessary LLM call (cost optimization).
"""

import json

from config.llm_config import generate_response
from utils.json_parser import parse_json_response


def _is_clean(feedback: str) -> bool:
    """True when feedback indicates nothing to fix."""
    if not feedback:
        return True
    upper = feedback.upper()
    return "LGTM" in upper and len(feedback) < 50


async def debugger_agent(code_files: list, feedback: str) -> list:
    """Return fixed code files given reviewer/test feedback.

    ``feedback`` may be reviewer comments or a deterministic error report
    produced by the validation/test tools.
    """
    if _is_clean(feedback):
        return code_files

    files_str = json.dumps(code_files, indent=2)
    system_prompt = f"""You are the Debugger Agent.
Based on the code files and the feedback (reviewer comments or test/validation errors), apply fixes and generate the corrected source code files.
You MUST output ONLY valid JSON in the following format. Do not include markdown blocks or any other text.
{{
    "files": [
        {{
            "path": "backend/main.py",
            "content": "fixed full source code..."
        }}
    ]
}}

Code Files: {files_str}
Feedback: {feedback}
"""
    response = await generate_response(system_prompt, task_type="debugging")
    data = parse_json_response(response)
    fixed = data.get("files", [])
    return fixed if fixed else code_files
