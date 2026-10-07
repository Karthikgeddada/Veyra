"""Tester agent.

Generates test cases and validation scripts for the provided code files.
Returns a list of {{"path", "content"}} dicts. Uses a coding model via the
central LLM factory.
"""

import json

from config.llm_config import generate_response
from utils.json_parser import parse_json_response


async def tester_agent(code_files: list) -> list:
    """Generate test files as a list of {path, content} dicts."""
    files_str = json.dumps(code_files, indent=2)
    system_prompt = f"""You are the Tester Agent.
Generate test cases and validation scripts for the provided code files.
You MUST output ONLY valid JSON in the following format. Do not include markdown blocks.
{{
    "files": [
        {{
            "path": "tests/test_main.py",
            "content": "pytest code..."
        }}
    ]
}}

Code Files: {files_str}
"""
    response = await generate_response(system_prompt, task_type="coding")
    data = parse_json_response(response)
    return data.get("files", [])
