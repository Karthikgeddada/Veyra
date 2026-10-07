"""Documentation agent.

Writes README.md, setup guide and API docs from the requirements and
generated code. Returns a list of {{"path", "content"}} dicts. Uses a
documentation/fast model via the central LLM factory.
"""

from config.llm_config import generate_response
from utils.json_parser import parse_json_response


async def documentation_agent(code_files: list, requirements: str) -> list:
    """Generate documentation files as a list of {path, content} dicts."""
    # Only send file paths (not full contents) to keep the prompt small.
    file_paths = [f.get("path", "") for f in code_files if f.get("path")]
    system_prompt = f"""You are the Documentation Agent.
Write the README.md, setup guide, and API docs based on the requirements and the generated code file list.
You MUST output ONLY valid JSON in the following format. Do not include markdown blocks.
{{
    "files": [
        {{
            "path": "README.md",
            "content": "markdown content..."
        }}
    ]
}}

Requirements: {requirements}
Generated files: {file_paths}
"""
    response = await generate_response(system_prompt, task_type="documentation")
    data = parse_json_response(response)
    return data.get("files", [])
