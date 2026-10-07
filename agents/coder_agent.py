"""Coder (Developer) agent.

Generates all source code files for the project from the prompt and
architecture. Returns a list of {{"path", "content"}} dicts. Uses a
coding-oriented model via the central LLM factory.
"""

from config.llm_config import generate_response
from utils.json_parser import parse_json_response


async def coder_agent(prompt: str, architecture: str) -> list:
    """Generate code files as a list of {path, content} dicts."""
    system_prompt = f"""You are the Code Generator Agent.
Based on the user's prompt and the provided architecture, generate all the required source code files.
You MUST output ONLY valid JSON in the following format. Do not include markdown blocks or any other text.
{{
    "files": [
        {{
            "path": "backend/main.py",
            "content": "full source code..."
        }},
        {{
            "path": "frontend/index.html",
            "content": "full source code..."
        }}
    ]
}}

User Prompt: {prompt}
Architecture: {architecture}
"""
    try:
        response = await generate_response(system_prompt, task_type="coding")
        if not response or not response.strip():
            raise ValueError(
                "Coder agent received empty response from LLM. "
                "Check your API key and model configuration."
            )
        data = parse_json_response(response)
        if not isinstance(data, dict):
            raise ValueError(
                f"Coder agent expected a dict, got {type(data).__name__}. "
                f"Response preview: {response[:500]}"
            )
        files = data.get("files", [])
        if not files:
            raise ValueError(
                f"Coder agent parsed JSON but found no files. "
                f"Parsed data keys: {list(data.keys())}. "
                f"Response preview: {response[:500]}"
            )
        return files
    except Exception as exc:
        # Re-raise with more context for debugging
        raise ValueError(
            f"Coder agent failed: {exc}. "
            f"Check logs for the full LLM response."
        ) from exc
