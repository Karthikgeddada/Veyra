import json
import logging
import re

logger = logging.getLogger("devteam_ai.json_parser")

def parse_json_response(response: str):
    """Parse JSON from LLM response with multiple fallback strategies.
    
    Handles:
    - Direct JSON
    - JSON in markdown code blocks
    - JSON embedded in text
    
    Returns empty dict if all parsing fails.
    """
    if not response or not isinstance(response, str):
        logger.warning(
            "parse_json_response received invalid input: type=%s, empty=%s",
            type(response).__name__, not response
        )
        return {}
    
    # Strategy 1: Direct parsing
    try:
        return json.loads(response)
    except json.JSONDecodeError as exc:
        logger.debug("Direct JSON parse failed: %s", exc)
    
    # Strategy 2: Extract from markdown code blocks
    match = re.search(r'```(?:json)?\s*(.*?)\s*```', response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            logger.debug("Markdown block JSON parse failed: %s", exc)
    
    # Strategy 3: Extract first JSON-like structure
    match = re.search(r'(\{.*\}|\[.*\])', response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            logger.debug("Embedded JSON parse failed: %s", exc)
    
    # All strategies failed
    preview = response[:300] + "..." if len(response) > 300 else response
    logger.error(
        "Failed to parse JSON from LLM response. Preview: %s",
        preview
    )
    return {}
