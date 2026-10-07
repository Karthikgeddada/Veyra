import json
import re

def parse_json_response(response: str):
    try:
        # First try direct parsing
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    
    # Try finding markdown code blocks
    match = re.search(r'```(?:json)?\s*(.*?)\s*```', response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
            
    # Try to extract anything that looks like a JSON object or array
    match = re.search(r'(\{.*\}|\[.*\])', response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
            
    # Return empty dict if all fails
    print("Failed to parse JSON response completely.")
    return {}
