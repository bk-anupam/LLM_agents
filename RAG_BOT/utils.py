from typing import Optional
import re
import json
from RAG_BOT.logger import logger

def parse_json_answer(content: str) -> Optional[dict]:
    """Helper to parse JSON from final AI message, handling markdown."""
    json_str = content.strip()
    if json_str.startswith("```"):
        # Use regex to extract content within ```json ... ``` or ``` ... ```
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", json_str, re.MULTILINE)
        if match:
            json_str = match.group(1).strip()
        else: # Fallback if regex fails but starts with ```
            json_str = re.sub(r"^```(?:json)?\s*", "", json_str)
            json_str = re.sub(r"\s*```$", "", json_str)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}\nContent was: {content}")
        return None # Return None if parsing fails