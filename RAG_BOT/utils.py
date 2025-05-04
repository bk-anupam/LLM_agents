# /home/bk_anupam/code/LLM_agents/RAG_BOT/utils.py
from typing import Optional
import re
import json
from RAG_BOT.logger import logger

def parse_json_answer(content: str) -> Optional[dict]:
    """
    Extracts and parses a JSON object embedded within a markdown code block.

    Args:
        content: The raw string output from the LLM, potentially containing ```json ... ```.

    Returns:
        The parsed dictionary if successful, None otherwise.
    """
    if not content:
        logger.warning("Attempted to parse empty LLM output.")
        return None

    json_str = content.strip()

    # Regex to find content within ```json ... ``` or ``` ... ```, handling potential variations
    # This pattern accounts for optional whitespace around the JSON content
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", json_str, re.DOTALL | re.IGNORECASE)

    if match:
        json_str = match.group(1).strip() # Extract the content and strip whitespace
        logger.debug("Extracted JSON string from markdown block.")
    else:
        # If no markdown block is found, assume the content *might* be raw JSON
        # We still proceed, but log a warning if it doesn't look like JSON
        if not json_str.startswith('{') and not json_str.startswith('['):
             logger.warning("LLM output did not contain a markdown JSON block and doesn't start with { or [. Attempting direct parse anyway.")
        else:
             logger.debug("No markdown block found, attempting to parse content directly as JSON.")


    try:
        # Attempt to parse the extracted (or original) string
        parsed_json = json.loads(json_str)

        if isinstance(parsed_json, dict):
            # Optionally, validate if the dict has the expected 'answer' key
            if "answer" in parsed_json:
                logger.debug("Successfully parsed JSON answer with 'answer' key.")
                return parsed_json
            else:
                logger.warning("Parsed JSON is a dictionary but does not contain 'answer' key. Returning dict anyway.")
                # Decide if you want to return the dict even without 'answer'
                # return parsed_json
                # Or return None if 'answer' is mandatory
                return None
        else:
            logger.warning(f"Parsed JSON is not a dictionary: {type(parsed_json)}. Content: {json_str[:100]}...")
            return None

    except json.JSONDecodeError as e:
        # Log the specific error and the problematic string portion
        error_context_start = max(0, e.pos - 30)
        error_context_end = min(len(json_str), e.pos + 30)
        error_snippet = json_str[error_context_start:error_context_end]
        # Replace newline characters in the snippet for cleaner logging
        error_snippet_oneline = error_snippet.replace('\n', '\\n')

        logger.error(f"Failed to parse JSON: {e}. Near char {e.pos}: '{error_snippet_oneline}'")
        # Log the full content only at DEBUG level to avoid flooding logs
        logger.debug(f"Full content that failed parsing:\n{content}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during JSON parsing: {e}", exc_info=True)
        logger.debug(f"Full content during unexpected error:\n{content}")
        return None

# Example usage (can be removed or kept for testing)
if __name__ == '__main__':
    test_cases = [
        '```json\n{\n  "answer": "This is a valid answer."\n}\n```',
        '```json \n { "answer": " Another valid answer with space. " } \n ```',
        '```\n{\n  "answer": "No language specified."\n}\n```',
        '{\n  "answer": "Raw JSON string."\n}',
        '   {\n  "answer": "Raw JSON with leading/trailing whitespace."\n}   ',
        '```json\n{\n  "answer": "Invalid JSON structure"\n', # Missing closing brace
        '```json\n{\n  "answer": "Contains\x00invalid control char."\n}\n```', # Example invalid char
        'Some text before ```json\n{\n  "answer": "Text around JSON."\n}\n``` and after.',
        'Plain text response, not JSON.',
        '```json\n{\n "other_key": "No answer key."\n}\n```',
        '' # Empty string
    ]

    print("--- Testing parse_json_answer ---")
    for i, case in enumerate(test_cases):
        print(f"\nTest Case {i+1}:")
        print(f"Input:\n{case}")
        result = parse_json_answer(case)
        print(f"Output: {result}")
        print("-" * 20)
