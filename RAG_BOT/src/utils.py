# /home/bk_anupam/code/LLM_agents/RAG_BOT/utils.py
from typing import Optional
import re
import json
import unicodedata # Added for character category checking
import codecs
from RAG_BOT.src.logger import logger
from langdetect import detect, LangDetectException, DetectorFactory
from langchain_core.documents import Document
from typing import List 
from datetime import datetime

def _devanagari_to_ascii_digits(devanagari_string: str) -> str:
    """Converts Devanagari numerals in a string to ASCII digits."""
    mapping = {
        '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
        '५': '5', '६': '6', '७': '7', '८': '8', '९': '9'
    }
    return "".join(mapping.get(char, char) for char in devanagari_string)


def extract_date_from_text(text: str, return_date_format: str = "%Y-%m-%d") -> Optional[str]:
    """
    Attempts to extract a date from the given text and returns it in return_date_format.
    Args:
        text (str): The text to search for a date.
        return_date_format (str): The format to return the date in. Default is "%Y-%m-%d"(YYYY-MM-DD).
    Returns:
        str or None: The extracted date in return_date_format if found, otherwise None.
    """
    # Specific date patterns to avoid ambiguity
    date_patterns = [
        (r"(\d{4})-(\d{2})-(\d{2})", "%Y-%m-%d"),  # YYYY-MM-DD
        (r"([०-९]{4})-([०-९]{2})-([०-९]{2})", "%Y-%m-%d"), # YYYY-MM-DD (Devanagari)

        (r"(\d{2})/(\d{2})/(\d{4})", "%d/%m/%Y"), # DD/MM/YYYY
        (r"([०-९]{2})/([०-९]{2})/([०-९]{4})", "%d/%m/%Y"), # DD/MM/YYYY (Devanagari)

        (r"(\d{2})\.(\d{2})\.(\d{4})", "%d.%m.%Y"), # DD.MM.YYYY
        (r"([०-९]{2})\.([०-९]{2})\.([०-९]{4})", "%d.%m.%Y"), # DD.MM.YYYY (Devanagari)

        (r"(\d{1,2})\.(\d{1,2})\.(\d{4})", "%d.%m.%Y"), # D.M.YYYY, DD.M.YYYY, D.MM.YYYY
        (r"([०-९]{1,2})\.([०-९]{1,2})\.([०-९]{4})", "%d.%m.%Y"), # D.M.YYYY (Devanagari)

        (r"(\d{1,2})/(\d{1,2})/(\d{4})", "%d/%m/%Y"), # D/M/YYYY, DD/M/YYYY, D/MM/YYYY
        (r"([०-९]{1,2})/([०-९]{1,2})/([०-९]{4})", "%d/%m/%Y"), # D/M/YYYY (Devanagari)

        (r"(\d{1,2})-(\d{1,2})-(\d{4})", "%d-%m-%Y"), # D-M-YYYY, DD-M-YYYY, D-MM-YYYY
        (r"([०-९]{1,2})-([०-९]{1,2})-([०-९]{4})", "%d-%m-%Y"), # D-M-YYYY (Devanagari)

        (r"(\d{2})\.(\d{2})\.(\d{2})", "%d.%m.%y"), # DD.MM.YY
        (r"([०-९]{2})\.([०-९]{2})\.([०-९]{2})", "%d.%m.%y"), # DD.MM.YY (Devanagari)

        (r"(\d{2})/(\d{2})/(\d{2})", "%d/%m/%y"), # DD/MM/YY
        (r"([०-९]{2})/([०-९]{2})/([०-९]{2})", "%d/%m/%y"), # DD/MM/YY (Devanagari)

        (r"(\d{2})-(\d{2})-(\d{2})", "%d-%m-%y"), # DD-MM-YY
        (r"([०-९]{2})-([०-९]{2})-([०-९]{2})", "%d-%m-%y"), # DD-MM-YY (Devanagari)

        (r"(\d{1,2})\.(\d{1,2})\.(\d{2})", "%d.%m.%y"), # D.M.YY, DD.M.YY, D.MM.YY
        (r"([०-९]{1,2})\.([०-९]{1,2})\.([०-९]{2})", "%d.%m.%y"), # D.M.YY (Devanagari)

        (r"(\d{1,2})/(\d{1,2})/(\d{2})", "%d/%m/%y"), # D/M/YY, DD/M/YY, D/MM/YY
        (r"([०-९]{1,2})/([०-९]{1,2})/([०-९]{2})", "%d/%m/%y"), # D/M/YY (Devanagari)

        (r"(\d{1,2})-(\d{1,2})-(\d{2})", "%d-%m-%y"), # D-M-YY, DD-M-YY, D-MM-YY
        (r"([०-९]{1,2})-([०-९]{1,2})-([०-९]{2})", "%d-%m-%y"), # D-M-YY (Devanagari)
        # Add other common formats if needed (e.g., "January 21, 1969")
    ]

    for pattern, date_format in date_patterns:
        match = re.search(pattern, text)
        if match:
            matched_date_str = match.group(0)
            ascii_date_str = _devanagari_to_ascii_digits(matched_date_str)
            try:
                # Attempt to parse the date using the specified format
                date_obj = datetime.strptime(ascii_date_str, date_format)
                return date_obj.strftime(return_date_format)
            except ValueError as e:
                logger.warning(f"Date format '{date_format}' matched for '{matched_date_str}' (converted to '{ascii_date_str}'), but couldn't parse. Error: {e}")                
            except Exception as e:
                    logger.error(f"Unexpected error parsing date '{matched_date_str}' (converted to '{ascii_date_str}') with format '{date_format}': {e}")                    

    logger.info(f"No date pattern matched in text: '{text[:100]}...'")
    return None # Return None if no pattern matched or parsing failed


def remove_control_characters(text: str) -> str:
    """
    Removes Unicode control characters (Cc category) from a string,
    except for common whitespace characters (newline, tab, carriage return).
    """
    if not isinstance(text, str):
        return text
    cleaned_chars = []
    for char_val in text:
        if char_val in ('\n', '\r', '\t'): # Keep common whitespace
            cleaned_chars.append(char_val)
        elif unicodedata.category(char_val) != 'Cc': # Keep if not a Control character
            cleaned_chars.append(char_val)
    cleaned_text = "".join(cleaned_chars)
    if text != cleaned_text:
        orig_snippet = text[:100].replace('\n', '\\n')
        cleaned_snippet = cleaned_text[:100].replace('\n', '\\n')
        logger.info(
            f"Cleaned control characters from text. Original snippet (first 100 chars, newlines escaped): "
            f" '{orig_snippet}', Cleaned snippet: '{cleaned_snippet}'"
        )
    return cleaned_text

def filter_to_devanagari_and_essentials(text: str) -> str:
    """
    Filters a string to keep only Devanagari characters, standard European digits,
    common punctuation, and standard whitespace.
    """
    if not isinstance(text, str):
        return text

    allowed_chars = []
    for char_val in text:
        # Devanagari Unicode block: U+0900 to U+097F
        if '\u0900' <= char_val <= '\u097F':
            allowed_chars.append(char_val)
        # Standard European Digits (0-9)
        elif '0' <= char_val <= '9':
            allowed_chars.append(char_val)
        # Common Punctuation (English and Devanagari Dandas)
        # Add or remove punctuation as needed for your specific use case
        elif char_val in ('.', ',', '?', '!', '-', '(', ')', '\'', '"', ':', ';', '।', '॥', '/', '*', '_'):
            allowed_chars.append(char_val)
        # Standard Whitespace (space, newline, tab, carriage return)
        elif char_val in (' ', '\n', '\r', '\t'):
            allowed_chars.append(char_val)
        # allow english characters (a-z, A-Z) for English text
        elif 'a' <= char_val <= 'z' or 'A' <= char_val <= 'Z':
            allowed_chars.append(char_val)
        else:
            # Optional: Log discarded characters for debugging
            logger.info(f"Discarding character: '{char_val}' (Unicode: U+{ord(char_val):04X})")

    filtered_text = "".join(allowed_chars)
    if text != filtered_text:
        logger.info(f"Filtered non-Devanagari/non-essential characters. Snippet before: '{text[:100]}', Snippet after: '{filtered_text[:100]}'")
    return filtered_text

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
        parsed_json = json.loads(json_str, strict=False)

        if isinstance(parsed_json, dict):
            # Optionally, validate if the dict has the expected 'answer' key
            if "answer" in parsed_json: # Check if 'answer' key exists
                answer_text = parsed_json.get("answer") # Use .get for safety
                if isinstance(answer_text, str):
                    # Apply the cleaning function to the extracted answer string
                    # First, remove invisible control characters
                    temp_cleaned_answer = remove_control_characters(answer_text)
                    # Then, filter to keep only Devanagari and essentials
                    cleaned_answer = filter_to_devanagari_and_essentials(temp_cleaned_answer)
                    parsed_json["answer"] = cleaned_answer
                    logger.debug("Successfully parsed JSON, 'answer' key found and content cleaned.")
                elif answer_text is not None: # If answer is not None but also not a string
                    logger.warning(f"'answer' key found but content is not a string: {type(answer_text)}. Skipping cleaning.")
                # If answer_text is None (key exists but value is null), it will be returned as is.
                return parsed_json # Return the dict with the (potentially cleaned) answer
            else:
                # If "answer" key is not present
                logger.warning("Parsed JSON is a dictionary but does not contain 'answer' key. Returning dict as is.")
                return parsed_json # Return the dictionary as is, if other keys might be useful.                                   
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
        logger.error(f"Failed to parse JSON: {e}. Near char {e.pos}: '{error_snippet_oneline}.\n Full content: {json_str}...'")

        # --- Fallback Regex Recovery ---
        # If standard parsing fails, it's often due to unescaped quotes in the 'answer' string.
        # We'll try to extract the 'answer' and 'references' content using regex as a fallback.
        logger.warning("Attempting to recover content using regex due to JSON parsing error.")
        try:
            answer_match = re.search(r'"answer"\s*:\s*"(.*?)",\s*"references"', json_str, re.DOTALL)
            references_match = re.search(r'"references"\s*:\s*(\[.*?\])', json_str, re.DOTALL)

            if answer_match and references_match:
                # Manually reconstruct the dictionary
                # Use codecs.decode to properly un-escape the raw string from the regex capture.
                # This correctly handles escaped newlines (e.g., '\\n' -> '\n') and
                # escaped quotes (e.g., '\\"' -> '"').
                answer_text = codecs.decode(answer_match.group(1), 'unicode_escape')

                recovered_data = {
                    "answer": answer_text,
                    "references": json.loads(references_match.group(1)) # Parse references separately
                }
                logger.info("Successfully recovered 'answer' and 'references' using regex.")
                # Re-run the cleaning process on the recovered data
                cleaned_answer = remove_control_characters(recovered_data["answer"])
                recovered_data["answer"] = filter_to_devanagari_and_essentials(cleaned_answer)
                return recovered_data
            else:
                logger.error("Regex recovery failed. Could not find 'answer' and 'references' blocks.")
                return None
        except (json.JSONDecodeError, AttributeError) as recovery_error:
            logger.error(f"An error occurred during regex recovery: {recovery_error}")
            logger.debug(f"Full content that failed parsing and recovery:\n{content}")
            return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during JSON parsing: {e}", exc_info=True)
        logger.debug(f"Full content during unexpected error:\n{content}")
        return None


def detect_document_language(documents: List[Document], file_name_for_logging: str = "uploaded document", 
                             default_lang: str = 'en') -> str:
    """
    Detects the language of the content within a list of Document objects.

    Args:
        documents: A list of Langchain Document objects.
        file_name_for_logging: The name of the file (for logging purposes).
        default_lang: The language code to return if detection fails.

    Returns:
        The detected language code (e.g., 'en', 'hi') or the default language.
    """
    logger.debug(f"Attempting to detect language for: {file_name_for_logging}")
    DetectorFactory.seed = 0  # Set seed for reproducibility
    try:
        if not documents:
            logger.warning(f"No documents provided for '{file_name_for_logging}'. Cannot detect language. Defaulting to '{default_lang}'.")
            return default_lang
        # Concatenate content from first few documents/pages for detection
        # Using page_content attribute of Langchain Document
        sample_text = " ".join([doc.page_content for doc in documents[:5]]).strip()
        if not sample_text:
            logger.warning(f"Document(s) '{file_name_for_logging}' contain no text to detect language from. Defaulting to '{default_lang}'.")
            return default_lang

        detected_lang = detect(sample_text)
        logger.info(f"Detected language '{detected_lang}' for: {file_name_for_logging}")
        if detected_lang not in ['en', 'hi']:
            logger.info(f"Sample text used for detection: '{sample_text}'")
        return detected_lang
    except LangDetectException as lang_err:
        logger.warning(f"Could not detect language for '{file_name_for_logging}': {lang_err}. Defaulting to '{default_lang}'.")
        return default_lang
    except Exception as e:
        logger.error(f"Error during language detection for '{file_name_for_logging}': {e}", exc_info=True)
        return default_lang


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
