# /home/bk_anupam/code/LLM_agents/RAG_BOT/utils.py
from typing import Optional
import re
import json
import unicodedata # Added for character category checking
import codecs
from RAG_BOT.src.logger import logger
from langdetect import detect, LangDetectException, DetectorFactory
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
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


def detect_text_language(text: str, default_lang: str = 'en') -> str:
    """
    Detects the language of the user question using langdetect.
    Falls back to default_lang if detection fails.
    """    
    DetectorFactory.seed = 0  # Set seed for reproducibility
    try:
        if not text.strip():
            logger.warning("Empty text provided for language detection. Defaulting to '%s'.", default_lang)
            return default_lang
        detected_lang = detect(text)
        logger.info(f"Detected language '{detected_lang}' for text.")
        return detected_lang
    except LangDetectException as lang_err:
        logger.warning(f"Could not detect language for text: {lang_err}. Defaulting to '{default_lang}'.")
        return default_lang
    except Exception as e:
        logger.error(f"Error during language detection for user question: {e}", exc_info=True)
        return default_lang


def get_conversational_history(messages: List[BaseMessage]) -> List[BaseMessage]:
    """
    Filters the message history to be more conversational for an LLM.

    This function strips out messages that are not part of the natural
    human-AI dialogue, such as tool invocation messages and their raw outputs.
    It keeps HumanMessages and AIMessages that contain a final, readable answer,
    while filtering out AIMessages that are only for invoking tools.

    Args:
        messages: The complete list of messages from the agent state.

    Returns:
        A cleaned list of messages suitable for conversational context.
    """
    conversational_messages = []
    for msg in messages:
        # Skip tool messages entirely
        if isinstance(msg, ToolMessage):
            continue
        # Skip AIMessages that are only tool calls with no textual content
        if isinstance(msg, AIMessage) and msg.tool_calls and not msg.content.strip():
            continue
        conversational_messages.append(msg)
    return conversational_messages