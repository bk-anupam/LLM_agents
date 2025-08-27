from typing import Optional, Dict, Any, List
import re
import json
import codecs
from RAG_BOT.src.utils import remove_control_characters, filter_to_devanagari_and_essentials
from RAG_BOT.src.logger import logger


class JsonParser:
    """
    Encapsulates enhanced JSON parsing logic for handling LLM responses.
    This class provides a robust way to parse JSON from raw LLM output,
    handling various edge cases and malformed structures.
    """

    def parse_json_answer(self, content: str) -> Optional[Dict[str, Any]]:
        """
        Enhanced JSON parser that can handle multiple edge cases from LLM responses.
        
        This method attempts multiple strategies to extract meaningful content:
        1. Standard markdown JSON block parsing
        2. Flexible JSON structure detection
        3. Content extraction from malformed JSON
        4. Plain text wrapping as fallback
        5. Multiple regex patterns for different scenarios
        
        Args:
            content: Raw string output from LLM
            
        Returns:
            Dictionary with 'answer' key if successful, None otherwise
        """
        if not content:
            logger.warning("Attempted to parse empty LLM output.")
            return None

        original_content = content
        content = content.strip()
        
        # List of parsing strategies to attempt in order
        parsing_strategies = [
            self._try_markdown_json_extraction,
            self._try_direct_json_parsing,
            self._try_flexible_json_patterns,
            self._try_malformed_json_recovery,
            self._try_quote_extraction,
            self._try_plain_text_fallback,
        ]

        for strategy in parsing_strategies:
            try:
                result = strategy(content)
                if result:
                    logger.debug(f"Successfully parsed using strategy '{strategy.__name__}'")
                    return self._clean_and_validate_result(result)
            except Exception as e:
                logger.error(f"Error in parsing strategy {strategy.__name__}: {e}")
                continue
        
        logger.error(f"All parsing strategies failed for content: {original_content[:200]}...")
        return None

    def _try_markdown_json_extraction(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from markdown code blocks with various formats."""
        patterns = [
            r"```json\s*(.*?)\s*```",           # Standard ```json ... ```
            r"```JSON\s*(.*?)\s*```",           # Uppercase JSON
            r"```\s*(.*?)\s*```",               # Generic code block
            r"```json\s*(.*?)$",                # Unclosed json block
            r"```\s*json\s*(.*?)\s*```",        # Space before json
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                json_str = match.group(1).strip()
                try:
                    return json.loads(json_str, strict=False)
                except json.JSONDecodeError:
                    continue
        return None

    def _try_direct_json_parsing(self, content: str) -> Optional[Dict[str, Any]]:
        """Try to parse content directly as JSON."""
        # Clean up common issues
        cleaned_content = content.strip()
        
        # Remove potential markdown remnants
        cleaned_content = re.sub(r'^```(?:json)?', '', cleaned_content, flags=re.IGNORECASE)
        cleaned_content = re.sub(r'```$', '', cleaned_content)
        cleaned_content = cleaned_content.strip()
        
        if cleaned_content.startswith('{') and cleaned_content.endswith('}'):
            try:
                return json.loads(cleaned_content, strict=False)
            except json.JSONDecodeError:
                pass
        
        return None

    def _try_flexible_json_patterns(self, content: str) -> Optional[Dict[str, Any]]:
        """Use multiple regex patterns to extract JSON-like structures."""
        
        # Pattern 1: Standard answer field
        patterns = [
            # Match: "answer": "content"
            r'"answer"\s*:\s*"([^"]*(?:\\.[^"]*)*)"',
            # Match: "answer":"content" (no spaces)
            r'"answer"\s*:\s*"([^"]*)"',
            # Match: 'answer': 'content' (single quotes)
            r"'answer'\s*:\s*'([^']*)'",
            # Match: answer: "content" (no quotes around key)
            r'answer\s*:\s*"([^"]*)"'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                answer_text = match.group(1)
                # Unescape common escape sequences
                answer_text = self._unescape_text(answer_text)
                
                # Try to extract references if they exist
                references = self._extract_references(content)
                
                result = {"answer": answer_text}
                if references:
                    result["references"] = references
                
                return result
        
        return None

    def _try_malformed_json_recovery(self, content: str) -> Optional[Dict[str, Any]]:
        """Recover content from malformed JSON structures."""
        
        # Look for JSON-like structure with potential issues
        json_like_patterns = [
            # Find JSON block even with missing closing brace
            r'\{\s*"answer"\s*:\s*"([^"]*(?:\\.[^"]*)*)"[^}]*', 
            # Find nested structure
            r'\{\s*[^{}]*"answer"\s*:\s*"([^"]*(?:\\.[^"]*)*)"[^}]*\}?',
            # More lenient pattern
            r'[{{\[].*?"answer"\s*:\s*"([^"]*(?:\\.[^"]*)*)".*?[}}\]]?',
        ]
        
        for pattern in json_like_patterns:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                answer_text = self._unescape_text(match.group(1))            
                # Try to fix and parse the JSON
                json_candidate = self._attempt_json_fix(content)
                if json_candidate:
                    return json_candidate
                
                # If fixing fails, return extracted content
                return {"answer": answer_text}
        
        return None

    def _try_quote_extraction(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract content between quotes as a last resort."""
        
        # Look for quoted content that might be the answer
        quote_patterns = [
            r'"([^"]{50,})"',  # Long quoted strings (likely answers)
            r"'([^']{50,})'",  # Single quoted long strings
            r'["""]([^"""]{50,})["""]',  # Smart quotes
        ]
        
        for pattern in quote_patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            if matches:
                # Take the longest match as it's likely the main content
                longest_match = max(matches, key=len)
                return {"answer": longest_match.strip()}
        
        return None

    def _try_plain_text_fallback(self, content: str) -> Optional[Dict[str, Any]]:
        """Use plain text content as answer if it looks meaningful."""
        
        # Clean the content
        cleaned = content.strip()
        
        # Remove common markdown/code artifacts
        cleaned = re.sub(r'^```.*?$', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'^```', '', cleaned)
        cleaned = re.sub(r'```$', '', cleaned)
        cleaned = cleaned.strip()
        
        # Check if it looks like meaningful content (not just error messages)
        if (len(cleaned) > 20 and 
            not cleaned.lower().startswith('error') and
            not cleaned.lower().startswith('sorry') and
            not cleaned.lower().startswith('i cannot')):
            
            logger.warning(f"Using plain text fallback for content: {cleaned[:100]}...")
            return {"answer": cleaned}
        
        return None

    def _extract_references(self, content: str) -> Optional[List[str]]:
        """Extract references from content if they exist."""
        # This single pattern handles references with double quotes, single quotes, or no quotes.
        # (?:'|")? makes the quote optional and non-capturing.
        pattern = r"""(?:'|")?references(?:'|")?\s*:\s*(\[.*?\])"""
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            try:
                # Using a non-greedy match `.*?` in the capture group is safer for complex content.
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                logger.warning(f"Regex found a references block, but it was not valid JSON: {match.group(1)}")
                return None
        return None

    def _attempt_json_fix(self, content: str) -> Optional[Dict[str, Any]]:
        """Attempt to fix common JSON issues and parse."""
        fixes_to_try = [
            # Add missing closing brace
            lambda x: x + '}' if x.count('{') > x.count('}') else x,
            # Add missing closing bracket
            lambda x: x + ']' if x.count('[') > x.count(']') else x,
            # Remove trailing commas
            lambda x: re.sub(r',\s*}', '}', x),
            lambda x: re.sub(r',\s*]', ']', x),
            # Fix quote issues
            lambda x: re.sub(r'(?<!\\)"(?=\w)', '\\"', x),  # Escape unescaped quotes
        ]
        
        for fix_func in fixes_to_try:
            try:
                fixed_content = fix_func(content)
                return json.loads(fixed_content, strict=False)
            except (json.JSONDecodeError, Exception):
                continue
        
        return None

    def _unescape_text(self, text: str) -> str:
        """Properly unescape text content."""
        try:
            # Handle unicode escapes
            unescaped = codecs.decode(text, 'unicode_escape')
            return unescaped
        except Exception:
            # Fallback to basic unescaping
            text = text.replace('"', '"')
            text = text.replace('\n', '\n')
            text = text.replace('\r', '\r')
            text = text.replace('\t', '\t')
            text = text.replace('\\', '\\')
            return text

    def _clean_and_validate_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and validate the final result."""
        if not isinstance(result, dict):
            return {"answer": str(result)}
        
        if "answer" in result and isinstance(result["answer"], str):
            # Apply existing cleaning functions
            cleaned_answer = remove_control_characters(result["answer"])
            result["answer"] = filter_to_devanagari_and_essentials(cleaned_answer)
        elif "answer" not in result:
            # If no answer key, try to find the main content
            for key in ["content", "response", "text", "message"]:
                if key in result and isinstance(result[key], str):
                    result["answer"] = result.pop(key) # Move value to "answer" and remove old key
                    break
            else:
                # If still no answer, use the entire result as string
                result = {"answer": str(result)}
        
        return result


# Test function to validate the enhanced parser
def test_enhanced_parser():
    """Test the enhanced parser with various edge cases."""
    
    # Instantiate the parser
    parser = JsonParser()
    
    test_cases = [
        # Standard cases
        '```json\n{"answer": "This is a standard response."}\n```',        
        # Malformed JSON
        '```json\n{"answer": "Missing closing brace"',        
        # Plain text
        'हमारी अब तक की बातचीत का सार यह है कि हमने कई महत्वपूर्ण आध्यात्मिक विषयों पर चर्चा की है।',        
        # Mixed content
        'Some text before ```json\n{"answer": "Mixed content"}\n``` and after.',        
        # Unescaped quotes
        '```json\n{"answer": "Text with \"quotes\" inside"}\n```',        
        # No markdown blocks
        '{"answer": "Direct JSON without markdown"}',        
        # Single quotes
        "{'answer': 'Single quoted JSON'}",        
        # Empty response
        '',        
        # Incomplete JSON
        '```json\n{"answer": "Incomplete',        
        # Multiple patterns
        'Response: "The answer is this content here" with more text.',
        # A case with a different key
        '{"response": "This is a response with a different key."}'
    ]
    
    print("Testing Enhanced JSON Parser:")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Input: {repr(test_case[:100])}")        
        result = parser.parse_json_answer(test_case)        
        if result:
            print(f"✓ Success: {result}")
        else:
            print("✗ Failed to parse")
        
        print("-" * 30)


if __name__ == "__main__":
    test_enhanced_parser()
