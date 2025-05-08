import os
import yaml 
from dotenv import load_dotenv, find_dotenv

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)  # This loads the variables from .env

# Define the path to the prompts file relative to this config file
PROMPTS_FILE_PATH = os.path.join(os.path.dirname(__file__), 'prompts.yaml') 

def load_prompts(file_path):
    """Loads prompts from a YAML file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            prompts = yaml.safe_load(f)
        if not isinstance(prompts, dict):
            print(f"Warning: Prompts file '{file_path}' did not load as a dictionary.")
            return {} # Return empty dict if not loaded correctly
        return prompts
    except FileNotFoundError:
        print(f"Error: Prompts file not found at '{file_path}'")
        return {} # Return empty dict if file not found
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file '{file_path}': {e}")
        return {} # Return empty dict on parsing error
    except Exception as e:
        print(f"An unexpected error occurred while loading prompts from '{file_path}': {e}")
        return {} # Return empty dict on other errors


class Config:
    # Load prompts from YAML file
    PROMPTS = load_prompts(PROMPTS_FILE_PATH)
    # Telegram Bot Token
    TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
    # For tracking conversation history
    USER_SESSIONS = {}
    # Optional: API keys for external services
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', None)
    VECTOR_STORE_PATH = os.environ.get('VECTOR_STORE_PATH', None)
    DATA_PATH = os.environ.get('DATA_PATH', None)
    INDEXED_DATA_PATH = os.environ.get('INDEXED_DATA_PATH', None)
    WEBHOOK_URL = os.environ.get('WEBHOOK_URL', None)
    PORT = int(os.environ.get('PORT', 5000))
    LLM_MODEL_NAME = os.environ.get('LLM_MODEL_NAME', 'gemini-2.5-flash-preview-04-17')
    # LANGUAGE = os.environ.get('LANGUAGE', 'en') # Removed global language setting
    EMBEDDING_MODEL_NAME = os.environ.get('EMBEDDING_MODEL_NAME', 'all-MiniLM-L6-v2')
    RERANKER_MODEL_NAME = os.environ.get('RERANKER_MODEL_NAME', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
    SEMANTIC_CHUNKING = False
    TEMPERATURE = 0
    CONVERSATION_HISTORY_LIMIT = 10    
    JUDGE_LLM_MODEL_NAME = "gemini-2.0-flash"    
    K = 10 # Initial retrieval K for non-reranking flow (can be kept or removed if INITIAL_RETRIEVAL_K is always used)
    SEARCH_TYPE = "similarity"
    SCORE_THRESHOLD = 0.5
    # Reranking specific config
    INITIAL_RETRIEVAL_K = 40 # Number of docs to fetch initially for reranking    
    RERANK_TOP_N = 10 # Number of docs to keep after reranking
    CHUNK_SIZE = 1000 # Size of chunks for semantic chunking
    CHUNK_OVERLAP = 100 # Overlap size for semantic chunking

    # --- Accessor methods for prompts (optional but good practice) ---
    @classmethod
    def get_bk_persona_language_instruction(cls, language_code: str):
        """Gets the language-specific instruction for the BK persona prompt."""
        # Use provided language_code
        lang = language_code.lower() # Ensure lowercase for matching keys
        return cls.PROMPTS.get('language_instructions', {}).get('bk_persona', {}).get(lang, '') # Default to empty string

    @classmethod
    def get_final_answer_language_instruction(cls, language_code: str):
        """Gets the language-specific instruction for the final answer system prompt."""
        lang = language_code.lower()
        return cls.PROMPTS.get('language_instructions', {}).get('final_answer_system', {}).get(lang, '') # Default to empty string

    @classmethod
    def get_system_prompt(cls, language_code: str):
        """Gets the combined system prompt including base persona, guidance, and language instruction."""
        base_persona = cls.PROMPTS.get('system_prompt', {}).get('bk_persona', '')
        guidance = cls.PROMPTS.get('system_prompt', {}).get('question_guidance', '')
        lang_instruction = cls.get_bk_persona_language_instruction(language_code) # Fetch dynamic instruction based on arg
        # Combine, adding a newline before the instruction if it exists
        return f"{base_persona}\n{guidance}\n{lang_instruction}".strip()

    @classmethod
    def get_bk_persona_prompt(cls):
        """Gets the base BK persona system prompt (without language instruction - remains language-agnostic)."""
        # This now returns only the base English text
        return cls.PROMPTS.get('system_prompt', {}).get('bk_persona', '')

    @classmethod
    def get_question_guidance_prompt(cls):
        """Gets the question guidance prompt."""
        return cls.PROMPTS.get('system_prompt', {}).get('question_guidance', '')

    @classmethod
    def get_evaluate_context_prompt(cls):
        """Gets the evaluate context prompt."""
        return cls.PROMPTS.get('evaluate_context_prompt', '')

    @classmethod
    def get_reframe_question_prompt(cls):
        """Gets the reframe question prompt."""
        return cls.PROMPTS.get('reframe_question_prompt', '')

    @classmethod
    def get_final_answer_system_prompt_template(cls, language_code: str):
        """Gets the final answer system prompt template including language instruction."""
        base_template = cls.PROMPTS.get('final_answer_prompt_system', '')
        lang_instruction = cls.get_final_answer_language_instruction(language_code) # Fetch dynamic instruction based on arg
        # Append the instruction. Add logic if needed to insert it cleanly.
        # For now, just appending. Consider placement relative to JSON format instruction.
        # Let's append it before the JSON format instruction for clarity.
        # Find the position of 'IMPORTANT: Provide your final answer...'
        insertion_point_str = "IMPORTANT: Provide your final answer strictly"
        insertion_point = base_template.find(insertion_point_str)
        if lang_instruction and insertion_point != -1:
             # Insert instruction before the JSON format part
             return f"{base_template[:insertion_point]}{lang_instruction}\n{base_template[insertion_point:]}".strip()
        else:
             # If instruction is empty or insertion point not found, return base
             return base_template

    @classmethod
    def get_final_answer_human_prompt_template(cls):
        """Gets the final answer human prompt template."""
        return cls.PROMPTS.get('final_answer_prompt_human', '')

    @classmethod
    def get_judge_prompt_template(cls):
        """Gets the judge prompt template."""
        return cls.PROMPTS.get('judge_prompt_template', '')

    @classmethod
    def get_user_message(cls, message_key: str, default: str = ""):
        """Gets a specific user-facing message string from prompts.yaml."""
        return cls.PROMPTS.get('user_messages', {}).get(message_key, default)

# Example usage (optional, for testing)
if __name__ == "__main__":    
    print("\nSystem Prompt:")
    print(Config.get_system_prompt('en')) # Example for English
    print("\nReframe Question Prompt:")
    print(Config.get_reframe_question_prompt())
    print("\nJudge Prompt Template:")
    print(Config.get_judge_prompt_template())
    print(f"\nTelegram Token: {Config.TELEGRAM_BOT_TOKEN}")
