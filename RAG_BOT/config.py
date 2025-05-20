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
    PROMPTS = load_prompts(PROMPTS_FILE_PATH) # Stays as class variable

    def __init__(self, **overrides):
        self._overrides = overrides

        # Paths and simple values become instance attributes
        self.TELEGRAM_BOT_TOKEN = self._get_config_value('TELEGRAM_BOT_TOKEN', os.environ.get('TELEGRAM_BOT_TOKEN'))
        self.GEMINI_API_KEY = self._get_config_value('GEMINI_API_KEY', os.environ.get('GEMINI_API_KEY', None))
        self.VECTOR_STORE_PATH = self._get_config_value('VECTOR_STORE_PATH', os.environ.get('VECTOR_STORE_PATH', None))
        self.DATA_PATH = self._get_config_value('DATA_PATH', os.environ.get('DATA_PATH', None))
        self.INDEXED_DATA_PATH = self._get_config_value('INDEXED_DATA_PATH', os.environ.get('INDEXED_DATA_PATH', None))
        self.WEBHOOK_URL = self._get_config_value('WEBHOOK_URL', os.environ.get('WEBHOOK_URL', None))
        self.PORT = self._get_config_value('PORT', int(os.environ.get('PORT', 5000)))
        self.LLM_MODEL_NAME = self._get_config_value('LLM_MODEL_NAME', os.environ.get('LLM_MODEL_NAME', 'gemini-2.5-flash-preview-04-17'))
        self.EMBEDDING_MODEL_NAME = self._get_config_value('EMBEDDING_MODEL_NAME', os.environ.get('EMBEDDING_MODEL_NAME', 'all-MiniLM-L6-v2'))
        self.RERANKER_MODEL_NAME = self._get_config_value('RERANKER_MODEL_NAME', os.environ.get('RERANKER_MODEL_NAME', 'cross-encoder/ms-marco-MiniLM-L-6-v2'))
        
        # Handle boolean conversion for SEMANTIC_CHUNKING if it could come from env as string
        default_semantic_chunking_str = os.environ.get('SEMANTIC_CHUNKING', 'False')
        default_semantic_chunking = default_semantic_chunking_str.lower() in ('true', '1', 't')
        self.SEMANTIC_CHUNKING = self._get_config_value('SEMANTIC_CHUNKING', default_semantic_chunking)
        
        self.TEMPERATURE = self._get_config_value('TEMPERATURE', int(os.environ.get('TEMPERATURE', 0)))
        self.CONVERSATION_HISTORY_LIMIT = self._get_config_value('CONVERSATION_HISTORY_LIMIT', int(os.environ.get('CONVERSATION_HISTORY_LIMIT', 10)))
        self.JUDGE_LLM_MODEL_NAME = self._get_config_value('JUDGE_LLM_MODEL_NAME', os.environ.get('JUDGE_LLM_MODEL_NAME', "gemini-2.0-flash"))
        self.K = self._get_config_value('K', int(os.environ.get('K', 10)))
        self.K_FALLBACK = self._get_config_value('K_FALLBACK', int(os.environ.get('K_FALLBACK', 10)))
        self.SEARCH_TYPE = self._get_config_value('SEARCH_TYPE', os.environ.get('SEARCH_TYPE', "similarity"))
        self.SCORE_THRESHOLD = self._get_config_value('SCORE_THRESHOLD', float(os.environ.get('SCORE_THRESHOLD', 0.5)))
        self.INITIAL_RETRIEVAL_K = self._get_config_value('INITIAL_RETRIEVAL_K', int(os.environ.get('INITIAL_RETRIEVAL_K', 40)))
        self.RERANK_TOP_N = self._get_config_value('RERANK_TOP_N', int(os.environ.get('RERANK_TOP_N', 10)))
        self.CHUNK_SIZE = self._get_config_value('CHUNK_SIZE', int(os.environ.get('CHUNK_SIZE', 1000)))
        self.CHUNK_OVERLAP = self._get_config_value('CHUNK_OVERLAP', int(os.environ.get('CHUNK_OVERLAP', 100)))

        # For USER_SESSIONS, make it an instance variable for better isolation in tests
        self.USER_SESSIONS = self._get_config_value('USER_SESSIONS', {})
        self.MAX_CHUNKS_FOR_DATE_FILTER = self._get_config_value('MAX_CHUNKS_FOR_DATE_FILTER', int(os.environ.get('MAX_CHUNKS_FOR_DATE_FILTER', 40)))

    def _get_config_value(self, key, default_value):
        """Helper to get value from overrides or use default."""
        return self._overrides.get(key, default_value)

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
        # For now, just appending. Consider placement relative to CRITICAL INSTRUCTION:.        
        # Find the position of 'CRITICAL INSTRUCTION:...'
        insertion_point_str = "CRITICAL INSTRUCTION:"
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
