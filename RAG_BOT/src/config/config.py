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
        self.TAVILY_API_KEY = self._get_config_value('TAVILY_API_KEY', os.environ.get('TAVILY_API_KEY', None))
        self.VECTOR_STORE_PATH = self._get_config_value('VECTOR_STORE_PATH', os.environ.get('VECTOR_STORE_PATH', None))
        self.DATA_PATH = self._get_config_value('DATA_PATH', os.environ.get('DATA_PATH', None))
        self.INDEXED_DATA_PATH = self._get_config_value('INDEXED_DATA_PATH', os.environ.get('INDEXED_DATA_PATH', None))
        self.WEBHOOK_URL = self._get_config_value('WEBHOOK_URL', os.environ.get('WEBHOOK_URL', None))
        self.PORT = self._get_config_value('MYPORT', int(os.environ.get('MYPORT', 5000)))
        self.LLM_MODEL_NAME = self._get_config_value('LLM_MODEL_NAME', os.environ.get('LLM_MODEL_NAME', 'gemini-2.5-flash'))
        self.TOOL_CALLING_LLM_MODEL_NAME = self._get_config_value('TOOL_CALLING_LLM_MODEL_NAME', os.environ.get('TOOL_CALLING_LLM_MODEL_NAME', 'gemini-2.5-pro'))
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
        self.SEARCH_TYPE = self._get_config_value('SEARCH_TYPE', os.environ.get('SEARCH_TYPE', "similarity")) # For Chroma semantic search
        self.SCORE_THRESHOLD = self._get_config_value('SCORE_THRESHOLD', float(os.environ.get('SCORE_THRESHOLD', 0.5)))
        self.INITIAL_RETRIEVAL_K = self._get_config_value('INITIAL_RETRIEVAL_K', int(os.environ.get('INITIAL_RETRIEVAL_K', 20))) # Semantic K
        self.RERANK_TOP_N = self._get_config_value('RERANK_TOP_N', int(os.environ.get('RERANK_TOP_N', 10)))
        self.CHUNK_SIZE = self._get_config_value('CHUNK_SIZE', int(os.environ.get('CHUNK_SIZE', 512)))
        self.CHUNK_OVERLAP = self._get_config_value('CHUNK_OVERLAP', int(os.environ.get('CHUNK_OVERLAP', 0)))

        # New BM25 related configs
        self.BM25_TOP_K = self._get_config_value('BM25_TOP_K', int(os.environ.get('BM25_TOP_K', 10)))
        self.BM25_MAX_CORPUS_SIZE = self._get_config_value('BM25_MAX_CORPUS_SIZE', int(os.environ.get('BM25_MAX_CORPUS_SIZE', 15000)))

        # New Murli Reconstruction Configs
        self.RECONSTRUCT_MURLIS = False
        self.MAX_RECON_MURLIS = self._get_config_value('MAX_RECON_MURLIS', int(os.environ.get('MAX_RECON_MURLIS', 5)))
        # Max chunks to fetch for a single Murli
        self.MAX_CHUNKS_PER_MURLI_RECON = self._get_config_value(
            'MAX_CHUNKS_PER_MURLI_RECON', 
            int(os.environ.get('MAX_CHUNKS_PER_MURLI_RECON', 25))
        ) 

        # Sentence window reconstruction
        self.SENTENCE_WINDOW_RECONSTRUCTION = self._get_config_value(
            'SENTENCE_WINDOW_RECONSTRUCTION', 
             os.environ.get('SENTENCE_WINDOW_RECONSTRUCTION', 'True').lower() in ('true', '1', 't')
        )
        self.SENTENCE_WINDOW_SIZE = self._get_config_value('SENTENCE_WINDOW_SIZE', int(os.environ.get('SENTENCE_WINDOW_SIZE', 2)))

        self.MAX_CHUNKS_FOR_DATE_FILTER = self._get_config_value(
            'MAX_CHUNKS_FOR_DATE_FILTER', 
            int(os.environ.get('MAX_CHUNKS_FOR_DATE_FILTER', 40))
        )
        self.ASYNC_OPERATION_TIMEOUT = self._get_config_value(
            'ASYNC_OPERATION_TIMEOUT', 
            int(os.environ.get('ASYNC_OPERATION_TIMEOUT', 240))
        )

        # Add USE_POLLING configuration
        default_use_polling_str = os.environ.get('USE_POLLING', 'False')
        default_use_polling = default_use_polling_str.lower() in ('true', '1', 't')
        self.USE_POLLING = self._get_config_value('USE_POLLING', default_use_polling)
        self.DEV_MODE = self._get_config_value(
            'DEV_MODE', 
            os.environ.get('DEV_MODE', 'False').lower() in ('true', '1', 't')
        )
        self.INDEX_ON_STARTUP = self._get_config_value(
            'INDEX_ON_STARTUP', 
            os.environ.get('INDEX_ON_STARTUP', 'False').lower() in ('true', '1', 't')
        )
        self.SQLITE_DB_PATH = self._get_config_value(
            'SQLITE_DB_PATH', 
            os.environ.get('SQLITE_DB_PATH', './RAG_BOT/rag_bot_sqlite.db')
        )
        self.GCP_PROJECT_ID = self._get_config_value(
            'GCP_PROJECT_ID',
            os.environ.get('GCP_PROJECT_ID', 'ardent-justice-466212-b8')
        )
        self.MAX_TOKENS = self._get_config_value('MAX_TOKENS', int(os.environ.get('MAX_TOKENS', 2500)))
        self.MAX_TOKENS_BEFORE_SUMMARY = self._get_config_value(
            'MAX_TOKENS_BEFORE_SUMMARY',
            int(os.environ.get('MAX_TOKENS_BEFORE_SUMMARY', 2500))
        )
        self.MAX_SUMMARY_TOKENS = self._get_config_value(
            'MAX_SUMMARY_TOKENS',
            int(os.environ.get('MAX_SUMMARY_TOKENS', 1000))
        )
        self.CHECKPOINTER_TYPE = self._get_config_value(
            'CHECKPOINTER_TYPE', 
            os.environ.get('CHECKPOINTER_TYPE', 'in_memory')).lower()
        
        # Persistence settings for thread management
        self.CONV_PERSISTENCE_BACKEND = self._get_config_value(
            'CONV_PERSISTENCE_BACKEND',
            os.environ.get('CONV_PERSISTENCE_BACKEND', 'sqlite')
        ).lower()

        # Conversation threading settings
        self.CONVERSATION_SUMMARY_THRESHOLD = self._get_config_value(
            'CONVERSATION_SUMMARY_THRESHOLD',
            int(os.environ.get('CONVERSATION_SUMMARY_THRESHOLD', 5))
        )        
        self.MAX_CONVERSATION_TURNS = self._get_config_value(
            'MAX_CONVERSATION_TURNS',
            int(os.environ.get('MAX_CONVERSATION_TURNS', 3))
        )
        self.GCS_VECTOR_STORE_PATH = self._get_config_value(
            'GCS_VECTOR_STORE_PATH',
            os.environ.get('GCS_VECTOR_STORE_PATH')
        )

    def _get_config_value(self, key, default_value):
        """Helper to get value from overrides or use default."""
        return self._overrides.get(key, default_value)

    # --- Accessor methods for prompts (optional but good practice) ---
    @classmethod
    def get_bk_persona_language_instruction(cls, language_code: str):
        """Gets the language-specific instruction for the BK persona prompt."""        
        lang = language_code.lower()
        return cls.PROMPTS.get('language_instructions', {}).get('bk_persona', {}).get(lang, '') 

    @classmethod
    def get_final_answer_language_instruction(cls, language_code: str):
        """Gets the language-specific instruction for the final answer system prompt."""
        lang = language_code.lower()
        return cls.PROMPTS.get('language_instructions', {}).get('final_answer_system', {}).get(lang, '') # Default to empty string

    @classmethod
    def get_response_guidelines(cls, mode: str):
        """Gets the response guidelines for the specified mode."""
        return cls.PROMPTS.get('response_guidelines', {}).get(mode, '')

    @classmethod
    def get_handle_question_system_prompt_template(cls):
        """Gets the system prompt template for the handle_question_node."""
        return cls.PROMPTS.get('handle_question_prompt_system', '')

    @classmethod
    def get_handle_question_human_prompt_template(cls):
        """Gets the human prompt template for the handle_question_node."""
        return cls.PROMPTS.get('handle_question_prompt_human', '')
    
    @classmethod
    def get_bk_persona_prompt(cls):
        """Gets the base BK persona system prompt (without language instruction - remains language-agnostic)."""
        # This now returns only the base English text
        return cls.PROMPTS.get('system_prompt', {}).get('bk_persona', '')

    @classmethod
    def get_tool_calling_instructions(cls):
        """Gets the dedicated tool calling instructions."""
        return cls.PROMPTS.get('tool_calling_instructions', '')

    @classmethod
    def get_router_system_prompt(cls):
        """Gets the system prompt for the router node."""
        return cls.PROMPTS.get('system_prompt', {}).get('router', '')

    @classmethod
    def get_conversational_system_prompt_template(cls):
        """Gets the system prompt template for the conversational_node."""
        return cls.PROMPTS.get('system_prompt', {}).get('conversation', '')

    @classmethod
    def get_question_guidance_prompt(cls):
        """Gets the question guidance prompt."""
        return cls.PROMPTS.get('system_prompt', {}).get('question_guidance', '')

    @classmethod
    def get_initial_summary_prompt_text(cls):
        """Gets the text for the initial summarization prompt."""
        return cls.PROMPTS.get('system_prompt', {}).get('summarization_prompts', {}).get('initial', '')

    @classmethod
    def get_existing_summary_prompt_text(cls):
        """Gets the text for the existing summarization prompt."""
        return cls.PROMPTS.get('system_prompt', {}).get('summarization_prompts', {}).get('existing', '')

    @classmethod
    def get_evaluate_context_prompt(cls):
        """Gets the evaluate context prompt."""
        return cls.PROMPTS.get('evaluate_context_prompt', '')

    @classmethod
    def get_reframe_question_prompt(cls):
        """Gets the reframe question prompt."""
        return cls.PROMPTS.get('reframe_question_prompt', '')

    @classmethod
    def get_final_answer_system_prompt_template(cls, language_code: str, mode: str = 'default'):
        """
        Gets the final answer system prompt template string based on the specified mode.
        The template string contains placeholders to be filled at runtime.
        """
        prompt_key = f'final_answer_prompt_system_{mode}'
        base_template = cls.PROMPTS.get(prompt_key, cls.PROMPTS.get('final_answer_prompt_system_default', ''))
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
    def get_json_format_instructions(cls):
        """Gets the json_format_prompt."""
        return cls.PROMPTS.get('json_format_prompt', '')

    @classmethod
    def get_user_message(cls, message_key: str, default: str = ""):
        """Gets a specific user-facing message string from prompts.yaml."""
        return cls.PROMPTS.get('user_messages', {}).get(message_key, default)
    
    @classmethod
    def get_murli_url_template(cls, language_code: str):
        """Returns the Murli URL template (to be used by tavily-extract) for the given language code."""
        lang = language_code.lower()
        return cls.PROMPTS.get('guidance_prompt', {}).get('murli_url', {}).get(lang, '')

    @classmethod
    def get_guidance_prompt(cls, language_code: str, current_query: str, formatted_date_for_url: str):
        """Returns the formatted guidance prompt with the correct Murli URL."""
        url_template = cls.get_murli_url_template(language_code)
        murli_url = url_template.format(date=formatted_date_for_url) if url_template and formatted_date_for_url else ""
        template = cls.PROMPTS.get('guidance_prompt', {}).get('template', '')
        return template.format(current_query=current_query, murli_url=murli_url)
