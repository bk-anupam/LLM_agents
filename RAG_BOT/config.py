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
    WEBHOOK_URL = os.environ.get('WEBHOOK_URL', None)
    PORT = int(os.environ.get('PORT', 5000))
    SEMANTIC_CHUNKING = False
    TEMPERATURE = 0
    CONVERSATION_HISTORY_LIMIT = 10
    LLM_MODEL_NAME = "gemini-2.5-flash-preview-04-17"
    # LLM_MODEL_NAME = "gemini-2.0-flash"
    # LLM_MODEL_NAME = "gemini-2.5-pro-exp-03-25"
    JUDGE_LLM_MODEL_NAME = "gemini-2.0-flash"
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    K = 10 # Initial retrieval K for non-reranking flow (can be kept or removed if INITIAL_RETRIEVAL_K is always used)
    SEARCH_TYPE = "similarity"
    SCORE_THRESHOLD = 0.5
    # Reranking specific config
    INITIAL_RETRIEVAL_K = 40 # Number of docs to fetch initially for reranking
    RERANKER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    RERANK_TOP_N = 10 # Number of docs to keep after reranking

    # --- Accessor methods for prompts (optional but good practice) ---
    @classmethod
    def get_system_prompt(cls):
        """Gets a specific system prompt."""        
        bk_persona_prompt = cls.PROMPTS.get('system_prompt', {}).get('bk_persona', '')
        question_guidance_prompt = cls.PROMPTS.get('system_prompt', {}).get('question_guidance', '')
        return bk_persona_prompt + question_guidance_prompt

    @classmethod
    def get_bk_persona_prompt(cls):
        """Gets the BK persona system prompt."""
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
    def get_final_answer_system_prompt_template(cls):
        """Gets the final answer system prompt template."""
        return cls.PROMPTS.get('final_answer_prompt_system', '')

    @classmethod
    def get_final_answer_human_prompt_template(cls):
        """Gets the final answer human prompt template."""
        return cls.PROMPTS.get('final_answer_prompt_human', '')

    @classmethod
    def get_judge_prompt_template(cls):
        """Gets the judge prompt template."""
        return cls.PROMPTS.get('judge_prompt_template', '')
    
# Example usage (optional, for testing)
if __name__ == "__main__":    
    print("\nSystem Prompt:")
    print(Config.get_system_prompt())
    print("\nReframe Question Prompt:")
    print(Config.get_reframe_question_prompt())
    print("\nJudge Prompt Template:")
    print(Config.get_judge_prompt_template())
    print(f"\nTelegram Token: {Config.TELEGRAM_BOT_TOKEN}")
