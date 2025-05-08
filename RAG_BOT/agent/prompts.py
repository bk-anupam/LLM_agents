# /home/bk_anupam/code/LLM_agents/RAG_BOT/agent/prompts.py
import os
import sys
from langchain_core.prompts import ChatPromptTemplate

# Add the project root to the Python path to import Config
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from RAG_BOT.config import Config

# --- Helper Prompt Functions ---

def get_evaluate_context_chat_prompt():
    return ChatPromptTemplate.from_messages(
        [
            ("system", Config.get_evaluate_context_prompt()),
            (
                "human",
                "Original User Question: {original_query}\n\nRetrieved Context:\n{context}"
            ),
        ]
    )

def get_reframe_question_chat_prompt():
    return ChatPromptTemplate.from_messages(
        [
            ("system", Config.get_reframe_question_prompt()),
            (
                "human",
                "Original User Question: {original_query}\nQuery Used for Failed Retrieval: {failed_query}"
            ),
        ]
    )

def get_final_answer_chat_prompt(language_code: str):
    """Creates the chat prompt template for generating the final answer, using the specified language."""
    system_template = Config.get_final_answer_system_prompt_template(language_code) # Use language
    human_template = Config.get_final_answer_human_prompt_template() 
    return ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", human_template),
    ])
