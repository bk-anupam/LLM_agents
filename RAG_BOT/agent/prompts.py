# /home/bk_anupam/code/LLM_agents/RAG_BOT/agent/prompts.py
import os
import sys
from langchain_core.prompts import ChatPromptTemplate

# Add the project root to the Python path to import Config
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from RAG_BOT.config import Config

# --- Helper Prompts ---
EVALUATE_CONTEXT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", Config.get_evaluate_context_prompt()),
        (
            "human",
            "Original User Question: {original_query}\n\nRetrieved Context:\n{context}"
        ),
    ]
)

REFRAME_QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", Config.get_reframe_question_prompt()),
        (
            "human",
            "Original User Question: {original_query}\nQuery Used for Failed Retrieval: {failed_query}"
        ),
    ]
)

# LangChain will expect 'system_base', 'context', and 'original_query' during invoke
FINAL_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [   # Pass raw template string
        ("system", Config.get_final_answer_system_prompt_template()),
        # Pass raw template string ({original_query})
        ("human", Config.get_final_answer_human_prompt_template()),
    ]
)
