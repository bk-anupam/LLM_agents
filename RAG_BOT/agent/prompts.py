# /home/bk_anupam/code/LLM_agents/RAG_BOT/agent/prompts.py
import os
import sys
from langchain_core.prompts import ChatPromptTemplate
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

def get_final_answer_chat_prompt(language_code: str, mode: str):
    """Creates the chat prompt template for generating the final answer, using the specified language and mode."""
    system_template = Config.get_final_answer_system_prompt_template(language_code=language_code, mode=mode)
    human_template = Config.get_final_answer_human_prompt_template() 
    return ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", human_template),
    ])

def get_custom_summary_prompt():
    """
    Returns a custom chat prompt template for the summarization node.
    This prompt correctly assembles the final message list by:
    1. Preserving the original SystemMessage.
    2. Injecting the generated summary as a HumanMessage for context.
    3. Appending the remaining, non-summarized messages.
    """
    return ChatPromptTemplate.from_messages([
        # This placeholder is filled by the node with the original SystemMessage.
        # ("system", "You are a helpful assistant")  # Hard-coded system message (This creates a new SystemMessage with this content)
        # ("placeholder", "{system_message}")  # Dynamic system message (This inserts existing SystemMessage object) 
        # placeholder is a special message type that tells the template "this spot will be filled with actual message objects at runtime"
        # Unlike specific message types like "human", "assistant", or "system", a placeholder doesn't create a new message
        # Instead, it reserves a spot where existing message objects will be inserted
        # "{system_message}" is a template variable name and is a pointer to the actual SystemMessage object

        ("placeholder", "{system_message}"),

        # This injects the summary as a HumanMessage, framing it as context.
        ("human", "This is a summary of our conversation so far:\n{summary}"),

        # This placeholder is filled with the latest, non-summarized messages.
        ("placeholder", "{messages}"),
    ])
