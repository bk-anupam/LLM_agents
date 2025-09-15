# /home/bk_anupam/code/LLM_agents/RAG_BOT/agent/prompts.py
import os
import sys
from langchain_core.prompts import ChatPromptTemplate
from RAG_BOT.src.config.config import Config

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
    system_template = Config.get_final_answer_system_prompt_template(language_code, mode=mode)
    human_template = Config.get_final_answer_human_prompt_template() 
    return ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", human_template),
    ])

def get_handle_question_chat_prompt():
    """
    Creates the chat prompt template for the handle_question_node, which decides on tool usage.
    The system and human messages are templates that will be filled at runtime.
    """
    system_template = Config.get_handle_question_system_prompt_template()
    human_template = Config.get_handle_question_human_prompt_template()
    return ChatPromptTemplate.from_messages([("system", system_template), ("human", human_template)])

def get_initial_summary_prompt():
    """
    Returns a ChatPromptTemplate for the initial conversation summary.
    It includes a placeholder for the token limit.
    """
    return ChatPromptTemplate.from_messages([
        ("placeholder", "{messages}"),
        ("user", Config.get_initial_summary_prompt_text()),
    ])

def get_existing_summary_prompt():
    """
    The function returns a langchain_core.prompts.ChatPromptTemplate object.
    This is not just a string. It's a reusable template. When the SummarizationNode in your agent's graph 
    needs to update a summary, it will invoke this template with the necessary data.
    For example, the SummarizationNode will provide:
    - A list of new HumanMessage and AIMessage objects to fill the {messages} placeholder.
    - The old summary string to fill {existing_summary}.
    - The token limit from your config to fill {max_summary_tokens}.
    """
    return ChatPromptTemplate.from_messages([
        # This is a special LangChain placeholder. It tells the template: "At runtime, a list of actual message 
        # objects (the new messages since the last summary) will be inserted here."
        ("placeholder", "{messages}"),
        # This creates a HumanMessage (a user-turn message) containing the core instructions for the LLM.
        # The instruction text includes two template variables:
        # {existing_summary} - This will be replaced with the current summary of the conversation.
        # {max_summary_tokens} - This will be replaced with the maximum allowed tokens for the summary
        ("user", Config.get_existing_summary_prompt_text()),
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
