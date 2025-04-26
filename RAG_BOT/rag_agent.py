# This file is now primarily for running the agent directly if needed.
# The core logic (state, prompts, nodes, graph builder) is in the RAG_BOT.agent package.

import json
import os
import sys
import re

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from RAG_BOT.config import Config
from RAG_BOT.logger import logger
# Import from the new agent package
from RAG_BOT.agent.state import AgentState
from RAG_BOT.agent.graph_builder import build_agent


# --- Example Invocation ---
if __name__ == '__main__':    
    try:
        persist_directory = Config.VECTOR_STORE_PATH
        embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL_NAME)
        vectordb_instance = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        logger.info(f"Chroma DB loaded from: {persist_directory}")

        app = build_agent(vectordb_instance)

        # Example run - User query no longer needs JSON instruction
        # user_question = "Can you summarize the murli from 1950-01-18?"
        user_question = "Can you summarize the murli of 1969-02-06"

        # Initialize state correctly
        initial_state = AgentState(
            messages=[HumanMessage(content=user_question)],
            original_query=None,
            current_query=None,
            context=None,
            retry_attempted=False,
            evaluation_result=None
        )

        print(f"\n--- Invoking Agent for query: '{user_question}' ---")
        print("\n--- Invoking Agent (using invoke) ---")
        # Use invoke to get the final state directly
        final_state_result = app.invoke(initial_state, {"recursion_limit": 15})
        print("\n--- Agent Invocation Complete ---")

        # Process final state
        if isinstance(final_state_result, dict) and 'messages' in final_state_result:
            print("\n--- Final State ---")
            print(f"Original Query: {final_state_result.get('original_query')}")
            print(f"Current Query: {final_state_result.get('current_query')}")
            print(f"Retry Attempted: {final_state_result.get('retry_attempted')}")
            print(f"Evaluation Result: {final_state_result.get('evaluation_result')}")
            print(f"Context Present: {bool(final_state_result.get('context'))}")

            print("\n--- Final State Messages ---")
            for m in final_state_result['messages']:
                m.pretty_print()

            final_answer_message = final_state_result['messages'][-1]
            if isinstance(final_answer_message, AIMessage):
                print("\nFinal Answer Content:")
                print(final_answer_message.content)
                # Try to parse JSON for verification
                try:
                    # Handle potential markdown code blocks
                    content_str = final_answer_message.content.strip()
                    if content_str.startswith("```json"):
                         content_str = re.sub(r"^```json\s*([\s\S]*?)\s*```$", r"\1", content_str, flags=re.MULTILINE)
                    parsed_json = json.loads(content_str)
                    print("\nParsed JSON Answer:", parsed_json.get("answer"))
                except json.JSONDecodeError:
                    print("\nWarning: Final answer content is not valid JSON.")
                except Exception as e:
                    print(f"\nWarning: Error processing final answer content: {e}")

            else:
                print("\nFinal message was not an AIMessage:", final_answer_message)
        else:
             print("\nError: Could not extract final messages from result:", final_state_result)

    except Exception as e:
        logger.error(f"Error during example run: {e}", exc_info=True)
        print(f"\nError during example run: {e}")
