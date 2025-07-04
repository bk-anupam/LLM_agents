# This file is now primarily for running the agent directly if needed.
import asyncio
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings
from RAG_BOT.config import Config
from RAG_BOT.logger import logger
from RAG_BOT.agent.state import AgentState
from RAG_BOT.agent.graph_builder import build_agent
from RAG_BOT.utils import parse_json_answer 
from RAG_BOT.vector_store import VectorStore


# --- Example Invocation ---
if __name__ == '__main__':    
    async def run_agent_example():
        try:
            config = Config()
            persist_directory = config.VECTOR_STORE_PATH
            # Instantiate the vector store instance
            logger.info("Initializing VectorStore...")
            # Pass the config instance to VectorStore
            vector_store_instance = VectorStore(persist_directory=config.VECTOR_STORE_PATH, config=config)
            vectordb = vector_store_instance.get_vectordb() # Get the db instance after init
            logger.info("VectorStore initialized.")
            # Create rag agent instance
            logger.info("Initializing RAG agent...")
            # Ensure vectordb is valid before passing to agent
            if vectordb is None:
                logger.error("VectorDB instance is None after initialization and indexing. Cannot build agent.")
                exit(1)
            agent = await build_agent(vectordb=vectordb, config_instance=config)
            logger.info("RAG agent initialized successfully")        

            # Example run - User query no longer needs JSON instruction
            # user_question = "Can you summarize the murli from 1950-01-18?"
            # user_question = "Can you summarize the murli of 2025-06-06"
            # user_question = "भट्ठी की जो सारी पढ़ाई वा शिक्षा ली उसका सार कौन से तीन शब्दों में याद रखने के लिए बाबा ने मुरलियों में जोर दिया"
            # user_question = "2025-06-29 की मुरली का सार क्या है?"
            user_question = "रिफाइन स्थिति की क्या पहचान बाबा ने 1972 में अव्यक्त मुरलियों में बताई है"
            # user_question = "दूसरों की चेकिंग करने के बारे में बाबा ने मुरली में क्या बताया है?"
            # user_question = "त्याग तपस्या और सेवा की परिभाषा के बारे में बाबा ने मुरली में क्या कहा है "
            language_code = "hi"

            # Initialize state correctly
            initial_state = AgentState(
                messages=[HumanMessage(content=user_question)],
                original_query=None,
                current_query=None,
                context=None,
                retry_attempted=False,
                evaluation_result=None,
                language_code=language_code,
                documents=None,
                web_search_attempted=False,
                last_retrieval_source=None
            )

            print(f"\n--- Invoking Agent for query: '{user_question}' ---")
            print("\n--- Invoking Agent (using invoke) ---")
            # Use invoke to get the final state directly
            final_state_result = await agent.ainvoke(initial_state, {"recursion_limit": 15})
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
                    # Use the robust parsing function from utils.py
                    parsed_json = parse_json_answer(final_answer_message.content)
                    if parsed_json and "answer" in parsed_json:
                        print("\nParsed JSON Answer:", parsed_json.get("answer"))
                    else:
                        print("\nWarning: Final answer content is not valid JSON.")

                else:
                    print("\nFinal message was not an AIMessage:", final_answer_message)
            else:
                print("\nError: Could not extract final messages from result:", final_state_result)

        except Exception as e:
            logger.error(f"Error during example run: {e}", exc_info=True)
            print(f"\nError during example run: {e}")

asyncio.run(run_agent_example())