import os
import re
import sys
import shutil
import unittest
import json
from unittest.mock import MagicMock
from langchain_core.messages import HumanMessage # Import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
# Removed import for should_retrieve_node

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from RAG_BOT.vector_store import VectorStore
from RAG_BOT.rag_agent import build_agent
from RAG_BOT.logger import logger
from RAG_BOT.config import Config

class TestIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup method that is called once before all tests in the class."""
        cls.config = Config()
        cls.delete_exisiting_test_vector_store()
        logger.info("Deleted existing test vector store.")
        cls.test_vector_store = cls.setup_test_environment()
        cls.vectordb = cls.test_vector_store.get_vectordb()

    @classmethod
    def tearDownClass(cls):
        """Teardown method that is called once after all tests in the class."""
        pass

    @classmethod
    def delete_exisiting_test_vector_store(cls):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        test_vector_store_dir = os.path.join(current_dir, "..", "test_vector_store")
        if os.path.exists(test_vector_store_dir):
            shutil.rmtree(test_vector_store_dir)

    @classmethod
    def setup_test_environment(cls):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        pdf_dir = os.path.join(current_dir, "..", "data")
        test_vector_store_dir = os.path.join(current_dir, "..", "test_vector_store")
        # Create a test vector store and index sample PDFs
        test_vector_store = VectorStore(persist_directory=test_vector_store_dir)
        pdf_files = [
            os.path.join(pdf_dir, f)
            for f in os.listdir(pdf_dir)
            if f.endswith(".pdf")
        ]
        for pdf_file in pdf_files:
            test_vector_store.build_index(pdf_file, semantic_chunk=cls.config.SEMANTIC_CHUNKING)
        return test_vector_store

    def test_indexing_documents(self):
        documents_dict = self.vectordb.get()
        # Verify the number of indexed documents
        self.assertGreater(len(documents_dict["documents"]), 0, "No documents were indexed.")


    def test_agent_with_retrieval(self):
        """Tests the agent's ability to retrieve context using the tool."""
        agent = build_agent(vectordb=self.vectordb, model_name=self.config.LLM_MODEL_NAME)
        # Define the query, explicitly mentioning the date to guide the LLM tool usage
        query = "From the murli dated 1969-01-18, answer the following: " \
                "'Together with love, also become an embodiment of what so that there will be success?' " \
                "Give the answer in one word. Please provide your answer strictly in the following JSON format: " \
                "{\"answer\": \"<your answer>\"}"

        # Invoke the agent using the LangGraph message format
        initial_state = {"messages": [HumanMessage(content=query)]}
        final_state = agent.invoke(initial_state)

        # Verify the output from the final message
        self.assertIsInstance(final_state, dict)
        self.assertIn("messages", final_state)
        self.assertGreater(len(final_state["messages"]), 1) # Should have Human and AIMessage

        # The final answer is typically the last AIMessage
        final_answer_message = final_state['messages'][-1]
        self.assertEqual(final_answer_message.type, "ai", "Last message should be from AI")
        final_answer_content = final_answer_message.content
        self.assertIsInstance(final_answer_content, str, "Final answer content is not a string.")
        json_str = final_answer_content.strip()
        logger.info(f"Final answer content: {json_str}")
        # Remove triple backtick code block if present
        if json_str.startswith("```"):
            json_str = re.sub(r"^```(?:json)?\s*([\s\S]*?)\s*```$", r"\1", json_str.strip(), flags=re.MULTILINE)

        try:
            # Attempt to parse the JSON directly from the content
            json_result = json.loads(json_str)
            self.assertIn("answer", json_result, "Answer key missing in JSON response.")
            self.assertEqual(json_result["answer"].lower(), "power", "Retrieved answer is incorrect.")
        except json.JSONDecodeError:
            self.fail(f"Final answer is not valid JSON: {final_answer_content}")
        except Exception as e:
            self.fail(f"An unexpected error occurred during result verification: {e}")


    def test_agent_without_retrieval(self):
        """Tests the agent's ability to answer a general question without retrieval."""
        agent = build_agent(vectordb=self.vectordb, model_name=self.config.LLM_MODEL_NAME)
        # Define a general knowledge query unlikely to trigger the retriever tool
        query = "What is the capital of France? Provide your answer strictly in the following JSON format: " \
                "{\"answer\": \"<your answer>\"}"

        # Invoke the agent using the LangGraph message format
        initial_state = {"messages": [HumanMessage(content=query)]}
        final_state = agent.invoke(initial_state)

        # Verify the output from the final message
        self.assertIsInstance(final_state, dict)
        self.assertIn("messages", final_state)
        from langchain_core.messages import AIMessage, ToolMessage # Import necessary message types

        messages = final_state['messages']
        self.assertGreater(len(messages), 1) # Should have Human and AIMessage

        # Verify that the 'retrieve_context' tool was not called.
        retrieve_context_called = False
        for msg in messages:
            # Check if an AIMessage requested the specific tool call
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    # The tool name might be structured, e.g., 'context_retriever_tool.retrieve_context'
                    # or just 'retrieve_context' depending on how the tool is defined and bound.
                    # Check for both possibilities or adjust based on your exact tool naming.
                    if tool_call.get('name') == 'retrieve_context' or tool_call.get('name') == 'context_retriever_tool':
                        retrieve_context_called = True
                        logger.warning(f"Unexpected 'retrieve_context' tool call found: {tool_call}")
                        break           

        self.assertFalse(retrieve_context_called, "The 'retrieve_context' tool was called unexpectedly.")

        # The final answer is typically the last AIMessage
        final_answer_message = messages[-1]
        self.assertEqual(final_answer_message.type, "ai", "Last message should be from AI")
        final_answer_content = final_answer_message.content
        self.assertIsInstance(final_answer_content, str, "Final answer content is not a string.")
        json_str = final_answer_content.strip()
        logger.info(f"Final answer content: {json_str}")
        # Remove triple backtick code block if present
        if json_str.startswith("```"):
            json_str = re.sub(r"^```(?:json)?\s*([\s\S]*?)\s*```$", r"\1", json_str.strip(), flags=re.MULTILINE)
        try:
            # Attempt to parse the JSON directly from the content
            json_result = json.loads(json_str)
            self.assertIn("answer", json_result, "Answer key missing in JSON response.")
            # Check for the expected general knowledge answer
            self.assertEqual(json_result["answer"].lower(), "paris", "General knowledge answer is incorrect.")
        except json.JSONDecodeError:
            self.fail(f"Final answer is not valid JSON: {final_answer_content}")
        except Exception as e:
            self.fail(f"An unexpected error occurred during result verification: {e}")


if __name__ == "__main__":
    unittest.main()
