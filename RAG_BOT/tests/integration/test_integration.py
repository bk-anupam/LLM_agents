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
        query = (
            "What is the title of the murli from 1969-01-23?"
            "Please provide your answer strictly in the following JSON format: "
            '{"answer": "<your answer>"}'
        )
        initial_state = {"messages": [HumanMessage(content=query)]}
        final_state = agent.invoke(initial_state)

        self.assertIsInstance(final_state, dict)
        self.assertIn("messages", final_state)
        messages = final_state["messages"]
        self.assertGreater(len(messages), 1)

        # Check that the tool was called at least once
        from langchain_core.messages import AIMessage
        tool_called = any(
            isinstance(msg, AIMessage) and msg.tool_calls and
            any(tc.get("name") == "retrieve_context" for tc in msg.tool_calls)
            for msg in messages
        )
        self.assertTrue(tool_called, "The 'retrieve_context' tool was not called as expected.")

        # Check the final answer
        final_answer_message = messages[-1]
        self.assertEqual(final_answer_message.type, "ai")
        final_answer_content = final_answer_message.content
        self.assertIsInstance(final_answer_content, str)
        json_str = final_answer_content.strip()
        if json_str.startswith("```"):
            json_str = re.sub(r"^```(?:json)?\s*([\s\S]*?)\s*```$", r"\1", json_str.strip(), flags=re.MULTILINE)
        try:
            json_result = json.loads(json_str)
            self.assertIn("answer", json_result)
            self.assertEqual(json_result["answer"].lower(), "the ashes are to remind you of the stage")
        except Exception as e:
            self.fail(f"Final answer is not valid JSON or incorrect: {final_answer_content} ({e})")


    def test_agent_without_retrieval(self):
        """Tests the agent's ability to answer a general question without retrieval."""
        agent = build_agent(vectordb=self.vectordb, model_name=self.config.LLM_MODEL_NAME)
        query = (
            "What is the capital of France? Provide your answer strictly in the following JSON format: "
            '{"answer": "<your answer>"}'
        )
        initial_state = {"messages": [HumanMessage(content=query)]}
        final_state = agent.invoke(initial_state)

        self.assertIsInstance(final_state, dict)
        self.assertIn("messages", final_state)
        messages = final_state["messages"]
        self.assertGreater(len(messages), 1)

        from langchain_core.messages import AIMessage
        # Ensure no tool call was made
        tool_called = any(
            isinstance(msg, AIMessage) and msg.tool_calls and
            any(tc.get("name") == "retrieve_context" for tc in msg.tool_calls)
            for msg in messages
        )
        self.assertFalse(tool_called, "The 'retrieve_context' tool was called unexpectedly.")

        final_answer_message = messages[-1]
        self.assertEqual(final_answer_message.type, "ai")
        final_answer_content = final_answer_message.content
        self.assertIsInstance(final_answer_content, str)
        json_str = final_answer_content.strip()
        if json_str.startswith("```"):
            json_str = re.sub(r"^```(?:json)?\s*([\s\S]*?)\s*```$", r"\1", json_str.strip(), flags=re.MULTILINE)
        try:
            json_result = json.loads(json_str)
            self.assertIn("answer", json_result)
            self.assertEqual(json_result["answer"].lower(), "paris")
        except Exception as e:
            self.fail(f"Final answer is not valid JSON or incorrect: {final_answer_content} ({e})")


    def test_agent_insufficient_context(self):
        """Test agent response when no relevant context is found in the database."""
        agent = build_agent(vectordb=self.vectordb, model_name=self.config.LLM_MODEL_NAME)
        query = (
            "Can you summarize the murli from 1970-01-18? Please provide your answer strictly in the following JSON format: "
            '{"answer": "<your answer>"}'
        )
        initial_state = {"messages": [HumanMessage(content=query)]}
        final_state = agent.invoke(initial_state)

        self.assertIsInstance(final_state, dict)
        self.assertIn("messages", final_state)
        messages = final_state["messages"]
        self.assertGreater(len(messages), 0)

        final_answer_message = messages[-1]
        self.assertEqual(final_answer_message.type, "ai")
        final_answer_content = final_answer_message.content.lower()
        self.assertTrue(
            "cannot be found" in final_answer_content or "cannot find" in final_answer_content,
            f"Agent did not return a 'cannot find' message: {final_answer_content}"
        )
        # Check state reflects insufficient evaluation
        self.assertEqual(final_state.get("evaluation_result"), "insufficient", 
                         "Agent state should indicate evaluation_result was insufficient")


    def test_agent_retry_logic_reframing(self):
        """Test agent retry logic by asking an ambiguous question that should trigger reframing."""
        agent = build_agent(vectordb=self.vectordb, model_name=self.config.LLM_MODEL_NAME)
        # This question is intentionally vague to trigger a reframe
        query = (
            "Can you summarize the murli from 1970-01-18? Please provide your answer strictly in the following JSON format: "
            '{"answer": "<your answer>"}'
        )
        initial_state = {"messages": [HumanMessage(content=query)]}
        final_state = agent.invoke(initial_state)

        self.assertIsInstance(final_state, dict)
        self.assertIn("messages", final_state)
        messages = final_state["messages"]
        self.assertGreater(len(messages), 1)

        from langchain_core.messages import AIMessage
        # Check that at least one tool call was made (original or after reframing)
        tool_calls = [
            msg for msg in messages
            if isinstance(msg, AIMessage) and msg.tool_calls and
            any(tc.get("name") == "retrieve_context" for tc in msg.tool_calls)
        ]
        self.assertGreaterEqual(len(tool_calls), 1, "No tool call was made during retry logic.")
        # Check state reflects retry attempt
        self.assertTrue(final_state.get("retry_attempted"), "Agent state should indicate retry_attempted was True")

        # Optionally, check that the answer is in JSON format
        final_answer_message = messages[-1]
        self.assertEqual(final_answer_message.type, "ai")
        final_answer_content = final_answer_message.content
        json_str = final_answer_content.strip()
        if json_str.startswith("```"):
            json_str = re.sub(r"^```(?:json)?\s*([\s\S]*?)\s*```$", r"\1", json_str.strip(), flags=re.MULTILINE)
        try:
            json_result = json.loads(json_str)
            self.assertIn("answer", json_result)
        except Exception as e:
            self.fail(f"Final answer is not valid JSON: {final_answer_content} ({e})")

if __name__ == "__main__":
    unittest.main()
