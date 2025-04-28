# /home/bk_anupam/code/LLM_agents/RAG_BOT/tests/integration/test_integration.py
import os
import re
import sys
import shutil
import unittest
import json
from unittest.mock import MagicMock
from typing import Optional
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage # Added ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from RAG_BOT.vector_store import VectorStore
# Updated imports for build_agent and AgentState
from RAG_BOT.agent.graph_builder import build_agent
from RAG_BOT.agent.state import AgentState
from RAG_BOT.logger import logger
from RAG_BOT.config import Config
from RAG_BOT import utils

class TestIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup method that is called once before all tests in the class."""
        cls.config = Config()
        cls.delete_exisiting_test_vector_store()
        logger.info("Deleted existing test vector store.")
        cls.test_vector_store = cls.setup_test_environment()
        cls.vectordb = cls.test_vector_store.get_vectordb()
        # Build agent once for the class
        cls.agent = build_agent(vectordb=cls.vectordb, model_name=cls.config.LLM_MODEL_NAME)


    @classmethod
    def tearDownClass(cls):
        """Teardown method that is called once after all tests in the class."""
        pass # Keep vector store for inspection if needed, or delete

    @classmethod
    def delete_exisiting_test_vector_store(cls):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        test_vector_store_dir = os.path.join(current_dir, "..", "test_vector_store")
        if os.path.exists(test_vector_store_dir):
            shutil.rmtree(test_vector_store_dir)
            logger.info(f"Deleted test vector store at: {test_vector_store_dir}")


    @classmethod
    def setup_test_environment(cls):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        pdf_dir = os.path.join(current_dir, "..", "data")
        test_vector_store_dir = os.path.join(current_dir, "..", "test_vector_store")
        os.makedirs(test_vector_store_dir, exist_ok=True) # Ensure dir exists
        logger.info(f"Setting up test vector store in: {test_vector_store_dir}")
        # Create a test vector store and index sample PDFs
        test_vector_store = VectorStore(persist_directory=test_vector_store_dir)
        pdf_files = [
            os.path.join(pdf_dir, f)
            for f in os.listdir(pdf_dir)
            if f.endswith(".pdf")
        ]
        if not pdf_files:
             logger.warning(f"No PDF files found in {pdf_dir} for indexing.")
             return test_vector_store # Return empty store if no PDFs

        for pdf_file in pdf_files:
            logger.info(f"Indexing test file: {pdf_file}")
            test_vector_store.build_index(pdf_file, semantic_chunk=cls.config.SEMANTIC_CHUNKING)
        logger.info("Test vector store setup complete.")
        return test_vector_store

    def _run_agent(self, query: str) -> AgentState:
        """Helper method to run the agent with a query."""
        initial_state = AgentState(messages=[HumanMessage(content=query)])
        # Add recursion limit for safety
        final_state = self.agent.invoke(initial_state, {"recursion_limit": 15})
        self.assertIsInstance(final_state, dict)
        self.assertIn("messages", final_state)
        return final_state


    def test_indexing_documents(self):
        """Verify that documents were indexed in the test vector store."""
        # Skip if vectordb wasn't created properly
        if not hasattr(self, 'vectordb') or self.vectordb is None:
            self.skipTest("VectorDB instance not available.")
        try:
            documents_dict = self.vectordb.get(limit=1) # Fetch just one to confirm collection exists
            # Check if the collection is empty or exists
            self.assertIsNotNone(documents_dict, "VectorDB get() returned None.")
            # Check if 'ids' list exists and is not empty
            self.assertIn("ids", documents_dict)
            self.assertIsInstance(documents_dict["ids"], list)
            # We only check if *any* document was indexed, as exact count depends on chunking
            self.assertGreater(len(documents_dict["ids"]), 0, "No documents were indexed.")
        except Exception as e:
             # Catch potential errors if the collection doesn't exist yet
             self.fail(f"Failed to get documents from VectorDB: {e}")


    def evaluate_response_with_llm(self, query: str, context: Optional[str], response: str) -> bool:
        """Uses an LLM to judge the quality of the agent's response."""
        judge_llm = ChatGoogleGenerativeAI(model=Config.JUDGE_LLM_MODEL_NAME, temperature=0.0)
        judge_prompt_template = Config.get_judge_prompt_template()
        # The judge prompt expects the raw response string, which includes the JSON structure
        judge_prompt = judge_prompt_template.format(
            query=query,
            context=context if context else "N/A",
            response=response # Pass the raw response string
        )
        try:
            evaluation = judge_llm.invoke([HumanMessage(content=judge_prompt)]).content.strip().upper()
            logger.info(f"LLM Judge Evaluation for query '{query[:50]}...': {evaluation}")
            return evaluation == 'PASS'
        except Exception as e:
            logger.error(f"LLM Judge call failed: {e}")
            return False # Fail the test if judge fails


    def test_agent_with_retrieval(self):
        """Tests the agent's ability to retrieve context and answer in JSON."""
        # Query without JSON instruction
        query = "What is the title of the murli from 1969-01-23?"
        final_state = self._run_agent(query)
        messages = final_state["messages"]
        self.assertGreater(len(messages), 1)

        # Check that the tool was called at least once
        tool_called = any(
            isinstance(msg, AIMessage) and msg.tool_calls and
            any(tc.get("name") == "retrieve_context" for tc in msg.tool_calls)
            for msg in messages
        )
        self.assertTrue(tool_called, "The 'retrieve_context' tool was not called as expected.")

        # Check the final answer format and content
        final_answer_message = messages[-1]
        self.assertEqual(final_answer_message.type, "ai")
        json_result = utils.parse_json_answer(final_answer_message.content)
        self.assertIsNotNone(json_result, f"Final answer is not valid JSON: {final_answer_message.content}")
        self.assertIn("answer", json_result)
        # Make comparison case-insensitive and check for substring
        self.assertIn("the ashes are to remind you of the stage", json_result["answer"].lower())


    def test_agent_without_retrieval(self):
        """Tests the agent's ability to answer a general question without retrieval, in JSON."""
        # Query without JSON instruction
        query = "What is the purpose of life?"
        final_state = self._run_agent(query)
        messages = final_state["messages"]
        self.assertGreater(len(messages), 1)

        # Ensure no tool call was made
        tool_called = any(
            isinstance(msg, AIMessage) and msg.tool_calls and
            any(tc.get("name") == "retrieve_context" for tc in msg.tool_calls)
            for msg in messages
        )
        self.assertFalse(tool_called, "The 'retrieve_context' tool was called unexpectedly.")

        # Check the final answer format and content
        final_answer_message = messages[-1]
        self.assertEqual(final_answer_message.type, "ai")
        json_result = utils.parse_json_answer(final_answer_message.content)
        self.assertIsNotNone(json_result, f"Final answer is not valid JSON: {final_answer_message.content}")
        self.assertIn("answer", json_result)
        # check that cannot find is not in the answer
        answer_lower = json_result["answer"].lower()
        self.assertNotIn("cannot be found", answer_lower,
                         f"Agent returned 'cannot be found' unexpectedly: {json_result['answer']}")
        self.assertNotIn("cannot find", answer_lower,
                         f"Agent returned 'cannot find' unexpectedly: {json_result['answer']}")


    def test_agent_insufficient_context(self):
        """Test agent response (in JSON) when no relevant context is found."""
        # Query without JSON instruction
        query = "Can you summarize the murli from 1950-01-18?"
        final_state = self._run_agent(query)
        messages = final_state["messages"]
        self.assertGreater(len(messages), 1)

        # --- Behavioral Assertions ---
        # 1. Check if retry was attempted (assuming the first retrieval yields nothing relevant)        
        self.assertTrue(final_state.get("retry_attempted"),
                         "Agent state should indicate retry_attempted was True if initial retrieval failed")

        # 2. Check that the tool was called (at least once)
        tool_call_count = sum(
            1 for msg in messages
            if isinstance(msg, AIMessage) and msg.tool_calls and
            any(tc.get("name") == "retrieve_context" for tc in msg.tool_calls)
        )
        self.assertGreaterEqual(tool_call_count, 1, "The 'retrieve_context' tool was not called.")

        # 3. Check the final answer format and content
        final_answer_message = messages[-1]
        self.assertEqual(final_answer_message.type, "ai")
        json_result = utils.parse_json_answer(final_answer_message.content)
        self.assertIsNotNone(json_result, f"Final 'cannot find' answer is not valid JSON: {final_answer_message.content}")
        self.assertIn("answer", json_result)
        self.assertTrue(
            "cannot be found" in json_result["answer"].lower() or "cannot find" in json_result["answer"].lower(),
            f"Agent did not return a 'cannot find' message within the JSON answer: {json_result['answer']}"
        )

        # 4. Check state reflects insufficient evaluation (if retry occurred) or final decision path        
        if final_state.get("retry_attempted"):
            self.assertEqual(final_state.get("evaluation_result"), "insufficient",
                             "Agent state should indicate evaluation_result was insufficient after retry")


    def test_agent_retry_logic_reframing(self):
        """Test agent retry logic (reframing) and final JSON output."""
        # Query without JSON instruction - date likely not in test data
        query = "Can you summarize the murli from 1970-01-18?"
        final_state = self._run_agent(query)
        messages = final_state["messages"]
        self.assertGreater(len(messages), 1)

        # Check that at least one tool call was made
        tool_calls = [
            msg for msg in messages
            if isinstance(msg, AIMessage) and msg.tool_calls and
            any(tc.get("name") == "retrieve_context" for tc in msg.tool_calls)
        ]
        self.assertGreaterEqual(len(tool_calls), 1, "No tool call was made during retry logic.")
        # Check that the retry logic was invoked        
        self.assertTrue(final_state.get("retry_attempted"), "Agent state should indicate retry_attempted was True")

        # Check the final answer format (should be JSON, likely a 'cannot find' message)
        final_answer_message = messages[-1]
        self.assertEqual(final_answer_message.type, "ai")
        json_result = utils.parse_json_answer(final_answer_message.content)
        self.assertIsNotNone(json_result, f"Final answer after retry is not valid JSON: {final_answer_message.content}")
        self.assertIn("answer", json_result)
        # Content could be a summary if found after retry, or 'cannot find'
        self.assertIsInstance(json_result["answer"], str)


    def test_summarization_for_a_date(self):
        """Test agent's ability to summarize a murli for a specific date in JSON."""
        # Query without JSON instruction
        query = "Can you summarize the murli from 1969-01-23?"
        final_state = self._run_agent(query)

        # --- Explicitly check context presence in final state ---
        self.assertIn("context", final_state, "The 'context' key is missing from the final agent state.")
        context = final_state.get("context")
        # Context could be None if retrieval failed, but the final answer should reflect that.
        # If context *is* present, it should be a string.
        if context is not None:
            self.assertIsInstance(context, str, "Context field in the final state is not a string.")
            # Optional: Check if context is not empty if retrieval was expected to succeed
            # self.assertTrue(len(context.strip()) > 0, "Context retrieved from final state appears to be empty.")

        # Evaluate the response using the LLM judge
        final_answer_content = final_state["messages"][-1].content
        evaluation_result = self.evaluate_response_with_llm(query, context, final_answer_content)
        json_result = utils.parse_json_answer(final_answer_content)
        response_answer = json_result.get("answer", "")
        self.assertTrue(evaluation_result, f"LLM Judge evaluation failed for query '{query}'. Response: {response_answer}")


if __name__ == "__main__":
    unittest.main()
