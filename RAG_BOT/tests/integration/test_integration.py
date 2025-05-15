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

from RAG_BOT.document_indexer import DocumentIndexer
from RAG_BOT.file_manager import FileManager
from RAG_BOT.vector_store import VectorStore
from RAG_BOT.agent.graph_builder import build_agent
from RAG_BOT.agent.state import AgentState
from RAG_BOT.logger import logger
from RAG_BOT.config import Config
from RAG_BOT import utils

class TestIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup method that is called once before all tests in the class."""
        cls.delete_exisiting_test_vector_store()
        logger.info("Deleted existing test vector store.")

        # Define test-specific paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        cls.test_data_dir = os.path.join(current_dir, "..", "data")
        cls.test_vector_store_dir = os.path.join(current_dir, "..", "test_vector_store")
        cls.test_indexed_data_dir = os.path.join(current_dir, "..", "indexed_test_data") # For moved files

        # Create a test-specific Config instance
        cls.config = Config(
            DATA_PATH=cls.test_data_dir,
            VECTOR_STORE_PATH=cls.test_vector_store_dir,
            INDEXED_DATA_PATH=cls.test_indexed_data_dir
        )
        cls.test_vector_store = cls.setup_test_environment(cls.config)
        cls.vectordb = cls.test_vector_store.get_vectordb()
        # Build agent once for the class
        cls.agent = build_agent(vectordb=cls.vectordb, config_instance=cls.config, model_name=cls.config.LLM_MODEL_NAME)


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
    def setup_test_environment(cls, test_config: Config):
        os.makedirs(test_config.VECTOR_STORE_PATH, exist_ok=True) # Ensure dir exists
        os.makedirs(test_config.INDEXED_DATA_PATH, exist_ok=True) # Ensure indexed data dir exists
        logger.info(f"Setting up test vector store in: {test_config.VECTOR_STORE_PATH}")
        
        # 1. Initialize VectorStore
        # Pass the test_config and explicit persist_directory
        test_vector_store = VectorStore(persist_directory=None, config=test_config)
        logger.info("Test VectorStore initialized.")        

        # Pass the test_config
        test_file_manager = FileManager(config=test_config)
        # 3. Initialize DocumentIndexer
        # Pass the test_config
        document_indexer = DocumentIndexer(
            vector_store_instance=test_vector_store,
            file_manager_instance=test_file_manager,
            config=test_config
        )
        logger.info("DocumentIndexer initialized.")        
        logger.info(f"Indexing documents from test data directory: {test_config.DATA_PATH}")
        document_indexer.index_directory(base_data_path=test_config.DATA_PATH, move_indexed_files=False)
        logger.info("Test data indexing complete.")
        # Return the initialized VectorStore instance
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
            # Retrieve all documents (or a sufficient number to cover test files)
            # In a real large index, you might filter or query, but for a test index, getting all is fine.
            documents_dict = self.vectordb.get(include=['metadatas', 'documents'])

            self.assertIsNotNone(documents_dict, "VectorDB get() returned None.")
            # Check if 'ids' list exists and is not empty
            self.assertIn("ids", documents_dict)
            self.assertIsInstance(documents_dict["ids"], list)
            self.assertGreater(len(documents_dict["ids"]), 0, "No documents were indexed.")
            self.assertIn("metadatas", documents_dict)
            self.assertIsInstance(documents_dict["metadatas"], list)
            self.assertEqual(len(documents_dict["ids"]), len(documents_dict["metadatas"]), "Mismatch between IDs and metadatas count.")

            indexed_metadatas = documents_dict["metadatas"]

            # Check for specific documents and their metadata
            # Assuming test data includes 'english/1969/1969-01-23.pdf' and 'hindi/1992/1992-09-24.htm'
            # Note: Source metadata is typically just the filename, not the full path
            english_pdf_indexed = False
            hindi_htm_indexed = False

            for metadata in indexed_metadatas:
                source = metadata.get('source')
                date = metadata.get('date')
                language = metadata.get('language')
                is_avyakt = metadata.get('is_avyakt') # This might be None if not Avyakt

                if source == '1969-01-23.pdf':
                    english_pdf_indexed = True
                    self.assertEqual(date, '1969-01-23', f"Incorrect date for {source}: {date}")
                    self.assertEqual(language, 'en', f"Incorrect language for {source}: {language}")
                    self.assertTrue(is_avyakt, f"Incorrect is_avyakt for {source}: {is_avyakt}") 
                elif source == '1992-09-24.htm':
                    hindi_htm_indexed = True
                    self.assertEqual(date, '1992-09-24', f"Incorrect date for {source}: {date}")
                    self.assertEqual(language, 'hi', f"Incorrect language for {source}: {language}")
                    self.assertTrue(is_avyakt, f"Incorrect is_avyakt for {source}: {is_avyakt}") 

        except Exception as e:
             # Catch potential errors if the collection doesn't exist yet
             self.fail(f"Failed to get documents from VectorDB: {e}")


    def evaluate_response_with_llm(self, query: str, context: Optional[str], response: str, test_config: Config = None) -> bool:
        """Uses an LLM to judge the quality of the agent's response."""
        judge_llm = ChatGoogleGenerativeAI(model=test_config.JUDGE_LLM_MODEL_NAME, temperature=0.0)
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
        # --- Check Context in Final State ---        
        self.assertIn("context", final_state, "Context missing from final state.")
        context = final_state.get("context")
        self.assertIsInstance(context, str, "Context in final state is not a string.")


    def test_agent_without_retrieval(self):
        """Tests the agent's ability to answer a general question without retrieval, in JSON."""
        # Query without JSON instruction
        query = "How are you doing today?"
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
        self.assertGreaterEqual(tool_call_count, 2, "The 'retrieve_context' tool was not called at least twice for retry.")

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
        self.assertGreaterEqual(len(tool_calls), 2, "No tool call was made during retry logic.")
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
        evaluation_result = self.evaluate_response_with_llm(query, context, final_answer_content, test_config=self.config) 
        json_result = utils.parse_json_answer(final_answer_content)
        response_answer = json_result.get("answer", "")
        self.assertTrue(evaluation_result, f"LLM Judge evaluation failed for query '{query}'. Response: {response_answer}")


if __name__ == "__main__":
    unittest.main()
