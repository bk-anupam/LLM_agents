import os
import re
import sys
import shutil
import unittest
import json
from unittest.mock import MagicMock
from langchain_google_genai import ChatGoogleGenerativeAI

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
        agent = build_agent(vectordb=self.vectordb, model_name=self.config.LLM_MODEL_NAME)
        # Define the query and date filter
        query = "'Together with love, also become an embodiment of what so that there will be success. " \
            "Give the answer in one word. Please provide your answer in the following JSON format: {\"answer\": \"<your answer>\"}'"
        date_filter = "1969-01-18"
        # Invoke the agent
        result = agent.invoke(
            {
                "query": query,
                "skip_retrieval": False,
                "date_filter": date_filter,
                "k": self.config.K,
                "search_type": self.config.SEARCH_TYPE,
                # "score_threshold": self.config.SCORE_THRESHOLD, # Removed this line
            }
        )
        # Verify the output
        self.assertIn("answer", result, "Agent did not return an answer.")
        self.assertIsInstance(result["answer"], str, "Answer is not a string.")
        try:
            json_result = json.loads(result["answer"])
            self.assertIn("answer", json_result, "Answer is not in JSON format.")
            self.assertEqual(json_result["answer"].lower(), "power", "Answer is incorrect.")
        except json.JSONDecodeError:
            self.fail("Answer is not valid JSON.")

    def test_agent_without_retrieval(self):
        agent = build_agent(vectordb=self.vectordb, model_name=self.config.LLM_MODEL_NAME)
        # Define the query
        query = "In what year did Brahma Baba became avyakt? Give the answer in one word. " \
                "Provide your answer in JSON format only, with no other text. " \
                "The JSON should have the following structure: {\"answer\": \"<your answer>\"}" 
                
        # Invoke the agent
        result = agent.invoke({"query": query, "skip_retrieval": True})
        # Verify the output
        self.assertIn("answer", result, "Agent did not return an answer.")
        self.assertIsInstance(result["answer"], str, "Answer is not a string.")
        try:
            # Extract JSON from the string using regular expressions
            match = re.search(r"```json\n(.*)\n```", result["answer"], re.DOTALL)
            if match:
                json_string = match.group(1)
                json_result = json.loads(json_string)
                self.assertIn("answer", json_result, "Answer is not in JSON format.")
                self.assertEqual(json_result["answer"].lower(), "1969", "Answer is incorrect.")
            else:
                self.fail("Answer is not in JSON format.")
        except json.JSONDecodeError:
            self.fail("Answer is not valid JSON.")    

if __name__ == "__main__":
    unittest.main()
