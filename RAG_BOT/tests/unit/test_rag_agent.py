import unittest
import os
import sys
from unittest.mock import MagicMock, patch
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage
from langgraph.graph.state import CompiledStateGraph


# Add the parent directory to the Python path
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# grand_parent_dir = os.path.dirname(parent_dir)
# sys.path.insert(0, grand_parent_dir)

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from RAG_BOT.rag_agent import should_retrieve_node, retriever_node, generator_node, build_agent

class TestRAGAgent(unittest.TestCase):

    @patch("RAG_BOT.rag_agent.logger")
    def test_should_retrieve_node(self, mock_logger):
        state = {"skip_retrieval": True}
        result = should_retrieve_node(state)
        self.assertEqual(result["next"], "generator")
        mock_logger.info.assert_called_with("Skipping retrieval and going directly to generator.")

        state = {"skip_retrieval": False}
        result = should_retrieve_node(state)
        self.assertEqual(result["next"], "retriever")
        mock_logger.info.assert_called_with("Proceeding with document retrieval.")


    @patch("RAG_BOT.rag_agent.logger")
    def test_retriever_node(self, mock_logger):
        mock_vectordb = MagicMock(spec=Chroma)
        mock_retriever = MagicMock()
        mock_vectordb.as_retriever.return_value = mock_retriever
        mock_retriever.invoke.return_value = [
            MagicMock(page_content="Document 1"),
            MagicMock(page_content="Document 2"),
        ]
        state = {
            "query": "Test query",
            "k": 2,
            "date_filter": "2023-01-01",
            "search_type": "similarity",
            "score_threshold": 0.5,
        }
        result = retriever_node(state, mock_vectordb)
        self.assertIn("context", result)
        self.assertIn("query", result)
        self.assertEqual(result["context"], "Document 1\n\nDocument 2")
        self.assertEqual(result["query"], "Test query")
        mock_logger.info.assert_any_call("Applying date filter: 2023-01-01")
        mock_logger.info.assert_any_call("Executed retriever node and retrieved 2 documents for query: test query")


    @patch("RAG_BOT.rag_agent.logger")
    @patch("langchain_google_genai.ChatGoogleGenerativeAI")
    def test_generator_node(self, mock_llm_class, mock_logger):
        mock_llm = MagicMock(spec=ChatGoogleGenerativeAI)
        mock_llm_class.return_value = mock_llm
        mock_llm.invoke.return_value = AIMessage(content="Generated response")
        state = {
            "query": "What is AI?",
            "context": "Artificial Intelligence is the simulation of human intelligence in machines.",
        }
        result = generator_node(state, mock_llm)
        self.assertIn("answer", result)
        self.assertEqual(result["answer"], "Generated response")
        mock_logger.info.assert_any_call("Executing generator node with query: What is AI? and context: Artificial Intelligence is the simulation of human intelligence in machines.")
        mock_logger.info.assert_any_call("Executed generator node and generated response: Generated response")


    # Outer patch for the Chroma class from rag_agent, inner patch for ChatGoogleGenerativeAI from rag_agent.
    @patch("RAG_BOT.rag_agent.Chroma")
    @patch("RAG_BOT.rag_agent.ChatGoogleGenerativeAI")
    def test_build_agent(self, mock_llm_class, mock_chroma_class):
        # Create instance mocks to pass into build_agent
        mock_vectordb_instance = MagicMock(spec=Chroma)
        mock_llm_instance = MagicMock(spec=ChatGoogleGenerativeAI)
        mock_llm_class.return_value = mock_llm_instance

        # Since build_agent receives the vectordb instance, we pass our instance mock directly.
        agent = build_agent(mock_vectordb_instance, model_name="test-model")

        # Assert that the built agent is of the expected type.
        self.assertIsInstance(agent, CompiledStateGraph)
        graph_nodes = agent.get_graph().nodes
        self.assertIn("should_retrieve", graph_nodes)
        self.assertIn("retriever", graph_nodes)
        self.assertIn("generator", graph_nodes)

        # Now, check that ChatGoogleGenerativeAI was instantiated as expected.
        mock_llm_class.assert_called_once_with(model="test-model", temperature=unittest.mock.ANY)


if __name__ == "__main__":
    unittest.main()