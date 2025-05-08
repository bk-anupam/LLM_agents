import unittest
from unittest.mock import patch, MagicMock, mock_open, call
import os
import sys
# shutil is no longer needed as _move_indexed_file and index_directory are removed
from langchain_core.documents import Document
from langchain.schema import AIMessage

# Ensure the RAG_BOT module can be found
project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root_path not in sys.path:
    sys.path.insert(0, project_root_path)

from RAG_BOT.vector_store import VectorStore
# Import DocumentProcessor for mocking
from RAG_BOT.document_processor import DocumentProcessor


class TestVectorStore(unittest.TestCase):

    @patch('RAG_BOT.vector_store.DocumentProcessor') 
    @patch('RAG_BOT.vector_store.HuggingFaceEmbeddings')
    @patch('RAG_BOT.vector_store.Chroma')
    @patch('RAG_BOT.vector_store.Config')
    @patch('RAG_BOT.vector_store.logger')
    @patch('RAG_BOT.vector_store.os')
    def setUp(self, mock_os, mock_logger, mock_config_cls, mock_chroma_cls, mock_hf_embeddings_cls, mock_document_processor_cls):
        """Set up for each test."""
        # Configure mock Config instance
        self.mock_config_instance = MagicMock()
        self.mock_config_instance.VECTOR_STORE_PATH = "dummy/vector_store"
        self.mock_config_instance.EMBEDDING_MODEL_NAME = "dummy_model"
        self.mock_config_instance.DATA_PATH = "dummy/data" # Kept for now, though index_directory is removed
        self.mock_config_instance.INDEXED_DATA_PATH = "dummy/indexed_data" # Kept for now
        self.mock_config_instance.SEMANTIC_CHUNKING = True
        self.mock_config_instance.CHUNK_SIZE = 1000 # Added for index_document tests
        self.mock_config_instance.CHUNK_OVERLAP = 200 # Added for index_document tests
        self.mock_config_instance.get_system_prompt.return_value = "System prompt:"
        mock_config_cls.return_value = self.mock_config_instance

        # Configure mock os
        self.mock_os = mock_os
        self.mock_os.path.exists.return_value = False # Default: new store
        self.mock_os.listdir.return_value = []      # Default: empty directory
        self.mock_os.path.join.side_effect = os.path.join # Use actual os.path.join
        self.mock_os.path.abspath.side_effect = os.path.abspath
        self.mock_os.path.dirname.side_effect = os.path.dirname
        self.mock_os.path.basename.side_effect = os.path.basename
        self.mock_os.path.isdir.return_value = True # Assume paths are dirs by default

        # Configure mock Chroma
        self.mock_chroma_instance = MagicMock()
        mock_chroma_cls.return_value = self.mock_chroma_instance

        # Configure mock Embeddings
        self.mock_embeddings_instance = MagicMock()
        mock_hf_embeddings_cls.return_value = self.mock_embeddings_instance
        
        # Configure mock DocumentProcessor
        self.mock_document_processor_instance = MagicMock()
        mock_document_processor_cls.return_value = self.mock_document_processor_instance

        self.mock_logger = mock_logger

        # Initialize VectorStore - this will call the mocked dependencies
        self.vector_store = VectorStore()

    @patch('RAG_BOT.vector_store.DocumentProcessor') # Changed
    @patch('RAG_BOT.vector_store.HuggingFaceEmbeddings')
    @patch('RAG_BOT.vector_store.Chroma')
    @patch('RAG_BOT.vector_store.Config')
    @patch('RAG_BOT.vector_store.logger')
    @patch('RAG_BOT.vector_store.os')
    def test_init_new_vector_store(self, mock_os, mock_logger, mock_config_cls, mock_chroma_cls, mock_hf_embeddings_cls, mock_document_processor_cls): # Changed
        """Test initialization of a new vector store."""
        mock_config_instance = MagicMock()
        mock_config_instance.VECTOR_STORE_PATH = "new_store_path" # Use specific path for this test
        mock_config_instance.EMBEDDING_MODEL_NAME = "dummy_model_new"
        mock_config_cls.return_value = mock_config_instance

        mock_os.path.exists.return_value = True # Dir exists
        mock_os.listdir.return_value = []      # Dir is empty, so new store
        mock_os.path.join.side_effect = os.path.join
        mock_os.makedirs.return_value = None

        mock_embeddings_instance = MagicMock()
        mock_hf_embeddings_cls.return_value = mock_embeddings_instance
        mock_chroma_instance = MagicMock()
        mock_chroma_cls.return_value = mock_chroma_instance

        # Re-initialize to test this specific scenario
        vs = VectorStore(persist_directory="new_store_path")

        # Check HuggingFaceEmbeddings initialization
        mock_hf_embeddings_cls.assert_called_with(model_name="dummy_model_new")

        # Check Chroma initialization for new store
        mock_chroma_cls.assert_called_with(
            persist_directory="new_store_path",
            embedding_function=mock_embeddings_instance
        )
        mock_logger.info.assert_any_call("Embedding model initialized successfully.")
        mock_logger.info.assert_any_call(f"New vector store created successfully at: new_store_path")
        self.assertIsNotNone(vs.document_processor) # Changed

    @patch('RAG_BOT.vector_store.DocumentProcessor') # Added DocumentProcessor mock
    @patch('RAG_BOT.vector_store.HuggingFaceEmbeddings')
    @patch('RAG_BOT.vector_store.Chroma')
    @patch('RAG_BOT.vector_store.Config')
    @patch('RAG_BOT.vector_store.logger')
    @patch('RAG_BOT.vector_store.os')
    def test_init_load_existing_vector_store(self, mock_os, mock_logger, mock_config_cls, mock_chroma_cls, mock_hf_embeddings_cls, mock_document_processor_cls): # Added mock_document_processor_cls
        """Test initialization loading an existing vector store."""
        mock_config_instance = MagicMock()
        mock_config_instance.VECTOR_STORE_PATH = "existing_store_path"
        mock_config_instance.EMBEDDING_MODEL_NAME = "dummy_model_existing"
        mock_config_cls.return_value = mock_config_instance

        mock_os.path.exists.return_value = True
        mock_os.listdir.return_value = ["some_file.db"] # Non-empty dir
        mock_os.path.join.side_effect = os.path.join
        mock_os.makedirs.return_value = None

        mock_embeddings_instance = MagicMock()
        mock_hf_embeddings_cls.return_value = mock_embeddings_instance
        mock_chroma_instance = MagicMock()
        mock_chroma_cls.return_value = mock_chroma_instance

        vs = VectorStore(persist_directory="existing_store_path")

        mock_hf_embeddings_cls.assert_called_with(model_name="dummy_model_existing")
        mock_chroma_cls.assert_called_with(
            persist_directory="existing_store_path",
            embedding_function=mock_embeddings_instance
        )
        mock_logger.info.assert_any_call(f"Existing vector store loaded successfully from: existing_store_path")
        self.assertIsNotNone(vs.document_processor) # Added assertion for document_processor

    @patch('RAG_BOT.vector_store.DocumentProcessor') # Added DocumentProcessor mock
    @patch('RAG_BOT.vector_store.HuggingFaceEmbeddings')
    @patch('RAG_BOT.vector_store.Chroma')
    @patch('RAG_BOT.vector_store.Config')
    @patch('RAG_BOT.vector_store.logger')
    @patch('RAG_BOT.vector_store.os')
    def test_init_load_error_fallback_to_new(self, mock_os, mock_logger, mock_config_cls, mock_chroma_cls, mock_hf_embeddings_cls, mock_document_processor_cls): # Added mock_document_processor_cls
        """Test fallback to new store creation if loading existing fails."""
        mock_config_instance = MagicMock()
        mock_config_instance.VECTOR_STORE_PATH = "fallback_store_path"
        mock_config_instance.EMBEDDING_MODEL_NAME = "dummy_model_fallback"
        mock_config_cls.return_value = mock_config_instance

        mock_os.path.exists.return_value = True
        mock_os.listdir.return_value = ["some_file.db"] # Non-empty dir
        mock_os.path.join.side_effect = os.path.join
        mock_os.makedirs.return_value = None

        mock_embeddings_instance = MagicMock()
        mock_hf_embeddings_cls.return_value = mock_embeddings_instance
        mock_chroma_instance_success = MagicMock() # For the successful creation

        # Simulate Chroma load failure then success on new creation
        mock_chroma_cls.side_effect = [Exception("Load failed"), mock_chroma_instance_success]

        vs = VectorStore(persist_directory="fallback_store_path")

        mock_hf_embeddings_cls.assert_called_with(model_name="dummy_model_fallback")
        self.assertEqual(mock_chroma_cls.call_count, 2)
        # First call (failed load)
        mock_chroma_cls.assert_any_call(
            persist_directory="fallback_store_path",
            embedding_function=mock_embeddings_instance
        )
        # Second call (successful creation)
        mock_chroma_cls.assert_any_call(
            persist_directory="fallback_store_path",
            embedding_function=mock_embeddings_instance
        )

        mock_logger.error.assert_any_call(f"Error loading existing vector store from fallback_store_path: Load failed", exc_info=True)
        mock_logger.warning.assert_any_call(f"Attempting to create a new vector store at fallback_store_path due to loading error.")
        mock_logger.info.assert_any_call("New vector store created after load failure.")
        self.assertIsNotNone(vs.document_processor) # Added assertion for document_processor

    @patch('RAG_BOT.vector_store.DocumentProcessor') # Added DocumentProcessor mock
    @patch('RAG_BOT.vector_store.HuggingFaceEmbeddings')
    @patch('RAG_BOT.vector_store.Chroma')
    @patch('RAG_BOT.vector_store.Config')
    @patch('RAG_BOT.vector_store.logger')
    @patch('RAG_BOT.vector_store.os')
    def test_init_critical_creation_failure(self, mock_os, mock_logger, mock_config_cls, mock_chroma_cls, mock_hf_embeddings_cls, mock_document_processor_cls): # Added mock_document_processor_cls
        """Test critical failure during new vector store creation."""
        mock_config_instance = MagicMock()
        mock_config_instance.VECTOR_STORE_PATH = "critical_fail_path"
        mock_config_instance.EMBEDDING_MODEL_NAME = "dummy_model_critical"
        mock_config_cls.return_value = mock_config_instance

        mock_os.path.exists.return_value = True # Dir exists
        mock_os.listdir.return_value = []      # Dir is empty, new store attempt
        mock_os.path.join.side_effect = os.path.join
        mock_os.makedirs.return_value = None

        mock_hf_embeddings_cls.return_value = MagicMock()
        mock_chroma_cls.side_effect = Exception("Critical creation failed")

        with self.assertRaises(Exception) as context:
            VectorStore(persist_directory="critical_fail_path")
        self.assertTrue("Critical creation failed" in str(context.exception))
        mock_hf_embeddings_cls.assert_called_with(model_name="dummy_model_critical")
        mock_logger.critical.assert_any_call(f"Failed to create new vector store at critical_fail_path: Critical creation failed", exc_info=True)

    def test_get_vectordb(self):
        """Test the get_vectordb getter."""
        self.assertEqual(self.vector_store.get_vectordb(), self.mock_chroma_instance)

    @patch('RAG_BOT.vector_store.logger')
    def test_add_documents_success(self, mock_logger):
        """Test adding documents successfully."""        
        docs = [Document(page_content="doc1", metadata={"source": "source1.txt"})]
        self.vector_store.add_documents(docs)
        self.mock_chroma_instance.add_documents.assert_called_once_with(documents=docs)
        mock_logger.info.assert_called_with("Vector store updated with 1 document chunks from source: source1.txt") # os.path.basename

    @patch('RAG_BOT.vector_store.logger')
    def test_add_documents_empty_list(self, mock_logger):
        """Test adding an empty list of documents."""
        #self.mock_logger.reset_mock()
        self.vector_store.add_documents([])
        self.mock_chroma_instance.add_documents.assert_not_called()
        mock_logger.warning.assert_called_with("Attempted to add an empty list of documents. Skipping.")

    @patch('RAG_BOT.vector_store.logger')
    def test_add_documents_failure(self, mock_logger):
        """Test failure during adding documents to ChromaDB."""        
        docs = [Document(page_content="doc1", metadata={"source": "source_fail.txt"})]
        self.mock_chroma_instance.add_documents.side_effect = Exception("DB add error")
        self.vector_store.add_documents(docs)
        mock_logger.error.assert_called_with("Failed to add documents from source source_fail.txt to ChromaDB: DB add error", exc_info=True) # os.path.basename

    def test_document_exists_true(self):
        """Test document_exists when document is found.""" # Renamed
        self.mock_chroma_instance.get.return_value = {"ids": ["id1"], "metadatas": [], "documents": []}
        exists = self.vector_store.document_exists("2023-01-01", "en") # Renamed
        self.assertTrue(exists)
        self.mock_chroma_instance.get.assert_called_once_with(
            where={"$and": [{"date": "2023-01-01"}, {"language": "en"}]},
            limit=1,
            include=[]
        )

    def test_document_exists_false(self):
        """Test document_exists when document is not found.""" # Renamed
        self.mock_chroma_instance.get.return_value = {"ids": [], "metadatas": [], "documents": []}
        exists = self.vector_store.document_exists("2023-01-02", "en") # Renamed
        self.assertFalse(exists)

    @patch('RAG_BOT.vector_store.logger')
    def test_document_exists_no_date(self, mock_logger):
        """Test document_exists with no date string.""" # Renamed        
        exists = self.vector_store.document_exists(None, "en") # Renamed
        self.assertFalse(exists)
        mock_logger.warning.assert_called_with("Cannot check for existing document without a date string. Assuming it does not exist.")

    @patch('RAG_BOT.vector_store.logger')
    def test_document_exists_no_language(self, mock_logger):
        """Test document_exists with no language string.""" # Renamed
        self.mock_logger.reset_mock()
        exists = self.vector_store.document_exists("2023-01-01", None) # Renamed
        self.assertFalse(exists)
        mock_logger.warning.assert_called_with("Cannot check for existing document without language metadata. Assuming it does not exist.")

    @patch('RAG_BOT.vector_store.logger')
    def test_document_exists_db_error(self, mock_logger):
        """Test document_exists when ChromaDB get fails.""" # Renamed
        self.mock_logger.reset_mock()
        self.mock_chroma_instance.get.side_effect = Exception("DB get error")
        exists = self.vector_store.document_exists("2023-01-03", "en") # Renamed
        self.assertFalse(exists) # Should assume not exists on error
        mock_logger.error.assert_called_with(
            "Error checking ChromaDB for existing date 2023-01-03 and language en: DB get error. Assuming document does not exist.",
            exc_info=True
        )

    @patch.object(VectorStore, 'document_exists') 
    @patch('RAG_BOT.vector_store.logger')    
    def test_index_document_semantic_success(self, mock_logger, mock_doc_exists): 
        """Test index_document for a new document with semantic chunking."""        
        mock_doc_exists.return_value = False
        docs_in = [Document(page_content="doc content", metadata={"source": "test.doc", "date": "2023-01-01", "language": "en"})] # Generic doc
        chunks_out = [Document(page_content="chunk1", metadata={"source": "test.doc", "date": "2023-01-01", "language": "en"})]
        
        # Ensure the mock_document_processor_instance from setUp is used and configured
        self.mock_document_processor_instance.semantic_chunking.return_value = chunks_out # Use mock_document_processor_instance
        self.mock_document_processor_instance.semantic_chunking.side_effect = None 

        # self.mock_config_instance.SEMANTIC_CHUNKING = True # This should be set in setUp or per test if varied
        
        was_indexed = self.vector_store.index_document(docs_in, semantic_chunk=True) # Renamed

        self.assertTrue(was_indexed)
        mock_doc_exists.assert_called_once_with("2023-01-01", "en")
        self.mock_document_processor_instance.semantic_chunking.assert_called_once_with( 
            docs_in,
            chunk_size=self.mock_config_instance.CHUNK_SIZE,
            chunk_overlap=self.mock_config_instance.CHUNK_OVERLAP,
            model_name=self.mock_config_instance.EMBEDDING_MODEL_NAME
        )
        self.mock_chroma_instance.add_documents.assert_called_once_with(documents=chunks_out)
        
        # Assert no error or warning logs that would indicate a False return path
        mock_logger.error.assert_not_called()
        mock_logger.warning.assert_not_called()

        # Assert all expected info logs for the success path
        mock_logger.info.assert_any_call("Proceeding with chunking and indexing for test.doc.")
        # The add_documents method also logs an info message
        mock_logger.info.assert_any_call(f"Vector store updated with {len(chunks_out)} document chunks from source: test.doc")
        mock_logger.info.assert_any_call(f"Successfully indexed {len(chunks_out)} chunks from test.doc.")

    @patch.object(VectorStore, 'document_exists')     
    def test_index_document_non_semantic_success(self, mock_doc_exists): 
        """Test index_document for a new document with non-semantic chunking.""" # Renamed        
        self.mock_logger.reset_mock()
        mock_doc_exists.return_value = False
        docs_in = [Document(page_content="doc content", metadata={"source": "test.doc", "date": "2023-01-02", "language": "hi"})] # Generic doc
        chunks_out = [Document(page_content="chunk_doc", metadata={"source": "test.doc", "date": "2023-01-02", "language": "hi"})]
        self.mock_document_processor_instance.split_text.return_value = chunks_out 
        self.mock_document_processor_instance.split_text.side_effect = None
        self.mock_chroma_instance.add_documents.side_effect = None                

        was_indexed = self.vector_store.index_document(docs_in, semantic_chunk=False) # Renamed, removed chunk_size/overlap args

        self.assertTrue(was_indexed)
        mock_doc_exists.assert_called_once_with("2023-01-02", "hi")
        self.mock_document_processor_instance.split_text.assert_called_once_with( # Use mock_document_processor_instance
            docs_in, 
            chunk_size=self.mock_config_instance.CHUNK_SIZE, 
            chunk_overlap=self.mock_config_instance.CHUNK_OVERLAP
        )
        self.mock_chroma_instance.add_documents.assert_called_once_with(documents=chunks_out)
        # Assert no error or warning logs that would indicate a False return path
        self.mock_logger.error.assert_not_called()
        self.mock_logger.warning.assert_not_called()
        

    @patch.object(VectorStore, 'document_exists') # Renamed
    def test_index_document_already_exists(self, mock_doc_exists):
        """Test index_document skips if document already exists.""" # Renamed
        self.mock_logger.reset_mock()
        mock_doc_exists.return_value = True
        docs_in = [Document(page_content="content", metadata={"source": "exist.doc", "date": "2023-01-03", "language": "en"})]
        
        was_indexed = self.vector_store.index_document(docs_in) # Renamed

        self.assertFalse(was_indexed)
        mock_doc_exists.assert_called_once_with("2023-01-03", "en")
        self.mock_document_processor_instance.semantic_chunking.assert_not_called() # Check generic processor
        self.mock_document_processor_instance.split_text.assert_not_called() # Check generic processor
        self.mock_chroma_instance.add_documents.assert_not_called()
        self.mock_logger.info.assert_called_with("Document with date 2023-01-03 and language en (source: exist.doc) already indexed. Skipping.")

    def test_index_document_empty_input(self):
        """Test index_document with empty document list.""" # Renamed
        self.mock_logger.reset_mock()
        was_indexed = self.vector_store.index_document([]) # Renamed
        self.assertFalse(was_indexed)
        self.mock_logger.warning.assert_called_with("Attempted to index an empty list of documents. Skipping.")

    @patch.object(VectorStore, 'document_exists') # Renamed
    def test_index_document_no_chunks_generated(self, mock_doc_exists):
        """Test index_document when no chunks are generated.""" # Renamed
        self.mock_logger.reset_mock()
        mock_doc_exists.return_value = False
        docs_in = [Document(page_content="doc content", metadata={"source": "no_chunks.doc", "date": "2023-01-04", "language": "en"})]
        self.mock_document_processor_instance.semantic_chunking.return_value = [] # No chunks
        
        was_indexed = self.vector_store.index_document(docs_in, semantic_chunk=True) # Renamed

        self.assertFalse(was_indexed)
        self.mock_logger.warning.assert_called_with("No text chunks generated after processing no_chunks.doc. Nothing to index.")

    @patch.object(VectorStore, 'document_exists') # Renamed
    def test_index_document_chunking_error(self, mock_doc_exists):
        """Test index_document when chunking raises an error.""" # Renamed
        self.mock_logger.reset_mock()
        mock_doc_exists.return_value = False
        docs_in = [Document(page_content="doc content", metadata={"source": "chunk_error.doc", "date": "2023-01-05", "language": "en"})]
        self.mock_document_processor_instance.semantic_chunking.side_effect = Exception("Chunking error")

        was_indexed = self.vector_store.index_document(docs_in, semantic_chunk=True) # Renamed
        self.assertFalse(was_indexed)
        self.mock_logger.error.assert_called_with("Error during chunking or adding documents for chunk_error.doc: Chunking error", exc_info=True)
        self.mock_logger.info.assert_any_call("Proceeding with chunking and indexing for chunk_error.doc.")

    # Removed all _move_indexed_file tests as the method is removed
    # Removed all index_directory tests as the method is removed

    def test_log_all_indexed_metadata_success(self):
        """Test log_all_indexed_metadata with data."""
        self.mock_logger.reset_mock()
        self.mock_chroma_instance.get.return_value = {
            "ids": ["id1", "id2", "id3"],
            "metadatas": [
                {"date": "2023-01-01", "is_avyakt": True, "language": "en"},
                {"date": "2023-01-01", "is_avyakt": True, "language": "en"},
                {"date": "2023-01-02", "is_avyakt": False, "language": "hi"},
            ]
        }
        self.vector_store.log_all_indexed_metadata()
        self.mock_logger.info.assert_any_call("Retrieved metadata for 3 documents.")
        self.mock_logger.info.assert_any_call("Date: 2023-01-01 - Type: Avyakt, Language: en, Count: 2")
        self.mock_logger.info.assert_any_call("Date: 2023-01-02 - Type: Sakar/Other, Language: hi, Count: 1")

    def test_log_all_indexed_metadata_empty_db(self):
        """Test log_all_indexed_metadata with an empty database."""
        self.mock_logger.reset_mock()
        self.mock_chroma_instance.get.return_value = {"ids": [], "metadatas": []}
        self.vector_store.log_all_indexed_metadata()
        self.mock_logger.info.assert_any_call("ChromaDB index appears to be empty. No metadata to retrieve.")

    def test_log_all_indexed_metadata_db_error(self):
        """Test log_all_indexed_metadata when ChromaDB get fails."""
        self.mock_logger.reset_mock()
        self.mock_chroma_instance.get.side_effect = Exception("DB get all error")
        self.vector_store.log_all_indexed_metadata()
        self.mock_logger.error.assert_called_with("Error retrieving all metadata from ChromaDB: DB get all error", exc_info=True)

    @patch('RAG_BOT.vector_store.ChatGoogleGenerativeAI')
    @patch('RAG_BOT.vector_store.PromptTemplate')
    # Removed RunnablePassthrough mock as it's not directly used in the chain logic being tested here
    def test_query_index_success(self, mock_prompt_template_cls, mock_chat_google_cls):
        """Test query_index successfully returns a response."""
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = AIMessage(content="Test LLM response")
        mock_chat_google_cls.return_value = mock_llm_instance

        mock_prompt_instance = MagicMock()
        mock_prompt_template_cls.return_value = mock_prompt_instance
        
        # Mock retriever and its invoke
        mock_retriever = MagicMock()
        retrieved_docs = [Document(page_content="context doc 1")]
        mock_retriever.invoke.return_value = retrieved_docs
        self.mock_chroma_instance.as_retriever.return_value = mock_retriever

        query = "What is love?"
        response = self.vector_store.query_index(query, k=5)

        self.assertEqual(response, "Test LLM response")
        self.mock_chroma_instance.as_retriever.assert_called_once_with(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        mock_retriever.invoke.assert_called_once_with(query)
        mock_prompt_template_cls.assert_called_once_with(
            input_variables=["context", "question"],
            template="System prompt:\n\nContext:\n{context}\n\nQuestion: {question}" # Assuming "System prompt:" is from mock_config
        )
        self.mock_config_instance.get_system_prompt.assert_called_with(language_code="en")
        mock_llm_instance.invoke.assert_called_once_with({"context": "context doc 1", "question": query})


    def test_query_index_vectordb_not_initialized(self):
        """Test query_index when vectordb is not initialized."""
        self.mock_logger.reset_mock()
        original_vectordb = self.vector_store.vectordb # Store original
        self.vector_store.vectordb = None # Simulate uninitialized DB
        response = self.vector_store.query_index("test query")
        self.assertEqual(response, "Error: Vector Store is not available.")
        self.mock_logger.error.assert_called_with("VectorDB not initialized. Cannot perform query.")

    @patch('RAG_BOT.vector_store.ChatGoogleGenerativeAI')
    def test_query_index_with_date_filter(self, mock_chat_google_cls):
        """Test query_index with a valid date filter."""
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = AIMessage(content="Filtered response")
        mock_chat_google_cls.return_value = mock_llm_instance
        
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        self.mock_chroma_instance.as_retriever.return_value = mock_retriever

        self.vector_store.query_index("query", date_filter="2023-01-01")
        self.mock_chroma_instance.as_retriever.assert_called_with(
            search_type="similarity",
            search_kwargs={"k": 25, "filter": {"date": "2023-01-01"}} # Default k is 25
        )
        self.mock_config_instance.get_system_prompt.assert_called_with(language_code="en")

    def test_query_index_invalid_date_filter(self):
        """Test query_index with an invalid date filter format."""
        self.mock_logger.reset_mock()
        response = self.vector_store.query_index("query", date_filter="01-01-2023")
        self.assertEqual(response, "Error: Invalid date format for filter. Please use YYYY-MM-DD.")
        self.mock_logger.error.assert_called_with("Invalid date format provided: 01-01-2023. Should be YYYY-MM-DD.")

    @patch('RAG_BOT.vector_store.ChatGoogleGenerativeAI')
    def test_query_index_llm_error(self, mock_chat_google_cls):
        self.mock_logger.reset_mock()
        """Test query_index when the LLM call fails."""
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.side_effect = Exception("LLM API error")
        mock_chat_google_cls.return_value = mock_llm_instance

        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [Document(page_content="context")]
        self.mock_chroma_instance.as_retriever.return_value = mock_retriever
        
        response = self.vector_store.query_index("query")
        self.assertEqual(response, "Sorry, an error occurred while processing your query.")
        self.mock_logger.error.assert_called_with("Error during query execution: LLM API error", exc_info=True)
        self.mock_config_instance.get_system_prompt.assert_called_with(language_code="en")


if __name__ == '__main__':
    unittest.main()
