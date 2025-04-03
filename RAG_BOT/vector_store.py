import os
import datetime
import re
from RAG_BOT.logger import logger
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from RAG_BOT.config import Config
from RAG_BOT.pdf_processor import load_pdf, split_text, semantic_chunking
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough


class VectorStore:
    def __init__(self, persist_directory=None):
        self.config = Config()
        self.persist_directory = persist_directory or self.config.VECTOR_STORE_PATH
        # Initialize the embedding model once.
        self.embeddings = HuggingFaceEmbeddings(model_name=self.config.EMBEDDING_MODEL_NAME)
        logger.info("Embedding model initialized successfully.")
        # Create or load the Chroma vector database.
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            self.vectordb = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
            logger.info("Existing vector store loaded successfully.")
        else:
            self.vectordb = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
            logger.info("New vector store created successfully.")

    def get_vectordb(self):
        return self.vectordb

    def add_documents(self, texts):
        self.vectordb.add_documents(documents=texts)
        logger.info(f"Vector store updated with {len(texts)} documents")

    def build_index(self, pdf_path, chunk_size=1000, chunk_overlap=200, semantic_chunk=True):
        documents = load_pdf(pdf_path)
        if semantic_chunk:
            texts = semantic_chunking(documents)
        else:
            texts = split_text(documents, chunk_size, chunk_overlap)
        self.add_documents(texts)
        return self.vectordb

    def query_index(self, query, chain_type="stuff", k=25, model_name="gemini-2.0-flash", date_filter=None):        
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.3)
        if date_filter:
            try:
                filter_date = datetime.datetime.strptime(date_filter, '%Y-%m-%d')
                formatted_date = filter_date.strftime('%Y-%m-%d')
                logger.info(f"Applying date filter: {formatted_date}")
            except ValueError:
                raise ValueError("Invalid date format. Please use YYYY-MM-DD.")
            filter_criteria = {"date": {"$eq": formatted_date}}
            retriever = self.vectordb.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k, "filter": filter_criteria}
            )
        else:
            retriever = self.vectordb.as_retriever(search_kwargs={"k": k})
        retrieved_docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        logger.info(f"Retrieved {len(retrieved_docs)} documents for query: {query}")
        custom_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                self.config.SYSTEM_PROMPT +
                "Context: {context}\n\n"
                "Question: {question}"
            ),
        )
        chain = custom_prompt | llm | RunnablePassthrough()
        response = chain.invoke({"context": context, "question": query})
        if isinstance(response, AIMessage):
            return response.content
        else:
            return str(response)

    def query_llm(query, model_name="gemini-2.0-flash"):
        """Queries the LLM directly and returns the answer."""
        try:
            # Initialize the LLM
            llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.3)
            # Create a custom prompt using the system prompt text
            custom_prompt = PromptTemplate(
                input_variables=["question"],
                template=(
                    Config.SYSTEM_PROMPT +
                    "Question: {question}"
                ),
            )        
            # Format the query using the custom prompt
            formatted_query = custom_prompt.format(question=query)        
            # Format the query properly as a message
            messages = [HumanMessage(content=formatted_query)]        
            # Get the response
            response = llm.invoke(messages)
            # Extract the content from the response
            return response.content
        except Exception as e:
            print(f"Error querying LLM: {str(e)}")
            return f"Sorry, I encountered an error: {str(e)}"
    

# --- Standalone script functions ---

def index_data():
    """
    Build the index for all PDFs in a given directory.
    This function uses an instance of VectorStore to index PDF files.
    """
    pdf_dir = "/home/bk_anupam/code/LLM_agents/RAG_BOT/data/1970"
    config = Config()
    print(f"Building index for {pdf_dir}")
    pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    vs = VectorStore()
    for pdf_file in pdf_files:
        print(f"Building index for {pdf_file}")
        vs.build_index(pdf_file, semantic_chunk=config.SEMANTIC_CHUNKING)
    print("Indexing complete.")


def test_query_index():
    """
    Test querying the index.
    """
    vs = VectorStore()
    query = "What are the main points regarding remembrance that Baba talks about? Summarize in 2-3 sentences."
    result = vs.query_index(query, k=10, date_filter="1969-02-02")
    print(result)


if __name__ == "__main__":
    index_data()
    # Optionally test query
    # test_query_index()
    
