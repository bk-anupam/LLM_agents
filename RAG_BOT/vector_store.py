import os
import sys
import datetime
import re
from logger import logger
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
from pdf_processor import load_pdf, split_text, create_embeddings, semantic_chunking
from config import Config

# Define custom system prompt
sytem_prompt_text = "You are a Brahmakumaris murli teacher and are an expert in understanding the murlis and \n"
"explaining the spiritual principles mentioned in the murlis to spiritual seekers. Think step by step, explaining your \n"
"reasoning for each point you make. Analyze the question in the context of core Brahmakumaris principles, such as soul \n "
"consciousness, karma, drama, yoga, dharna, seva and the role of the Supreme Soul (Baba). Explain the underlying spiritual \n"
"logic behind your answer, drawing connections between different murli concepts. \n"
"Based on the factual information provided to you in the context, which consists of excerpts from Brahmakumaris murlis, \n"
"and the knowledge you already possess about Brahmakumaris murlis, be as detailed and as accurate in your answer as possible. \n"
"When possible, quote directly from the provided context to support your answer. \n"
"Remember, the murlis are spiritual discourses spoken by Baba, containing deep insights into self-realization \n"
"and spiritual living. Your role is to convey these teachings with clarity and understanding. \n"
"Answer in a clear, compassionate, and instructive tone, as a spiritual teacher guiding a student. \n"
"Use simple, accessible language while maintaining the depth of the murli teachings. \n"
"Where applicable, suggest practical ways the spiritual seeker can apply these principles in their daily life. \n"
"Offer insights into how these teachings can help the seeker overcome challenges and achieve spiritual progress. \n"
"Now answer the following question: \n\n"

def create_vectorstore(texts, embeddings, persist_directory=Config.VECTOR_STORE_PATH):
    """
    Creates a Chroma vectorstore from the texts and embeddings, or loads an existing one and updates it with new texts.
    Args:
        texts (list): A list of text documents to be added to the vector store.
        embeddings (callable): A function or model that generates embeddings for the texts.
        persist_directory (str, optional): The directory path where the vector store will be persisted. 
        Defaults to Config.VECTOR_STORE_PATH.
    Returns:
        Chroma: The updated or newly created Chroma vector store.
    """    
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        # Load existing vector store
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        # vectordb.load()  # Load the vectorstore from disk
        # Add new documents to the existing vector store
        vectordb.add_documents(documents=texts)
    else:
        # Create a new vector store
        vectordb = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=persist_directory)        
    logger.info(f"Vector store updated with {len(texts)} documents")    
    return vectordb


def build_index(pdf_path, chunk_size=1000, chunk_overlap=200, persist_directory=Config.VECTOR_STORE_PATH, 
                semantic_chunk=True): 
    """Builds the index from the PDF document."""
    documents = load_pdf(pdf_path)
    if semantic_chunk:
        texts = semantic_chunking(documents) # if semantic_chunk is set to True, do semantic chunking
    else:
        texts = split_text(documents, chunk_size, chunk_overlap) # otherwise do the previous chunking
    embeddings = create_embeddings()
    vectordb = create_vectorstore(texts, embeddings, persist_directory)
    return vectordb


def load_existing_index(persist_directory=Config.VECTOR_STORE_PATH):
    """Loads an existing Chroma vectorstore from disk."""
    embeddings = create_embeddings()  # Make sure you use the same embedding model used during indexing
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return vectordb


def query_index(vectordb, query, chain_type="stuff", k=25, model_name="gemini-2.0-flash", 
                date_filter=None):
    """
    Queries the vectorstore with an optional date filter and returns the answer.
    
    Args:
        vectordb: The vector database.
        query (str): The query string.
        chain_type (str): The chain type.
        k (int): The number of documents to retrieve.
        model_name (str): The name of the language model.
        date_filter (str, optional): A date string (YYYY-MM-DD) to filter documents by. Defaults to None.
    
    Returns:
        str: The answer from the language model.
    """
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.3)  # Using chat generative model
    # Create the retriever with filter if date is provided
    if date_filter:
        try:
            filter_date = datetime.datetime.strptime(date_filter, '%Y-%m-%d')
            formatted_date = filter_date.strftime('%Y-%m-%d')
            logger.info(f"Date filter with query: {formatted_date}")
        except ValueError:
            raise ValueError("Invalid date format. Please use YYYY-MM-DD.")

        filter_criteria = {"date": { "$eq": formatted_date}}

        retriever = vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k, "filter": filter_criteria}  
        )
    else:
        retriever = vectordb.as_retriever(search_kwargs={"k": k})
    # Retrieve documents
    retrieved_docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])  # Concatenate retrieved docs
    # Log retrieved documents
    logger.info(f"Retrieved {len(retrieved_docs)} documents for query: {query}")    
    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            sytem_prompt_text +
            "Context: {context}\n\n"
            "Question: {question}"
        ),
    )
    # Create an LLM chain with the custom prompt
    chain = custom_prompt | llm | RunnablePassthrough()    
    # Run the LLM chain
    response = chain.invoke({"context": context, "question": query})
     # Extract text from AIMessage
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
                sytem_prompt_text +
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
    

# add code to execute vector_store.py as a script
def index_data(build_index):
    pdf_path = "/home/bk_anupam/code/ML/LLM_apps/RAG_BOT/data"
    config = Config()
    print(f"Building index for {pdf_path}")
    pdf_files = [os.path.join(pdf_path, f) for f in os.listdir(pdf_path) if f.endswith(".pdf")]
    for pdf_file in pdf_files:
        print(f"Building index for {pdf_file}")
        vectordb = build_index(pdf_file)
    print("Indexing complete.")


def test_query_index():
    # Load the existing index
    vectordb = load_existing_index()
    # Query the index
    query = "What are the main points regarding remembrance that Baba talks about? Summarize in 2-3 sentences."
    result = query_index(vectordb, query, k=10, date_filter="1969-02-02")
    print(result)

if __name__ == "__main__":        
    index_data(build_index)    
    # test_query_index()
    
