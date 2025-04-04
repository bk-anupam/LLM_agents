from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage
from langgraph.graph import START, StateGraph, END
from RAG_BOT.config import Config
import datetime
from RAG_BOT.logger import logger

def should_retrieve_node(state: dict) -> str:
    """Decides whether to perform document retrieval based on the skip_retrieval flag."""
    if state.get("skip_retrieval", False):        
        state["next"] = "generator"
        logger.info("Skipping retrieval and going directly to generator.")        
    else:                
        state["next"] = "retriever"
        logger.info("Proceeding with document retrieval.")
    return state        


def retriever_node(state: dict, vectordb: Chroma) -> dict:
    """Retrieves documents from the vector database with improved query normalization and filtering."""
    raw_query = state["query"]
    # Normalize query
    query = raw_query.strip().lower()  
    # Extract parameters from the state
    k = state.get("k", 25)
    date_filter = state.get("date_filter", None)
    search_type = state.get("search_type", "similarity")
    #score_threshold = state.get("score_threshold", 0.5)
    # extra_kwargs = {"k": k, "score_threshold": score_threshold}
    extra_kwargs = {"k": k}
    if date_filter:
        try:
            filter_date = datetime.datetime.strptime(date_filter, '%Y-%m-%d')
            formatted_date = filter_date.strftime('%Y-%m-%d')
            logger.info(f"Applying date filter: {formatted_date}")
        except ValueError:
            raise ValueError("Invalid date format. Please use YYYY-MM-DD.")
        filter_criteria = {"date": {"$eq": formatted_date}}
        extra_kwargs["filter"] = filter_criteria
    # Create retriever with refined kwargs and search type option
    retriever = vectordb.as_retriever(search_type=search_type, search_kwargs=extra_kwargs)
    retrieved_docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    logger.info(f"Executed retriever node and retrieved {len(retrieved_docs)} documents for query: {query}")
    logger.info(f"Context for query: {context}")
    return {"context": context, "query": raw_query}


def generator_node(state: dict, llm: ChatGoogleGenerativeAI) -> dict:
    """Generates a response using the LLM and retrieved context."""
    logger.info(f"Executing generator node with state: {state}") 
    system_prompt_text = Config.SYSTEM_PROMPT
    # Get context, default to empty string if not present
    context = state.get("context", "")  
    query = state["query"]
    logger.info(f"Executing generator node with query: {query} and context: {context}")
    if context:
        custom_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                system_prompt_text +
                "Context: {context}\n"
                "Question: {question}\n"
            ),
        )
        formatted_query = custom_prompt.format(context=context, question=query)
    else:
        custom_prompt = PromptTemplate(
            input_variables=["question"],
            template=(
                system_prompt_text +
                "Question: {question}\n"
            ),
        )
        formatted_query = custom_prompt.format(question=query)
    messages = [HumanMessage(content=formatted_query)]
    response = llm.invoke(messages)
    logger.info(f"Generated response: {response}")
    if isinstance(response, AIMessage):
        logger.info(f"Executed generator node and generated response: {response.content}")
        return {"answer": response.content}
    else:
        return {"answer": str(response)}


def build_agent(vectordb: Chroma, model_name: str = "gemini-2.0-flash") -> StateGraph:
    """Builds and returns a persistent LangGraph agent."""
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=Config.TEMPERATURE)
    # In this context, dict is not a variable holding a dictionary. It's a type hint or a type annotation.
    # It's telling StateGraph that the state of the graph (the data that will be passed between nodes) 
    # will be a dictionary.
    builder = StateGraph(dict)
    builder.add_node("should_retrieve", should_retrieve_node)
    builder.add_node("retriever", lambda state: retriever_node(state, vectordb))
    builder.add_node("generator", lambda state: generator_node(state, llm))    
    builder.add_edge(START, "should_retrieve")
    builder.add_conditional_edges(
        "should_retrieve",
        lambda state: state.get("next", "retriever"),
        {
            "retriever": "retriever",
            "generator": "generator",
        },
    )
    builder.add_edge("retriever", "generator")
    builder.set_entry_point("should_retrieve")
    builder.add_edge("generator", END)
    return builder.compile()