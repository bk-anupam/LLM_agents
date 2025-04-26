# /home/bk_anupam/code/LLM_agents/RAG_BOT/agent/graph_builder.py
import functools
import os
import sys
from typing import Literal

from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from sentence_transformers import CrossEncoder

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from RAG_BOT.config import Config
from RAG_BOT.logger import logger
from RAG_BOT.context_retriever_tool import create_context_retriever_tool
from RAG_BOT.agent.state import AgentState
from RAG_BOT.agent.agent_node import agent_node
from RAG_BOT.agent.retrieval_nodes import rerank_context_node
from RAG_BOT.agent.evaluation_nodes import evaluate_context_node, reframe_query_node


# --- Conditional Edge Logic ---

def decide_next_step(state: AgentState) -> Literal["reframe_query", "agent_final_answer", "__end__"]:
    """
    Determines the next node based on evaluation result and retry status.
    """
    logger.info("--- Deciding Next Step ---")
    evaluation = state.get('evaluation_result')
    retry_attempted = state.get('retry_attempted', False)

    logger.info(f"Evaluation: {evaluation}, Retry Attempted: {retry_attempted}")

    if evaluation == "sufficient":
        logger.info("Decision: Context sufficient, proceed to final answer generation.")
        return "agent_final_answer" # Route to agent node for final answer
    elif not retry_attempted:
        logger.info("Decision: Context insufficient, attempt retry.")
        return "reframe_query" # Route to reframe node
    else:
        logger.info("Decision: Context insufficient after retry, proceed to 'cannot find' message.")
        return "agent_final_answer" # Route to agent node for "cannot find" message


# --- Graph Builder ---
def build_agent(vectordb: Chroma, model_name: str = Config.LLM_MODEL_NAME) -> StateGraph:
    """Builds the multi-node LangGraph agent."""
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=Config.TEMPERATURE)
    logger.info(f"LLM model '{model_name}' initialized with temperature {Config.TEMPERATURE}.")

    # --- Reranker Model Initialization ---
    reranker_model = None # Initialize as None
    try:
        reranker_model_name = Config.RERANKER_MODEL_NAME
        logger.info(f"Loading reranker model: {reranker_model_name}")
        reranker_model = CrossEncoder(reranker_model_name)
        logger.info("Reranker model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load reranker model '{Config.RERANKER_MODEL_NAME}': {e}", exc_info=True)
        # The graph will proceed, but rerank_context_node will skip reranking

    # --- Tool Preparation ---
    # Use INITIAL_RETRIEVAL_K for the retriever tool that feeds the reranker
    ctx_retriever_tool_instance = create_context_retriever_tool(
        vectordb=vectordb,
        k=Config.INITIAL_RETRIEVAL_K, # Use the larger K for initial retrieval
        search_type=Config.SEARCH_TYPE
    )
    logger.info(f"Context retriever tool created with k={Config.INITIAL_RETRIEVAL_K}, search_type='{Config.SEARCH_TYPE}'.")
    available_tools = [ctx_retriever_tool_instance]

    # --- LLM Binding (for initial decision in agent_node) ---
    llm_with_tools = llm.bind_tools(available_tools)
    logger.info("LLM bound with tools successfully.")
    # Create ToolNode specifically for context retrieval
    retrieve_context_node = ToolNode(tools=[ctx_retriever_tool_instance])

    # --- Bind LLM and Reranker to Nodes ---
    agent_node_runnable = functools.partial(agent_node, llm=llm, llm_with_tools=llm_with_tools)
    # Bind the loaded reranker model (or None if loading failed)
    rerank_context_node_runnable = functools.partial(rerank_context_node, reranker_model=reranker_model)
    evaluate_context_node_runnable = functools.partial(evaluate_context_node, llm=llm)
    reframe_query_node_runnable = functools.partial(reframe_query_node, llm=llm)

    # --- Define the Graph ---
    builder = StateGraph(AgentState)

    # --- Add Nodes ---
    builder.add_node("agent_initial", agent_node_runnable) # Handles initial query & first decision
    builder.add_node("retrieve_context", retrieve_context_node)
    builder.add_node("rerank_context", rerank_context_node_runnable) # Add the new reranker node
    builder.add_node("evaluate_context", evaluate_context_node_runnable)
    builder.add_node("reframe_query", reframe_query_node_runnable)
    # Use a distinct node name for final answer generation step
    builder.add_node("agent_final_answer", agent_node_runnable)

    # --- Define Edges ---
    builder.set_entry_point("agent_initial")

    # Decide whether to retrieve or answer directly from the start
    builder.add_conditional_edges(
        "agent_initial",
        tools_condition, # Checks if the AIMessage from agent_initial has tool_calls
        {
            "tools": "retrieve_context", # If tool call exists, go retrieve
            "__end__": "agent_final_answer", # If no tool call, go directly to final answer generation
        },
    )

    # --- Main RAG loop with Reranking ---
    builder.add_edge("retrieve_context", "rerank_context") # Retrieve -> Rerank
    builder.add_edge("rerank_context", "evaluate_context") # Rerank -> Evaluate

    # Conditional logic after evaluation remains the same
    builder.add_conditional_edges(
        "evaluate_context",
        decide_next_step, # Use the dedicated decision function based on evaluation of reranked context
        {
            "reframe_query": "reframe_query",
            "agent_final_answer": "agent_final_answer", # Route to final answer generation
        }
    )
    # Loop back to retrieve after reframing
    builder.add_edge("reframe_query", "retrieve_context")
    # Final answer generation leads to end
    builder.add_edge("agent_final_answer", END)
    # Compile the graph
    graph = builder.compile()
    # # Optional: Save graph visualization
    # try:
    #     graph.get_graph().draw_mermaid_png(output_file_path="rag_agent_graph.png")
    #     logger.info("Saved graph visualization to rag_agent_graph.png")
    # except Exception as e:
    #     logger.warning(f"Could not save graph visualization: {e}")
    logger.info("LangGraph agent compiled successfully...")
    return graph
