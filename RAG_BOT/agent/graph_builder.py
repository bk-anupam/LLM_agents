# /home/bk_anupam/code/LLM_agents/RAG_BOT/agent/graph_builder.py
import functools
from typing import Literal, Optional, List, Dict, Any
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_mcp_adapters.client import MultiServerMCPClient
from sentence_transformers import CrossEncoder # type: ignore
from langchain_core.messages import AIMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from RAG_BOT.config import Config
from RAG_BOT.logger import logger
from RAG_BOT.context_retriever_tool import create_context_retriever_tool
from RAG_BOT.agent.state import AgentState
from RAG_BOT.agent.agent_node import agent_node
from RAG_BOT.agent.retrieval_nodes import rerank_context_node
from RAG_BOT import utils # Ensure utils is importable
from RAG_BOT.agent.evaluation_nodes import evaluate_context_node, reframe_query_node
from RAG_BOT.agent.process_tool_output_node import process_tool_output_node
from RAG_BOT.agent.prompts import get_custom_summary_prompt
from RAG_BOT.agent.custom_nodes import LoggingSummarizationNode

# --- Conditional Edge Logic ---

def decide_next_step_after_evaluation(state: AgentState) -> Literal["reframe_query", "agent_final_answer"]:
    """
    Determines the next node based on evaluation result and retry status.    
    """
    logger.info("--- Deciding Next Step After Evaluation ---")
    evaluation = state.get('evaluation_result')
    retry_attempted = state.get('retry_attempted', False)
    logger.info(f"Evaluation: {evaluation}, Retry Attempted: {retry_attempted}")
    if evaluation == "sufficient":
        logger.info("Decision: Context sufficient, proceed to final answer generation.")
        return "agent_final_answer"
    elif not retry_attempted:
        logger.info("Decision: Context insufficient, attempt to reframe query.")
        return "reframe_query"
    else:
        logger.info("Decision: Context insufficient after retry, proceed to 'cannot find' message.")
        return "agent_final_answer"


async def get_mcp_server_tools(config_instance: Config):
    """
    Returns a list of MCP server tools.
    """
    if config_instance.DEV_MODE:
        tavily_mcp_command = f"export TAVILY_API_KEY='{config_instance.TAVILY_API_KEY}' && " \
                            "source /home/bk_anupam/.nvm/nvm.sh > /dev/null 2>&1 && "\
                            "nvm use v22.14.0 > /dev/null 2>&1 && "\
                            "npx --quiet -y tavily-mcp@0.2.1"
    else:
        tavily_mcp_command = f"export TAVILY_API_KEY='{config_instance.TAVILY_API_KEY}' && "\
                            "npx --quiet -y tavily-mcp@0.2.1"

    mcp_client = MultiServerMCPClient(
        {
            "tavily-mcp": {
                "command": "bash",
                "args": [
                    "-c",
                    (
                        f"{tavily_mcp_command}"
                    )
                ],
                "transport": "stdio",
            },
            # Add other MCP servers as needed
        }
    )
    tools = await mcp_client.get_tools()
    return tools


async def force_web_search_node(state: AgentState) -> Dict[str, Any]:
    """
    Forces a call to the tavily-extract tool if local retrieval failed.
    This node does not invoke the LLM for a decision; it directly prepares a tool call.
    """
    logger.info("--- Executing force_web_search_node ---")
    current_query = state.get("current_query")
    language_code = state.get("language_code", "en")
    messages = state.get("messages", [])

    if not current_query:
        logger.warning("No current_query in state for force_web_search_node. Cannot proceed.")
        # This might lead to an issue, consider how to handle. For now, pass through.
        return {"messages": messages, "web_search_attempted": True} # Mark as attempted to avoid loops

    formatted_date_for_url = utils.extract_date_from_text(current_query, return_date_format="%d.%m.%y")
    if not formatted_date_for_url:
        logger.warning(f"Could not extract date from query '{current_query}' for web search URL. Skipping web search.")
        return {"messages": messages, "web_search_attempted": True}

    murli_url = Config.get_murli_url_template(language_code).format(date=formatted_date_for_url)
    if not murli_url:
        logger.warning(f"Could not generate Murli URL for lang '{language_code}' and date '{formatted_date_for_url}'. Skipping web search.")
        return {"messages": messages, "web_search_attempted": True}

    logger.info(f"Forcing tavily-extract tool call for URL: {murli_url}")
    # Construct a ToolMessage as if the LLM decided to call it.
    # The 'tool_input' for tavily-extract is typically a dictionary with 'url'.
    tool_call_message = AIMessage(
        content="",  # AIMessage can be empty if it's just for tool calls
        tool_calls=[{"name": "tavily-extract", "args": {"urls": murli_url}, "id": "tool_call_forced_web_search"}]
    )
    return {"messages": messages + [tool_call_message], "web_search_attempted": True}

# --- Helper Functions for Graph Building ---

def _initialize_llm_and_reranker(config_instance: Config):
    """Initializes the LLM and reranker model."""
    effective_model_name = config_instance.LLM_MODEL_NAME
    llm = ChatGoogleGenerativeAI(model=effective_model_name, temperature=config_instance.TEMPERATURE)
    logger.info(f"LLM model '{effective_model_name}' initialized with temperature {config_instance.TEMPERATURE}.")

    reranker_model = None
    try:
        reranker_model_name = config_instance.RERANKER_MODEL_NAME
        logger.info(f"Loading reranker model: {reranker_model_name}")
        reranker_model = CrossEncoder(reranker_model_name)
        logger.info("Reranker model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load reranker model '{config_instance.RERANKER_MODEL_NAME}': {e}", exc_info=True)
    return llm, reranker_model


async def _prepare_tools(vectordb: Chroma, config_instance: Config):
    """Prepares context retriever and MCP tools."""
    ctx_retriever_tool_instance = create_context_retriever_tool(
        vectordb=vectordb,
        config=config_instance
    )
    logger.info("Context retriever tool created using config instance.")
    mcp_tools = await get_mcp_server_tools(config_instance)
    available_tools = [ctx_retriever_tool_instance] + mcp_tools
    logger.info(f"Available tools: {[tool.name for tool in available_tools]}")
    return ctx_retriever_tool_instance, available_tools


def route_after_retrieval(state: AgentState) -> Literal["rerank_context", "force_web_search", "agent_final_answer"]:
    """
    Decides the next step after tool output has been processed.
    - If documents found: rerank.
    - If local search failed and web not tried: go back to agent_initial (which should be prompted for web search).
    - Else (web search failed or no other options): go to final answer (likely "cannot find").
    """
    logger.info("--- Routing After Retrieval Attempt ---")
    documents = state.get("documents")
    last_retrieval_source = state.get("last_retrieval_source")
    web_search_attempted = state.get("web_search_attempted", False)

    logger.info(f"Routing with: docs_count={len(documents) if documents else 0}, last_retrieval_source='{last_retrieval_source}', "
                f"web_search_attempted={web_search_attempted}")

    if documents and len(documents) > 0:
        logger.info(f"Documents found ({len(documents)}) from '{last_retrieval_source}'. Proceeding to rerank.")
        return "rerank_context"
    else: # No documents found
        logger.info(f"No documents found from '{last_retrieval_source}' retrieval attempt.")
        if last_retrieval_source == "local" and not web_search_attempted:
            logger.info("Local retrieval failed, web search not yet attempted. Routing to force_web_search.")
            return "force_web_search"
        else:
            logger.info("Web search already attempted and failed, or was the primary failed source, or agent decided against "
                        f"web search. Proceeding to final answer.")
            return "agent_final_answer"


def clear_state_node(state: AgentState) -> AgentState:
    """
    Clears the transient state fields to prepare for a new query within the same conversation.
    Preserves long-term state like message history and user preferences.
    """
    logger.info("--- Clearing Transient Agent State for New Query ---")
    
    # Fields to clear
    state['original_query'] = None
    state['current_query'] = None
    state['retrieved_context'] = None
    state['documents'] = None
    state['evaluation_result'] = None
    state['retry_attempted'] = False
    state['web_search_attempted'] = False
    state['last_retrieval_source'] = None
    state['raw_retrieved_docs'] = None
    
    logger.info("State cleared. Preserved fields: 'messages', 'context', 'language_code', 'mode'.")
    return state

def _define_edges_for_graph(builder: StateGraph):
    """Defines edges for the LangGraph builder."""
    builder.set_entry_point("clear_state")
    builder.add_edge("clear_state", "summarize_messages")
    builder.add_edge("summarize_messages", "agent_initial")
    builder.add_conditional_edges(
        "agent_initial",
        tools_condition,
        {
            "tools": "tool_invoker",
            "__end__": "agent_final_answer",
        },
    )
    builder.add_edge("force_web_search", "tool_invoker")
    builder.add_edge("tool_invoker", "process_tool_output")
    builder.add_conditional_edges(
        "process_tool_output",
        route_after_retrieval,
        {
            "rerank_context": "rerank_context",
            "force_web_search": "force_web_search",
            "agent_final_answer": "agent_final_answer"
        }
    )
    builder.add_edge("rerank_context", "evaluate_context")
    builder.add_conditional_edges(
        "evaluate_context",
        decide_next_step_after_evaluation,
        {
            "reframe_query": "reframe_query",
            "agent_final_answer": "agent_final_answer",
        }
    )
    builder.add_edge("reframe_query", "summarize_messages")
    builder.add_edge("agent_final_answer", END)


# --- Graph Builder ---
async def build_agent(vectordb: Chroma, config_instance: Config, checkpointer: Optional[BaseCheckpointSaver] = None) -> StateGraph:
    """Builds the multi-node LangGraph agent."""
    llm, reranker_model = _initialize_llm_and_reranker(config_instance)
    ctx_retriever_tool_instance, available_tools = await _prepare_tools(vectordb, config_instance) 
    llm_with_tools = llm.bind_tools(available_tools)

    # Single ToolNode for all available tools
    tool_invoker_node = ToolNode(tools=available_tools) 

    # Bind LLM and Reranker to Nodes
    agent_node_runnable = functools.partial(
        agent_node,
        llm=llm,
        llm_with_tools=llm_with_tools
    )
    # Bind the loaded reranker model (or None if loading failed)
    # Use a lambda to ensure correct argument passing, especially for config_instance
    # The lambda will receive the 'state' from LangGraph as its first argument.
    rerank_context_node_runnable = lambda state_arg: rerank_context_node(
        state_arg, # This is the state passed by LangGraph
        reranker_model=reranker_model, # This is from graph_builder's scope
        app_config=config_instance    # This is also from graph_builder's scope
    )
    evaluate_context_node_runnable = functools.partial(evaluate_context_node, llm=llm)    
    reframe_query_node_runnable = functools.partial(reframe_query_node, llm=llm)
    
    # Instantiate the summarization node
    summarization_model = llm.bind(generation_config={"max_output_tokens": config_instance.MAX_SUMMARY_TOKENS})
    custom_summary_prompt = get_custom_summary_prompt()
    summarization_node = LoggingSummarizationNode(
        input_messages_key="messages",
        output_messages_key="messages",
        token_counter=llm.get_num_tokens_from_messages,
        model=summarization_model,
        max_tokens=config_instance.MAX_TOKENS,
        max_tokens_before_summary=config_instance.MAX_TOKENS_BEFORE_SUMMARY,
        max_summary_tokens=config_instance.MAX_SUMMARY_TOKENS,
        final_prompt=custom_summary_prompt
    )

    # Define the Graph
    builder = StateGraph(AgentState)
    # Add Nodes
    builder.add_node("clear_state", clear_state_node)
    builder.add_node("summarize_messages", summarization_node)
    builder.add_node("agent_initial", agent_node_runnable)    
    builder.add_node("tool_invoker", tool_invoker_node) 
    builder.add_node("force_web_search", force_web_search_node) 
    builder.add_node("process_tool_output", process_tool_output_node) 
    builder.add_node("rerank_context", rerank_context_node_runnable) 
    builder.add_node("evaluate_context", evaluate_context_node_runnable)
    builder.add_node("reframe_query", reframe_query_node_runnable)
    builder.add_node("agent_final_answer", agent_node_runnable)

    _define_edges_for_graph(builder)

    if checkpointer is not None:
        graph = builder.compile(checkpointer=checkpointer) 
    else:
        graph = builder.compile()
    # Optional: Save graph visualization
    # try:
    #     graph.get_graph().draw_mermaid_png(output_file_path="rag_agent_graph.png")
    #     logger.info("Saved graph visualization to rag_agent_graph.png")
    # except Exception as e:
    #     logger.warning(f"Could not save graph visualization: {e}")
    logger.info("LangGraph agent compiled successfully...")
    return graph
