import functools
from typing import Optional,  Dict, Any
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_mcp_adapters.client import MultiServerMCPClient
from sentence_transformers import CrossEncoder
from langchain_core.messages import AIMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from RAG_BOT.src.agent.conversational_node import conversational_node
from RAG_BOT.src.config.config import Config
from RAG_BOT.src.logger import logger
from RAG_BOT.src.context_retrieval.context_retriever_tool import create_context_retriever_tool
from RAG_BOT.src.agent.state import AgentState
from RAG_BOT.src.agent.agent_node import handle_question_node, generate_final_response
from RAG_BOT.src.agent.retrieval_nodes import rerank_context_node
from RAG_BOT.src import utils 
from RAG_BOT.src.agent.evaluation_nodes import evaluate_context_node, reframe_query_node
from RAG_BOT.src.agent.process_tool_output_node import process_tool_output_node
from RAG_BOT.src.agent.prompts import get_custom_summary_prompt, get_initial_summary_prompt, get_existing_summary_prompt
from RAG_BOT.src.agent.custom_nodes import LoggingSummarizationNode
from RAG_BOT.src.agent.router_node import router_node, route_query_decision
from RAG_BOT.src.agent.indexing_node import get_indexing_and_sync_node
from RAG_BOT.src.persistence.vector_store import VectorStore
from RAG_BOT.src.services.gcs_uploader import GCSUploaderService
from RAG_BOT.src.agent.conditional_edges import (
    decide_next_step_after_evaluation,
    should_invoke_tool_after_web_search_force,
    route_after_retrieval
)

async def get_mcp_server_tools(config_instance: Config):
    """
    Returns a list of MCP server tools.
    """
    if config_instance.DEV_MODE:
        tavily_mcp_command = f"export TAVILY_API_KEY='{config_instance.TAVILY_API_KEY}' && " \
                            "source /home/bk_anupam/.nvm/nvm.sh > /dev/null 2>&1 && " \
                            "nvm use v22.14.0 > /dev/null 2>&1 && " \
                            "npx --quiet -y tavily-mcp@0.2.1"
    else:
        tavily_mcp_command = f"export TAVILY_API_KEY='{config_instance.TAVILY_API_KEY}' && " \
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


def _initialize_models(config_instance: Config):
    """Initializes the LLMs and reranker model."""
    llm_flash = ChatGoogleGenerativeAI(model=config_instance.LLM_MODEL_NAME, temperature=config_instance.TEMPERATURE)
    logger.info(f"'{config_instance.LLM_MODEL_NAME}' initialized with temperature {config_instance.TEMPERATURE}.")

    llm_pro = ChatGoogleGenerativeAI(model=config_instance.TOOL_CALLING_LLM_MODEL_NAME, temperature=config_instance.TEMPERATURE)
    logger.info(f"'{config_instance.TOOL_CALLING_LLM_MODEL_NAME}' initialized with temperature {config_instance.TEMPERATURE}.")

    reranker_model = None
    try:
        reranker_model_name = config_instance.RERANKER_MODEL_NAME
        logger.info(f"Loading reranker model: {reranker_model_name}")
        reranker_model = CrossEncoder(reranker_model_name)
        logger.info("Reranker model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load reranker model '{config_instance.RERANKER_MODEL_NAME}': {e}", exc_info=True)
    return llm_flash, llm_pro, reranker_model


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
    state['docs_to_index'] = None
    state['evaluation_result'] = None
    state['retry_attempted'] = False
    state['web_search_attempted'] = False
    state['last_retrieval_source'] = None
    state['raw_retrieved_docs'] = None
    state['route_decision'] = None    
    logger.info("State cleared. Preserved fields: 'messages', 'context', 'language_code', 'mode'.")
    return state


def _define_edges_for_graph(builder: StateGraph):
    """Defines edges for the LangGraph builder with router logic."""
    builder.set_entry_point("clear_state")
    builder.add_edge("clear_state", "summarize_messages")
    builder.add_edge("summarize_messages", "router")    
    # Add conditional routing after router decision
    builder.add_conditional_edges(
        "router",
        route_query_decision,
        {
            "rag_path": "agent_initial",
            "conversational_path": "conversational_handler"
        }
    )    
    # Conversational path goes directly to end
    builder.add_edge("conversational_handler", END)
    # RAG path continues with existing logic
    builder.add_conditional_edges(
        "agent_initial",
        tools_condition,
        {
            "tools": "tool_invoker",
            "__end__": "agent_final_answer",
        },
    )
    builder.add_conditional_edges(
        "force_web_search",
        should_invoke_tool_after_web_search_force,
        {
            "tool_invoker": "tool_invoker",
            "agent_final_answer": "agent_final_answer"
        }
    )
    builder.add_edge("tool_invoker", "process_tool_output")
    builder.add_edge("process_tool_output", "index_and_sync")  # New edge
    builder.add_conditional_edges(
        "index_and_sync",  # Edge from the new node
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
async def build_agent(
    vector_store: VectorStore, 
    config_instance: Config, 
    gcs_uploader: GCSUploaderService, 
    checkpointer: Optional[BaseCheckpointSaver] = None
) -> StateGraph:
    """Builds the multi-node LangGraph agent."""
    llm_flash, llm_pro, reranker_model = _initialize_models(config_instance)
    # Note: create_context_retriever_tool expects the raw Chroma DB, not the wrapper
    ctx_retriever_tool_instance, available_tools = await _prepare_tools(vector_store.get_vectordb(), config_instance) 
    llm_with_tools = llm_flash.bind_tools(available_tools)

    # Single ToolNode for all available tools
    tool_invoker_node = ToolNode(tools=available_tools) 

    # Bind LLM and Reranker to Nodes
    handle_question_runnable  = functools.partial(handle_question_node, llm_with_tools=llm_with_tools, app_config=config_instance)
    generate_final_response_runnable = functools.partial(generate_final_response, llm=llm_flash)
    router_node_runnable = functools.partial(router_node,llm=llm_flash)
    conversational_node_runnable = functools.partial(conversational_node, llm=llm_flash)
    # Bind the loaded reranker model (or None if loading failed)
    # Use a lambda to ensure correct argument passing, especially for config_instance
    # The lambda will receive the 'state' from LangGraph as its first argument.
    rerank_context_node_runnable = lambda state_arg: rerank_context_node(
        state_arg, # This is the state passed by LangGraph
        reranker_model=reranker_model, # This is from graph_builder's scope
        app_config=config_instance    # This is also from graph_builder's scope
    )
    evaluate_context_node_runnable = functools.partial(evaluate_context_node, llm=llm_flash)    
    reframe_query_node_runnable = functools.partial(reframe_query_node, llm=llm_flash)
    
    # New node for indexing and syncing
    index_and_sync_node_runnable = get_indexing_and_sync_node(
        vector_store=vector_store, 
        gcs_uploader=gcs_uploader, 
        config=config_instance
    )
        
    # Instantiate the summarization node. Limit the max output tokens to avoid exceeding MAX_SUMMARY_TOKENS.
    summarization_model = llm_pro.bind(generation_config={"max_output_tokens": config_instance.MAX_SUMMARY_TOKENS})
    # Get all three required prompts
    custom_summary_prompt = get_custom_summary_prompt()
    initial_summary_prompt = get_initial_summary_prompt()
    existing_summary_prompt = get_existing_summary_prompt()
    summarization_node = LoggingSummarizationNode(
        input_messages_key="messages",
        output_messages_key="messages",
        # token_counter=llm.get_num_tokens_from_messages,
        model=summarization_model,
        max_tokens=config_instance.MAX_TOKENS,
        max_tokens_before_summary=config_instance.MAX_TOKENS_BEFORE_SUMMARY,
        max_summary_tokens=config_instance.MAX_SUMMARY_TOKENS,
        initial_summary_prompt=initial_summary_prompt,
        existing_summary_prompt=existing_summary_prompt,
        final_prompt=custom_summary_prompt
    )

    # Define the Graph
    builder = StateGraph(AgentState)
    # Add Nodes
    builder.add_node("clear_state", clear_state_node)
    builder.add_node("summarize_messages", summarization_node)
    builder.add_node("router", router_node_runnable)
    builder.add_node("conversational_handler", conversational_node_runnable)  
    builder.add_node("agent_initial", handle_question_runnable)    
    builder.add_node("tool_invoker", tool_invoker_node) 
    builder.add_node("force_web_search", force_web_search_node) 
    builder.add_node("process_tool_output", process_tool_output_node) 
    builder.add_node("index_and_sync", index_and_sync_node_runnable) # New node
    builder.add_node("rerank_context", rerank_context_node_runnable) 
    builder.add_node("evaluate_context", evaluate_context_node_runnable) 
    builder.add_node("reframe_query", reframe_query_node_runnable)
    builder.add_node("agent_final_answer", generate_final_response_runnable)

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
