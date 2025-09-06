import json
from langchain_core.messages import ToolMessage, HumanMessage
from langchain_core.documents import Document
from typing import Any, Dict, List, Optional, Tuple
from RAG_BOT.src.logger import logger
from RAG_BOT.src.agent.state import AgentState


def _convert_to_documents(
    doc_items: List[Tuple[str, Dict[str, Any]]]  # Expects list of (content, metadata) tuples
) -> List[Document]:
    """Helper to convert list of (content, metadata) tuples to list of Documents."""
    return [Document(page_content=content, metadata=meta) for content, meta in doc_items if isinstance(content, str)]


def _create_default_state(state: AgentState) -> Dict[str, Any]:
    """Create default updated state structure."""
    return {
        "documents": [],
        "last_retrieval_source": None,
        # Preserve the web_search_attempted flag from the current state
        "web_search_attempted": state.get("web_search_attempted", False)
    }


def _validate_tool_message(messages: List[Any]) -> Optional[ToolMessage]:
    """Validate and return the last ToolMessage if valid."""
    if not messages:
        logger.warning("No messages found.")
        return None
    
    last_message = messages[-1]
    if not isinstance(last_message, ToolMessage):
        logger.warning("No valid ToolMessage found. Skipping processing.")
        return None
    
    return last_message


def _process_retrieve_context_success(doc_items_artifact: List[Tuple[str, Dict[str, Any]]], state: AgentState) -> Dict[str, Any]:
    """Process successful retrieve_context tool output."""    
    valid_doc_items = [
        item for item in doc_items_artifact
        if isinstance(item, tuple) and len(item) == 2 and 
           isinstance(item[0], str) and isinstance(item[1], dict)
    ]
    
    if not valid_doc_items:
        logger.warning("Artifact from retrieve_context contained no valid (content, metadata) items.")
        return {
            "documents": [],
            "last_retrieval_source": "local",
            "web_search_attempted": state.get("web_search_attempted", False),
        }
    
    # Log invalid items
    invalid_count = len(doc_items_artifact) - len(valid_doc_items)
    if invalid_count > 0:
        logger.warning(f"Skipped {invalid_count} invalid items in artifact from retrieve_context")
    
    updated_state = _create_default_state(state)
    updated_state.update({
        "documents": _convert_to_documents(valid_doc_items),
        "last_retrieval_source": "local"
    })    
    logger.info(f"Processed {len(valid_doc_items)} hybrid local documents retrieved using retrieve_context and updated agent state.")
    return updated_state


def _process_retrieve_context_failure(messages: List[Any], current_query: Optional[str], state: AgentState) -> Dict[str, Any]:
    """Process failed retrieve_context tool output."""
    logger.warning("No valid document items were returned by retrieve_context tool.")    
    updated_state = _create_default_state(state)
    updated_state.update({
        "last_retrieval_source": "local"
    })    
    return updated_state


def _process_retrieve_context(tool_message: ToolMessage, messages: List[Any], current_query: Optional[str], state: AgentState) -> Dict[str, Any]:
    """Process retrieve_context tool output."""
    logger.info("Processing retrieve_context tool output.")    
    doc_items_artifact = tool_message.artifact
    if not (doc_items_artifact and isinstance(doc_items_artifact, list)):
        return _process_retrieve_context_failure(messages, current_query, state)
    
    return _process_retrieve_context_success(doc_items_artifact, state)


def _extract_tavily_extract_content(tool_content: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Extract content from tavily-extract tool output."""
    logger.info("Processing tavily-extract tool output (string parsing for extracted content)")
    logger.debug(f"Tool content for extraction: {tool_content}")
    
    docs_content = []
    docs_metadata = []
    
    try:
        # Extract main content
        raw_content_marker = "Raw Content: "
        raw_content_start_idx = tool_content.find(raw_content_marker)
        if raw_content_start_idx == -1:
            logger.warning("Could not find 'Raw Content:' marker in tavily-extract output")
            return docs_content, docs_metadata
        
        text_start_pos = raw_content_start_idx + len(raw_content_marker)
        extracted_text = tool_content[text_start_pos:].strip()
        
        if not extracted_text:
            logger.warning("No extracted text found in tavily-extract output")
            return docs_content, docs_metadata
        
        # Extract URL
        url = ""
        url_marker = "URL: "
        url_start_idx = tool_content.find(url_marker)
        if url_start_idx != -1:
            url_text_start = url_start_idx + len(url_marker)
            url_end_idx = tool_content.find("\n", url_text_start)
            if url_end_idx == -1:
                url_end_idx = len(tool_content)
            url = tool_content[url_text_start:url_end_idx].strip()
        
        docs_content.append(extracted_text)
        docs_metadata.append({"source": url} if url else {})
        
    except Exception as e:
        logger.error(f"Error processing tavily-extract output: {e}, Content: {tool_content}", exc_info=True)
    
    return docs_content, docs_metadata


def _extract_tavily_json_content(tool_name: str, tool_content: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Extract content from tavily tools that return JSON."""
    logger.info(f"Processing {tool_name} tool output (expected JSON list)")
    logger.debug(f"Tool content: {tool_content}")
    
    docs_content = []
    docs_metadata = []
    
    try:
        search_results = json.loads(tool_content)
        
        # Normalize to list format
        if not isinstance(search_results, list):
            logger.warning(f"{tool_name} output was not a list after JSON parsing. Content: {tool_content}")
            search_results = [search_results] if isinstance(search_results, dict) else []
        
        # Extract content from each result
        for item in search_results:
            if not isinstance(item, dict):
                continue
            
            content = item.get("content") or item.get("snippet") or item.get("raw_content")
            if content:
                docs_content.append(content)
                metadata = {k: v for k, v in item.items() if k not in ["content", "raw_content", "snippet"]}
                docs_metadata.append(metadata)
        
        if not docs_content:
            logger.info(f"No content extracted from {tool_name} search results after filtering.")
            
    except json.JSONDecodeError as e:
        logger.error(f"JSONDecodeError processing {tool_name} results: {e}. Content was: {tool_content}", exc_info=True)
    except Exception as e:
        logger.error(f"Error processing Tavily results for {tool_name}: {e}", exc_info=True)
    
    return docs_content, docs_metadata


def _process_tavily_tools(tool_name: str, tool_content: str, state: AgentState) -> Dict[str, Any]:
    """Process Tavily web search tool output."""
    if tool_name == "tavily-extract":
        docs_content, docs_metadata = _extract_tavily_extract_content(tool_content)
    else:
        docs_content, docs_metadata = _extract_tavily_json_content(tool_name, tool_content)
    
    updated_state = _create_default_state(state)
    updated_state.update({
        "last_retrieval_source": "web",
        "web_search_attempted": True
    })
    
    if docs_content:
        tavily_documents = [
            Document(page_content=c, metadata=m) 
            for c, m in zip(docs_content, docs_metadata)
        ]
        updated_state["documents"] = tavily_documents
        logger.info(f"Processed {len(tavily_documents)} documents from {tool_name} (web) and updated agent state.")
    else:
        logger.warning(f"No document content extracted from Tavily tool {tool_name}.")
    
    return updated_state


def process_tool_output_node(state: AgentState) -> Dict[str, Any]:
    """
    Parses the last ToolMessage, extracts documents, and updates state.
    Handles retrieve_context (artifact) and Tavily tools (JSON content).
    """
    logger.info("--- Executing process_tool_output_node ---")
    
    messages = state.get("messages", [])
    current_query = state.get("current_query")
    
    # Validate input
    tool_message = _validate_tool_message(messages)
    if not tool_message:
        return _create_default_state(state)
    
    tool_name = tool_message.name
    tool_content = tool_message.content
    
    # Route to appropriate processor
    if tool_name == "retrieve_context":
        return _process_retrieve_context(tool_message, messages, current_query, state)
    elif tool_name and tool_name.startswith("tavily"):
        return _process_tavily_tools(tool_name, tool_content, state)
    else:
        logger.warning(f"Unknown tool name: {tool_name}")
        return _create_default_state(state)