import json
from langchain_core.messages import ToolMessage, HumanMessage
from langchain_core.documents import Document
from typing import Any, Dict, List, Optional, Tuple
from RAG_BOT.logger import logger
from RAG_BOT.agent.state import AgentState

def _convert_to_documents(
    doc_items: List[Tuple[str, Dict[str, Any]]] # Expects list of (content, metadata) tuples
) -> List[Document]:
    """Helper to convert list of (content, metadata) tuples to list of Documents."""
    return [Document(page_content=content, metadata=meta) for content, meta in doc_items if isinstance(content, str)]


def process_tool_output_node(state: AgentState) -> Dict[str, Any]:
    """
    Parses the last ToolMessage, extracts documents, and updates state.
    Handles retrieve_context (artifact) and Tavily tools (JSON content).
    """
    updated_state = {
        "documents": [],
        "last_retrieval_source": None,
        "web_search_attempted": False
    }
    logger.info("--- Executing process_tool_output_node ---")
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None
    current_query = state.get("current_query") # Get current_query from state
    
    if not isinstance(last_message, ToolMessage):
        logger.warning("No valid ToolMessage found. Skipping processing.")
        return updated_state
        
    tool_name = last_message.name
    tool_content = last_message.content    
    
    # Handle local retrieval
    if tool_name == "retrieve_context":
        logger.info("Processing retrieve_context tool output.")
        # Artifact is now List[Tuple[str, Dict[str, Any]]]
        doc_items_artifact: Optional[List[Tuple[str, Dict[str, Any]]]] = last_message.artifact
        
        if doc_items_artifact and isinstance(doc_items_artifact, list):
            valid_doc_items = []
            for item in doc_items_artifact:
                if isinstance(item, tuple) and len(item) == 2 and \
                   isinstance(item[0], str) and isinstance(item[1], dict):
                    valid_doc_items.append(item)
                else:
                    logger.warning(f"Skipping invalid item in artifact from retrieve_context: {item}")

            if valid_doc_items:
                updated_state.update({
                    "documents": _convert_to_documents(valid_doc_items), 
                    "last_retrieval_source": "local", 
                    "web_search_attempted": False 
                })
                logger.info(f"Processed {len(valid_doc_items)} hybrid local documents retrieved using {tool_name} and updated agent state.")
                return updated_state
            else:
                logger.warning("Artifact from retrieve_context was empty or contained no valid (content, metadata) items.")
        
        logger.warning("No valid document items were returned by retrieve_context tool.")
        updated_state.update({
            "last_retrieval_source": "local", 
            "web_search_attempted": False
        })
        if current_query:
            updated_state["messages"] = messages + [HumanMessage(content=current_query)]
            logger.info(f"Appended HumanMessage for '{current_query}' for agent_initial re-run after local retrieval failure.")
        else:
            logger.warning("current_query not found in state, cannot append HumanMessage for agent_initial re-run.")
        return updated_state
    
    # Handle Tavily web search
    if tool_name and tool_name.startswith("tavily"):
        # Convert Tavily output (list of strings or dicts) to Document objects
        # For tavily-extract, tool_content is a string.
        # For other tavily tools, tool_content is a JSON string of a list of dicts.
        
        tavily_docs_content: List[str] = []
        tavily_docs_metadata: List[Dict[str, Any]] = []

        if tool_name == "tavily-extract":
            try:
                logger.info(f"Processing {tool_name} tool output (string parsing for extracted content)")
                logger.debug(f"Tool content for extraction: {tool_content}")
                
                extracted_text = ""
                raw_content_marker = "Raw Content: "                
                raw_content_start_idx = tool_content.find(raw_content_marker)
                if raw_content_start_idx != -1:
                    text_start_pos = raw_content_start_idx + len(raw_content_marker)
                    extracted_text = tool_content[text_start_pos:].strip()                

                if extracted_text:
                    url = ""
                    url_marker = "URL: "
                    url_start_idx = tool_content.find(url_marker)
                    if url_start_idx != -1:
                        url_text_start = url_start_idx + len(url_marker)
                        url_end_idx = tool_content.find("\n", url_text_start)
                        if url_end_idx == -1: url_end_idx = len(tool_content)
                        url = tool_content[url_text_start:url_end_idx].strip()
                    
                    tavily_docs_content.append(extracted_text)
                    tavily_docs_metadata.append({"source": url} if url else {})
                else:
                    logger.warning(f"Could not extract text from {tool_name} output using string parsing. Content: {tool_content}")
            except Exception as e:
                logger.error(f"Error processing {tool_name} output: {e}, Content: {tool_content}", exc_info=True)
        else: # For other tavily tools like tavily-search, tavily-crawl, tavily-map
            try:
                logger.info(f"Processing {tool_name} tool output (expected JSON list)")
                logger.debug(f"Tool content: {tool_content}")
                search_results = json.loads(tool_content)
                if not isinstance(search_results, list):
                    logger.warning(f"{tool_name} output was not a list after JSON parsing. Content: {tool_content}")
                    if isinstance(search_results, dict): 
                        search_results = [search_results]
                    else: # Neither list nor dict, cannot process
                        search_results = [] 
                    
                for item in search_results:
                    if isinstance(item, dict):
                        content = item.get("content") or item.get("snippet") or item.get("raw_content")
                        if content:
                            tavily_docs_content.append(content)
                            metadata = {k: v for k, v in item.items() if k not in ["content", "raw_content", "snippet"]}
                            tavily_docs_metadata.append(metadata)
                
                if not tavily_docs_content:
                    logger.info(f"No content extracted from {tool_name} search results after filtering.")
            except json.JSONDecodeError as e:
                logger.error(f"JSONDecodeError processing {tool_name} results: {e}. Content was: {tool_content}", exc_info=True)
            except Exception as e:
                logger.error(f"Error processing Tavily results for {tool_name}: {e}", exc_info=True)
        
        if tavily_docs_content:
            # Create Document objects from the extracted content and metadata
            # Need to adapt _convert_to_documents or do it manually here
            tavily_documents = [
                Document(page_content=c, metadata=m) for c, m in zip(tavily_docs_content, tavily_docs_metadata)
            ]
            updated_state.update({
                "documents": tavily_documents,
                "last_retrieval_source": "web",
                "web_search_attempted": True
            })
            logger.info(f"Processed {len(tavily_documents)} documents from {tool_name} (web) and updated agent state.")
        else: # No content from Tavily tools
            logger.warning(f"No document content extracted from Tavily tool {tool_name}.")
            updated_state.update({ # Still mark the attempt
                "last_retrieval_source": "web",
                "web_search_attempted": True
            })
            
    return updated_state