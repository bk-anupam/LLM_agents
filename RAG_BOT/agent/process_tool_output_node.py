import json
from langchain_core.messages import ToolMessage, HumanMessage
from langchain_core.documents import Document
from typing import Any, Dict, List, Optional
from RAG_BOT.logger import logger
from RAG_BOT.agent.state import AgentState

def _convert_to_documents(doc_strings: List[str], metadata_list: Optional[List[Dict[str, Any]]] = None) -> List[Document]:
    """Helper to convert list of strings to list of Documents."""
    if metadata_list and len(doc_strings) == len(metadata_list):
        return [Document(page_content=s, metadata=m) for s, m in zip(doc_strings, metadata_list)]
    return [Document(page_content=s) for s in doc_strings]


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
        doc_strings = last_message.artifact
        if doc_strings:
            updated_state.update({
                "documents": _convert_to_documents(doc_strings),
                "last_retrieval_source": "local",
                "web_search_attempted": False
            })
            logger.info(f"Processed {len(doc_strings)} local documents retrieved using {tool_name} and updated agent state.")
            return updated_state
        else:
            logger.warning("No documents were returned by retrieve_context tool. ")
            updated_state.update({
                "last_retrieval_source": "local",
                "web_search_attempted": False # This attempt was local
            })
            # If no docs from local, prepare for potential agent_initial re-run for web search
            if current_query:
                 # Add a new HumanMessage to represent the query for the next agent_initial run
                updated_state["messages"] = messages + [HumanMessage(content=current_query)]
                logger.info(f"Appended HumanMessage for '{current_query}' for agent_initial re-run after local retrieval failure.")
            else:
                logger.warning("current_query not found in state, cannot append HumanMessage for agent_initial re-run.")
            return updated_state
    
    # Handle Tavily web search
    if tool_name and tool_name.startswith("tavily"):
        if tool_name == "tavily-extract":
            try:
                logger.info(f"Processing {tool_name} tool output (string parsing for extracted content)")
                logger.debug(f"Tool content for extraction: {tool_content}")
                
                extracted_text = ""
                # Prioritize "Raw Content:" as it seems to contain the main body from logs
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
                    
                    doc_metadata = {"source": url} if url else {}
                    
                    updated_state.update({
                        "documents": _convert_to_documents([extracted_text], [doc_metadata] if doc_metadata else None),
                        "last_retrieval_source": "web",
                        "web_search_attempted": True
                    })
                    logger.info(f"Processed 1 document from {tool_name} (string parsing) and updated agent state.")
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
                    if isinstance(search_results, dict): # Handle single dict result by wrapping in list
                        search_results = [search_results]
                    else:
                        return updated_state # No valid list or dict
                    
                docs_data = [(item.get("content") or item.get("snippet") or item.get("raw_content"),
                             {k: v for k, v in item.items() 
                              if k not in ["content", "raw_content", "snippet"]})
                            for item in search_results if isinstance(item, dict)]
                
                unzipped_data = [(c, m) for c, m in docs_data if c]
                if not unzipped_data:
                    logger.info(f"No content extracted from {tool_name} search results after filtering.")
                    return updated_state
                
                docs_content, docs_metadata = zip(*unzipped_data)
                
                if docs_content:
                    updated_state.update({
                        "documents": _convert_to_documents(list(docs_content), list(docs_metadata)),
                        "last_retrieval_source": "web",
                        "web_search_attempted": True
                    })
                    logger.info(f"Processed {len(docs_content)} documents from {tool_name} and updated agent state.")
            except json.JSONDecodeError as e:
                logger.error(f"JSONDecodeError processing {tool_name} results: {e}. Content was: {tool_content}", exc_info=True)
            except Exception as e:
                logger.error(f"Error processing Tavily results for {tool_name}: {e}", exc_info=True)
            
    return updated_state