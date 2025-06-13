# /home/bk_anupam/code/LLM_agents/RAG_BOT/agent/state.py
from typing import List, TypedDict, Optional, Annotated, Literal
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_core.documents import Document

# --- Agent State ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    original_query: Optional[str]
    # Query for the *next* retrieval
    current_query: Optional[str] 
    # Last retrieved context
    context: Optional[str]       
    retry_attempted: bool
    # Evaluation result: 'sufficient', 'insufficient', or None
    evaluation_result: Optional[Literal['sufficient', 'insufficient']]
    # Store the raw retrieved docs before reranking (optional, for debugging/analysis)
    raw_retrieved_docs: Optional[List[str]]
     # Language code for the final answer generation
    language_code: Optional[Literal['en', 'hi']]
    # To store retrieved docs from any source
    documents: Optional[List[Document]]
    # Whether web search was attempted for context retrieval for the current query
    web_search_attempted: bool
    # Last retrieval source: 'local' for local vector store, 'web' for web search
    last_retrieval_source: Optional[Literal["local", "web"]]
