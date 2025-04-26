# /home/bk_anupam/code/LLM_agents/RAG_BOT/agent/state.py
from typing import List, TypedDict, Optional, Annotated, Literal
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# --- Agent State ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    original_query: Optional[str]
    current_query: Optional[str] # Query for the *next* retrieval
    context: Optional[str]       # Last retrieved context
    retry_attempted: bool
    # Evaluation result: 'sufficient', 'insufficient', or None
    evaluation_result: Optional[Literal['sufficient', 'insufficient']]
    # Store the raw retrieved docs before reranking (optional, for debugging/analysis)
    raw_retrieved_docs: Optional[List[str]]
