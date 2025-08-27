from typing import List, TypedDict, Optional, Annotated, Literal, Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_core.documents import Document

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    original_query: Optional[str]
    # Query for the *next* retrieval
    current_query: Optional[str] 
    # Last retrieved context
    retrieved_context: Optional[str]       
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
    # Current mode of the agent: 'default' or 'research'
    mode: Optional[Literal['default', 'research']]
    # Context for the summarization node. Must be named 'context' for SummarizationNode default behavior.
    context: Optional[dict[str, Any]]
    # Router decision: determines whether to use RAG or conversational path
    route_decision: Optional[Literal['RAG_QUERY', 'CONVERSATIONAL_QUERY']]
    # Flag to indicate that the conversation history was just summarized
    summary_was_triggered: bool
