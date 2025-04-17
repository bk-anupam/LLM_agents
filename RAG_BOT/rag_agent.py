import datetime
import functools
from typing import List, TypedDict, Optional, Annotated, Literal
from operator import itemgetter
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    BaseMessage,
    ToolMessage,
    SystemMessage
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from RAG_BOT.config import Config
from RAG_BOT.logger import logger
from RAG_BOT.context_retriever_tool import create_context_retriever_tool

# --- Helper Prompts ---
EVALUATE_CONTEXT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", Config.EVALUATE_CONTEXT_PROMPT),
        (
            "human",
            "Original User Question: {original_query}\n\nRetrieved Context:\n{context}"
        ),
    ]
)

REFRAME_QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", Config.REFRAME_QUESTION_PROMPT),
        (
            "human", 
            "Original User Question: {original_query}\nQuery Used for Failed Retrieval: {failed_query}"
        ),
    ]
)

FINAL_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", Config.SYSTEM_PROMPT + "\n\nUse the following retrieved context to answer the question:\nContext:\n{context}"),
        ("human", "{original_query}"),
    ]
)

# --- Agent State ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    original_query: Optional[str]
    current_query: Optional[str] # Query for the *next* retrieval
    context: Optional[str]       # Last retrieved context
    retry_attempted: bool
    # Evaluation result: 'sufficient', 'insufficient', or None
    evaluation_result: Optional[Literal['sufficient', 'insufficient']]

# --- Node Functions ---

def agent_node(state: AgentState, llm: ChatGoogleGenerativeAI, llm_with_tools: ChatGoogleGenerativeAI):
    """
    Handles initial query, decides first action, and generates final response.
    """
    logger.info("--- Executing Agent Node ---")
    messages = state['messages']
    last_message = messages[-1]

    # 1. Handle Initial User Query
    if isinstance(last_message, HumanMessage):
        logger.info("Handling initial user query: " + last_message.content)
        original_query_content = last_message.content
        # Decide whether to retrieve context or answer directly (usually retrieve)
        system_prompt_msg = SystemMessage(content=Config.SYSTEM_PROMPT)
        # Use LLM with tools to decide if tool call is needed
        response = llm_with_tools.invoke([system_prompt_msg] + messages)
        logger.info("LLM invoked for initial decision.")
        # Update state for the first retrieval attempt
        return {
            "messages": [response],
            "original_query": original_query_content,
            "current_query": original_query_content, # Start with original query
            "retry_attempted": False,
            "evaluation_result": None, # Reset evaluation
            "context": None # Reset context
        }

    # 2. Generate Final Answer or "Cannot Find" Message
    # This part is reached after evaluation (either sufficient or failed retry)
    else:
        logger.info("Generating final response.")
        # Check if the last message is the direct answer from agent_initial
        if isinstance(last_message, AIMessage) and not last_message.tool_calls and not state.get('evaluation_result'):
            logger.info("Passing through direct answer from initial agent call.")
            return {"messages": [last_message]} # Return the direct answer
        evaluation = state.get('evaluation_result')
        original_query = state.get('original_query')
        context = state.get('context')

        if evaluation == 'sufficient' and context is not None:
            logger.info("Context sufficient. Generating final answer.")
            # Use base LLM without tools for response generation
            final_answer_chain = FINAL_ANSWER_PROMPT | llm 
            final_answer = final_answer_chain.invoke({
                "original_query": original_query,
                "context": context
            })
            if not isinstance(final_answer, AIMessage):
                 final_answer = AIMessage(content=str(final_answer.content), 
                                          response_metadata=getattr(final_answer, 'response_metadata', {}))
            return {"messages": [final_answer]}
        else: # Evaluation was insufficient after retry, or some other edge case
            logger.info("Context insufficient after retry or error. Generating 'cannot find' message.")
            cannot_find_message = AIMessage(
                content="Relevant information cannot be found in the database to answer the question. " \
                "Please try reframing your question."
            )
            return {"messages": [cannot_find_message]}


def evaluate_context_node(state: AgentState, llm: ChatGoogleGenerativeAI):
    """
    Evaluates the retrieved context based on the original query.
    """
    logger.info("--- Executing Evaluate Context Node ---")
    original_query = state['original_query']    
    messages = state['messages']
    last_message = messages[-1]

    # Ensure context was actually retrieved
    if not isinstance(last_message, ToolMessage) or last_message.name != 'retrieve_context':
         logger.warning("Evaluate node reached without valid preceding ToolMessage. Skipping evaluation.")
         # Potentially route to an error state or back to agent
         return {"evaluation_result": "insufficient"} # Treat as insufficient

    retrieved_context = last_message.content
    state['context'] = retrieved_context # Store context explicitly in state

    if not original_query or not retrieved_context or retrieved_context.startswith("Error:"):
        logger.warning("Missing original query or context for evaluation.")
        return {"evaluation_result": "insufficient", "context": retrieved_context}

    logger.info("Evaluating retrieved context...")
    eval_chain = EVALUATE_CONTEXT_PROMPT | llm
    evaluation_result_str = eval_chain.invoke({
        "original_query": original_query,
        "context": retrieved_context
    }).content.strip().upper()

    logger.info(f"Context evaluation result: {evaluation_result_str}")
    evaluation = "sufficient" if evaluation_result_str == "YES" else "insufficient"

    return {"evaluation_result": evaluation, "context": retrieved_context}


def reframe_query_node(state: AgentState, llm: ChatGoogleGenerativeAI):
    """
    Reframes the query if the first retrieval attempt was insufficient.
    """
    logger.info("--- Executing Reframe Query Node ---")
    original_query = state['original_query']
    # The query used in the failed attempt
    failed_query = state['current_query'] 

    if not original_query or not failed_query:
        logger.error("Missing original or current query for reframing.")
        # Handle error state - perhaps skip reframing and go to final "cannot find"
        # For now, just update state to prevent infinite loops
        return {"retry_attempted": True, "current_query": original_query} # Fallback

    logger.info("Reframing question...")
    reframe_chain = REFRAME_QUESTION_PROMPT | llm
    reframed_question = reframe_chain.invoke({
        "original_query": original_query,
        "failed_query": failed_query
    }).content.strip()

    logger.info(f"Reframed question: {reframed_question}")

    # Create an AIMessage to trigger the tool call again
    tool_call_id = f"call_retry_{datetime.datetime.now().isoformat()}"
    retry_tool_call_message = AIMessage(
        content=f"Reframing query to: {reframed_question}", # Optional content for logging/tracing
        tool_calls=[{
            "name": "retrieve_context",
            "args": {"query": reframed_question},
            "id": tool_call_id
        }]
    )

    # Return state updates AND the message to trigger the tool
    return {
        "messages": [retry_tool_call_message], # Add this message to the state
        "current_query": reframed_question,
        "retry_attempted": True,
        "evaluation_result": None # Reset evaluation for the next attempt
    }

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

# --- Tool Node Definition ---
# We need a wrapper function for the ToolNode that extracts the 'current_query' from the state
def get_retriever_tool_input(state: AgentState) -> dict:
    """Extracts the current query for the retriever tool."""
    query = state.get('current_query')
    # Potentially extract date filter from state if needed in the future
    # date_filter = state.get('date_filter')
    if not query:
        logger.error("Current query not found in state for retrieval tool!")
        # Handle error - maybe raise exception or return a default query?
        return {"query": state.get('original_query', " ")} # Fallback to original or empty
    return {"query": query} # Add other args like date_filter if necessary


# Initial decision: retrieve or end?
# We need a condition to check if the agent_initial decided to call the tool
def should_retrieve(state: AgentState) -> Literal["retrieve_context", "__end__"]:
    messages = state['messages']
    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        # Check if the tool call is for our retriever
        if any(tc['name'] == 'retrieve_context' for tc in last_message.tool_calls):
                logger.info("Agent decided to retrieve context.")
                return "retrieve_context"
    logger.info("Agent decided to answer directly or error occurred.")
    return "__end__" # If no tool call, end (or go to final answer directly)
    

# --- Graph Builder ---
def build_agent(vectordb: Chroma, model_name: str = Config.LLM_MODEL_NAME) -> StateGraph:
    """Builds the multi-node LangGraph agent."""    
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=Config.TEMPERATURE)
    # Tool Preparation
    ctx_retriever_tool_instance = create_context_retriever_tool(
        vectordb=vectordb,
        k=Config.K,
        search_type=Config.SEARCH_TYPE
    )
    logger.info(f"Context retriever tool created with k={Config.K}, search_type='{Config.SEARCH_TYPE}'.")
    available_tools = [ctx_retriever_tool_instance]
    # LLM Binding (for initial decision in agent_node)
    llm_with_tools = llm.bind_tools(available_tools)
    logger.info("LLM bound with tools successfully.")
    # Create ToolNode specifically for context retrieval
    # It needs to get its input ('current_query') from the state
    retrieve_context_node = ToolNode(
        tools=[ctx_retriever_tool_instance],
        # Use the wrapper to extract input from state
        # tool_input=get_retriever_tool_input
    )
    # Bind LLM to nodes where needed
    agent_node_runnable = functools.partial(agent_node, llm=llm, llm_with_tools=llm_with_tools)
    evaluate_context_node_runnable = functools.partial(evaluate_context_node, llm=llm)
    reframe_query_node_runnable = functools.partial(reframe_query_node, llm=llm)
    # Define the graph
    builder = StateGraph(AgentState)
    # Add nodes
    # Handles initial query & first decision
    builder.add_node("agent_initial", agent_node_runnable) 
    builder.add_node("retrieve_context", retrieve_context_node)
    builder.add_node("evaluate_context", evaluate_context_node_runnable)
    builder.add_node("reframe_query", reframe_query_node_runnable)
    # Add a separate node name for the agent when generating the final answer
    # This avoids ambiguity in conditional routing if agent_node handled both start and end
    builder.add_node("agent_final_answer", agent_node_runnable)
    # Define edges
    builder.set_entry_point("agent_initial")    
    # Decide if we should retrieve context or end
    # builder.add_conditional_edges("agent_initial", should_retrieve)
    # Decide whether to retrieve
    builder.add_conditional_edges(
        "agent_initial",
        # Assess agent decision
        tools_condition,
        {
            # Translate the condition outputs to nodes in our graph
            "tools": "retrieve_context",
            "__end__": "agent_final_answer", # Directly to final answer generation
        },
    )
    # Main loop
    builder.add_edge("retrieve_context", "evaluate_context")
    builder.add_conditional_edges(
        "evaluate_context",
        decide_next_step, # Use the dedicated decision function
        {
            "reframe_query": "reframe_query",
            "agent_final_answer": "agent_final_answer", # Route to final answer generation
            # "__end__": END # Should not happen if decide_next_step is correct
        }
    )
    # Loop back to retrieve after reframing
    builder.add_edge("reframe_query", "retrieve_context")
    # Final answer generation leads to end
    builder.add_edge("agent_final_answer", END)
    # Compile the graph
    graph = builder.compile()
    # Optional: Save graph visualization
    try:
        graph.get_graph().draw_mermaid_png(output_file_path="rag_agent_graph.png")
        logger.info("Saved graph visualization to rag_agent_graph.png")
    except Exception as e:
        logger.warning(f"Could not save graph visualization: {e}")
    logger.info("LangGraph agent compiled successfully...")
    return graph


# --- Example Invocation ---
if __name__ == '__main__':    
    try:
        persist_directory = Config.VECTOR_STORE_PATH
        embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL_NAME)
        vectordb_instance = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        logger.info(f"Chroma DB loaded from: {persist_directory}")

        app = build_agent(vectordb_instance)

        # Example run
        user_question = "Can you summarize the murli from 1970-01-18"

        # Initialize state correctly for the new structure
        initial_state = AgentState(
            messages=[HumanMessage(content=user_question)],
            original_query=None,
            current_query=None,
            context=None,
            retry_attempted=False,
            evaluation_result=None
        )

        print(f"\n--- Invoking Agent for query: '{user_question}' ---")
        print("\n--- Invoking Agent (using invoke) ---")
        # Use invoke to get the final state directly, avoiding stream issues
        final_state_result = app.invoke(initial_state, {"recursion_limit": 15})
        print("\n--- Agent Invocation Complete ---")

        # The rest of the final state processing logic remains the same
        if final_state_result:
            if isinstance(final_state_result, dict) and 'messages' in final_state_result:
                print("\n--- Final State Messages ---")
                for m in final_state_result['messages']:
                    m.pretty_print()

                final_answer_message = final_state_result['messages'][-1]
                if isinstance(final_answer_message, AIMessage):
                    print("\nFinal Answer:", final_answer_message.content)
                else:
                    print("\nFinal message was not an AIMessage:", final_answer_message)
            else:
                 print("\nError: Could not extract final messages from result:", final_state_result)
        else:
            print("\nError: Could not determine final state from stream.")

    except Exception as e:
        logger.error(f"Error during example run: {e}", exc_info=True)
        print(f"\nError during example run: {e}")
