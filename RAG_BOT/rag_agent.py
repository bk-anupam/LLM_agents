# /home/bk_anupam/code/LLM_agents/RAG_BOT/rag_agent.py
import datetime
import functools
import json 
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
from sentence_transformers import CrossEncoder # Added for reranking
import os
import sys
import re

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from RAG_BOT.config import Config
from RAG_BOT.logger import logger
from RAG_BOT.context_retriever_tool import create_context_retriever_tool

# --- Helper Prompts ---
EVALUATE_CONTEXT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", Config.get_evaluate_context_prompt()),
        (
            "human",
            "Original User Question: {original_query}\n\nRetrieved Context:\n{context}"
        ),
    ]
)

REFRAME_QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", Config.get_reframe_question_prompt()),
        (
            "human",
            "Original User Question: {original_query}\nQuery Used for Failed Retrieval: {failed_query}"
        ),
    ]
)

# LangChain will expect 'system_base', 'context', and 'original_query' during invoke
FINAL_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [   # Pass raw template string
        ("system", Config.get_final_answer_system_prompt_template()), 
        # Pass raw template string ({original_query})
        ("human", Config.get_final_answer_human_prompt_template()),   
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
    # Store the raw retrieved docs before reranking (optional, for debugging/analysis)
    raw_retrieved_docs: Optional[List[str]]


# --- Node Functions ---

def agent_node(state: AgentState, llm: ChatGoogleGenerativeAI, llm_with_tools: ChatGoogleGenerativeAI):
    """
    Handles initial query, decides first action, and generates final response.
    Now ensures final response adheres to JSON format defined in FINAL_ANSWER_PROMPT.
    """
    logger.info("--- Executing Agent Node ---")
    messages = state['messages']
    last_message = messages[-1]

    # 1. Handle Initial User Query
    if isinstance(last_message, HumanMessage):
        logger.info("Handling initial user query: " + last_message.content)
        original_query_content = last_message.content
        # Decide whether to retrieve context or answer directly (usually retrieve)
        system_prompt_msg = SystemMessage(content=Config.get_system_prompt())
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

    # 2. Generate Final Answer or "Cannot Find" Message (in JSON format)
    else:
        logger.info("Generating final response.")
        evaluation = state.get('evaluation_result')
        original_query = state.get('original_query')
        context = state.get('context') # Context should be populated by evaluate_context_node

        # --- Removed direct pass-through logic ---
        # We now always generate the final answer using the FINAL_ANSWER_PROMPT
        # or generate the "cannot find" message in JSON format.

        # If context was sufficient OR if the initial decision was to answer directly
        # (which also routes here via tools_condition -> agent_final_answer)
        # and we have an original query.
        # Note: Context might be None if answering directly without retrieval.
        if evaluation == 'sufficient' or (evaluation is None and not state.get('retry_attempted')):
            logger.info("Context sufficient or answering directly. Generating final answer using FINAL_ANSWER_PROMPT.")
            # Use base LLM without tools for response generation
            final_answer_chain = FINAL_ANSWER_PROMPT | llm
            # Provide empty context if none was retrieved (direct answer case)
            final_answer = final_answer_chain.invoke({
                "system_base": Config.get_system_prompt(), # Provide system_base here
                "original_query": original_query,
                "context": context if context else "N/A" # Provide N/A if no context
            })
            # Ensure the output is AIMessage
            if not isinstance(final_answer, AIMessage):
                 final_answer = AIMessage(content=str(final_answer.content),
                                          response_metadata=getattr(final_answer, 'response_metadata', {}))
            return {"messages": [final_answer]}
        else: # Evaluation was insufficient after retry, or some other error state
            logger.info("Context insufficient after retry or error. Generating 'cannot find' message in JSON format.")
            # Format the "cannot find" message as JSON
            cannot_find_content = {
                "answer": "Relevant information cannot be found in the database to answer the question. Please try reframing your question."
            }
            # Wrap in markdown code block as per the prompt's example output
            cannot_find_json_string = f"```json\n{json.dumps(cannot_find_content, indent=2)}\n```"
            cannot_find_message = AIMessage(content=cannot_find_json_string)
            return {"messages": [cannot_find_message]}


def rerank_context_node(state: AgentState, reranker_model: CrossEncoder):
    """
    Reranks the initially retrieved documents based on the current query.
    Updates the state with the final, concatenated context string.
    """
    logger.info("--- Executing Rerank Context Node ---")
    current_query = state.get('current_query')
    logger.info(f"Current query for reranking: '{current_query}'")
    messages = state.get('messages')
    last_message = messages[-1] if messages else None
    logger.info(f"Last message type: {type(last_message)}")
    # retrieved_docs = last_message.content 
    # logger.info(f"Retrieved docs type: {type(retrieved_docs)}")
    # logger.info(f"isinstance(retrieved_docs, list): {isinstance(retrieved_docs, list)}")
    # logger.info(f"Last message content: {last_message.content if last_message else 'None'}")

    retrieved_docs_artifact = None
    # *** KEY CHANGE: Access the artifact from the last message ***
    if isinstance(last_message, ToolMessage) and hasattr(last_message, 'artifact'):
        # Optional: Check if it's the correct tool message
        if last_message.name == 'retrieve_context':
            retrieved_docs_artifact = last_message.artifact
            logger.info(f"Artifact received from tool '{last_message.name}': type {type(retrieved_docs_artifact)}")
        else:
            logger.error(f"Last message is a ToolMessage, but not from 'retrieve_context' (name: {last_message.name}). Skipping reranking.")
            # Skip reranking if not from the correct tool
            return {"context": None} 
    else:
        logger.warning(f"Last message is not a ToolMessage with an artifact (type: {type(last_message)}). Skipping reranking.")    
        return {"context": None}

    # Proceed only if we got a valid list artifact
    if not isinstance(retrieved_docs_artifact, list):
        logger.error("No valid document list artifact found to rerank.")        
        # Set context to empty or None depending on desired downstream handling
        return {"context": ""} # Or {"context": None}        

    # Validate reranker and input
    if reranker_model is None:
        logger.error("Reranker model not loaded. Concatenating original artifact docs.")
        final_context = "\n\n".join(retrieved_docs_artifact)
        return {"context": final_context}    
    
    if not retrieved_docs_artifact:
        logger.info("No documents in artifact to rerank.")
        return {"context": ""}

    logger.info(f"Reranking {len(retrieved_docs_artifact)} documents for query: '{current_query}'")
    # Prepare pairs for the cross-encoder
    pairs = [[current_query, doc] for doc in retrieved_docs_artifact]
    try:
        # Get scores from the cross-encoder
        scores = reranker_model.predict(pairs)
        logger.info(f"Reranking scores obtained (Top {Config.RERANK_TOP_N}): {scores[:Config.RERANK_TOP_N]}")
        # Combine docs with scores and sort
        scored_docs = list(zip(scores, retrieved_docs_artifact))
        scored_docs.sort(key=itemgetter(0), reverse=True)
        # Select top N documents based on config        
        reranked_docs = [doc for score, doc in scored_docs[:Config.RERANK_TOP_N]]
        logger.info(f"Selected top {len(reranked_docs)} documents after reranking.")
        # Concatenate the final context
        final_context = "\n\n".join(reranked_docs)
        logger.info(f"Final reranked context snippet: {final_context[:400]}...")
        # Update state with the final context
        # Optionally store raw docs too: "raw_retrieved_docs": retrieved_docs
        return {"context": final_context}
    except Exception as e:
        logger.error(f"Error during reranking: {e}. Using original context without reranking", exc_info=True)
        # Fallback: use the original concatenated context
        final_context = "\n\n".join(retrieved_docs_artifact)
        return {"context": final_context}


def evaluate_context_node(state: AgentState, llm: ChatGoogleGenerativeAI):
    """
    Evaluates the reranked context (read from state) based on the original query.
    """
    logger.info("--- Executing Evaluate Context Node ---")
    original_query = state.get('original_query')
    # Context is now expected to be populated by the rerank_context_node
    context_to_evaluate = state.get('context')

    # Check if context exists in the state
    if context_to_evaluate is None: # Check for None explicitly
         logger.warning("Evaluate context node reached but no context found in state (potentially due to retrieval/reranking failure). Treating as insufficient.")
         return {"evaluation_result": "insufficient"} # Context remains None

    if not original_query:
        logger.warning("Missing original query for evaluation.")
        # Context is already stored, just return insufficient
        return {"evaluation_result": "insufficient"}

    # Handle empty string context (e.g., if reranker returned no docs)
    if not context_to_evaluate:
        logger.info("Context in state is empty. Treating as insufficient.")
        return {"evaluation_result": "insufficient"}

    # Check for error messages passed as context (though reranker should return None on error)
    if isinstance(context_to_evaluate, str) and context_to_evaluate.startswith("Error:"):
        logger.warning(f"Context contains an error message: {context_to_evaluate}. Treating as insufficient.")
        return {"evaluation_result": "insufficient"}


    logger.info("Evaluating context from state...")
    eval_chain = EVALUATE_CONTEXT_PROMPT | llm
    evaluation_result_str = eval_chain.invoke({
        "original_query": original_query,
        "context": context_to_evaluate # Use context from state
    }).content.strip().upper()

    logger.info(f"Context evaluation result: {evaluation_result_str}")
    evaluation = "sufficient" if evaluation_result_str == "YES" else "insufficient"

    # Context is already in state, just return the evaluation result
    return {"evaluation_result": evaluation}


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
        "evaluation_result": None, # Reset evaluation for the next attempt
        "context": None # Clear context before retry
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


# --- Example Invocation ---
if __name__ == '__main__':
    try:
        persist_directory = Config.VECTOR_STORE_PATH
        embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL_NAME)
        vectordb_instance = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        logger.info(f"Chroma DB loaded from: {persist_directory}")

        app = build_agent(vectordb_instance)

        # Example run - User query no longer needs JSON instruction
        # user_question = "Can you summarize the murli from 1950-01-18?"
        user_question = "Can you summarize the murli of 1969-02-06"

        # Initialize state correctly
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
        # Use invoke to get the final state directly
        final_state_result = app.invoke(initial_state, {"recursion_limit": 15})
        print("\n--- Agent Invocation Complete ---")

        # Process final state
        if isinstance(final_state_result, dict) and 'messages' in final_state_result:
            print("\n--- Final State ---")
            print(f"Original Query: {final_state_result.get('original_query')}")
            print(f"Current Query: {final_state_result.get('current_query')}")
            print(f"Retry Attempted: {final_state_result.get('retry_attempted')}")
            print(f"Evaluation Result: {final_state_result.get('evaluation_result')}")
            print(f"Context Present: {bool(final_state_result.get('context'))}")

            print("\n--- Final State Messages ---")
            for m in final_state_result['messages']:
                m.pretty_print()

            final_answer_message = final_state_result['messages'][-1]
            if isinstance(final_answer_message, AIMessage):
                print("\nFinal Answer Content:")
                print(final_answer_message.content)
                # Try to parse JSON for verification
                try:
                    # Handle potential markdown code blocks
                    content_str = final_answer_message.content.strip()
                    if content_str.startswith("```json"):
                         content_str = re.sub(r"^```json\s*([\s\S]*?)\s*```$", r"\1", content_str, flags=re.MULTILINE)
                    parsed_json = json.loads(content_str)
                    print("\nParsed JSON Answer:", parsed_json.get("answer"))
                except json.JSONDecodeError:
                    print("\nWarning: Final answer content is not valid JSON.")
                except Exception as e:
                    print(f"\nWarning: Error processing final answer content: {e}")

            else:
                print("\nFinal message was not an AIMessage:", final_answer_message)
        else:
             print("\nError: Could not extract final messages from result:", final_state_result)

    except Exception as e:
        logger.error(f"Error during example run: {e}", exc_info=True)
        print(f"\nError during example run: {e}")
