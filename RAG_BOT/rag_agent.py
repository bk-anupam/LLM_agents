import datetime
import functools
from typing import List, TypedDict, Optional, Annotated
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
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import os
import sys
# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from RAG_BOT.config import Config # Import Config
from RAG_BOT.logger import logger
# Import the factory function instead of the direct tool
from RAG_BOT.context_retriever_tool import create_context_retriever_tool

# Define the state structure for the graph
class AgentState(TypedDict):
    # messages is the primary list for interaction history and tool calls/results    
    messages: Annotated[List[BaseMessage], add_messages]
    # Optional: Keep query for logging or specific needs, but flow relies on messages
    query: Optional[str]    


# --- Agent Node ---
# This node now decides if tools are needed OR generates the final response
def agent_node(state: AgentState, llm_with_tools: ChatGoogleGenerativeAI):
    """
    Invokes the LLM with tools to decide the next step or generate a response.

    Args:
        state: The current graph state.
        llm_with_tools: The LLM instance bound with tools.

    Returns:
        A dictionary updating the 'messages' state.
    """
    logger.info(f"Executing agent node...")
    # logger.debug(f"Current messages in state: {state['messages']}") # Can be verbose

    # Ensure the system prompt is included.
    # Prepending it ensures it's considered in every LLM call within this node.
    system_prompt = SystemMessage(content=Config.SYSTEM_PROMPT)
    messages_for_llm = [system_prompt] + state['messages']

    # Store the original query if the last message is Human (first entry)
    last_message = state['messages'][-1]
    if isinstance(last_message, HumanMessage):
        state['query'] = last_message.content # Store for potential logging
        logger.info(f"User query: {state['query']}")

    # Invoke the LLM. It will either return a response or a tool call request.
    # The message history (including previous tool results) is passed.
    logger.info("Invoking LLM...")
    # logger.debug(f"Messages sent to LLM: {messages_for_llm}") # Can be verbose
    response = llm_with_tools.invoke(messages_for_llm)
    logger.info("LLM invocation complete.")        
    # We return the response message to be appended to the state by LangGraph
    return {"messages": [response]}


def build_agent(vectordb: Chroma, model_name: str = "gemini-2.0-flash") -> StateGraph:
    """Builds and returns a LangGraph agent incorporating the murli_retriever tool."""
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=Config.TEMPERATURE)

    # --- Tool Preparation ---
    # 1. Create the tool instance using the factory function, passing the vectordb and config values
    ctx_retriever_tool = create_context_retriever_tool(
        vectordb=vectordb,
        k=Config.K,                
        search_type=Config.SEARCH_TYPE 
    )
    logger.info(f"Context retriever tool created successfully with k={Config.K} and search_type='{Config.SEARCH_TYPE}'.")
    # Now 'available_tools' contains a valid callable tool that ToolNode and bind_tools expect.
    available_tools = [ctx_retriever_tool]

    # --- ToolNode ---
    # ToolNode can now directly use the tool returned by the factory
    tool_node = ToolNode(available_tools) 
    logger.info("ToolNode created successfully.")
    # This creates a pre-built LangGraph node specifically designed to execute tools.
    # When activated, it looks at the last message in the state. If it's an AIMessage
    # with tool_calls matching one of the 'tools' provided here, it executes the
    # corresponding tool function (retrieve_murli_context_tool) with the arguments from the
    # tool_call. It then formats the result as a ToolMessage.

    # --- LLM Binding ---
    llm_with_tools = llm.bind_tools(available_tools) # Use the new name
    logger.info("LLM bound with tools successfully.")
    # This modifies the LLM instance. It attaches the schema (name, description, args)
    # of the provided 'available_tools' to the LLM's configuration. Now, when this LLM instance
    # is invoked, it's aware of the 'retrieve_murli_context_tool' tool and can choose
    # to generate a tool_call for it in its AIMessage response if it deems it necessary.

    # --- Agent Node Runnable ---
    agent_node_runnable = functools.partial(agent_node, llm_with_tools=llm_with_tools)
    # We use partial to pre-fill the 'llm_with_tools' argument for our agent_node
    # function. This makes it easier to register with the graph, as the graph runner
    # just needs to pass the 'state'.

    # --- Graph Definition ---
    builder = StateGraph(AgentState) # Initialize the graph builder with our state definition

    # --- Add Nodes ---
    builder.add_node("agent", agent_node_runnable) # Register agent_node under the name "agent"
    builder.add_node("action", tool_node)         # Register the ToolNode under the name "action"

    # --- Set Entry Point ---
    builder.set_entry_point("agent") # Execution always starts at the "agent" node

    # --- Add Conditional Edges ---
    builder.add_conditional_edges(
        "agent",         # Starting node name
        tools_condition, # Function to decide the next step
        {                # Mapping: {condition_output: destination_node_name}
            "tools": "action", # If tools_condition returns "action", go to "action" node
            "__end__": END            # If tools_condition returns END, finish the graph
        }
    )
    # How tools_condition works: It examines the *last message added* to the state
    # (which would be the AIMessage returned by the "agent" node). If that message
    # contains tool_calls, it returns "action". Otherwise, it returns END.

    # --- Add Regular Edge ---
    builder.add_edge("action", "agent")
    # After the "action" node (ToolNode) runs and adds its ToolMessage result to the
    # state, this edge unconditionally routes the execution flow *back* to the "agent" node.
    # This allows the agent to process the tool's output.

    # --- Compile Graph ---
    graph = builder.compile() # Finalize the graph structure into an executable object
    logger.info("LangGraph agent compiled successfully...")
    return graph 


# --- Example Invocation (Conceptual) ---
if __name__ == '__main__':    
    try:
        persist_directory = '/home/bk_anupam/code/LLM_agents/RAG_BOT/chroma_db'
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb_instance = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    
        app = build_agent(vectordb_instance)
    
        # Example run
        user_question = "From the murli of 18th Jan 1969 answer the following question: \n" \
                        "Question: Together with love, also become an embodiment of what so that there will be success. \n" \
                        
        initial_state = {"messages": [HumanMessage(content=user_question)]}
        final_state = app.invoke(initial_state)

        for m in final_state['messages']:
            m.pretty_print()
    
        # The final answer is typically the last AIMessage
        final_answer = final_state['messages'][-1].content
        print("Final Answer:", final_answer)                    

    except Exception as e:
        print(f"Error during example run: {e}")
    # pass # Keep pass if not running example directly
