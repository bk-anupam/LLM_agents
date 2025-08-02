from typing import Any, TypedDict, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AnyMessage
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langmem.short_term import SummarizationNode
import os
from dotenv import load_dotenv, find_dotenv

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)  # This loads the variables from .env

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
# For Google GenAI, max tokens should be set via generation_config
summarization_model = model.bind(generation_config={"max_output_tokens": 128})

class State(MessagesState):
    # The context is used by the SummarizationNode to store its running summary.
    # It's not present in the initial input, so we can mark it as Optional.
    context: Optional[dict[str, Any]]

class LLMInputState(TypedDict):
    summarized_messages: list[AnyMessage]
    context: Optional[dict[str, Any]]

summarization_node = SummarizationNode(
    model=summarization_model,
    # This is the budget for the summarizer. It should be large enough to include
    # all messages up to the point where summarization is triggered.
    max_tokens=4096,
    max_tokens_before_summary=256,    
)

def call_model(state: LLMInputState):
    response = model.invoke(state["summarized_messages"])
    return {"messages": [response]}

def print_result(result: dict[str, Any]):
    print("\n--- Result Messages ---")    
    for m in result['messages']:
        m.pretty_print()

checkpointer = InMemorySaver()
workflow = StateGraph(State)
workflow.add_node(call_model)
workflow.add_node("summarize", summarization_node)
workflow.add_edge(START, "summarize")
workflow.add_edge("summarize", "call_model")
graph = workflow.compile(checkpointer=checkpointer)

def run_graph():
    config = {"configurable": {"thread_id": "1"}}
    result1 = graph.invoke({"messages": "hi, my name is bob"}, config)
    print_result(result1)
    result2 = graph.invoke({"messages": "write a short poem about cats"}, config)
    print_result(result2)
    result3 = graph.invoke({"messages": "now do the same but for dogs"}, config)
    print_result(result3)
    result4 = graph.invoke({"messages": "now do the same but for a cow"}, config)
    print_result(result4)
    result5 = graph.invoke({"messages": "what's my name?"}, config)
    print_result(result5)

if __name__ == "__main__":
    run_graph()    