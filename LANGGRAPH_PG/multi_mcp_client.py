import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv, find_dotenv

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)  # This loads the variables from .env

async def main():
    # Initialize the MultiServerMCPClient with the server URL
    client = MultiServerMCPClient(
        {
            "murli": {
                "url": "http://localhost:8000/mcp",
                "transport": "streamable_http",
            },
            "math": {
                "command": "python",
                "args": ["/home/bk_anupam/code/LLM_agents/LANGGRAPH_PG/demo_mcp_server.py"],
                "transport": "stdio",
            }
        }
    )

    tools = await client.get_tools()
    # Create and run the agent
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)
    agent = create_react_agent(llm, tools)    
    # Invoke the agent with a sample message
    math_response = await agent.ainvoke({"messages": "what's 3 + 5 ?"})
    murli_response = await agent.ainvoke({"messages": "Summarize the murli of 1969-01-18"})      
    for m in math_response['messages']:
        m.pretty_print()
    for m in murli_response['messages']:
        m.pretty_print()

asyncio.run(main())