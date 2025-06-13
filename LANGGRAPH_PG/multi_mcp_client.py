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
            # "murli": {
            #     "url": "http://localhost:8000/mcp",
            #     "transport": "streamable_http",
            # },
            "math": {
                "command": "python",
                "args": [
                    "/home/bk_anupam/code/LLM_agents/LANGGRAPH_PG/demo_mcp_server.py"
                ],
                "transport": "stdio",
            },
            "tavily-mcp": {
                "command": "bash",
                "args": [
                    "-c",
                    (
                        "export TAVILY_API_KEY='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' && "
                        "source /home/bk_anupam/.nvm/nvm.sh > /dev/null 2>&1 && "
                        "nvm use v22.14.0 > /dev/null 2>&1 && "
                        "npx --quiet -y tavily-mcp@0.2.1"
                    )
                ],
                "transport": "stdio",
            },
            "filesystem": {
                "command": "bash",
                "args": [
                    "-c",
                    (
                        "source /home/bk_anupam/.nvm/nvm.sh > /dev/null 2>&1 && "
                        "nvm use v22.14.0 > /dev/null 2>&1 && "
                        "npx --quiet -y @modelcontextprotocol/server-filesystem "
                        "/home/bk_anupam/code"
                    )
                ],
                "transport": "stdio",
            },
            "playwright": {
                "command": "bash",
                "args": [
                    "-c",
                    (
                        "cd /home/bk_anupam && "
                        "source /home/bk_anupam/.nvm/nvm.sh > /dev/null 2>&1 && "
                        "nvm use v22.14.0 > /dev/null 2>&1 && "
                        "export PLAYWRIGHT_BROWSERS_PATH=0 && "
                        "export BROWSER=chromium && "
                        "npx --quiet -y @playwright/mcp@latest "
                        "--executable-path "
                        "'/home/bk_anupam/.cache/ms-playwright/chromium-1169/chrome-linux/chrome'"
                    )
                ],
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
    #murli_response = await agent.ainvoke({"messages": "Summarize the murli of 1969-01-18"})
    tavily_response = await agent.ainvoke(
        {
            "messages": (
                "Use tavily extract to extract the information from "
                "'https://www.babamurli.com/01.%20Daily%20Murli/02.%20English/01.%20Eng%20Murli%20-%20Htm/01.06.25-E.htm' and present to me "
                "the summary of the murli contents"
            )
        }
    )
    for m in math_response['messages']:
        m.pretty_print()
    # for m in murli_response['messages']:
    #     m.pretty_print()
    for m in tavily_response['messages']:
        m.pretty_print()
asyncio.run(main())