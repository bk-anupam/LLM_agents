# Create server parameters for stdio connection
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
import os
from dotenv import load_dotenv, find_dotenv

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)  # This loads the variables from .env

# StdioServerParameters from mcp is used to define how to connect to an MCP server that communicates over standard 
# input/output (stdio).
# command="python": Specifies that the server is a Python script.
# args=["/path/to/demo_mcp_server.py"]: This is a crucial placeholder. It tells the client to execute the Python interpreter 
# with the demo_mcp_server.py script. 
server_params = StdioServerParameters(
    command="python",
    # Make sure to update to the full absolute path to your math_server.py file
    args=["/home/bk_anupam/code/LLM_agents/LANGGRAPH_PG/demo_mcp_server.py"],
)

async def main():
    # This async context manager establishes a connection to the MCP server using the stdio_client. It yields read and write 
    # streams for communication. The stdio_client will start the demo_mcp_server.py process.
    async with stdio_client(server_params) as (read, write):
        #  Inside the stdio_client context, an MCP ClientSession is created using the read/write streams. 
        # This session object will be used for all further interactions with the server.
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # load_mcp_tools queries the connected MCP server (our demo_mcp_server.py) for its available tools. 
            # The load_mcp_tools function then converts these MCP-native tools into BaseTool objects that Langchain 
            # can understand and use.
            tools = await load_mcp_tools(session)

            # Create and run the agent
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)
            agent = create_react_agent(llm, tools)
            messages = await agent.ainvoke({"messages": "what's 3 + 5 ?"})
            for m in messages['messages']:
                m.pretty_print()

asyncio.run(main())