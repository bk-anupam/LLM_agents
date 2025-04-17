from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Demo MCP Server", "0.1.0")

# Add an addition tool
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


# Add a dynamic greeting resource
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"

# Run the MCP Server
if __name__ == "__main__":
    print("Starting MCP Server...")    
    mcp.run(transport='stdio')    
    print("MCP Server is running.")