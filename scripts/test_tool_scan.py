import asyncio
import logging
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Added logging configuration
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__) # Optional: for script-specific logs
mcp_client_logger = logging.getLogger("mcp.client") # Get mcp.client logger
mcp_client_logger.setLevel(logging.DEBUG) # Set mcp.client logger to DEBUG

async def main():
    logger.debug("About to start stdio_client connection...")
    # Use sys.executable to ensure the correct Python interpreter from the venv is used
    server_command = sys.executable 
    server_args = ["mcp_stdio_server.py"]
    logger.debug(f"Calculated server command: {server_command}")
    server_params = StdioServerParameters(command=server_command, args=server_args)
    logger.debug(f"Server params: {server_params}")
    try:
        async with stdio_client(server_params) as (read, write):
            logger.debug("Successfully started server process for stdio_client. Initializing session...")
            client = ClientSession(read, write)
            logger.debug("ClientSession object created. Attempting to initialize...")
            try:
                await client.initialize()
                logger.debug("Client session initialized successfully.")
            except Exception as e:
                logger.error(f"Error during client.initialize(): {e}", exc_info=True) # Log with traceback
                return

            logger.debug("Attempting to list tools...")
            try:
                # Since the server has no tools now, this should return an empty list or similar
                tools = await client.list_tools()
                logger.debug("List tools call completed.") # New log
                logger.debug("Available tools:")
                if tools:
                    for tool in tools:
                        logger.debug(f"- {tool['name']}: {tool.get('description', 'No description')}")
                else:
                    logger.debug("No tools found (as expected with bare server).") # Modified log
            except Exception as e:
                logger.error(f"Error during client.list_tools(): {e}", exc_info=True) # Log with traceback
            finally:
                logger.debug("Attempting to close client session...")
                await client.close()
                logger.debug("Client session closed.")
    except Exception as e:
        logger.error(f"Error with stdio_client or top-level operation: {e}", exc_info=True) # Log with traceback

if __name__ == "__main__":
    asyncio.run(main()) 