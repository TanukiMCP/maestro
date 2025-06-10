import uvicorn
import logging
import os
import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# It's good practice to have a single, clear entrypoint.
# This script is now simplified to focus on one thing: running the server.
# Configuration is loaded from environment variables inside the app factory,
# following the 12-factor app methodology.

def main():
    """
    Run the MAESTRO MCP server using Uvicorn.
    
    This entrypoint handles both local development and Smithery deployment
    by properly reading the PORT environment variable.
    """
    logger = logging.getLogger(__name__)

    # Get port from environment (Smithery sets this)
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("MAESTRO_HOST", "0.0.0.0")
    
    logger.info(f"🎭 Starting MAESTRO MCP server on {host}:{port}")
    
    # The application path 'src.main:app' tells Uvicorn where to find the app.
    # Uvicorn will import `src/main.py` and look for the `app` variable.
    # The app factory pattern ensures that `create_app()` is called at that time.
    
    # We don't need to handle reload or workers here, as that's better
    # managed by deployment configurations (e.g., Docker Compose, systemd)
    # or command-line arguments to uvicorn directly.
    # For development, `uvicorn src.main:app --reload` is recommended.

    uvicorn.run(
        "src.app_factory:create_app",
        host=host,
        port=port,
        factory=True, # Treat the import string as a factory function to call
        log_level="info"
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main() 