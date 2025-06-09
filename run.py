import sys
from pathlib import Path
import uvicorn
import logging

# Add src directory to Python path
# This allows us to import modules from the 'src' directory as if they were top-level packages
src_path = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(src_path))

logger = logging.getLogger(__name__)

def main():
    """
    Run the MAESTRO MCP server.
    This script serves as the entrypoint for the application, ensuring that the
    'src' directory is added to the Python path before attempting to import
    and run the server.
    """
    try:
        # We can now import from 'main' which refers to 'src/main.py'
        # This gives us access to the config object for logging.
        from main import config
        
        is_dev_mode = config.engine.mode.value == "development"
        workers = 1 if is_dev_mode else config.server.workers

        logger.info("🎭 Starting MAESTRO MCP server...")
        logger.info(f"Mode: {config.engine.mode.value}")
        logger.info(f"Host: {config.server.host}")
        logger.info(f"Port: {config.server.port}")
        logger.info(f"Workers: {workers}")
        logger.info(f"Reloading: {is_dev_mode}")

        uvicorn.run(
            "main:app",
            host=config.server.host,
            port=config.server.port,
            workers=workers,
            reload=is_dev_mode,
        )
    except ImportError as e:
        logger.error(f"❌ Failed to import server application. Please ensure all dependencies are installed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Server startup failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main() 