"""
Entry point script for Docker container to ensure BookRecommender class is loaded before unpickling
"""
import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("docker_entrypoint")

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, src_dir)

logger.info(f"Python path: {sys.path}")
logger.info(f"Current directory: {os.getcwd()}")
logger.info(f"Python version: {sys.version}")

try:
    # Import BookRecommender class to make it available for unpickling
    logger.info("Importing BookRecommender class...")
    from src.models.train_model import BookRecommender
    logger.info("BookRecommender class imported successfully")

    # Try importing numpy and scipy to verify they're working
    import numpy as np
    logger.info(f"Numpy version: {np.__version__}")
    import scipy as sp
    logger.info(f"Scipy version: {sp.__version__}")
    import pandas as pd
    logger.info(f"Pandas version: {pd.__version__}")
except Exception as e:
    logger.error(f"Error during imports: {str(e)}")
    raise

# Start the API server
if __name__ == "__main__":
    try:
        logger.info("Starting API server with --test flag to disable reload mode")
        import uvicorn
        uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=False)
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        raise
