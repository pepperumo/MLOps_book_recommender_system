"""
Script to run the Book Recommender API locally
"""
import os
import sys
import uvicorn

# Ensure we can import from parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(src_dir)

sys.path.insert(0, root_dir)  # Add project root to path
sys.path.insert(0, src_dir)   # Add src directory to path

# Import the BookRecommender class to make it available when unpickling
from src.models.train_model import BookRecommender

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the Book Recommender API")
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    args = parser.parse_args()
    
    print("Starting Book Recommender API server...")
    print("API documentation will be available at http://localhost:8000/docs")
    
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(root_dir, "logs")
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
        print(f"Created logs directory: {logs_dir}")
    
    # Create data/results directory if it doesn't exist
    results_dir = os.path.join(root_dir, "data", "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created results directory: {results_dir}")
    
    # Run the API with app reload for development
    if args.test:
        print("Running in TEST mode without reload enabled on port 8001")
        uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=False)
    else:
        uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
