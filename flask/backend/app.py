"""
Flask application for book recommendation system API.
This API follows the same structure and functionality as the FastAPI implementation.
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, Blueprint, current_app
from flask_cors import CORS
import traceback
from flask_jwt_extended import JWTManager

# Configure logging
log_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), 'logs')
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(log_dir, f'flask_api_{timestamp}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('flask_book_recommender')

# Set project paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))
sys.path.append(os.path.join(project_root, "src", "models"))

# Import model-related functions
try:
    from src.models.predict_model import (
        recommend_for_user, 
        recommend_similar_books, 
        get_popular_books,
        get_book_metadata,
        load_recommender_model
    )
    # Import the collaborative model class to ensure it's in the global namespace
    # This helps with unpickling the model
    from src.models.train_model import CollaborativeRecommender
    logger.info("Successfully imported recommender model functions")
except ImportError as e:
    logger.error(f"Error importing from src.models.predict_model: {e}")
    # Fallback imports if needed
    try:
        from models.predict_model import (
            recommend_for_user, 
            recommend_similar_books, 
            get_popular_books,
            get_book_metadata,
            load_recommender_model
        )
        # Import the collaborative model class for fallback
        from models.train_model import CollaborativeRecommender
        logger.info("Successfully imported recommender model functions from fallback location")
    except ImportError as e:
        logger.error(f"Error importing from fallback location: {e}")

# Create Flask app
def create_app(config_name='default'):
    app = Flask(__name__)
    
    # Configure CORS to allow requests from any origin with proper settings
    CORS(app, 
         resources={r"/*": {"origins": "*"}}, 
         supports_credentials=True,
         allow_headers=["Content-Type", "Accept", "Authorization"],
         methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    
    # Initialize JWT
    app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'super-secret-key-change-in-production')
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = 86400  # 1 day
    jwt = JWTManager(app)
    
    # Set data directory paths for the app
    app.config['DATA_DIR'] = os.path.abspath(os.path.join(project_root, "data"))
    app.config['MODELS_DIR'] = os.path.abspath(os.path.join(project_root, "models"))
    
    # Make these available to the predict_model functions
    os.environ['BOOK_RECOMMENDER_DATA_DIR'] = app.config['DATA_DIR']
    os.environ['BOOK_RECOMMENDER_MODELS_DIR'] = app.config['MODELS_DIR']
    
    logger.info(f"Set DATA_DIR to {app.config['DATA_DIR']}")
    logger.info(f"Set MODELS_DIR to {app.config['MODELS_DIR']}")
    
    # Load data at startup
    with app.app_context():
        try:
            # Load merged dataset instead of separate files
            merged_paths = [
                os.path.join(app.config['DATA_DIR'], 'processed', 'merged.csv'),
                os.path.join(app.config['DATA_DIR'], 'processed', 'merged_train.csv'),
                os.path.join(project_root, 'data', 'processed', 'merged.csv'),
                os.path.join(project_root, 'data', 'processed', 'merged_train.csv')
            ]
            
            merged_found = False
            for merged_path in merged_paths:
                if os.path.exists(merged_path):
                    logger.info(f"Loading merged dataset from {merged_path}")
                    merged_df = pd.read_csv(merged_path)
                    app.config['MERGED_DF'] = merged_df
                    
                    # Create book-specific view for compatibility with existing code
                    # First, create a temporary DataFrame with just the unique book entries
                    unique_books = merged_df.drop_duplicates(subset=['book_id'])[
                        ['book_id', 'title', 'authors', 'average_rating', 'ratings_count', 
                         'image_url', 'publisher', 'published_year', 'original_publication_year',
                         'isbn', 'isbn13', 'description', 'genres', 'language_code', 'num_pages']
                    ]
                    
                    # Set this as the books dataframe without any aggregation
                    app.config['BOOKS_DF'] = unique_books
                    
                    # Extract ratings data from merged dataset for compatibility
                    app.config['RATINGS_DF'] = merged_df[['user_id', 'book_id', 'rating']]
                    
                    merged_found = True
                    logger.info(f"Successfully loaded merged dataset with {len(merged_df)} rows")
                    logger.info(f"Extracted {len(unique_books)} unique books from merged dataset")
                    break
            
            if not merged_found:
                logger.error("Merged dataset not found in any of the expected locations")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            logger.debug(traceback.format_exc())
    
    # Register blueprints
    from routes.api_routes import api_bp
    from routes.main_routes import main_bp
    
    app.register_blueprint(api_bp, url_prefix='/api')
    app.register_blueprint(main_bp, url_prefix='/')
    
    # Load model at startup
    with app.app_context():
        try:
            # Load collaborative filtering model
            collaborative_model = load_recommender_model(
                'collaborative', 
                models_dir=os.path.join(project_root, "models")
            )
            
            if collaborative_model:
                logger.info("Collaborative model loaded successfully")
                app.config['collaborative_model'] = collaborative_model
            else:
                logger.error("Failed to load collaborative model")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            logger.debug(traceback.format_exc())
    
    # Root endpoint (identical to FastAPI)
    @app.route('/')
    def root():
        return jsonify({
            "app_name": "Book Recommender API",
            "version": "1.0.0",
            "endpoints": [
                {"path": "/", "description": "This root endpoint"},
                {"path": "/health", "description": "Health check endpoint"},
                {"path": "/api/docs", "description": "API documentation"},
                {"path": "/api/recommend/user/{user_id}", "description": "Get book recommendations for a user"},
                {"path": "/api/similar-books/{book_id}", "description": "Get similar books to a given book"},
                {"path": "/api/books", "description": "Get a list of books with their IDs, titles, and authors"},
                {"path": "/api/ratings", "description": "Get ratings data with various filters"},
                {"path": "/api/users/{user_id}/ratings", "description": "Get all ratings for a specific user"},
                {"path": "/api/books/{book_id}/ratings", "description": "Get all ratings for a specific book"}
            ]
        })
    
    # Health check endpoint (identical to FastAPI)
    @app.route('/health')
    def health_check():
        return jsonify({
            "status": "ok", 
            "service": "book-recommender-api",
            "version": current_app.config.get('VERSION', None),
            "timestamp": datetime.now().isoformat()
        })
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            "status": "error",
            "message": "The requested resource was not found",
            "detail": str(error)
        }), 404
    
    @app.errorhandler(500)
    def server_error(error):
        return jsonify({
            "status": "error",
            "message": "An internal server error occurred",
            "detail": str(error)
        }), 500
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)
