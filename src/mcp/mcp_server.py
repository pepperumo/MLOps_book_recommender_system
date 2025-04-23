#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Book Recommendation MCP Server using Model Context Protocol."""

import os
import sys
import traceback
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import time
import json

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Read configuration from environment variables
HOST = os.getenv("MCP_HOST", "0.0.0.0")
PORT = int(os.getenv("MCP_PORT", "8080"))
DATA_DIR = os.getenv("MCP_DATA_DIR", None)
MODELS_DIR = os.getenv("MCP_MODELS_DIR", None)

# Set up project path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

# Use custom DATA_DIR if provided
if DATA_DIR:
    data_dir = DATA_DIR
else:
    data_dir = os.path.join(project_root, 'data')

# Use custom MODELS_DIR if provided
if MODELS_DIR:
    models_dir = MODELS_DIR
else:
    models_dir = os.path.join(project_root, 'models')

# Import shared metrics
try:
    from src.metrics import (
        RECOMMENDATION_COUNTER,
        RECOMMENDATION_TIME,
        MODEL_LOAD_TIME
    )
except ImportError:
    # Create dummy metrics if not available
    from prometheus_client import Counter, Histogram, Gauge
    
    RECOMMENDATION_COUNTER = Counter(
        'recommendations_total', 
        'Number of recommendations generated',
        ['endpoint', 'status']
    )
    
    RECOMMENDATION_TIME = Histogram(
        'recommendation_generation_seconds',
        'Time to generate recommendations',
        ['model_type', 'endpoint']
    )
    
    MODEL_LOAD_TIME = Gauge(
        'model_load_time_seconds',
        'Time taken to load model',
        ['model_type']
    )

# Import the necessary modules
try:
    from src.models.predict_model import (
        recommend_for_user, 
        recommend_similar_books,
        get_popular_books,
        get_book_metadata,
        load_recommender_model
    )
except ImportError as e:
    try:
        from models.predict_model import (
            recommend_for_user, 
            recommend_similar_books,
            get_popular_books,
            get_book_metadata,
            load_recommender_model
        )
    except ImportError as e:
        # Add parent directory to path to ensure correct imports
        parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sys.path.append(parent_dir)
        
        # Try importing from both namespaces
        try:
            from src.models.predict_model import (
                recommend_for_user, 
                recommend_similar_books,
                get_popular_books,
                get_book_metadata,
                load_recommender_model
            )
        except ImportError:
            from models.predict_model import (
                recommend_for_user, 
                recommend_similar_books,
                get_popular_books,
                get_book_metadata,
                load_recommender_model
            )

# Set up logging
log_dir = os.path.join(project_root, 'logs')
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(log_dir, f'mcp_{timestamp}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('mcp_recommendation_server')

# Data cache for reuse
_BOOKS_DF = None
_RATINGS_DF = None
_POPULAR_BOOKS_CACHE = {}

# MCP Protocol Models
class MCPModelInfoResponse(BaseModel):
    """Model info response according to MCP protocol"""
    id: str
    name: str
    description: str
    version: str
    inputs: List[Dict[str, Any]]
    outputs: List[Dict[str, Any]]

class HealthResponse(BaseModel):
    """Health check response for the MCP server"""
    status: str

class UserRecommendationInput(BaseModel):
    """Input schema for user recommendation"""
    user_id: int
    num_recommendations: int = 5
    model_type: str = "collaborative"
    include_images: bool = True
    force_diverse: bool = True

class SimilarBooksInput(BaseModel):
    """Input schema for similar books recommendation"""
    book_id: int
    num_recommendations: int = 5
    model_type: str = "collaborative"
    include_images: bool = True

class BookRecommendation(BaseModel):
    """Single book recommendation"""
    book_id: int
    title: str
    authors: str
    rank: int
    image_url: Optional[str] = None
    average_rating: Optional[float] = None
    ratings_count: Optional[int] = None
    similarity_score: Optional[float] = None

class RecommendationResponse(BaseModel):
    """Response for recommendations"""
    recommendations: List[BookRecommendation]
    user_id: Optional[int] = None
    book_id: Optional[int] = None

# Helper functions
def get_books_df():
    """Get and cache the books dataframe from merged_train.csv only"""
    global _BOOKS_DF
    if (_BOOKS_DF is not None):
        return _BOOKS_DF
    
    # Use merged_train.csv directly without fallback
    books_path = os.path.join(data_dir, 'processed', 'merged_train.csv')
    
    if os.path.exists(books_path):
        logger.info(f"Loading books data from {books_path}")
        try:
            _BOOKS_DF = pd.read_csv(books_path)
            
            # If merged data has user_id column, get unique books only
            if 'user_id' in _BOOKS_DF.columns:
                _BOOKS_DF = _BOOKS_DF.drop_duplicates(subset=['book_id'])
                
            logger.info(f"Successfully loaded books dataframe with {len(_BOOKS_DF)} books")
            return _BOOKS_DF
        except Exception as e:
            logger.error(f"Error loading books data from {books_path}: {e}")
    else:
        logger.error(f"Books data not found at {books_path}")
    
    # Return empty DataFrame if loading failed
    return pd.DataFrame()

def get_ratings_df():
    """Get and cache the ratings dataframe from merged_train.csv only"""
    global _RATINGS_DF
    if _RATINGS_DF is not None:
        return _RATINGS_DF
    
    # Use merged_train.csv directly without fallback
    ratings_path = os.path.join(data_dir, 'processed', 'merged_train.csv')
    
    if os.path.exists(ratings_path):
        logger.info(f"Loading ratings data from {ratings_path}")
        try:
            _RATINGS_DF = pd.read_csv(ratings_path)
            logger.info(f"Successfully loaded ratings dataframe with {len(_RATINGS_DF)} ratings")
            return _RATINGS_DF
        except Exception as e:
            logger.error(f"Error loading ratings data from {ratings_path}: {e}")
    else:
        logger.error(f"Ratings data not found at {ratings_path}")
    
    # Return empty DataFrame if loading failed
    return pd.DataFrame()

# Create FastAPI app
app = FastAPI(
    title="Book Recommender MCP Server",
    description="MCP API for book recommendations using collaborative filtering",
    version="1.0.0"
)

# Configure CORS to allow requests from anywhere
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

collaborative_model = None

# Startup event to check models availability
@app.on_event("startup")
async def startup_event():
    """Load models and check if they're available"""
    logger.info("Checking model and data file availability...")
    global collaborative_model
    
    try:
        start_time = time.time()
        
        # Preload the data
        books_df = get_books_df()
        if not books_df.empty:
            logger.info(f"Successfully loaded books dataframe with {len(books_df)} books")
        
        ratings_df = get_ratings_df()
        if not ratings_df.empty:
            logger.info(f"Successfully loaded ratings dataframe with {len(ratings_df)} ratings")
        
        # Load collaborative filtering model
        model = load_recommender_model('collaborative', models_dir=models_dir)
        if model:
            collaborative_model = model
            logger.info("Collaborative model loaded successfully")
            
            # Record model load time using the shared metrics
            load_time = time.time() - start_time
            MODEL_LOAD_TIME.labels(model_type="collaborative").set(load_time)
            logger.info(f"Model load time: {load_time:.2f} seconds")
        else:
            logger.error("Failed to load collaborative model")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        logger.debug(traceback.format_exc())

# MCP Endpoint - Health Check
@app.get("/v1/health", response_model=HealthResponse)
async def health_check():
    return {"status": "healthy"}

# MCP Endpoint - Model Info
@app.get("/v1/models/book-recommender", response_model=MCPModelInfoResponse)
async def model_info():
    """MCP compliant model info endpoint"""
    return {
        "id": "book-recommender",
        "name": "Book Recommender System",
        "description": "Collaborative filtering model for recommending books to users",
        "version": "1.0.0",
        "inputs": [
            {
                "name": "user_recommendations",
                "description": "Get book recommendations for a specific user",
                "schema": UserRecommendationInput.schema()
            },
            {
                "name": "similar_books",
                "description": "Get similar books to a given book",
                "schema": SimilarBooksInput.schema()
            }
        ],
        "outputs": [
            {
                "name": "recommendations",
                "description": "List of book recommendations",
                "schema": RecommendationResponse.schema()
            }
        ]
    }

# MCP Endpoint - Generate User Recommendations
@app.post("/v1/models/book-recommender/user-recommendations")
async def user_recommendations(request: Request, response: Response):
    """MCP compliant endpoint for user recommendations"""
    start_time = time.time()
    
    try:
        # Parse the request body as JSON
        body = await request.json()
        
        if 'inputs' not in body:
            response.status_code = 400
            return {"error": "Missing 'inputs' field in request body"}
        
        inputs = body['inputs']
        
        # Extract input parameters
        user_id = inputs.get('user_id')
        if user_id is None:
            response.status_code = 400
            return {"error": "Missing required field 'user_id' in inputs"}
        
        try:
            user_id = int(user_id)
        except (ValueError, TypeError):
            response.status_code = 400
            return {"error": "Field 'user_id' must be an integer"}
        
        # Optional parameters with defaults
        num_recommendations = int(inputs.get('num_recommendations', 5))
        model_type = inputs.get('model_type', 'collaborative')
        include_images = inputs.get('include_images', True)
        force_diverse = inputs.get('force_diverse', True)
        
        # Check for valid user ID range
        if user_id < 1 or user_id > 10000:  # Adjust the upper limit as needed
            response.status_code = 404
            return {"error": f"User ID {user_id} not found"}
        
        logger.info(f"Generating recommendations for user {user_id} using collaborative model")
        
        # Add a random offset to the user_id to create more diversity between users 
        # when they are close in ID number (for popular books fallback)
        diverse_user_id = user_id
        if force_diverse:
            np.random.seed(user_id)
            # Create a unique but reproducible offset for each user
            offset = np.random.randint(1, 1000)
            diverse_user_id = user_id * offset
        
        recommendations_df = recommend_for_user(
            user_id=diverse_user_id,
            model_type='collaborative',  # Always use collaborative model
            n=num_recommendations,
            data_dir=data_dir
        )
        
        if recommendations_df.empty:
            logger.warning(f"No recommendations found for user {user_id}")
            # Record metrics before returning
            RECOMMENDATION_COUNTER.labels(endpoint="user-recommendations", status="empty").inc()
            RECOMMENDATION_TIME.labels(model_type=model_type, endpoint="user-recommendations").observe(time.time() - start_time)
            # Return empty recommendations
            return {
                "outputs": {
                    "recommendations": [],
                    "user_id": user_id
                }
            }
        
        recommendations = []
        books_df = get_books_df()
        
        for _, row in recommendations_df.iterrows():
            try:
                book_id = int(row['book_id'])
                
                # Get complete book data from books dataframe
                book_data = books_df[books_df['book_id'] == book_id]
                
                if not book_data.empty:
                    book_row = book_data.iloc[0]
                    
                    recommendation = {
                        "book_id": book_id,
                        "title": book_row.get('title', f"Book {book_id}"),
                        "authors": book_row.get('authors', 'Unknown Author'),
                        "rank": int(row.get('rank', 0)) + 1,
                        "average_rating": float(book_row.get('average_rating', 0.0)),
                        "ratings_count": int(book_row.get('ratings_count', 0))
                    }
                    
                    # Add image URL if requested and available
                    if include_images and 'image_url' in book_row and book_row['image_url']:
                        recommendation["image_url"] = book_row['image_url']
                    
                    recommendations.append(recommendation)
                else:
                    # If book not found in books dataframe, use the data from recommendations dataframe
                    recommendation = {
                        "book_id": book_id,
                        "title": row.get('title', f"Book {book_id}"),
                        "authors": row.get('authors', 'Unknown Author'),
                        "rank": int(row.get('rank', 0)) + 1
                    }
                    
                    # Add image URL if requested and available
                    if include_images and 'image_url' in row and row['image_url']:
                        recommendation["image_url"] = row['image_url']
                    
                    recommendations.append(recommendation)
            except Exception as e:
                logger.warning(f"Error processing recommendation row: {e}")
                continue
        
        # Record success metrics
        RECOMMENDATION_COUNTER.labels(endpoint="user-recommendations", status="success").inc()
        RECOMMENDATION_TIME.labels(model_type=model_type, endpoint="user-recommendations").observe(time.time() - start_time)
        
        # Return MCP formatted response
        return {
            "outputs": {
                "recommendations": recommendations,
                "user_id": user_id
            }
        }
    
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}", exc_info=True)
        RECOMMENDATION_COUNTER.labels(endpoint="user-recommendations", status="error").inc()
        RECOMMENDATION_TIME.labels(model_type='collaborative', endpoint="user-recommendations").observe(time.time() - start_time)
        response.status_code = 500
        return {"error": f"Internal server error: {str(e)}"}

# MCP Endpoint - Generate Similar Books
@app.post("/v1/models/book-recommender/similar-books")
async def similar_books(request: Request, response: Response):
    """MCP compliant endpoint for similar books recommendations"""
    start_time = time.time()
    
    try:
        # Parse the request body as JSON
        body = await request.json()
        
        if 'inputs' not in body:
            response.status_code = 400
            return {"error": "Missing 'inputs' field in request body"}
        
        inputs = body['inputs']
        
        # Extract input parameters
        book_id = inputs.get('book_id')
        if book_id is None:
            response.status_code = 400
            return {"error": "Missing required field 'book_id' in inputs"}
        
        try:
            book_id = int(book_id)
        except (ValueError, TypeError):
            response.status_code = 400
            return {"error": "Field 'book_id' must be an integer"}
        
        # Optional parameters with defaults
        num_recommendations = int(inputs.get('num_recommendations', 5))
        model_type = inputs.get('model_type', 'collaborative')
        include_images = inputs.get('include_images', True)
        
        # Check for valid book ID range
        if book_id < 1 or book_id > 1000000:  # Adjust upper limit as needed
            response.status_code = 404
            return {"error": f"Book ID {book_id} not found"}
        
        # Check if the book exists in our dataset
        books_df = get_books_df()
        if not books_df.empty and book_id not in books_df['book_id'].values:
            response.status_code = 404
            return {"error": f"Book ID {book_id} not found"}
            
        logger.info(f"Finding similar books to book {book_id} using collaborative model")
        
        # Check if we need to map the book_id to an internal ID
        mapping_path = os.path.join(data_dir, 'processed', 'book_id_mapping.csv')
        if os.path.exists(mapping_path):
            try:
                # Load the mapping file
                mapping_df = pd.read_csv(mapping_path)
                logger.info(f"Loaded book ID mapping with {len(mapping_df)} entries")
                
                # Check if the book_id needs mapping
                if 'original_id' in mapping_df.columns and 'mapped_id' in mapping_df.columns:
                    # Look for the mapping
                    mapping_row = mapping_df[mapping_df['original_id'] == book_id]
                    if not mapping_row.empty:
                        mapped_id = mapping_row.iloc[0]['mapped_id']
                        logger.info(f"Mapped book ID {book_id} to internal ID {mapped_id}")
                        book_id = mapped_id  # Use the mapped ID
            except Exception as mapping_error:
                logger.warning(f"Error using book ID mapping: {mapping_error}")
        
        # Now get similar books using potentially mapped ID
        similar_books_df = recommend_similar_books(
            book_id=book_id,
            model_type='collaborative',  # Always use collaborative model
            n=num_recommendations,
            data_dir=data_dir
        )
        
        # If we have mappings, map the result IDs back to original IDs
        if 'mapping_df' in locals() and not similar_books_df.empty and 'mapped_id' in mapping_df.columns:
            try:
                # Map the book_ids in results back to original IDs
                for i, row in similar_books_df.iterrows():
                    result_id = row['book_id']
                    # Find the original ID
                    reverse_mapping = mapping_df[mapping_df['mapped_id'] == result_id]
                    if not reverse_mapping.empty:
                        original_id = reverse_mapping.iloc[0]['original_id']
                        similar_books_df.at[i, 'book_id'] = original_id
                        logger.info(f"Mapped result ID {result_id} back to original ID {original_id}")
            except Exception as reverse_mapping_error:
                logger.warning(f"Error in reverse ID mapping: {reverse_mapping_error}")
        
        if similar_books_df.empty:
            logger.warning(f"No similar books found for book {book_id}")
            # Record metrics and return empty response
            RECOMMENDATION_COUNTER.labels(endpoint="similar-books", status="empty").inc()
            RECOMMENDATION_TIME.labels(model_type=model_type, endpoint="similar-books").observe(time.time() - start_time)
            return {
                "outputs": {
                    "recommendations": [],
                    "book_id": book_id
                }
            }
        
        recommendations = []
        books_df = get_books_df()
        
        for _, row in similar_books_df.iterrows():
            try:
                similar_book_id = int(row['book_id'])
                
                # Get complete book data from books dataframe
                book_data = books_df[books_df['book_id'] == similar_book_id]
                
                if not book_data.empty:
                    book_row = book_data.iloc[0]
                    
                    recommendation = {
                        "book_id": similar_book_id,
                        "title": book_row.get('title', f"Book {similar_book_id}"),
                        "authors": book_row.get('authors', 'Unknown Author'),
                        "rank": int(row.get('rank', 0)) + 1,
                        "similarity_score": float(row.get('similarity', 0.0)),
                        "average_rating": float(book_row.get('average_rating', 0.0)),
                        "ratings_count": int(book_row.get('ratings_count', 0))
                    }
                    
                    # Add image URL if requested and available
                    if include_images and 'image_url' in book_row and book_row['image_url']:
                        recommendation["image_url"] = book_row['image_url']
                    
                    recommendations.append(recommendation)
                else:
                    # If book not found in books dataframe, use the data from recommendations dataframe
                    recommendation = {
                        "book_id": similar_book_id,
                        "title": row.get('title', f"Book {similar_book_id}"),
                        "authors": row.get('authors', 'Unknown Author'),
                        "rank": int(row.get('rank', 0)) + 1,
                        "similarity_score": float(row.get('similarity', 0.0))
                    }
                    
                    # Add image URL if requested and available
                    if include_images and 'image_url' in row and row['image_url']:
                        recommendation["image_url"] = row['image_url']
                    
                    recommendations.append(recommendation)
            except Exception as e:
                logger.warning(f"Error processing recommendation row: {e}")
                continue
        
        # Record metrics
        RECOMMENDATION_COUNTER.labels(endpoint="similar-books", status="success").inc()
        RECOMMENDATION_TIME.labels(model_type=model_type, endpoint="similar-books").observe(time.time() - start_time)
        
        # Return MCP formatted response
        return {
            "outputs": {
                "recommendations": recommendations,
                "book_id": book_id
            }
        }
    
    except Exception as e:
        logger.error(f"Error finding similar books: {e}", exc_info=True)
        # Record error metrics
        RECOMMENDATION_COUNTER.labels(endpoint="similar-books", status="error").inc()
        RECOMMENDATION_TIME.labels(model_type='collaborative', endpoint="similar-books").observe(time.time() - start_time)
        response.status_code = 500
        return {"error": f"Internal server error: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    print(f"Starting MCP server on {HOST}:{PORT}")
    print(f"Using data directory: {data_dir}")
    print(f"Using models directory: {models_dir}")
    uvicorn.run(app, host=HOST, port=PORT)