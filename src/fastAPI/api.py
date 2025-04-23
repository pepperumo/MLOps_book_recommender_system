#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Book Recommendation API using FastAPI with Collaborative Filtering."""

import os
import sys
import traceback
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Depends, BackgroundTasks, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import logging
import pandas as pd
import numpy as np
import time

# Add near your other imports
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from prometheus_client import Counter, Histogram, Gauge, push_to_gateway

# Set up project path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, os.path.join(project_root, "src", "models"))

# Import shared metrics instead of defining them here
from src.metrics import (
    RECOMMENDATION_COUNTER,
    RECOMMENDATION_TIME,
    MODEL_LOAD_TIME
)

# Import the necessary modules
from src.models.model_utils import BaseRecommender, load_data
from src.models.train_model import CollaborativeRecommender

# Import recommender modules
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
        logging.info(f"Adjusted Python path to {parent_dir}")

# Set up logging
log_dir = os.path.join(project_root, 'logs')
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(log_dir, f'api_{timestamp}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('recommendation_api')

# Data cache for reuse
_BOOKS_DF = None
_RATINGS_DF = None
_POPULAR_BOOKS_CACHE = {}

# Helper function to download files from DagsHub
def download_from_dagshub(file_path, dagshub_url):
    """
    Download a file from DagsHub if it doesn't exist locally.
    
    Parameters
    ----------
    file_path : str
        Local path where the file should be saved
    dagshub_url : str
        URL to download the file from
        
    Returns
    -------
    bool
        True if download successful or file already exists, False otherwise
    """
    try:
        if os.path.exists(file_path):
            logger.info(f"File already exists at {file_path}")
            return True
            
        logger.info(f"File not found at {file_path}, downloading from {dagshub_url}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        import requests
        response = requests.get(dagshub_url)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            logger.info(f"Successfully downloaded file to {file_path}")
            return True
        else:
            logger.error(f"Failed to download from DagsHub. Status code: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        logger.debug(traceback.format_exc())
        return False

# Define response models
class BookRecommendation(BaseModel):
    book_id: int
    title: str
    authors: str
    rank: int
    image_url: Optional[str] = None
    average_rating: Optional[float] = None
    ratings_count: Optional[int] = None
    similarity_score: Optional[float] = None

class RecommendationResponse(BaseModel):
    recommendations: List[BookRecommendation]
    user_id: Optional[int] = None
    book_id: Optional[int] = None

class BookDetail(BaseModel):
    book_id: int
    title: str
    authors: str
    average_rating: float = 0.0
    ratings_count: int = 0
    image_url: Optional[str] = None
    description: Optional[str] = None
    genres: Optional[str] = None

class UserDetail(BaseModel):
    user_id: int
    total_ratings: int = 0
    avg_rating: Any = "N/A"
    favorite_genres: List[str] = []
    recent_books: List[Dict[str, Any]] = []

# Create FastAPI app
app = FastAPI(
    title="Book Recommender API",
    description="API for book recommendations using collaborative filtering",
    version="1.0.0"
)

# Setup Prometheus instrumentation with additional metrics
instrumentator = Instrumentator()
instrumentator.add(metrics.latency())
instrumentator.add(metrics.requests())
instrumentator.instrument(app).expose(app)

# Configure CORS to allow requests from anywhere
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Helper functions
def get_books_df():
    """Get and cache the books dataframe from merged_train.csv only"""
    global _BOOKS_DF
    if (_BOOKS_DF is not None):
        return _BOOKS_DF
    
    # Use merged_train.csv directly without fallback
    books_path = os.path.join(project_root, 'data', 'processed', 'merged_train.csv')
    
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
    ratings_path = os.path.join(project_root, 'data', 'processed', 'merged_train.csv')
    
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

# Startup event to check models availability
@app.on_event("startup")
async def startup_event():
    """Load models and check if they're available"""
    logger.info("Checking model and data file availability...")
    global collaborative_model
    
    try:
        start_time = time.time()
        
        # Define the essential files and their DagsHub URLs
        essential_files = {
            os.path.join(project_root, 'data', 'processed', 'book_id_mapping.csv'): 
                "https://dagshub.com/pepperumo/MLOps_book_recommender_system/raw/master/data/processed/book_id_mapping.csv",
            os.path.join(project_root, 'data', 'processed', 'merged_train.csv'):
                "https://dagshub.com/pepperumo/MLOps_book_recommender_system/raw/master/data/processed/merged_train.csv",
            os.path.join(project_root, 'models', 'collaborative.pkl'):
                "https://dagshub.com/pepperumo/MLOps_book_recommender_system/raw/master/models/collaborative.pkl"
        }
        
        # Check and download each essential file if needed
        for file_path, dagshub_url in essential_files.items():
            if not os.path.exists(file_path):
                logger.warning(f"Required file not found: {file_path}")
                if download_from_dagshub(file_path, dagshub_url):
                    logger.info(f"Successfully downloaded {os.path.basename(file_path)}")
                else:
                    logger.error(f"Failed to download {os.path.basename(file_path)} from DagsHub")
        
        # Preload the data
        books_df = get_books_df()
        if not books_df.empty:
            logger.info(f"Successfully loaded books dataframe with {len(books_df)} books")
        
        ratings_df = get_ratings_df()
        if not ratings_df.empty:
            logger.info(f"Successfully loaded ratings dataframe with {len(ratings_df)} ratings")
        
        # Load collaborative filtering model
        model = load_recommender_model('collaborative', models_dir=os.path.join(project_root, "models"))
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

# Root endpoint
@app.get("/")
async def root():
    return {
        "app_name": "Book Recommender API",
        "version": "1.0.0",
        "endpoints": [
            {"path": "/", "description": "This root endpoint"},
            {"path": "/health", "description": "Health check endpoint"},
            {"path": "/docs", "description": "API documentation"},
            {"path": "/recommend/user/{user_id}", "description": "Get book recommendations for a user"},
            {"path": "/similar-books/{book_id}", "description": "Get similar books to a given book"},
            {"path": "/books", "description": "Get a list of books with their IDs, titles, and authors"},
            {"path": "/genres", "description": "Get a list of all genres"},
            {"path": "/authors", "description": "Get a list of all authors"},
            {"path": "/users", "description": "Get a list of users"},
            {"path": "/users/{user_id}", "description": "Get details for a specific user"},
            {"path": "/users/{user_id}/ratings", "description": "Get ratings for a specific user"},
            {"path": "/books/{book_id}/ratings", "description": "Get ratings for a specific book"},
            {"path": "/ratings", "description": "Get ratings data"},
            {"path": "/popular-books", "description": "Get a list of popular books"}
        ]
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Create a router with /api prefix
api_router = APIRouter(prefix="/api")

# Move your endpoint definitions to use this router (keep the rest of the function the same)
@api_router.get("/recommend/user/{user_id}", response_model=RecommendationResponse)
async def get_user_recommendations(
    user_id: int, 
    model_type: str = Query("collaborative", enum=["collaborative"]),
    num_recommendations: int = Query(5, ge=1, le=20),
    n: Optional[int] = Query(None, ge=1, le=20),
    include_images: bool = Query(True, description="Include book image URLs in the response"),
    force_diverse: bool = Query(True, description="Force diversity in recommendations")
):
    """Get book recommendations for a user"""
    start_time = time.time()
    try:
        # Check for valid user ID range
        if user_id < 1 or user_id > 10000:  # Adjust the upper limit as needed
            raise HTTPException(status_code=404, detail=f"User ID {user_id} not found")
            
        logger.info(f"Generating recommendations for user {user_id} using collaborative model")
        
        # Use 'n' parameter if provided, otherwise use num_recommendations
        if n is not None:
            num_recommendations = n
        
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
            data_dir=os.path.join(project_root, 'data')
        )
        
        if recommendations_df.empty:
            logger.warning(f"No recommendations found for user {user_id}")
            # Instead of raising a 404, return an empty recommendations list
            # Record metrics before returning
            RECOMMENDATION_COUNTER.labels(endpoint="user-recommendations", status="empty").inc()
            RECOMMENDATION_TIME.labels(model_type=model_type, endpoint="user-recommendations").observe(time.time() - start_time)
            return RecommendationResponse(recommendations=[], user_id=user_id)
        
        recommendations = []
        books_df = get_books_df()
        
        for _, row in recommendations_df.iterrows():
            try:
                book_id = int(row['book_id'])
                
                # Get complete book data from books dataframe
                book_data = books_df[books_df['book_id'] == book_id]
                
                if not book_data.empty:
                    book_row = book_data.iloc[0]
                    
                    recommendation = BookRecommendation(
                        book_id=book_id,
                        title=book_row.get('title', f"Book {book_id}"),
                        authors=book_row.get('authors', 'Unknown Author'),
                        rank=int(row.get('rank', 0)) + 1,
                        average_rating=float(book_row.get('average_rating', 0.0)),
                        ratings_count=int(book_row.get('ratings_count', 0))
                    )
                    
                    # Add image URL if requested and available
                    if include_images and 'image_url' in book_row and book_row['image_url']:
                        recommendation.image_url = book_row['image_url']
                    
                    recommendations.append(recommendation)
                else:
                    # If book not found in books dataframe, use the data from recommendations dataframe
                    recommendation = BookRecommendation(
                        book_id=book_id,
                        title=row.get('title', f"Book {book_id}"),
                        authors=row.get('authors', 'Unknown Author'),
                        rank=int(row.get('rank', 0)) + 1
                    )
                    
                    # Add image URL if requested and available
                    if include_images and 'image_url' in row and row['image_url']:
                        recommendation.image_url = row['image_url']
                    
                    recommendations.append(recommendation)
            except Exception as e:
                logger.warning(f"Error processing recommendation row: {e}")
                continue
        
        # Record success metrics
        RECOMMENDATION_COUNTER.labels(endpoint="user-recommendations", status="success").inc()
        RECOMMENDATION_TIME.labels(model_type=model_type, endpoint="user-recommendations").observe(time.time() - start_time)
        
        return RecommendationResponse(recommendations=recommendations, user_id=user_id)
    
    except HTTPException:
        # Record error metrics
        RECOMMENDATION_COUNTER.labels(endpoint="user-recommendations", status="error").inc()
        RECOMMENDATION_TIME.labels(model_type=model_type, endpoint="user-recommendations").observe(time.time() - start_time)
        raise
    except ValueError as e:
        # This could be raised by the model or when a user doesn't exist
        if "user not found" in str(e).lower():
            RECOMMENDATION_COUNTER.labels(endpoint="user-recommendations", status="error").inc()
            RECOMMENDATION_TIME.labels(model_type=model_type, endpoint="user-recommendations").observe(time.time() - start_time)
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")
        # Any other value errors likely mean a request validation issue
        RECOMMENDATION_COUNTER.labels(endpoint="user-recommendations", status="error").inc()
        RECOMMENDATION_TIME.labels(model_type=model_type, endpoint="user-recommendations").observe(time.time() - start_time)
        raise HTTPException(status_code=422, detail=f"Invalid request: {str(e)}")
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}", exc_info=True)
        RECOMMENDATION_COUNTER.labels(endpoint="user-recommendations", status="error").inc()
        RECOMMENDATION_TIME.labels(model_type=model_type, endpoint="user-recommendations").observe(time.time() - start_time)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Books endpoint to return book list
@api_router.get("/books", response_model=Dict[str, Any])
async def get_books(
    limit: int = Query(1000, description="Limit the number of books returned"),
    offset: int = Query(0, description="Offset for pagination")
):
    """Get a list of books with their IDs, titles, and authors"""
    try:
        books_df = get_books_df()
        
        if books_df.empty:
            logger.error("Books data not found")
            return {"status": "error", "message": "Books data not found", "books": []}
        
        # Get total count before pagination
        total_count = len(books_df)
        
        # Apply pagination
        books_df = books_df.iloc[offset:offset+limit]
        
        # Prepare book list
        books = []
        for _, row in books_df.iterrows():
            book = {
                "book_id": int(row['book_id']),
                "title": row.get('title', f"Book {row['book_id']}"),
                "authors": row.get('authors', 'Unknown Author')
            }
            
            # Add rating information if available
            if 'average_rating' in row:
                book['average_rating'] = float(row['average_rating'])
            else:
                book['average_rating'] = 0.0
                
            if 'ratings_count' in row:
                book['ratings_count'] = int(row['ratings_count'])
            else:
                book['ratings_count'] = 0
            
            # Add description if available
            if 'description' in row and row['description']:
                book['description'] = row['description']
            
            # Add image URL if available
            if 'image_url' in row and row['image_url']:
                book['image_url'] = row['image_url']
                
            books.append(book)
            
        # Return formatted response
        return {
            "status": "success",
            "count": len(books),
            "total": total_count,
            "offset": offset,
            "limit": limit,
            "books": books
        }
        
    except Exception as e:
        logger.error(f"Error getting books: {str(e)}")
        logger.error(traceback.format_exc())
        return {"status": "error", "message": str(e), "books": []}

# Similar books endpoint
@api_router.get("/similar-books/{book_id}", response_model=RecommendationResponse)
async def get_similar_books(
    book_id: int,
    model_type: str = Query("collaborative", enum=["collaborative"]),
    num_recommendations: int = Query(5, ge=1, le=20),
    n: Optional[int] = Query(None, ge=1, le=20),
    include_images: bool = Query(True, description="Include book image URLs in the response")
):
    start_time = time.time()
    try:
        # Check for valid book ID range
        if book_id < 1 or book_id > 1000000:  # Adjust upper limit as needed
            raise HTTPException(status_code=404, detail=f"Book ID {book_id} not found")
        
        # Check if the book exists in our dataset
        books_df = get_books_df()
        if not books_df.empty and book_id not in books_df['book_id'].values:
            raise HTTPException(status_code=404, detail=f"Book ID {book_id} not found")
            
        logger.info(f"Finding similar books to book {book_id} using collaborative model")
        
        # Use 'n' parameter if provided, otherwise use num_recommendations
        if n is not None:
            num_recommendations = n
        
        # Check if we need to map the book_id to an internal ID
        mapping_path = os.path.join(project_root, 'data', 'processed', 'book_id_mapping.csv')
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
            data_dir=os.path.join(project_root, 'data')
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
            # Instead of raising a 404, return an empty recommendations list
            return RecommendationResponse(recommendations=[], book_id=book_id)
        
        recommendations = []
        books_df = get_books_df()
        
        for _, row in similar_books_df.iterrows():
            try:
                similar_book_id = int(row['book_id'])
                
                # Get complete book data from books dataframe
                book_data = books_df[books_df['book_id'] == similar_book_id]
                
                if not book_data.empty:
                    book_row = book_data.iloc[0]
                    
                    recommendation = BookRecommendation(
                        book_id=similar_book_id,
                        title=book_row.get('title', f"Book {similar_book_id}"),
                        authors=book_row.get('authors', 'Unknown Author'),
                        rank=int(row.get('rank', 0)) + 1,
                        similarity_score=float(row.get('similarity', 0.0)),
                        average_rating=float(book_row.get('average_rating', 0.0)),
                        ratings_count=int(book_row.get('ratings_count', 0))
                    )
                    
                    # Add image URL if requested and available
                    if include_images and 'image_url' in book_row and book_row['image_url']:
                        recommendation.image_url = book_row['image_url']
                    
                    recommendations.append(recommendation)
                else:
                    # If book not found in books dataframe, use the data from recommendations dataframe
                    recommendation = BookRecommendation(
                        book_id=similar_book_id,
                        title=row.get('title', f"Book {similar_book_id}"),
                        authors=row.get('authors', 'Unknown Author'),
                        rank=int(row.get('rank', 0)) + 1,
                        similarity_score=float(row.get('similarity', 0.0))
                    )
                    
                    # Add image URL if requested and available
                    if include_images and 'image_url' in row and row['image_url']:
                        recommendation.image_url = row['image_url']
                    
                    recommendations.append(recommendation)
            except Exception as e:
                logger.warning(f"Error processing recommendation row: {e}")
                continue
        
        # Record metrics
        RECOMMENDATION_COUNTER.labels(endpoint="similar-books", status="success").inc()
        RECOMMENDATION_TIME.labels(model_type=model_type, endpoint="similar-books").observe(time.time() - start_time)
        
        return RecommendationResponse(recommendations=recommendations, book_id=book_id)
    
    except HTTPException:
        # Record error metrics
        RECOMMENDATION_COUNTER.labels(endpoint="similar-books", status="error").inc()
        RECOMMENDATION_TIME.labels(model_type=model_type, endpoint="similar-books").observe(time.time() - start_time)
        raise
    except ValueError as e:
        # This could be raised by the model or when a book doesn't exist
        if "book not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=f"Book {book_id} not found")
        # Any other value errors likely mean a request validation issue
        raise HTTPException(status_code=422, detail=f"Invalid request: {str(e)}")
    except Exception as e:
        logger.error(f"Error finding similar books: {e}", exc_info=True)
        # Record error metrics
        RECOMMENDATION_COUNTER.labels(endpoint="similar-books", status="error").inc()
        RECOMMENDATION_TIME.labels(model_type=model_type, endpoint="similar-books").observe(time.time() - start_time)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# New endpoints from Flask API
@api_router.get("/genres", response_model=Dict[str, Any])
async def get_genres():
    """Get a list of all genres in the dataset"""
    try:
        books_df = get_books_df()
        
        if books_df.empty:
            # If no data is found, return default genres
            default_genres = ["Fiction", "Non-Fiction", "Fantasy", "Science Fiction", "Mystery", 
                             "Romance", "Biography", "History", "Self-Help"]
            return {
                "genres": default_genres,
                "count": len(default_genres),
                "note": "Using default genres as books data not found"
            }
        
        # Extract genres from the dataframe
        all_genres = []
        
        # Check for genre columns with different possible names
        genre_col = None
        for col_name in ['genres', 'genre', 'category', 'categories']:
            if col_name in books_df.columns:
                genre_col = col_name
                break
        
        if genre_col is not None:
            # Handle different storage formats (comma-separated string or list)
            for genres in books_df[genre_col].dropna():
                if isinstance(genres, str):
                    genre_list = [g.strip() for g in genres.split(',')]
                    all_genres.extend(genre_list)
                elif isinstance(genres, list):
                    all_genres.extend(genres)
        
        # Get unique genres, sort them and return
        unique_genres = sorted(list(set([g for g in all_genres if g and str(g).lower() != 'unknown'])))
        
        return {
            "genres": unique_genres,
            "count": len(unique_genres)
        }
    
    except Exception as e:
        logger.error(f"Error in get_genres: {str(e)}")
        logger.debug(traceback.format_exc())
        return {"error": str(e)}

@api_router.get("/authors", response_model=Dict[str, Any])
async def get_authors():
    """Get a list of all authors in the dataset"""
    try:
        books_df = get_books_df()
        
        if books_df.empty:
            # If no data is found, return default authors
            default_authors = ["Jane Austen", "J.K. Rowling", "Stephen King", "Agatha Christie", 
                              "Mark Twain", "George Orwell", "Ernest Hemingway"]
            return {
                "authors": default_authors,
                "count": len(default_authors),
                "note": "Using default authors as books data not found"
            }
        
        # Check if authors column exists
        if 'authors' not in books_df.columns:
            # Try other possible column names
            for col in ['author', 'writer', 'creator']:
                if col in books_df.columns:
                    books_df['authors'] = books_df[col]
                    break
            else:
                # Default authors if no column found
                default_authors = ["Jane Austen", "J.K. Rowling", "Stephen King", "Agatha Christie", 
                                  "Mark Twain", "George Orwell", "Ernest Hemingway"]
                return {
                    "authors": default_authors,
                    "count": len(default_authors),
                    "note": "Using default authors as no author data found in the dataset"
                }
        
        # Extract authors
        authors_list = []
        for author in books_df['authors'].dropna():
            # Handle case where multiple authors are separated by commas
            if isinstance(author, str) and ',' in author:
                for single_author in author.split(','):
                    authors_list.append(single_author.strip())
            else:
                authors_list.append(str(author).strip())
        
        # Get unique authors, sort them and return
        unique_authors = sorted(list(set([a for a in authors_list if a and str(a).lower() != 'unknown'])))
        
        return {
            "authors": unique_authors,
            "count": len(unique_authors)
        }
    
    except Exception as e:
        logger.error(f"Error in get_authors: {str(e)}")
        logger.debug(traceback.format_exc())
        return {"error": str(e)}

@api_router.get("/users", response_model=Dict[str, Any])
async def get_users(limit: int = Query(1000, description="Maximum number of user IDs to return")):
    """Get a list of available user IDs for recommendations"""
    try:
        ratings_df = get_ratings_df()
        
        if ratings_df.empty:
            # If no data is found, return default users
            default_users = list(range(1, 11))  # Default is users 1-10
            return {
                "users": default_users,
                "count": len(default_users),
                "note": "Using default users as ratings data not found"
            }
        
        # Get unique user IDs
        user_ids = sorted(ratings_df['user_id'].unique().tolist())
        
        # Apply limit if specified
        if limit > 0 and limit < len(user_ids):
            user_ids = user_ids[:limit]
        
        return {
            "users": user_ids,
            "count": len(user_ids),
            "total": len(ratings_df['user_id'].unique())
        }
    
    except Exception as e:
        logger.error(f"Error in get_users: {str(e)}")
        logger.debug(traceback.format_exc())
        return {"error": str(e)}

@api_router.get("/users/{user_id}", response_model=UserDetail)
async def get_user_details(user_id: int):
    """Get details for a specific user"""
    try:
        # Prepare response with default values
        user_details = UserDetail(
            user_id=user_id,
            total_ratings=0,
            avg_rating="N/A",
            favorite_genres=[],
            recent_books=[]
        )
        
        # Load ratings data to get user's reading history
        ratings_df = get_ratings_df()
        books_df = get_books_df()
        
        if ratings_df.empty:
            return user_details
        
        # Get user's ratings
        user_ratings = ratings_df[ratings_df['user_id'] == user_id]
        
        if not user_ratings.empty:
            # Calculate total ratings and average rating
            user_details.total_ratings = len(user_ratings)
            user_details.avg_rating = round(float(user_ratings['rating'].mean()), 1)
            
            # Get books rated by the user
            user_book_ids = user_ratings['book_id'].tolist()
            
            # Join with books data to get genres
            if not books_df.empty and user_book_ids:
                # Get user's books
                user_books = books_df[books_df['book_id'].isin(user_book_ids)]
                
                # Get and count genres
                if not user_books.empty and 'genres' in user_books.columns:
                    genres = []
                    for g in user_books['genres'].dropna():
                        if isinstance(g, str):
                            genres.extend([genre.strip() for genre in g.split(',')])
                    
                    # Count genre occurrences and get top 5
                    if genres:
                        genre_counts = pd.Series(genres).value_counts()
                        user_details.favorite_genres = genre_counts.head(5).index.tolist()
                
                # Get 5 most recent books (assuming higher ratings are more recent)
                if not user_ratings.empty:
                    recent_ratings = user_ratings.sort_values(by='rating', ascending=False).head(5)
                    recent_book_ids = recent_ratings['book_id'].tolist()
                    
                    recent_books = books_df[books_df['book_id'].isin(recent_book_ids)]
                    if not recent_books.empty:
                        # Format data for recent books
                        user_details.recent_books = [
                            {
                                "book_id": int(row['book_id']),
                                "title": row.get('title', 'Unknown Title'),
                                "rating": float(user_ratings[user_ratings['book_id'] == row['book_id']]['rating'].iloc[0])
                            }
                            for _, row in recent_books.iterrows()
                        ][:5]  # Limit to 5 books
        
        return user_details
        
    except Exception as e:
        logger.error(f"Error retrieving user details: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error retrieving user details: {str(e)}")

@api_router.get("/users/{user_id}/ratings", response_model=Dict[str, Any])
async def get_user_ratings(
    user_id: int,
    limit: int = Query(50, description="Limit the number of ratings returned"),
    offset: int = Query(0, description="Offset for pagination"),
    include_books: bool = Query(False, description="Include full book details in response")
):
    """Get all ratings for a specific user"""
    logger.info(f"Getting ratings for user {user_id}")
    
    try:
        # Access the ratings data
        ratings_df = get_ratings_df()
        books_df = get_books_df()
        
        if ratings_df.empty:
            return {
                "status": "error",
                "message": "Ratings data not found"
            }
        
        # Filter ratings by user ID
        user_ratings = ratings_df[ratings_df['user_id'] == user_id].copy()
        
        if user_ratings.empty:
            return {
                "status": "success",
                "user_id": user_id,
                "count": 0,
                "ratings": []
            }
        
        # Apply pagination
        total_ratings = len(user_ratings)
        paginated_ratings = user_ratings.iloc[offset:offset + limit]
        
        # Format the response
        ratings_list = paginated_ratings.to_dict(orient='records')
        
        # Include book details if requested
        if include_books and not books_df.empty:
            for rating in ratings_list:
                book_id = rating.get('book_id')
                book_info = books_df[books_df['book_id'] == book_id]
                if not book_info.empty:
                    book_row = book_info.iloc[0]
                    rating['book'] = {
                        'book_id': book_id,
                        'title': book_row.get('title', 'Unknown'),
                        'authors': book_row.get('authors', 'Unknown'),
                        'image_url': book_row.get('image_url', '')
                    }
        
        return {
            "status": "success",
            "user_id": user_id,
            "count": len(ratings_list),
            "total": total_ratings,
            "ratings": ratings_list
        }
    
    except Exception as e:
        logger.error(f"Error getting ratings for user {user_id}: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error getting ratings for user {user_id}: {str(e)}")

@api_router.get("/books/{book_id}/ratings", response_model=Dict[str, Any])
async def get_book_ratings(
    book_id: int,
    limit: int = Query(50, description="Limit the number of ratings returned"),
    offset: int = Query(0, description="Offset for pagination")
):
    """Get all ratings for a specific book"""
    logger.info(f"Getting ratings for book {book_id}")
    
    try:
        # Access the ratings data
        ratings_df = get_ratings_df()
        books_df = get_books_df()
        
        if ratings_df.empty:
            return {
                "status": "error",
                "message": "Ratings data not found"
            }
        
        # Get the book title
        book_title = "Unknown"
        if not books_df.empty:
            book_info = books_df[books_df['book_id'] == book_id]
            if not book_info.empty:
                book_title = book_info.iloc[0].get('title', 'Unknown')
        
        # Filter ratings by book ID
        book_ratings = ratings_df[ratings_df['book_id'] == book_id].copy()
        
        if book_ratings.empty:
            return {
                "status": "success",
                "book_id": book_id,
                "book_title": book_title,
                "count": 0,
                "average_rating": 0,
                "rating_distribution": {},
                "ratings": []
            }
        
        # Calculate statistics
        average_rating = round(book_ratings['rating'].mean(), 2)
        rating_counts = book_ratings['rating'].value_counts().to_dict()
        rating_distribution = {str(i): rating_counts.get(i, 0) for i in range(1, 6)}
        
        # Apply pagination
        total_ratings = len(book_ratings)
        paginated_ratings = book_ratings.iloc[offset:offset + limit]
        
        # Format the response
        ratings_list = paginated_ratings.to_dict(orient='records')
        
        return {
            "status": "success",
            "book_id": book_id,
            "book_title": book_title,
            "count": len(ratings_list),
            "total": total_ratings,
            "average_rating": average_rating,
            "rating_distribution": rating_distribution,
            "ratings": ratings_list
        }
    
    except Exception as e:
        logger.error(f"Error getting ratings for book {book_id}: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error getting ratings for book {book_id}: {str(e)}")

@api_router.get("/ratings", response_model=Dict[str, Any])
async def get_ratings(
    user_id: Optional[int] = Query(None, description="Filter ratings by user ID"),
    book_id: Optional[int] = Query(None, description="Filter ratings by book ID"),
    min_rating: Optional[float] = Query(None, description="Filter ratings by minimum rating value"),
    max_rating: Optional[float] = Query(None, description="Filter ratings by maximum rating value"),
    limit: int = Query(100, description="Limit the number of results returned"),
    offset: int = Query(0, description="Offset for pagination")
):
    """Get ratings data with various filtering options"""
    try:
        # Get ratings data
        ratings_df = get_ratings_df()
        
        if ratings_df.empty:
            return {
                'error': 'Ratings data not found',
                'status': 'error'
            }
        
        # Apply filters if specified
        if user_id is not None:
            ratings_df = ratings_df[ratings_df['user_id'] == user_id]
        
        if book_id is not None:
            ratings_df = ratings_df[ratings_df['book_id'] == book_id]
        
        if min_rating is not None:
            ratings_df = ratings_df[ratings_df['rating'] >= min_rating]
        
        if max_rating is not None:
            ratings_df = ratings_df[ratings_df['rating'] <= max_rating]
        
        # Get total count before pagination
        total_count = len(ratings_df)
        
        # Apply pagination
        ratings_df = ratings_df.iloc[offset:offset+limit]
        
        # Convert to dictionary format for JSON serialization
        ratings_list = ratings_df.to_dict(orient='records')
        
        # Return response
        return {
            'status': 'success',
            'count': len(ratings_list),
            'total': total_count,
            'offset': offset,
            'limit': limit,
            'ratings': ratings_list
        }
    
    except Exception as e:
        logger.error(f"Error fetching ratings: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error fetching ratings: {str(e)}")

@api_router.get("/popular-books", response_model=Dict[str, List[Dict[str, Any]]])
async def get_popular_books_endpoint(
    limit: int = Query(6, description="Number of books to return"),
    randomize: bool = Query(True, description="Whether to randomize the results"),
    include_genres: bool = Query(True, description="Whether to include genre information"),
    seed: Optional[int] = Query(None, description="Random seed for reproducibility")
):
    """Get popular books with high ratings (currently returns random books)"""
    try:
        global _POPULAR_BOOKS_CACHE
        
        # Create a cache key based on the query parameters
        cache_key = f"popular_books_{limit}_{randomize}_{include_genres}_{seed}"
        
        # Check if we have a valid cache entry
        now = datetime.now()
        if cache_key in _POPULAR_BOOKS_CACHE:
            cache_entry = _POPULAR_BOOKS_CACHE[cache_key]
            # Cache is valid for 24 hours (86400 seconds)
            if (now - cache_entry['timestamp']).total_seconds() < 86400:
                logger.info(f"Using cached popular books data for key: {cache_key}")
                return {'books': cache_entry['data']}
            else:
                logger.info(f"Cache expired for key: {cache_key}, fetching fresh data")
        
        # Get all books directly
        books_df = get_books_df()
        
        if books_df.empty:
            logger.error("Books data not found")
            return {'books': []}
        
        # Set seed for reproducibility
        seed_value = seed if seed is not None else int(datetime.now().timestamp())
        np.random.seed(seed_value)
        
        # Sort by ratings if available, otherwise just use random selection
        if 'average_rating' in books_df.columns and 'ratings_count' in books_df.columns:
            # Create a "popularity score" combining rating and number of ratings
            books_df['popularity'] = books_df['average_rating'] * np.log1p(books_df['ratings_count'])
            sorted_books = books_df.sort_values('popularity', ascending=False)
            # Get top books (2x limit to have some variety)
            top_books = sorted_books.head(min(limit*10, len(sorted_books)))
            
            # Randomly select from top books
            if randomize and len(top_books) > limit:
                selected_books = top_books.sample(n=limit, random_state=seed_value)
            else:
                selected_books = top_books.head(limit)
        else:
            # No ratings data, just random selection
            selected_books = books_df.sample(n=min(limit, len(books_df)), random_state=seed_value)
        
        # Convert to list of dictionaries for JSON response
        books_data = []
        for _, book in selected_books.iterrows():
            book_data = {
                'book_id': int(book['book_id']),
                'title': str(book['title']),
                'authors': str(book['authors']),
                'average_rating': float(book.get('average_rating', 0)),
                'ratings_count': int(book.get('ratings_count', 0)),
                'image_url': str(book.get('image_url', ''))
            }
            
            # Add genres if requested and available
            if include_genres and 'genres' in book and book['genres']:
                book_data['genres'] = str(book['genres'])
                
            books_data.append(book_data)
            
        # Cache the result for future use
        _POPULAR_BOOKS_CACHE[cache_key] = {
            'timestamp': now,
            'data': books_data
        }
        logger.info(f"Cached popular books data for key: {cache_key}")
        
        return {'books': books_data}
    
    except Exception as e:
        logger.error(f"Error in get_popular_books_endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        # Return empty list instead of error to keep frontend working
        return {'books': []}

# At the bottom of your file, after defining all endpoints:
app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

