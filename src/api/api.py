#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Book Recommendation API using FastAPI with Collaborative Filtering."""

import os
import sys
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import logging
import pandas as pd
import numpy as np

# Set up project path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, os.path.join(project_root, "src", "models"))

# Import the necessary modules
from src.models.model_utils import BaseRecommender, load_data
from src.models.train_model import CollaborativeRecommender

# Import recommender modules
try:
    from src.models.predict_model import (
        recommend_for_user, 
        recommend_similar_books,
        load_recommender_model
    )
except ImportError as e:
    try:
        from models.predict_model import (
            recommend_for_user, 
            recommend_similar_books,
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
                load_recommender_model
            )
        except ImportError:
            from models.predict_model import (
                recommend_for_user, 
                recommend_similar_books,
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

# Create FastAPI app
app = FastAPI(
    title="Book Recommender API",
    description="API for book recommendations using collaborative filtering",
    version="1.0.0"
)

# Configure CORS to allow requests from the Streamlit app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Define response models
class BookRecommendation(BaseModel):
    book_id: int
    title: str
    authors: str
    rank: int
    image_url: Optional[str] = None

class RecommendationResponse(BaseModel):
    recommendations: List[BookRecommendation]

# Startup event to check models availability
@app.on_event("startup")
async def startup_event():
    """Load models and check if they're available"""
    logger.info("Checking model availability...")
    global collaborative_model
    
    try:
        # Load collaborative filtering model
        model = load_recommender_model('collaborative', models_dir=os.path.join(project_root, "models"))
        if model:
            collaborative_model = model
            logger.info("Collaborative model loaded successfully")
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
            {"path": "/books", "description": "Get a list of books with their IDs, titles, and authors"}
        ]
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# User recommendation endpoint
@app.get("/recommend/user/{user_id}", response_model=RecommendationResponse)
async def get_user_recommendations(
    user_id: int, 
    model_type: str = Query("collaborative", enum=["collaborative"]),
    num_recommendations: int = Query(5, ge=1, le=20),
    n: Optional[int] = Query(None, ge=1, le=20),
    include_images: bool = Query(False, description="Include book image URLs in the response"),
    force_diverse: bool = Query(True, description="Force diversity in recommendations")
):
    """Get book recommendations for a user"""
    # Check for valid user ID range (assuming we have users 1-500 for example)
    if user_id < 1 or user_id > 500:
        raise HTTPException(status_code=404, detail=f"User ID {user_id} not found")
        
    logger.info(f"Generating recommendations for user {user_id} using collaborative model")
    
    # Use 'n' parameter if provided, otherwise use num_recommendations
    if n is not None:
        num_recommendations = n
    
    try:
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
            num_recommendations=num_recommendations,
            data_dir=os.path.join(project_root, 'data')
        )
        
        if recommendations_df.empty:
            logger.warning(f"No recommendations found for user {user_id}")
            # Instead of raising a 404, return an empty recommendations list
            return RecommendationResponse(recommendations=[])
        
        recommendations = []
        for _, row in recommendations_df.iterrows():
            try:
                recommendation = BookRecommendation(
                    book_id=int(row['book_id']),
                    title=row.get('title', f"Book {row['book_id']}"),
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
        
        return RecommendationResponse(recommendations=recommendations)
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except ValueError as e:
        # This could be raised by the model or when a user doesn't exist
        if "user not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")
        # Any other value errors likely mean a request validation issue
        raise HTTPException(status_code=422, detail=f"Invalid request: {str(e)}")
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Books endpoint to return book list
@app.get("/books", response_model=List[dict])
async def get_books(limit: int = Query(1000, description="Limit the number of books returned")):
    """Get a list of books with their IDs, titles, and authors"""
    try:
        # Get book data path
        books_path = os.path.join(project_root, 'data', 'processed', 'books.csv')
        
        # Fallback paths if the main one doesn't exist
        fallback_paths = [
            os.path.join(project_root, 'data', 'processed', 'book_id_mapping.csv'),
            os.path.join(project_root, 'data', 'raw', 'books.csv')
        ]
        
        # Try to load from main path first
        if os.path.exists(books_path):
            books_df = pd.read_csv(books_path)
        else:
            # Try fallback paths
            for path in fallback_paths:
                if os.path.exists(path):
                    books_df = pd.read_csv(path)
                    break
            else:
                # If no paths work, return an empty list
                logger.error("No book data files found")
                return []
        
        # Ensure required columns exist
        if 'book_id' not in books_df.columns and 'original_id' in books_df.columns:
            books_df = books_df.rename(columns={'original_id': 'book_id'})
            
        if 'authors' not in books_df.columns and 'author' in books_df.columns:
            books_df = books_df.rename(columns={'author': 'authors'})
        elif 'authors' not in books_df.columns:
            books_df['authors'] = 'Unknown'
            
        if 'title' not in books_df.columns:
            logger.error("Required column 'title' missing from book data")
            return []
            
        # Get a subset of the DataFrame with just the columns we need
        required_columns = ['book_id', 'title', 'authors']
        available_columns = [col for col in required_columns if col in books_df.columns]
        books_df = books_df[available_columns]
        
        # Limit the number of books
        books_df = books_df.head(limit)
        
        # Convert to list of dictionaries
        books_list = books_df.fillna('').to_dict(orient='records')
        
        return books_list
        
    except Exception as e:
        logger.error(f"Error fetching books: {e}")
        return []

# Similar books endpoint
@app.get("/similar-books/{book_id}", response_model=RecommendationResponse)
async def get_similar_books(
    book_id: int,
    model_type: str = Query("collaborative", enum=["collaborative"]),
    num_recommendations: int = Query(5, ge=1, le=20),
    n: Optional[int] = Query(None, ge=1, le=20),
    include_images: bool = Query(False, description="Include book image URLs in the response")
):
    """Get similar books to a given book using collaborative filtering"""
    # Check for valid book ID range (assuming we have books 1-10000 for example)
    if book_id < 1 or book_id > 10000:
        raise HTTPException(status_code=404, detail=f"Book ID {book_id} not found")
        
    logger.info(f"Finding similar books to book {book_id} using collaborative model")
    
    # Use 'n' parameter if provided, otherwise use num_recommendations
    if n is not None:
        num_recommendations = n
    
    try:
        similar_books_df = recommend_similar_books(
            book_id=book_id,
            model_type='collaborative',  # Always use collaborative model
            num_recommendations=num_recommendations,
            data_dir=os.path.join(project_root, 'data')
        )
        
        if similar_books_df.empty:
            logger.warning(f"No similar books found for book {book_id}")
            # Instead of raising a 404, return an empty recommendations list
            return RecommendationResponse(recommendations=[])
        
        recommendations = []
        for _, row in similar_books_df.iterrows():
            try:
                recommendation = BookRecommendation(
                    book_id=int(row['book_id']),
                    title=row['title'],
                    authors=row['authors'],
                    rank=int(row.get('rank', 0)) + 1
                )
                
                # Add image URL if requested and available
                if include_images and 'image_url' in row and row['image_url']:
                    recommendation.image_url = row['image_url']
                
                recommendations.append(recommendation)
            except Exception as e:
                logger.warning(f"Error processing recommendation row: {e}")
                continue
        
        return RecommendationResponse(recommendations=recommendations)
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except ValueError as e:
        # This could be raised by the model or when a book doesn't exist
        if "book not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=f"Book {book_id} not found")
        # Any other value errors likely mean a request validation issue
        raise HTTPException(status_code=422, detail=f"Invalid request: {str(e)}")
    except Exception as e:
        logger.error(f"Error finding similar books: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9998)
