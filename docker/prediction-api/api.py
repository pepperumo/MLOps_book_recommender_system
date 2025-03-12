#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Book Recommendation API using FastAPI with Collaborative Filtering."""

import os
import sys
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
import logging

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

# Define response models
class BookRecommendation(BaseModel):
    book_id: int
    title: str
    authors: str
    rank: int

class RecommendationResponse(BaseModel):
    recommendations: List[BookRecommendation]

# Startup event to check models availability
@app.on_event("startup")
async def startup_event():
    logger.info("Checking model availability...")
    model = load_recommender_model('collaborative', model_dir=os.path.join(project_root, "models"))
    if model is None:
        logger.warning("Collaborative model not available")
    else:
        logger.info("Collaborative model loaded successfully")

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
            {"path": "/similar-books/{book_id}", "description": "Get similar books to a given book"}
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
    n: Optional[int] = Query(None, ge=1, le=20)
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
        recommendations_df = recommend_for_user(
            user_id=user_id,
            model_type='collaborative',  # Always use collaborative model
            num_recommendations=num_recommendations,
            data_dir=os.path.join(project_root, 'data')
        )
        
        if recommendations_df.empty:
            logger.warning(f"No recommendations found for user {user_id}")
            raise HTTPException(
                status_code=404, 
                detail=f"No recommendations found for user {user_id}. Please check if the user exists in the training data."
            )
        
        recommendations = []
        for _, row in recommendations_df.iterrows():
            recommendations.append(
                BookRecommendation(
                    book_id=int(row['book_id']),
                    title=row['title'],
                    authors=row['authors'],
                    rank=int(row.get('rank', 0)) + 1
                )
            )
        
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

# Similar books endpoint
@app.get("/similar-books/{book_id}", response_model=RecommendationResponse)
async def get_similar_books(
    book_id: int,
    model_type: str = Query("collaborative", enum=["collaborative"]),
    num_recommendations: int = Query(5, ge=1, le=20),
    n: Optional[int] = Query(None, ge=1, le=20)
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
            raise HTTPException(
                status_code=404, 
                detail=f"No similar books found for book {book_id}. Please check if the book exists in the training data."
            )
        
        recommendations = []
        for _, row in similar_books_df.iterrows():
            recommendations.append(
                BookRecommendation(
                    book_id=int(row['book_id']),
                    title=row['title'],
                    authors=row['authors'],
                    rank=int(row.get('rank', 0)) + 1
                )
            )
        
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
    uvicorn.run(app, host="0.0.0.0", port=9999)
