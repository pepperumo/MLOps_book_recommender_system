from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
import logging
import os
import sys
from datetime import datetime

# Set up logging
log_dir = os.path.join('logs')
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

# Import recommender modules
try:
    from src.models.predict_model import (
        recommend_for_user, 
        recommend_similar_books,
        load_recommender_model
    )
except ImportError as e:
    logger.error(f"Error importing modules: {e}")
    raise

# Create FastAPI app
app = FastAPI(
    title="Book Recommender API",
    description="API for book recommendations using collaborative filtering, content-based filtering, and hybrid approaches",
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
    for model_type in ['collaborative', 'content', 'hybrid']:
        model = load_recommender_model(model_type)
        if model is None:
            logger.warning(f"{model_type.capitalize()} model not available")
        else:
            logger.info(f"{model_type.capitalize()} model loaded successfully")

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
    model_type: str = Query("hybrid", enum=["collaborative", "content", "hybrid"]),
    num_recommendations: int = Query(5, ge=1, le=20)
):
    logger.info(f"Generating recommendations for user {user_id} using {model_type} model")
    
    try:
        # Validate model_type parameter
        if model_type not in ["collaborative", "content", "hybrid"]:
            raise HTTPException(
                status_code=422, 
                detail=f"Invalid model_type parameter. Must be one of: collaborative, content, hybrid"
            )
            
        recommendations_df = recommend_for_user(
            user_id=user_id,
            model_type=model_type,
            num_recommendations=num_recommendations
        )
        
        if recommendations_df.empty:
            if model_type == "content":
                # Provide a more detailed explanation for content-based filtering limitations
                raise HTTPException(
                    status_code=404, 
                    detail=(
                        f"No content-based recommendations found for user {user_id}. "
                        "This can occur when: (1) the user hasn't rated enough books, "
                        "(2) the books rated don't have strong content patterns, or "
                        "(3) there aren't enough similar books in the database. "
                        "Try using the 'hybrid' or 'collaborative' model type instead."
                    )
                )
            else:
                raise HTTPException(status_code=404, detail=f"No recommendations found for user {user_id}")
        
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
    model_type: str = Query("content", enum=["collaborative", "content", "hybrid"]),
    num_recommendations: int = Query(5, ge=1, le=20)
):
    logger.info(f"Finding similar books for book {book_id} using {model_type} model")
    
    try:
        # Validate model_type parameter
        if model_type not in ["collaborative", "content", "hybrid"]:
            raise HTTPException(
                status_code=422, 
                detail=f"Invalid model_type parameter. Must be one of: collaborative, content, hybrid"
            )
            
        similar_books_df = recommend_similar_books(
            book_id=book_id,
            model_type=model_type,
            num_recommendations=num_recommendations
        )
        
        if similar_books_df.empty:
            raise HTTPException(status_code=404, detail=f"No similar books found for book {book_id}")
        
        # Filter out the source book (which has rank -1 or matches the original book_id)
        similar_books_df = similar_books_df[
            (similar_books_df['rank'] >= 0) & 
            (similar_books_df['book_id'] != book_id)
        ]
        
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
