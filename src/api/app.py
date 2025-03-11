import logging
import os
import sys
import pandas as pd
import pickle
from datetime import datetime
from typing import List, Dict, Optional, Any, Union, Tuple
from fastapi import FastAPI, HTTPException, Depends, Query, status
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.responses import HTMLResponse
import scipy.sparse as sp
import numpy as np
import json
import traceback

# Add proper paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(src_dir)
sys.path.insert(0, root_dir)
sys.path.insert(0, src_dir)

# Import model components
from src.models.train_model import BookRecommender
from src.models.predict_model import get_book_metadata

# Define custom load_model function for the API
def load_model(model_path: str = "models/book_recommender.pkl", 
               data_dir: str = "data") -> Tuple[BookRecommender, sp.csr_matrix, sp.csr_matrix]:
    """
    Load the recommender model and required matrices.
    
    Args:
        model_path (str): Path to the saved model file
        data_dir (str): Directory containing data files
        
    Returns:
        Tuple containing:
        - BookRecommender: The loaded recommender model
        - sp.csr_matrix: User-item matrix
        - sp.csr_matrix: Book features matrix
    """
    logger = logging.getLogger("book_recommender_api")
    logger.info(f"Loading recommender model from {model_path}")
    
    try:
        # First, let's directly load the matrices and data instead of unpickling the whole model
        # This avoids the pickle class loading issue
        features_dir = os.path.join(os.path.dirname(model_path), "..", "data", "features")
        
        # Load matrices
        user_item_matrix = sp.load_npz(os.path.join(features_dir, "user_item_matrix.npz"))
        book_features = sp.load_npz(os.path.join(features_dir, "book_feature_matrix.npz"))
        book_similarity = sp.load_npz(os.path.join(features_dir, "book_similarity_matrix.npz"))
        
        # Load book IDs and feature names
        book_ids = pd.read_csv(os.path.join(features_dir, "book_ids.csv"))["book_id"].values
        feature_names = pd.read_csv(os.path.join(features_dir, "feature_names.csv"))["feature"].values
        
        # Create a new BookRecommender instance with the loaded data
        recommender = BookRecommender()
        recommender.user_item_matrix = user_item_matrix
        recommender.book_features = book_features
        recommender.book_similarity_matrix = book_similarity
        recommender.book_ids = book_ids
        recommender.feature_names = feature_names
        
        logger.info(f"Loaded model components with user-item matrix shape: {user_item_matrix.shape}")
        return recommender, user_item_matrix, book_features
    except Exception as e:
        logger.error(f"Error loading model components: {str(e)}")
        raise RuntimeError(f"Failed to load model: {str(e)}")

# Configure logging
log_file = os.path.join("logs", f"api_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("book_recommender_api")

# Initialize FastAPI app
app = FastAPI(
    title="Book Recommender API",
    description="API for book recommendations and information",
    version="1.0.0",
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API key authentication (simple implementation)
API_KEY = os.environ.get("API_KEY", "your-default-api-key")  # Should be changed in production
api_key_header = APIKeyHeader(name="X-API-Key")

def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        logger.warning(f"Invalid API key attempt: {api_key[:5]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    return api_key

# Load model and data at startup
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Book Recommender API")
    
    # Set default empty values
    app.state.model_loaded = False
    app.state.recommender = None
    app.state.user_item_matrix = None
    app.state.book_features = None
    app.state.book_metadata = None
    app.state.book_id_mapping = None
    
    try:
        # Create proper paths - look in models directory instead of data/features
        models_dir = os.path.join(root_dir, "models")
        processed_dir = os.path.join(root_dir, "data", "processed")
        
        logger.info(f"Checking for model components in {models_dir}")
        
        required_files = [
            os.path.join(models_dir, "user_item_matrix.npz"),
            os.path.join(models_dir, "book_feature_matrix.npz"),
            os.path.join(models_dir, "book_similarity_matrix.npz"),
            os.path.join(models_dir, "book_ids.npy"),  # Note: using .npy instead of .csv
            os.path.join(processed_dir, "merged_train.csv")
        ]
        
        # Check if all required files exist
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            logger.warning(f"Model files missing: {missing_files}")
            logger.warning("API will start but recommendation endpoints will return errors")
            logger.warning("Please train the model first or ensure model files are in the correct location")
            return
            
        # Load model components
        logger.info("Loading model components...")
        try:
            # Load the sparse matrices from NPZ files
            user_item_matrix = sp.load_npz(os.path.join(models_dir, "user_item_matrix.npz"))
            book_features = sp.load_npz(os.path.join(models_dir, "book_feature_matrix.npz"))
            book_similarity = sp.load_npz(os.path.join(models_dir, "book_similarity_matrix.npz"))
            
            # Load book IDs and feature names
            book_ids = np.load(os.path.join(models_dir, "book_ids.npy"))
            
            try:
                with open(os.path.join(models_dir, "feature_names.json"), 'r') as f:
                    feature_names = json.load(f)
            except:
                # If feature names JSON isn't available, use default placeholder
                feature_names = [f"feature_{i}" for i in range(book_features.shape[1])]
                logger.warning(f"Feature names file not found, using placeholders for {len(feature_names)} features")
            
            # Create the recommender object
            recommender = BookRecommender(
                user_item_matrix=user_item_matrix,
                book_feature_matrix=book_features,
                book_similarity_matrix=book_similarity,
                book_ids=book_ids,
                feature_names=feature_names
            )
            
            # Store in app state
            app.state.recommender = recommender
            app.state.user_item_matrix = user_item_matrix
            app.state.book_features = book_features
            
            logger.info(f"Loaded model with user-item matrix shape: {user_item_matrix.shape}")
            logger.info(f"Loaded book features with shape: {book_features.shape}")
        except Exception as e:
            logger.error(f"Error loading model components: {str(e)}")
            raise
        
        # Load book metadata
        try:
            logger.info("Loading book metadata...")
            app.state.book_metadata = pd.read_csv(os.path.join(processed_dir, "merged_train.csv")).drop_duplicates(subset=["book_id"])
            
            # Try to load mapping file if it exists
            mapping_path = os.path.join(models_dir, "book_id_mapping.csv")
            if os.path.exists(mapping_path):
                app.state.book_id_mapping = pd.read_csv(mapping_path)
                logger.info(f"Loaded book ID mapping for {len(app.state.book_id_mapping)} books")
            else:
                logger.warning("Book ID mapping file not found, using book_metadata IDs directly")
                app.state.book_id_mapping = None
            
            logger.info(f"Loaded book metadata for {len(app.state.book_metadata)} books")
        except Exception as e:
            logger.error(f"Error loading book metadata: {str(e)}")
            raise
            
        # Mark model as successfully loaded
        app.state.model_loaded = True
        logger.info("Model and data successfully loaded")
            
    except Exception as e:
        logger.error(f"Failed to load model or data: {str(e)}")
        logger.warning("API will start but recommendation endpoints will return errors")

# Pydantic models for request/response validation
class BookResponse(BaseModel):
    book_id: int = Field(..., description="The book's ID")
    title: str = Field(..., description="The book's title")
    authors: str = Field(..., description="The book's authors")
    average_rating: float = Field(..., description="Average rating (1-5)")
    isbn: Optional[str] = Field(None, description="ISBN")
    language_code: Optional[str] = Field(None, description="Language code")
    original_publication_year: Optional[float] = Field(None, description="Year of publication")
    
    class Config:
        schema_extra = {
            "example": {
                "book_id": 1,
                "title": "The Hunger Games",
                "authors": "Suzanne Collins",
                "average_rating": 4.34,
                "isbn": "439023483",
                "language_code": "eng",
                "original_publication_year": 2008.0
            }
        }

class RatingRequest(BaseModel):
    user_id: int = Field(..., description="User ID")
    book_id: int = Field(..., description="Book ID")
    rating: float = Field(..., ge=1, le=5, description="Rating (1-5)")

class RecommendationResponse(BaseModel):
    book_id: int
    title: str
    authors: str
    average_rating: float
    score: float = Field(..., description="Recommendation score")

class ErrorResponse(BaseModel):
    detail: str

# API endpoint implementations
@app.get("/", response_class=HTMLResponse)
async def root():
    """
    Root endpoint providing a welcome page for the API.
    
    Returns:
        HTMLResponse: Welcome page HTML
    """
    model_status = "✅ Loaded" if getattr(app.state, "model_loaded", False) else "❌ Not loaded"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
        <head>
            <title>Book Recommender API</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                    color: #333;
                }}
                h1 {{
                    color: #2c3e50;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                }}
                .status {{
                    display: inline-block;
                    padding: 5px 10px;
                    border-radius: 4px;
                    background-color: #f5f5f5;
                    margin-bottom: 20px;
                }}
                .links {{
                    margin-top: 20px;
                }}
                .links a {{
                    display: inline-block;
                    margin-right: 15px;
                    padding: 8px 16px;
                    background-color: #3498db;
                    color: white;
                    text-decoration: none;
                    border-radius: 4px;
                }}
                .links a:hover {{
                    background-color: #2980b9;
                }}
                .endpoint {{
                    background-color: #f9f9f9;
                    padding: 10px;
                    border-left: 3px solid #3498db;
                    margin-bottom: 10px;
                }}
            </style>
        </head>
        <body>
            <h1>Book Recommender API</h1>
            <div class="status">
                <strong>API Version:</strong> 1.0.0 | 
                <strong>Model Status:</strong> {model_status}
            </div>
            
            <h2>Description</h2>
            <p>This API provides access to book metadata and recommendation features based on collaborative filtering and content-based approaches.</p>
            
            <div class="links">
                <a href="/docs">API Documentation</a>
                <a href="/health">Health Check</a>
                <a href="/books?limit=10">Browse Books</a>
            </div>
            
            <h2>Available Endpoints</h2>
            
            <div class="endpoint">
                <strong>GET /books</strong> - Browse and search the book catalog
            </div>
            
            <div class="endpoint">
                <strong>GET /books/{'{book_id}'}</strong> - Get details for a specific book
            </div>
            
            <div class="endpoint">
                <strong>GET /recommendations/user/{'{user_id}'}</strong> - Get personalized recommendations
            </div>
            
            <div class="endpoint">
                <strong>GET /recommendations/similar-to/{'{book_id}'}</strong> - Find similar books
            </div>
            
            <div class="endpoint">
                <strong>POST /ratings</strong> - Submit a new rating (requires API key)
            </div>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Check if the API is running.
    
    Returns:
        dict: Status of the API and model
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": getattr(app.state, "model_loaded", False)
    }

# GET books with pagination and filtering
@app.get("/books", response_model=List[BookResponse])
async def get_books(
    skip: int = Query(0, ge=0, description="Number of books to skip"),
    limit: int = Query(10, ge=1, le=100, description="Number of books to return"),
    title: Optional[str] = Query(None, description="Filter by book title"),
    author: Optional[str] = Query(None, description="Filter by book author")
):
    """
    Get a list of books with optional filtering.
    
    Args:
        skip: Number of books to skip
        limit: Maximum number of books to return
        title: Optional filter by book title
        author: Optional filter by book author
        
    Returns:
        List[BookResponse]: List of books
    """
    try:
        # Check if book metadata is loaded
        if not hasattr(app.state, "book_metadata") or app.state.book_metadata is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Book metadata not available. The model data hasn't been loaded."
            )
        
        books = app.state.book_metadata
        
        # Apply filters
        if title:
            books = books[books['title'].str.contains(title, case=False, na=False)]
        if author:
            books = books[books['authors'].str.contains(author, case=False, na=False)]
        
        # Apply pagination
        books = books.iloc[skip:skip+limit]
        
        # Convert to list of dictionaries
        return books.to_dict(orient="records")
    except Exception as e:
        logger.exception(f"Error retrieving books: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving books: {str(e)}"
        )

# GET book details by ID
@app.get("/books/{book_id}", response_model=BookResponse)
async def get_book(book_id: int):
    """
    Get details for a specific book by ID.
    
    Args:
        book_id: ID of the book to retrieve
        
    Returns:
        BookResponse: Book details
    """
    try:
        # Check if book metadata is loaded
        if not hasattr(app.state, "book_metadata") or app.state.book_metadata is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Book metadata not available. The model data hasn't been loaded."
            )
        
        # Find the book by ID
        book = app.state.book_metadata[app.state.book_metadata["book_id"] == book_id]
        
        if book.empty:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Book with ID {book_id} not found"
            )
        
        # Return the first matching book
        return book.iloc[0].to_dict()
    except HTTPException:
        # Re-raise HTTP exceptions to maintain their status codes
        raise
    except Exception as e:
        logger.exception(f"Error retrieving book {book_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving book: {str(e)}"
        )

# GET recommendations for a user
@app.get("/recommendations/user/{user_id}", response_model=List[BookResponse], responses={404: {"model": ErrorResponse}})
async def get_user_recommendations(
    user_id: int,
    limit: int = Query(10, ge=1, le=50, description="Number of recommendations to return")
):
    """
    Get book recommendations for a specific user.
    
    Args:
        user_id: ID of the user to get recommendations for
        limit: Maximum number of recommendations to return
        
    Returns:
        List[BookResponse]: List of recommended books
    """
    if not app.state.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
    
    try:
        # Verify user exists in the matrix
        if app.state.recommender is None:
            raise HTTPException(status_code=503, detail="Recommender model not loaded")
            
        # Check if user exists in our dataset
        if user_id not in app.state.recommender.user_ids:
            raise HTTPException(status_code=404, detail=f"User with ID {user_id} not found")
            
        # Get recommendations
        recommended_book_ids = app.state.recommender.recommend_for_user(user_id, limit)
        
        if not recommended_book_ids:
            # The user exists but has no recommendations
            logger.warning(f"No recommendations found for user {user_id}")
            return []
        
        # Get book metadata for recommendations
        recommendations = []
        for book_id in recommended_book_ids:
            book_data = app.state.book_metadata[app.state.book_metadata["book_id"] == book_id]
            if not book_data.empty:
                book_dict = book_data.iloc[0].to_dict()
                recommendations.append(book_dict)
        
        return recommendations
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error getting user recommendations: {str(e)}")
        logger.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error getting recommendations: {str(e)}")

# GET similar books
@app.get("/recommendations/similar-to/{book_id}", response_model=List[RecommendationResponse], responses={404: {"model": ErrorResponse}})
async def get_similar_books(
    book_id: int,
    limit: int = Query(10, ge=1, le=50, description="Number of similar books to return")
):
    """
    Get books similar to a given book.
    
    Args:
        book_id: ID of the book to find similar books for
        limit: Maximum number of similar books to return
        
    Returns:
        List[BookResponse]: List of similar books
    """
    if not app.state.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
    
    try:
        # Check if model is loaded
        if app.state.recommender is None:
            raise HTTPException(status_code=503, detail="Recommender model not loaded")
            
        # Check if the book exists in our dataset by using numpy's where() 
        # This is consistent with how the model itself checks book existence
        if app.state.book_metadata is None:
            raise HTTPException(status_code=404, detail="Book metadata not loaded")
            
        # Find the book ID in the recommender's book_ids array
        book_exists = False
        try:
            # This is how the recommender model checks if a book exists
            np.where(app.state.recommender.book_ids == book_id)[0][0]
            book_exists = True
        except (IndexError, TypeError):
            book_exists = False
            
        if not book_exists:
            raise HTTPException(status_code=404, detail=f"Book with ID {book_id} not found")
        
        # Get similar book recommendations
        similar_book_ids = app.state.recommender.recommend_similar_books(book_id=book_id, n=limit)
        
        if not similar_book_ids:
            # Return empty list if no similar books found
            logger.warning(f"No similar books found for book {book_id}")
            return []
        
        # Get metadata for similar books
        similar_books = []
        for similar_id in similar_book_ids:
            book_data = app.state.book_metadata[app.state.book_metadata["book_id"] == similar_id]
            if not book_data.empty:
                book_row = book_data.iloc[0]
                
                # Calculate similarity score (placeholder for now)
                score = 0.0
                idx = similar_book_ids.index(similar_id)
                if idx < len(similar_book_ids):
                    # Higher score for higher ranked books
                    score = 1.0 - (idx / len(similar_book_ids))
                    
                similar_books.append({
                    "book_id": int(book_row["book_id"]),
                    "title": str(book_row["title"]),
                    "authors": str(book_row["authors"]),
                    "average_rating": float(book_row["average_rating"]),
                    "score": score
                })
        
        return similar_books
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error finding similar books: {str(e)}")
        logger.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error finding similar books: {str(e)}")

# POST a new rating
@app.post("/ratings", status_code=status.HTTP_201_CREATED)
async def submit_rating(
    rating: RatingRequest,
    api_key: str = Depends(api_key_header)
):
    """
    Submit a new user rating for a book.
    
    Args:
        rating: The rating information
        api_key: API key for authentication
        
    Returns:
        dict: Confirmation message
    """
    # Check API key
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    try:
        # Check if book metadata is loaded
        if not hasattr(app.state, "book_metadata") or app.state.book_metadata is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Book metadata not available. The model data hasn't been loaded."
            )
        
        # Validate that the book exists
        book = app.state.book_metadata[app.state.book_metadata["book_id"] == rating.book_id]
        if book.empty:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Book with ID {rating.book_id} not found"
            )
        
        # Validate rating value
        if rating.rating < 1 or rating.rating > 5:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Rating must be between 1 and 5"
            )
        
        # Save rating to CSV file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ratings_dir = os.path.join(root_dir, "data", "interim")
        os.makedirs(ratings_dir, exist_ok=True)
        
        # Create a DataFrame for the new rating
        new_rating_df = pd.DataFrame({
            "user_id": [rating.user_id],
            "book_id": [rating.book_id],
            "rating": [rating.rating],
            "timestamp": [datetime.now().isoformat()]
        })
        
        # Save to CSV
        ratings_file = os.path.join(ratings_dir, f"new_ratings_{timestamp}.csv")
        new_rating_df.to_csv(ratings_file, index=False)
        
        logger.info(f"Saved new rating: user_id={rating.user_id}, book_id={rating.book_id}, rating={rating.rating}")
        
        return {
            "message": "Rating submitted successfully",
            "user_id": rating.user_id,
            "book_id": rating.book_id,
            "rating": rating.rating,
            "saved_to": ratings_file
        }
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.exception(f"Error submitting rating: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error submitting rating: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    # Run the FastAPI app
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
