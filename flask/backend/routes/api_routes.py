"""
API routes for Book Recommender System
Following FastAPI structure for consistency
"""
import os
import sys
import pandas as pd
import logging
from flask import Blueprint, jsonify, request, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
import json
import traceback
from pathlib import Path
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Union, Any

# Configure logging
logger = logging.getLogger(__name__)

# Add project root to path for module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.append(project_root)

# Import model-related functions
try:
    from src.models.predict_model import (
        recommend_for_user, 
        recommend_similar_books, 
        get_popular_books,
        get_book_metadata,
        load_recommender_model
    )
except ImportError:
    logger.error("Error importing from src.models.predict_model. Check project structure.")
    # Fallback imports if needed
    from models.predict_model import (
        recommend_for_user, 
        recommend_similar_books, 
        get_popular_books,
        get_book_metadata,
        load_recommender_model
    )

# Create Blueprint
api_bp = Blueprint('api', __name__)

# FastAPI-compatible endpoints

# Comment out or delete the conflicting health check endpoint
# @api_bp.route('/health', methods=['GET'])
# def health_check():
#     """Health check endpoint matching FastAPI"""
#     return jsonify({
#         "status": "healthy", 
#         "timestamp": datetime.now().isoformat()
#     })

@api_bp.route('/books', methods=['GET'])
def get_books():
    """
    Get a list of books with their IDs, titles, and authors
    
    Parameters:
    - limit (int, optional): Limit the number of books returned (default: 1000)
    - offset (int, optional): Offset for pagination (default: 0)
    
    Returns:
    - JSON with list of books
    """
    try:
        # Parse query parameters
        limit = request.args.get('limit', 1000, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        # Get data directory
        data_dir = os.path.join(project_root, 'data')
        
        # Load books data - prioritize merged.csv over books.csv
        books_path = os.path.join(data_dir, 'processed', 'merged.csv')
        if not os.path.exists(books_path):
            books_path = os.path.join(data_dir, 'processed', 'books.csv')
            if not os.path.exists(books_path):
                books_path = os.path.join(data_dir, 'raw', 'books.csv')
            
        if not os.path.exists(books_path):
            return jsonify({
                "status": "error",
                "message": "Books data not found"
            }), 404
            
        # Load and process books data
        books_df = pd.read_csv(books_path)
        
        # If using merged.csv, get unique books only
        if 'user_id' in books_df.columns:
            logger.info("Using merged dataset with ratings information")
            # Get unique books from merged dataset
            books_df = books_df.drop_duplicates(subset=['book_id'])
        
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
        return jsonify({
            "status": "success",
            "count": len(books),
            "total": total_count,
            "offset": offset,
            "limit": limit,
            "books": books
        })
        
    except Exception as e:
        logger.error(f"Error getting books: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@api_bp.route('/recommend/user/<int:user_id>', methods=['GET'])
def get_user_recommendations_fastapi(user_id):
    """
    Get book recommendations for a user using collaborative filtering
    
    Parameters:
    - user_id (int): User ID to generate recommendations for
    - n (optional, int): Number of recommendations to return
    - include_images (optional, bool): Whether to include image URLs
    
    Returns:
    - JSON with list of recommended books
    """
    try:
        # Get query parameters
        n = request.args.get('n', default=5, type=int)
        include_images = request.args.get('include_images', default=True, type=lambda v: v.lower() == 'true')
        
        # Determine model type
        model_type = request.args.get('model_type', default='collaborative', type=str)
        
        # Log the request
        logger.info(f"Generating recommendations for user {user_id} using {model_type} model")
        
        # Generate recommendations
        recommendations_df = recommend_for_user(
            user_id=user_id,
            n=n,
            model_type=model_type
        )
        
        if recommendations_df.empty:
            logger.warning(f"No recommendations found for user {user_id}")
            return jsonify({
                "user_id": user_id,
                "recommendations": []
            })
        
        # Format recommendations
        recommendations = []
        books_df = current_app.config['BOOKS_DF']
        
        for _, row in recommendations_df.iterrows():
            book_id = int(row['book_id'])
            
            # Get complete book data from books dataframe
            book_data = books_df[books_df['book_id'] == book_id]
            
            if not book_data.empty:
                book_row = book_data.iloc[0]
                
                recommendation = {
                    "book_id": book_id,
                    "title": book_row.get('title', f"Book {book_id}"),
                    "authors": book_row.get('authors', 'Unknown Author'),
                    "rank": int(row.get('rank', 0)),
                    "average_rating": float(book_row.get('average_rating', 0.0)),
                    "ratings_count": int(book_row.get('ratings_count', 0))
                }
                
                # Add image URL if requested and available
                if include_images and 'image_url' in book_row and book_row['image_url']:
                    recommendation['image_url'] = book_row['image_url']
                
                recommendations.append(recommendation)
            
        # Return formatted response
        return jsonify({
            "user_id": user_id,
            "recommendations": recommendations
        })
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "detail": str(e)
        }), 500

@api_bp.route('/similar-books/<int:book_id>', methods=['GET'])
def get_similar_books_fastapi(book_id):
    """
    Get similar books to a given book using collaborative filtering
    
    Parameters:
    - book_id (int): Book ID to find similar books for
    - n (optional, int): Number of similar books to return
    - include_images (optional, bool): Whether to include image URLs
    
    Returns:
    - JSON with list of similar books
    """
    try:
        # Get query parameters
        n = request.args.get('n', default=5, type=int)
        include_images = request.args.get('include_images', default=True, type=lambda v: v.lower() == 'true')
        
        # Determine model type
        model_type = request.args.get('model_type', default='collaborative', type=str)
        
        # Log the request
        logger.info(f"Finding similar books to book {book_id} using {model_type} model")
        
        # Ensure model imports are available when needed
        try:
            # These imports are needed to properly unpickle the model
            from src.models.train_model import CollaborativeRecommender, ContentRecommender
            logger.info("Successfully imported model classes for similar books endpoint")
        except ImportError:
            try:
                from models.train_model import CollaborativeRecommender, ContentRecommender
                logger.info("Successfully imported model classes from alternate path")
            except ImportError:
                logger.warning("Could not import model classes explicitly. Proceeding with hope that model loading will work.")
        
        # Generate similar books recommendations
        similar_books_df = recommend_similar_books(
            book_id=book_id,
            n=n,
            model_type=model_type
        )
        
        if similar_books_df.empty:
            logger.warning(f"No similar books found for book {book_id}")
            return jsonify({
                "book_id": book_id,
                "recommendations": []
            })
        
        # Log that we have successful recommendations
        logger.info(f"Found {len(similar_books_df)} similar books for book {book_id}")
        logger.info(f"Similar book IDs: {similar_books_df['book_id'].tolist()}")
        
        # Format similar books
        similar_books = []
        books_df = current_app.config['BOOKS_DF']
        
        for _, row in similar_books_df.iterrows():
            similar_book_id = int(row['book_id'])
            
            # Get complete book data from books dataframe
            book_data = books_df[books_df['book_id'] == similar_book_id]
            
            if not book_data.empty:
                book_row = book_data.iloc[0]
                
                book = {
                    "book_id": similar_book_id,
                    "title": book_row.get('title', f"Book {similar_book_id}"),
                    "authors": book_row.get('authors', 'Unknown Author'),
                    "rank": int(row.get('rank', 0)),
                    "similarity_score": float(row.get('similarity', 0.0)),
                    "average_rating": float(book_row.get('average_rating', 0.0)),
                    "ratings_count": int(book_row.get('ratings_count', 0))
                }
                
                # Add description if available
                if 'description' in book_row and book_row['description']:
                    book['description'] = book_row['description']
                
                # Add image URL if requested and available
                if include_images and 'image_url' in book_row and book_row['image_url']:
                    book['image_url'] = book_row['image_url']
                
                similar_books.append(book)
            
        # Return formatted response
        return jsonify({
            "book_id": book_id,
            "recommendations": similar_books
        })
        
    except Exception as e:
        logger.error(f"Error finding similar books: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "detail": str(e)
        }), 500

@api_bp.route('/genres', methods=['GET'])
def get_genres():
    """
    Get a list of all genres in the dataset
    """
    try:
        # First try to use the cached books dataframe
        books_df = current_app.config.get('BOOKS_DF')
        
        # If not in cache, try to load from the data directory
        if books_df is None:
            # Get data directory from config (or use a fallback)
            data_dir = current_app.config.get('DATA_DIR', os.path.join(project_root, "data"))
            
            # Load books data from either processed or raw data
            books_path_processed = os.path.join(data_dir, 'processed', 'books.csv')
            books_path_raw = os.path.join(data_dir, 'raw', 'books.csv')
            
            if os.path.exists(books_path_processed):
                books_df = pd.read_csv(books_path_processed)
                # Cache for future use
                current_app.config['BOOKS_DF'] = books_df
            elif os.path.exists(books_path_raw):
                books_df = pd.read_csv(books_path_raw)
                # Cache for future use
                current_app.config['BOOKS_DF'] = books_df
        
        if books_df is None:
            # If no data is found, return default genres
            default_genres = ["Fiction", "Non-Fiction", "Fantasy", "Science Fiction", "Mystery", 
                             "Romance", "Biography", "History", "Self-Help"]
            return jsonify({
                "genres": default_genres,
                "count": len(default_genres),
                "note": "Using default genres as books data not found"
            })
        
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
        
        return jsonify({
            "genres": unique_genres,
            "count": len(unique_genres)
        })
    
    except Exception as e:
        logger.error(f"Error in get_genres: {str(e)}")
        logger.debug(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@api_bp.route('/authors', methods=['GET'])
def get_authors():
    """
    Get a list of all authors in the dataset
    """
    try:
        # First try to use the cached books dataframe
        books_df = current_app.config.get('BOOKS_DF')
        
        # If not in cache, try to load from the data directory
        if books_df is None:
            # Get data directory from config (or use a fallback)
            data_dir = current_app.config.get('DATA_DIR', os.path.join(project_root, "data"))
            
            # Load books data from either processed or raw data
            books_path_processed = os.path.join(data_dir, 'processed', 'books.csv')
            books_path_raw = os.path.join(data_dir, 'raw', 'books.csv')
            
            if os.path.exists(books_path_processed):
                books_df = pd.read_csv(books_path_processed)
                # Cache for future use
                current_app.config['BOOKS_DF'] = books_df
            elif os.path.exists(books_path_raw):
                books_df = pd.read_csv(books_path_raw)
                # Cache for future use
                current_app.config['BOOKS_DF'] = books_df
        
        if books_df is None:
            # If no data is found, return default authors
            default_authors = ["Jane Austen", "J.K. Rowling", "Stephen King", "Agatha Christie", 
                              "Mark Twain", "George Orwell", "Ernest Hemingway"]
            return jsonify({
                "authors": default_authors,
                "count": len(default_authors),
                "note": "Using default authors as books data not found"
            })
        
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
                return jsonify({
                    "authors": default_authors,
                    "count": len(default_authors),
                    "note": "Using default authors as no author data found in the dataset"
                })
        
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
        
        return jsonify({
            "authors": unique_authors,
            "count": len(unique_authors)
        })
    
    except Exception as e:
        logger.error(f"Error in get_authors: {str(e)}")
        logger.debug(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": f"Error retrieving authors: {str(e)}"
        }), 500

@api_bp.route('/users', methods=['GET'])
def get_users():
    """
    Get a list of available user IDs for recommendations
    
    Query parameters:
    - limit (int): Maximum number of user IDs to return
    """
    try:
        # Get query parameters
        limit = request.args.get('limit', 1000, type=int)
        
        # First try to use the cached ratings dataframe
        ratings_df = current_app.config.get('RATINGS_DF')
        
        # If not in cache, try to load from the data directory
        if ratings_df is None:
            # Get data directory from config (or use a fallback)
            data_dir = current_app.config.get('DATA_DIR', os.path.join(project_root, "data"))
            
            # Load ratings data from either processed or raw data
            ratings_path_processed = os.path.join(data_dir, 'processed', 'ratings.csv')
            ratings_path_raw = os.path.join(data_dir, 'raw', 'ratings.csv')
            
            if os.path.exists(ratings_path_processed):
                ratings_df = pd.read_csv(ratings_path_processed)
                # Cache for future use
                current_app.config['RATINGS_DF'] = ratings_df
            elif os.path.exists(ratings_path_raw):
                ratings_df = pd.read_csv(ratings_path_raw)
                # Cache for future use
                current_app.config['RATINGS_DF'] = ratings_df
        
        if ratings_df is None:
            # If no data is found, return default users
            default_users = list(range(1, 11))  # Default is users 1-10
            return jsonify({
                "users": default_users,
                "count": len(default_users),
                "note": "Using default users as ratings data not found"
            })
        
        # Get unique user IDs
        user_ids = sorted(ratings_df['user_id'].unique().tolist())
        
        # Apply limit if specified
        if limit > 0 and limit < len(user_ids):
            user_ids = user_ids[:limit]
        
        return jsonify({
            "users": user_ids,
            "count": len(user_ids),
            "total": len(ratings_df['user_id'].unique())
        })
    
    except Exception as e:
        logger.error(f"Error in get_users: {str(e)}")
        logger.debug(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": f"Error retrieving users: {str(e)}"
        }), 500

@api_bp.route('/users/<int:user_id>', methods=['GET'])
def get_user_details(user_id):
    """
    Get details for a specific user
    
    Path parameters:
    - user_id (int): User ID to get details for
    """
    try:
        # Load ratings data to get user's reading history
        data_dir = current_app.config.get('DATA_DIR')
        
        # First priority: merged.csv which contains the actual data
        ratings_path = os.path.join(data_dir, 'processed', 'merged.csv')
        
        # Fallbacks if the main path doesn't exist
        if not os.path.exists(ratings_path):
            merged_train_path = os.path.join(data_dir, 'processed', 'merged_train.csv')
            if os.path.exists(merged_train_path):
                ratings_path = merged_train_path
        
        # Last resort: look for ratings.csv
        if not os.path.exists(ratings_path):
            ratings_path = os.path.join(data_dir, 'processed', 'ratings.csv')
            
        if not os.path.exists(ratings_path):
            ratings_path = os.path.join(data_dir, 'raw', 'ratings.csv')
        
        # Path for books data
        books_path = os.path.join(data_dir, 'processed', 'books.csv')
        if not os.path.exists(books_path):
            books_path = os.path.join(data_dir, 'raw', 'books.csv')
        
        # Prepare response with default values
        user_details = {
            "user_id": user_id,
            "total_ratings": 0,
            "avg_rating": "N/A",
            "favorite_genres": [],
            "recent_books": []
        }
        
        # Load user ratings data
        user_ratings = None
        
        if os.path.exists(ratings_path):
            try:
                current_app.logger.info(f"Loading user ratings from: {ratings_path}")
                # Try loading from ratings file
                ratings_df = pd.read_csv(ratings_path)
                
                # Column names might vary
                user_col = None
                for col in ['user_id', 'reader_id', 'USER_ID']:
                    if col in ratings_df.columns:
                        user_col = col
                        break
                
                book_col = None
                for col in ['book_id', 'BOOK_ID', 'item_id']:
                    if col in ratings_df.columns:
                        book_col = col
                        break
                        
                rating_col = None
                for col in ['rating', 'RATING', 'score']:
                    if col in ratings_df.columns:
                        rating_col = col
                        break
                
                if user_col and book_col and rating_col:
                    current_app.logger.info(f"Using columns: {user_col}, {book_col}, {rating_col}")
                    
                    # Get user's ratings
                    user_ratings = ratings_df[ratings_df[user_col] == user_id]
                    
                    if not user_ratings.empty:
                        # Calculate total ratings and average rating
                        user_details["total_ratings"] = len(user_ratings)
                        user_details["avg_rating"] = round(float(user_ratings[rating_col].mean()), 1)
                        
                        # Get books rated by the user
                        user_book_ids = user_ratings[book_col].tolist()
                        
                        # Join with books data to get genres
                        if os.path.exists(books_path) and user_book_ids:
                            books_df = pd.read_csv(books_path)
                            
                            # Get user's books
                            user_books = books_df[books_df[book_col].isin(user_book_ids)]
                            
                            # Get and count genres
                            if not user_books.empty and 'genres' in user_books.columns:
                                genres = []
                                for g in user_books['genres'].dropna():
                                    if isinstance(g, str):
                                        genres.extend([genre.strip() for genre in g.split(',')])
                                
                                # Count genre occurrences and get top 5
                                genre_counts = pd.Series(genres).value_counts()
                                user_details["favorite_genres"] = genre_counts.head(5).index.tolist()
                            
                            # Get 5 most recent books (assuming higher ratings are more recent)
                            if not user_ratings.empty:
                                recent_ratings = user_ratings.sort_values(by=rating_col, ascending=False).head(5)
                                recent_book_ids = recent_ratings[book_col].tolist()
                                
                                recent_books = books_df[books_df[book_col].isin(recent_book_ids)]
                                if not recent_books.empty:
                                    # Format data for recent books
                                    user_details["recent_books"] = [
                                        {
                                            "book_id": row[book_col],
                                            "title": row.get('title', 'Unknown Title'),
                                            "rating": float(user_ratings[user_ratings[book_col] == row[book_col]][rating_col].iloc[0])
                                        }
                                        for _, row in recent_books.iterrows()
                                    ][:5]  # Limit to 5 books
            except Exception as e:
                current_app.logger.error(f"Error processing user ratings data: {str(e)}")
                # We'll fall back to default user details
        
        # If still no user ratings found, try to use the user-item matrix directly
        if (user_ratings is None or user_ratings.empty) and current_app.config.get('MODEL_DIR'):
            try:
                from scipy.sparse import load_npz
                import numpy as np
                
                matrix_path = os.path.join(data_dir, 'features', 'user_item_matrix.npz')
                book_ids_path = os.path.join(data_dir, 'features', 'book_ids.npy')
                
                if os.path.exists(matrix_path) and os.path.exists(book_ids_path):
                    # Load user-item matrix and book IDs
                    user_item_matrix = load_npz(matrix_path)
                    book_ids = np.load(book_ids_path)
                    
                    # Assuming user_id is the index into the matrix
                    if user_id < user_item_matrix.shape[0]:
                        user_row = user_item_matrix[user_id].toarray().flatten()
                        nonzero_indices = np.nonzero(user_row)[0]
                        
                        # Count nonzero elements (ratings)
                        user_details["total_ratings"] = len(nonzero_indices)
                        
                        if len(nonzero_indices) > 0:
                            # Calculate average rating
                            ratings = user_row[nonzero_indices]
                            user_details["avg_rating"] = round(float(ratings.mean()), 1)
            except Exception as e:
                current_app.logger.error(f"Error accessing user-item matrix: {str(e)}")
                # Still fall back to default user details
        
        # Return user details
        return jsonify(user_details)
        
    except Exception as e:
        current_app.logger.error(f"Error retrieving user details: {str(e)}")
        return jsonify({
            "error": "Failed to retrieve user details",
            "message": str(e)
        }), 500

@api_bp.route('/users/<int:user_id>/ratings', methods=['GET'])
def get_user_ratings(user_id):
    """
    Get all ratings for a specific user
    
    Parameters:
    - user_id (int): User ID to get ratings for
    - limit (int, optional): Limit the number of ratings returned (default: 50)
    - offset (int, optional): Offset for pagination (default: 0)
    - include_books (bool, optional): Include full book details in response (default: False)
    
    Returns:
    - JSON with list of user ratings
    """
    logger.info(f"Getting ratings for user {user_id}")
    
    # Get query parameters
    limit = request.args.get('limit', 50, type=int)
    offset = request.args.get('offset', 0, type=int)
    include_books = request.args.get('include_books', 'false').lower() == 'true'
    
    try:
        # Access the cached ratings data
        ratings_df = current_app.config.get('RATINGS_DF')
        books_df = current_app.config.get('BOOKS_DF')
        
        if ratings_df is None:
            # If ratings data is not in the cache, load it
            ratings_path = Path(project_root) / "data" / "processed" / "ratings.csv"
            if not ratings_path.exists():
                return jsonify({
                    "status": "error",
                    "message": "Ratings data not found"
                }), 500
            
            ratings_df = pd.read_csv(ratings_path)
            current_app.config['RATINGS_DF'] = ratings_df
        
        # Filter ratings by user ID
        user_ratings = ratings_df[ratings_df['user_id'] == user_id].copy()
        
        if user_ratings.empty:
            return jsonify({
                "status": "success",
                "user_id": user_id,
                "count": 0,
                "ratings": []
            })
        
        # Apply pagination
        total_ratings = len(user_ratings)
        paginated_ratings = user_ratings.iloc[offset:offset + limit]
        
        # Format the response
        ratings_list = paginated_ratings.to_dict(orient='records')
        
        # Include book details if requested
        if include_books and books_df is not None:
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
        
        return jsonify({
            "status": "success",
            "user_id": user_id,
            "count": len(ratings_list),
            "total": total_ratings,
            "ratings": ratings_list
        })
    
    except Exception as e:
        logger.error(f"Error getting ratings for user {user_id}: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": f"Error retrieving user ratings: {str(e)}"
        }), 500

@api_bp.route('/books/<int:book_id>/ratings', methods=['GET'])
def get_book_ratings(book_id):
    """
    Get all ratings for a specific book
    
    Parameters:
    - book_id (int): Book ID to get ratings for
    - limit (int, optional): Limit the number of ratings returned (default: 50)
    - offset (int, optional): Offset for pagination (default: 0)
    
    Returns:
    - JSON with list of book ratings and statistics
    """
    logger.info(f"Getting ratings for book {book_id}")
    
    # Get query parameters
    limit = request.args.get('limit', 50, type=int)
    offset = request.args.get('offset', 0, type=int)
    
    try:
        # Access the cached ratings data
        ratings_df = current_app.config.get('RATINGS_DF')
        books_df = current_app.config.get('BOOKS_DF')
        
        if ratings_df is None:
            # If ratings data is not in the cache, load it
            ratings_path = Path(project_root) / "data" / "processed" / "ratings.csv"
            if not ratings_path.exists():
                return jsonify({
                    "status": "error",
                    "message": "Ratings data not found"
                }), 500
            
            ratings_df = pd.read_csv(ratings_path)
            current_app.config['RATINGS_DF'] = ratings_df
        
        # Get the book title
        book_title = "Unknown"
        if books_df is not None:
            book_info = books_df[books_df['book_id'] == book_id]
            if not book_info.empty:
                book_title = book_info.iloc[0].get('title', 'Unknown')
        
        # Filter ratings by book ID
        book_ratings = ratings_df[ratings_df['book_id'] == book_id].copy()
        
        if book_ratings.empty:
            return jsonify({
                "status": "success",
                "book_id": book_id,
                "book_title": book_title,
                "count": 0,
                "average_rating": 0,
                "rating_distribution": {},
                "ratings": []
            })
        
        # Calculate statistics
        average_rating = round(book_ratings['rating'].mean(), 2)
        rating_counts = book_ratings['rating'].value_counts().to_dict()
        rating_distribution = {str(i): rating_counts.get(i, 0) for i in range(1, 6)}
        
        # Apply pagination
        total_ratings = len(book_ratings)
        paginated_ratings = book_ratings.iloc[offset:offset + limit]
        
        # Format the response
        ratings_list = paginated_ratings.to_dict(orient='records')
        
        return jsonify({
            "status": "success",
            "book_id": book_id,
            "book_title": book_title,
            "count": len(ratings_list),
            "total": total_ratings,
            "average_rating": average_rating,
            "rating_distribution": rating_distribution,
            "ratings": ratings_list
        })
    
    except Exception as e:
        logger.error(f"Error getting ratings for book {book_id}: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": f"Error retrieving book ratings: {str(e)}"
        }), 500

@api_bp.route('/auth/login', methods=['POST'])
def login():
    """
    User login endpoint
    
    Request body:
    - username: User's username
    - password: User's password
    """
    # This is a simplified version - in a real app you'd use proper authentication
    from flask_jwt_extended import create_access_token
    
    username = request.json.get('username', None)
    password = request.json.get('password', None)
    
    # Demo login - replace with real authentication in production
    if username == 'demo' and password == 'password':
        access_token = create_access_token(identity=username)
        return jsonify(access_token=access_token)
    
    return jsonify({"error": "Invalid username or password"}), 401

@api_bp.route('/profile', methods=['GET'])
@jwt_required()
def profile():
    """
    Get user profile information
    """
    current_user = get_jwt_identity()
    return jsonify({"username": current_user})

@api_bp.route('/ratings', methods=['GET'])
def get_ratings():
    """
    Get ratings data from the processed ratings.csv file
    
    Query parameters:
    - user_id (int, optional): Filter ratings by user ID
    - book_id (int, optional): Filter ratings by book ID
    - min_rating (float, optional): Filter ratings by minimum rating value
    - max_rating (float, optional): Filter ratings by maximum rating value
    - limit (int, optional): Limit the number of results returned
    - offset (int, optional): Offset for pagination
    
    Returns:
    - JSON object with ratings data
    """
    try:
        # Get query parameters
        user_id = request.args.get('user_id', type=int)
        book_id = request.args.get('book_id', type=int)
        min_rating = request.args.get('min_rating', type=float)
        max_rating = request.args.get('max_rating', type=float)
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        # Locate the ratings.csv file
        data_dir = os.path.join(project_root, 'data')
        ratings_path = os.path.join(data_dir, 'processed', 'ratings.csv')
        
        if not os.path.exists(ratings_path):
            # Try the raw data directory as fallback
            ratings_path = os.path.join(data_dir, 'raw', 'ratings.csv')
            if not os.path.exists(ratings_path):
                return jsonify({
                    'error': 'Ratings data not found',
                    'status': 'error'
                }), 404
        
        # Load the ratings data
        logger.info(f"Loading ratings data from {ratings_path}")
        ratings_df = pd.read_csv(ratings_path)
        
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
        
        # Return response in a FastAPI-like structure
        return jsonify({
            'status': 'success',
            'count': len(ratings_list),
            'total': total_count,
            'offset': offset,
            'limit': limit,
            'ratings': ratings_list
        })
    
    except Exception as e:
        logger.error(f"Error fetching ratings: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

# Global cache storage for popular books with expiration
_popular_books_cache = {}

@api_bp.route('/popular-books', methods=['GET'])
def get_popular_books_endpoint():
    """
    Get a list of popular books based on ratings count and average rating.
    
    Query Parameters:
    - limit: Number of books to return (default: 10)
    - randomize: Whether to randomize the results (default: false)
    - include_genres: Whether to include genre information (default: false)
    
    Returns:
    - JSON object with list of popular books
    """
    try:
        limit = int(request.args.get('limit', 10))
        randomize = request.args.get('randomize', 'false').lower() == 'true'
        include_genres = request.args.get('include_genres', 'false').lower() == 'true'
        
        # Create a cache key based on the query parameters
        cache_key = f"popular_books_{limit}_{randomize}_{include_genres}"
        
        # Check if we have a valid cache entry
        now = datetime.now()
        if cache_key in _popular_books_cache:
            cache_entry = _popular_books_cache[cache_key]
            # Cache is valid for 24 hours (86400 seconds)
            if (now - cache_entry['timestamp']).total_seconds() < 86400:
                logger.info(f"Using cached popular books data for key: {cache_key}")
                return jsonify({'books': cache_entry['data']})
            else:
                logger.info(f"Cache expired for key: {cache_key}, fetching fresh data")
        
        # Seed for reproducibility if randomizing
        seed = int(datetime.now().timestamp()) if randomize else None
        
        # Get popular book IDs - request more than needed to ensure we have enough after filtering
        book_ids = get_popular_books(
            n=limit*2,  # Request more books than needed to account for filtering
            data_dir=os.path.join(os.getcwd(), '..', '..'),  # Adjust path as needed
            randomize=randomize,
            seed=seed
        )
        
        if not book_ids:
            current_app.logger.error("Failed to get popular books")
            return jsonify({'error': 'Failed to get popular books'}), 500
        
        # Get metadata for the books
        books_df = get_book_metadata(book_ids, data_dir=os.path.join(os.getcwd(), '..', '..'))
        
        if books_df.empty:
            current_app.logger.error("Failed to get book metadata")
            return jsonify({'error': 'Failed to get book metadata'}), 500
        
        # Limit to the requested number of books
        books_df = books_df.head(limit)
        
        # Convert to list of dictionaries for JSON response
        books_data = []
        for _, book in books_df.iterrows():
            book_data = {
                'book_id': int(book['book_id']),
                'title': book['title'],
                'authors': book['authors'],
                'average_rating': float(book.get('average_rating', 0.0)),
                'ratings_count': int(book.get('ratings_count', 0))
            }
            
            # Include image URL if available
            if 'image_url' in book and pd.notna(book['image_url']):
                book_data['image_url'] = book['image_url']
            else:
                book_data['image_url'] = '/static/book-placeholder.png'
            
            # Include genres if requested and available
            if include_genres and 'genres' in book and pd.notna(book['genres']):
                book_data['genres'] = book['genres']
            
            books_data.append(book_data)
        
        # Cache the result for future use
        _popular_books_cache[cache_key] = {
            'timestamp': now,
            'data': books_data
        }
        logger.info(f"Cached popular books data for key: {cache_key}")
        
        return jsonify({'books': books_data})
    
    except Exception as e:
        current_app.logger.error(f"Error in get_popular_books_endpoint: {str(e)}")
        current_app.logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
