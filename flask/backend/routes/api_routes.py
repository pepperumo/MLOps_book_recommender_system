"""
API routes for Book Recommender System
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
        get_book_metadata
    )
except ImportError:
    logger.error("Error importing from src.models.predict_model. Check project structure.")
    # Fallback imports if needed
    from models.predict_model import (
        recommend_for_user, 
        recommend_similar_books, 
        get_popular_books,
        get_book_metadata
    )

# Create Blueprint
api_bp = Blueprint('api', __name__)

@api_bp.route('/books', methods=['GET'])
def get_books():
    """
    Get available books from the dataset
    
    Query parameters:
    - limit (int): Maximum number of books to return
    - search (str): Search term for filtering books
    - language (str): Filter by language code
    - sort (str): Sort field (default: popularity)
    - genre (str): Filter by genre
    - author (str): Filter by author
    - book_id (int): Filter by specific book ID
    """
    try:
        limit = request.args.get('limit', default=100, type=int)
        search = request.args.get('search', default='', type=str)
        language = request.args.get('language', default='', type=str)
        sort = request.args.get('sort', default='popularity', type=str)
        genre = request.args.get('genre', default='', type=str)
        author = request.args.get('author', default='', type=str)
        book_id = request.args.get('book_id', default=None, type=int)
        
        # Load book data
        data_dir = current_app.config.get('DATA_DIR')
        books_path = os.path.join(data_dir, 'processed', 'books.csv')
        
        if not os.path.exists(books_path):
            # Try raw data directory as fallback
            books_path = os.path.join(data_dir, 'raw', 'books.csv')
            
            if not os.path.exists(books_path):
                # Try merged_train.csv as fallback
                books_path = os.path.join(data_dir, 'processed', 'merged_train.csv')
            
        if os.path.exists(books_path):
            books_df = pd.read_csv(books_path)
            
            # Deduplicate by book_id if needed
            if 'book_id' in books_df.columns:
                books_df = books_df.drop_duplicates(subset=['book_id'])
            
            # Filter by book_id if provided
            if book_id is not None:
                books_df = books_df[books_df['book_id'] == book_id]
            
            # Apply search filter
            if search:
                search_lower = search.lower()
                # Search in title, authors, etc.
                title_col = 'title' if 'title' in books_df.columns else 'book_title'
                author_col = 'authors' if 'authors' in books_df.columns else 'author'
                
                title_mask = books_df[title_col].astype(str).str.lower().str.contains(search_lower, na=False)
                author_mask = books_df[author_col].astype(str).str.lower().str.contains(search_lower, na=False)
                books_df = books_df[title_mask | author_mask]
            
            # Filter by language
            if language and 'language_code' in books_df.columns:
                books_df = books_df[books_df['language_code'] == language]
                
            # Filter by genre
            if genre:
                # Check for genre column with different possible names
                genre_col = None
                for col_name in ['genres', 'genre', 'category', 'categories']:
                    if col_name in books_df.columns:
                        genre_col = col_name
                        break
                
                if genre_col is not None:
                    # Filter books that contain the specified genre
                    books_df = books_df[books_df[genre_col].astype(str).str.lower().str.contains(genre.lower(), na=False)]
            
            # Filter by author
            if author:
                # Check for author column with different possible names
                author_col = None
                for col_name in ['authors', 'author', 'book_author']:
                    if col_name in books_df.columns:
                        author_col = col_name
                        break
                
                if author_col is not None:
                    # Filter books by the specified author
                    books_df = books_df[books_df[author_col].astype(str).str.lower().str.contains(author.lower(), na=False)]
            
            # Sort the results
            if sort == 'rating' and 'average_rating' in books_df.columns:
                books_df = books_df.sort_values('average_rating', ascending=False)
            elif sort == 'title':
                title_col = 'title' if 'title' in books_df.columns else 'book_title'
                books_df = books_df.sort_values(title_col)
            elif sort == 'popularity' and 'ratings_count' in books_df.columns:
                books_df = books_df.sort_values('ratings_count', ascending=False)
                
            # Apply limit
            books_df = books_df.head(limit)
            
            # Prepare the response
            books_list = books_df.to_dict(orient='records')
            
            # Add placeholder image URLs
            for book in books_list:
                if 'image_url' not in book or not book['image_url']:
                    book['image_url'] = "https://islandpress.org/sites/default/files/default_book_cover_2015.jpg"
            
            return jsonify({"books": books_list, "count": len(books_list)})
        else:
            return jsonify({"error": "Book data not found"}), 404
    
    except Exception as e:
        logger.error(f"Error in get_books: {str(e)}")
        logger.debug(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@api_bp.route('/genres', methods=['GET'])
def get_genres():
    """
    Get a list of all genres in the dataset
    """
    try:
        # Get data directory from config
        data_dir = current_app.config.get('DATA_DIR')
        
        # Load books data from either processed or raw data
        books_path_processed = os.path.join(data_dir, 'processed', 'books.csv')
        books_path_raw = os.path.join(data_dir, 'raw', 'books.csv')
        
        if os.path.exists(books_path_processed):
            books_df = pd.read_csv(books_path_processed)
        elif os.path.exists(books_path_raw):
            books_df = pd.read_csv(books_path_raw)
        else:
            return jsonify({"error": "Books data not found"}), 404
        
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
        # Get data directory from config
        data_dir = current_app.config.get('DATA_DIR')
        
        # Load books data from either processed or raw data
        books_path_processed = os.path.join(data_dir, 'processed', 'books.csv')
        books_path_raw = os.path.join(data_dir, 'raw', 'books.csv')
        
        if os.path.exists(books_path_processed):
            books_df = pd.read_csv(books_path_processed)
        elif os.path.exists(books_path_raw):
            books_df = pd.read_csv(books_path_raw)
        else:
            return jsonify({"error": "Books data not found"}), 404
        
        # Check for author column with different possible names
        author_col = None
        for col_name in ['authors', 'author', 'book_author']:
            if col_name in books_df.columns:
                author_col = col_name
                break
        
        if author_col is None:
            return jsonify({"error": "Author column not found in books data"}), 404
        
        # Get unique authors, sort them and return
        # Only include the most common authors to avoid overwhelming the UI
        author_counts = books_df[author_col].value_counts().head(100)
        top_authors = author_counts.index.tolist()
        
        # Filter out unknown authors
        top_authors = [a for a in top_authors if a and str(a).lower() != 'unknown']
        
        return jsonify({
            "authors": top_authors,
            "count": len(top_authors)
        })
    
    except Exception as e:
        logger.error(f"Error in get_authors: {str(e)}")
        logger.debug(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@api_bp.route('/recommend/user/<int:user_id>', methods=['GET'])
def get_user_recommendations(user_id):
    """
    Get book recommendations for a specific user
    
    Path parameters:
    - user_id (int): User ID to get recommendations for
    
    Query parameters:
    - num_recommendations (int): Number of recommendations to return
    - model_type (str): Model type to use (default: collaborative)
    - include_images (bool): Whether to include image URLs
    """
    try:
        # Get parameters
        num_recommendations = request.args.get('num_recommendations', 
                                               default=current_app.config.get('DEFAULT_NUM_RECOMMENDATIONS', 5), 
                                               type=int)
        model_type = request.args.get('model_type', 
                                      default=current_app.config.get('DEFAULT_MODEL_TYPE', 'collaborative'),
                                      type=str)
        include_images = request.args.get('include_images', default=True, type=bool)
        
        # Get data directory from config
        data_dir = current_app.config.get('DATA_DIR')
        
        # Get recommendations
        recommendations_df = recommend_for_user(
            user_id=user_id,
            model_type=model_type,
            num_recommendations=num_recommendations,
            data_dir=data_dir
        )
        
        if recommendations_df.empty:
            return jsonify({"error": f"No recommendations found for user {user_id}"}), 404
            
        # Convert to dictionaries for JSON serialization
        recommendations = recommendations_df.to_dict(orient='records')
        
        # Add placeholder image URLs if requested
        if include_images:
            for rec in recommendations:
                # Add placeholder image URL if not already present
                if 'image_url' not in rec or not rec['image_url']:
                    rec['image_url'] = "https://islandpress.org/sites/default/files/default_book_cover_2015.jpg"
        
        return jsonify({
            "user_id": user_id,
            "recommendations": recommendations,
            "count": len(recommendations)
        })
    
    except Exception as e:
        logger.error(f"Error in get_user_recommendations: {str(e)}")
        logger.debug(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@api_bp.route('/similar-books/<int:book_id>', methods=['GET'])
def get_similar_books(book_id):
    """
    Get similar books to a specific book
    
    Path parameters:
    - book_id (int): Book ID to find similar books for
    
    Query parameters:
    - num_recommendations (int): Number of recommendations to return
    - model_type (str): Model type to use (default: collaborative)
    - include_images (bool): Whether to include image URLs
    """
    try:
        # Get parameters
        num_recommendations = request.args.get('num_recommendations', 
                                               default=current_app.config.get('DEFAULT_NUM_RECOMMENDATIONS', 5), 
                                               type=int)
        model_type = request.args.get('model_type', 
                                      default=current_app.config.get('DEFAULT_MODEL_TYPE', 'collaborative'),
                                      type=str)
        include_images = request.args.get('include_images', default=True, type=bool)
        
        # Get data directory from config
        data_dir = current_app.config.get('DATA_DIR')
        
        # Get similar books
        similar_books_df = recommend_similar_books(
            book_id=book_id,
            model_type=model_type,
            num_recommendations=num_recommendations,
            data_dir=data_dir
        )
        
        if similar_books_df.empty:
            return jsonify({"error": f"No similar books found for book ID {book_id}"}), 404
            
        # Convert to dictionaries for JSON serialization
        recommendations = similar_books_df.to_dict(orient='records')
        
        # Add placeholder image URLs if requested
        if include_images:
            for rec in recommendations:
                # Add placeholder image URL if not already present
                if 'image_url' not in rec or not rec['image_url']:
                    rec['image_url'] = "https://islandpress.org/sites/default/files/default_book_cover_2015.jpg"
        
        return jsonify({
            "book_id": book_id,
            "recommendations": recommendations,
            "count": len(recommendations)
        })
    
    except Exception as e:
        logger.error(f"Error in get_similar_books: {str(e)}")
        logger.debug(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@api_bp.route('/popular-books', methods=['GET'])
def get_top_books():
    """
    Get a list of popular books
    
    Query parameters:
    - limit (int): Number of books to return
    - randomize (bool): Whether to randomize the results
    """
    try:
        # Get parameters
        limit = request.args.get('limit', default=10, type=int)
        randomize = request.args.get('randomize', default=False, type=bool)
        
        # Get data directory from config
        data_dir = current_app.config.get('DATA_DIR')
        
        # Try to get popular books through the prediction model
        try:
            # Get popular book IDs
            popular_book_ids = get_popular_books(
                num_books=limit,
                data_dir=data_dir,
                randomize=randomize
            )
            
            if popular_book_ids and len(popular_book_ids) > 0:
                # Get metadata for the popular books
                popular_books_df = get_book_metadata(
                    book_ids=popular_book_ids,
                    data_dir=data_dir
                )
                
                if not popular_books_df.empty:
                    # Convert to dictionaries for JSON serialization
                    books = popular_books_df.to_dict(orient='records')
                    
                    # Add placeholder image URLs if missing
                    for book in books:
                        if 'image_url' not in book or not book['image_url']:
                            book['image_url'] = "https://islandpress.org/sites/default/files/default_book_cover_2015.jpg"
                    
                    return jsonify({
                        "books": books,
                        "count": len(books)
                    })
        except Exception as e:
            logger.warning(f"Error using predict_model: {str(e)}, falling back to direct data loading")
            # Fall through to backup method
        
        # Backup method: directly load books from CSV
        books_path = os.path.join(data_dir, 'processed', 'books.csv')
        if not os.path.exists(books_path):
            books_path = os.path.join(data_dir, 'raw', 'books.csv')
            
        if os.path.exists(books_path):
            try:
                # Use low_memory=False and handle encoding issues
                books_df = pd.read_csv(books_path, low_memory=False, encoding='utf-8', 
                                    on_bad_lines='skip', engine='python')
                
                # Make sure required columns exist
                if 'book_id' not in books_df.columns:
                    return jsonify({"error": "Invalid books data format"}), 500
                
                # Sort by ratings_count if available, otherwise random sample
                if 'ratings_count' in books_df.columns and 'average_rating' in books_df.columns:
                    books_df['popularity'] = books_df['ratings_count'] * books_df['average_rating']
                    books_df = books_df.sort_values('popularity', ascending=False)
                
                # Take a sample of books
                sample_size = min(limit, len(books_df))
                if randomize:
                    books_df = books_df.sample(n=sample_size)
                else:
                    books_df = books_df.head(sample_size)
                
                # Convert to dictionaries for JSON
                books = []
                for _, row in books_df.iterrows():
                    book = {}
                    for col in books_df.columns:
                        # Handle the specific issue with publisher column
                        if col == 'publisher' and isinstance(row[col], str) and '\n' in row[col]:
                            book[col] = row[col].split('\n')[0].strip()
                        else:
                            # For normal columns
                            try:
                                if pd.isna(row[col]):
                                    book[col] = None
                                else:
                                    book[col] = row[col]
                            except:
                                book[col] = None
                    
                    # Ensure image_url is provided
                    if 'image_url' not in book or not book['image_url']:
                        book['image_url'] = "https://islandpress.org/sites/default/files/default_book_cover_2015.jpg"
                    
                    books.append(book)
                
                return jsonify({
                    "books": books,
                    "count": len(books)
                })
            except Exception as e:
                logger.error(f"Error reading books CSV: {str(e)}")
                logger.debug(traceback.format_exc())
                return jsonify({"error": f"Error loading books data: {str(e)}"}), 500
        
        return jsonify({"error": "No popular books found"}), 404
    
    except Exception as e:
        logger.error(f"Error in get_top_books: {str(e)}")
        logger.debug(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@api_bp.route('/users', methods=['GET'])
def get_users():
    """
    Get a list of available user IDs for recommendations
    
    Query parameters:
    - limit (int): Maximum number of user IDs to return
    """
    try:
        limit = request.args.get('limit', default=100, type=int)
        
        # Load ratings data to extract user IDs
        data_dir = current_app.config.get('DATA_DIR')
        ratings_path = os.path.join(data_dir, 'processed', 'ratings.csv')
        
        if not os.path.exists(ratings_path):
            # Try raw data directory as fallback
            ratings_path = os.path.join(data_dir, 'raw', 'ratings.csv')
            
        if os.path.exists(ratings_path):
            ratings_df = pd.read_csv(ratings_path)
            
            # Get unique user IDs
            user_col = 'user_id' if 'user_id' in ratings_df.columns else 'reader_id'
            user_ids = ratings_df[user_col].unique().tolist()
            
            # Limit the number of user IDs
            user_ids = user_ids[:limit]
            
            return jsonify({"user_ids": user_ids, "total": len(user_ids)})
        else:
            # Return mock user IDs for testing if no data is available
            mock_user_ids = [11676, 98391, 153662, 221008, 24829, 70683]
            return jsonify({"user_ids": mock_user_ids, "total": len(mock_user_ids)})
    except Exception as e:
        logger.error(f"Error getting user IDs: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

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
        ratings_path = os.path.join(data_dir, 'processed', 'ratings.csv')
        books_path = os.path.join(data_dir, 'processed', 'books.csv')
        
        if not os.path.exists(ratings_path):
            ratings_path = os.path.join(data_dir, 'raw', 'ratings.csv')
        
        if not os.path.exists(books_path):
            books_path = os.path.join(data_dir, 'raw', 'books.csv')
        
        # Prepare response with default values
        user_details = {
            "user_id": user_id,
            "total_ratings": 0,
            "avg_rating": 0.0,
            "favorite_genres": [],
            "recent_books": []
        }
        
        if os.path.exists(ratings_path) and os.path.exists(books_path):
            ratings_df = pd.read_csv(ratings_path)
            books_df = pd.read_csv(books_path)
            
            # Column names might vary
            user_col = 'user_id' if 'user_id' in ratings_df.columns else 'reader_id'
            book_col = 'book_id' if 'book_id' in ratings_df.columns else 'book_id'
            rating_col = 'rating' if 'rating' in ratings_df.columns else 'rating'
            
            # Get user's ratings
            user_ratings = ratings_df[ratings_df[user_col] == user_id]
            
            if not user_ratings.empty:
                # Calculate total ratings and average rating
                user_details["total_ratings"] = len(user_ratings)
                user_details["avg_rating"] = user_ratings[rating_col].mean()
                
                # Get books rated by the user
                user_book_ids = user_ratings[book_col].tolist()
                
                # Join with books data to get genres
                if not books_df.empty:
                    user_books = books_df[books_df[book_col].isin(user_book_ids)]
                    
                    # Extract genres
                    genres = []
                    genre_col = 'genres' if 'genres' in books_df.columns else 'genre'
                    
                    if genre_col in user_books.columns:
                        for g in user_books[genre_col].dropna():
                            if isinstance(g, str):
                                genres.extend([genre.strip() for genre in g.split('|')])
                    
                    # Count genre frequencies
                    from collections import Counter
                    genre_counts = Counter(genres)
                    
                    # Get top 5 favorite genres
                    user_details["favorite_genres"] = [genre for genre, count in genre_counts.most_common(5)]
                    
                    # Get 5 recent books with details
                    recent_books = []
                    title_col = 'title' if 'title' in books_df.columns else 'book_title'
                    author_col = 'authors' if 'authors' in books_df.columns else 'author'
                    
                    for _, book in user_books.head(5).iterrows():
                        book_info = {
                            "book_id": int(book[book_col]),
                            "title": book[title_col] if title_col in book.index else "Unknown Title",
                            "author": book[author_col] if author_col in book.index else "Unknown Author"
                        }
                        
                        # Add image URL if available
                        if 'image_url' in book.index:
                            book_info["image_url"] = book['image_url']
                        
                        # Add user's rating for this book
                        book_rating = user_ratings[user_ratings[book_col] == book[book_col]][rating_col].values
                        if len(book_rating) > 0:
                            book_info["user_rating"] = float(book_rating[0])
                        
                        recent_books.append(book_info)
                    
                    user_details["recent_books"] = recent_books
        
        return jsonify(user_details)
    except Exception as e:
        logger.error(f"Error getting user details: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# User authentication and management endpoints
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
