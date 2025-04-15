"""
Utility functions for the book recommender Streamlit app.
"""
import os
import sys
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('streamlit_model_utils')

# Get current directory (inside streamlit folder)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Import the custom pickle loader that contains model class definitions
try:
    from pickle_loader import load_model, CollaborativeRecommender, BaseRecommender
    logger.info("Successfully imported model classes from pickle_loader")
except ImportError as e:
    logger.error(f"Error importing model classes from pickle_loader: {e}")

# Cache for data
_BOOKS_DF = None
_RATINGS_DF = None
_MODEL = None
_POPULAR_BOOKS_CACHE = {}

def get_books_df():
    """Get and cache the books dataframe"""
    global _BOOKS_DF
    if (_BOOKS_DF is not None):
        return _BOOKS_DF
    
    # Look for the data in the streamlit folder
    books_path = os.path.join(current_dir, 'data', 'processed', 'cleaned_books.csv')
    
    if os.path.exists(books_path):
        logger.info(f"Loading books data from {books_path}")
        _BOOKS_DF = pd.read_csv(books_path)
        
        # If merged.csv has user_id column, get unique books only
        if 'user_id' in _BOOKS_DF.columns:
            _BOOKS_DF = _BOOKS_DF.drop_duplicates(subset=['book_id'])
            
        # Ensure we have no duplicate titles by keeping the highest rated version
        if 'average_rating' in _BOOKS_DF.columns:
            _BOOKS_DF = _BOOKS_DF.sort_values('average_rating', ascending=False)
            _BOOKS_DF = _BOOKS_DF.drop_duplicates(subset=['title', 'authors'], keep='first')
            
        return _BOOKS_DF
    
    logger.error(f"Books data not found at {books_path}")
    return pd.DataFrame()

def get_ratings_df():
    """Get and cache the ratings dataframe"""
    global _RATINGS_DF
    if _RATINGS_DF is not None:
        return _RATINGS_DF
    
    # Look for the data in the streamlit folder
    ratings_path = os.path.join(current_dir, 'data', 'processed', 'ratings.csv')
    
    if os.path.exists(ratings_path):
        logger.info(f"Loading ratings data from {ratings_path}")
        _RATINGS_DF = pd.read_csv(ratings_path)
        return _RATINGS_DF
    
    logger.error("Ratings data not found in expected location")
    return pd.DataFrame()

def get_recommender_model():
    """Load the collaborative filtering model"""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    
    # Look for the model in the streamlit folder
    model_path = os.path.join(current_dir, 'models', 'collaborative.pkl')
    
    if os.path.exists(model_path):
        logger.info(f"Loading model from {model_path}")
        try:
            # Use the custom pickle loader instead of the standard pickle.load
            _MODEL = load_model(model_path)
            return _MODEL
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # If we can't load the model, we'll use a simple fallback recommendation approach
    
    logger.error(f"Model not found at {model_path}")
    return None

def get_all_users(limit=1000):
    """Get a list of all user IDs"""
    ratings_df = get_ratings_df()
    if ratings_df.empty:
        return list(range(1, 11))  # Default user IDs
    
    users = sorted(ratings_df['user_id'].unique().tolist())
    if limit and limit < len(users):
        return users[:limit]
    return users

def get_book_by_id(book_id):
    """Get details for a specific book"""
    books_df = get_books_df()
    if books_df.empty:
        return None
    
    # Check if we need to use ID mapping
    mapping_path = os.path.join(current_dir, 'data', 'processed', 'book_id_mapping.csv')
    if os.path.exists(mapping_path):
        try:
            mapping_df = pd.read_csv(mapping_path)
            if 'book_id' in mapping_df.columns and 'book_id_encoded' in mapping_df.columns:
                mapping_row = mapping_df[mapping_df['book_id'] == book_id]
                if not mapping_row.empty:
                    # Get the mapped ID
                    mapped_id = mapping_row.iloc[0]['book_id_encoded']
                    logger.info(f"Mapped book ID {book_id} to internal ID {mapped_id}")
                    # Try both IDs
                    book_data = books_df[books_df['book_id'] == book_id]
                    if book_data.empty:
                        book_data = books_df[books_df['book_id'] == mapped_id]
                else:
                    book_data = books_df[books_df['book_id'] == book_id]
            else:
                book_data = books_df[books_df['book_id'] == book_id]
        except Exception as e:
            logger.warning(f"Error using book ID mapping: {e}")
            book_data = books_df[books_df['book_id'] == book_id]
    else:
        book_data = books_df[books_df['book_id'] == book_id]
    
    if book_data.empty:
        return None
    
    # Convert to dictionary
    book = book_data.iloc[0].to_dict()
    return book

def recommend_for_user(user_id, n=5, force_diverse=True):
    """Get book recommendations for a user"""
    try:
        logger.info(f"Starting recommendations for user {user_id}")
        
        # Add a random offset to the user_id to create more diversity
        diverse_user_id = user_id
        if force_diverse:
            np.random.seed(user_id)
            offset = np.random.randint(1, 1000)
            diverse_user_id = user_id * offset
            logger.info(f"Using diverse user ID: {diverse_user_id}")
        
        books_df = get_books_df()
        ratings_df = get_ratings_df()
        model = get_recommender_model()
        
        logger.info(f"Data loaded - Books: {len(books_df)} rows, Ratings: {len(ratings_df)} rows, Model loaded: {model is not None}")
        
        if books_df.empty or ratings_df.empty:
            logger.error("Missing data for recommendations")
            return []
        
        # Get user's rated books
        user_ratings = ratings_df[ratings_df['user_id'] == user_id]
        logger.info(f"User has {len(user_ratings)} ratings")
        
        if user_ratings.empty:
            # User not found, return popular books
            logger.info(f"No ratings found for user {user_id}, returning popular books")
            popular_books = get_popular_books(limit=n)
            logger.info(f"Returning {len(popular_books)} popular books as fallback")
            return popular_books
        
        # Get rated book IDs
        rated_books = user_ratings['book_id'].tolist()
        logger.info(f"User has rated {len(rated_books)} books")
        
        # Try to use the model for recommendations if available
        if model is not None:
            try:
                # Check if the model has a predict method
                if hasattr(model, 'predict'):
                    logger.info("Model has predict method, using it for recommendations")
                    all_books = books_df['book_id'].unique()
                    logger.info(f"Total unique books: {len(all_books)}")
                    
                    unrated_books = [b for b in all_books if b not in rated_books]
                    logger.info(f"Unrated books count: {len(unrated_books)}")
                    
                    # Use model to predict ratings
                    predicted_ratings = []
                    prediction_count = 0
                    error_count = 0
                    
                    # Sample a subset of books for prediction if there are too many
                    books_to_predict = unrated_books[:100] if len(unrated_books) > 100 else unrated_books
                    logger.info(f"Predicting ratings for {len(books_to_predict)} books")
                    
                    for book_id in books_to_predict:
                        try:
                            # Use mapped ID for prediction if needed
                            pred_rating = model.predict(user_id, book_id)
                            predicted_ratings.append((book_id, pred_rating))
                            prediction_count += 1
                        except Exception as e:
                            error_count += 1
                            if error_count < 5:  # Log only first few errors
                                logger.error(f"Error predicting rating for user {user_id}, book {book_id}: {e}")
                    
                    logger.info(f"Successfully predicted {prediction_count} ratings with {error_count} errors")
                    
                    if predicted_ratings:
                        # Sort by predicted rating
                        predicted_ratings.sort(key=lambda x: x[1], reverse=True)
                        
                        # Get top n books
                        top_book_ids = [book_id for book_id, rating in predicted_ratings[:n]]
                        logger.info(f"Top {len(top_book_ids)} book IDs: {top_book_ids}")
                        
                        # Get book details
                        recommendations = []
                        for i, book_id in enumerate(top_book_ids):
                            book_details = get_book_by_id(book_id)
                            if book_details:
                                book_details['rank'] = i + 1
                                book_details['predicted_rating'] = next(rating for bid, rating in predicted_ratings if bid == book_id)
                                recommendations.append(book_details)
                        
                        if recommendations:
                            logger.info(f"Returning {len(recommendations)} model-based recommendations")
                            return recommendations
                        else:
                            logger.warning("No book details found for top recommendations")
                    else:
                        logger.warning("No predicted ratings available")
                else:
                    logger.warning("Model doesn't have predict method")
            except Exception as e:
                logger.error(f"Error using model for recommendations: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
        else:
            logger.warning("No model available for recommendations")
        
        # Fallback: recommend popular books
        logger.info("Using fallback recommendation approach based on popularity")
        popular_books = get_popular_books(limit=n*2)
        logger.info(f"Got {len(popular_books)} popular books")
        
        # Filter out books the user has already rated
        recommendations = [b for b in popular_books if b['book_id'] not in rated_books]
        logger.info(f"After filtering out rated books: {len(recommendations)} recommendations")
        
        # If the user has rated all popular books, recommend their highest-rated books
        if not recommendations:
            logger.info("User has rated all popular books - recommending their highest-rated books")
            
            # Get user's top rated books
            if 'rating' in user_ratings.columns:
                top_user_ratings = user_ratings.sort_values('rating', ascending=False).head(n)
                logger.info(f"Selecting from {len(top_user_ratings)} of user's highest-rated books")
                
                recommendations = []
                for _, row in top_user_ratings.iterrows():
                    book_id = row['book_id']
                    book_details = get_book_by_id(book_id)
                    if book_details:
                        book_details['user_rating'] = float(row['rating'])
                        recommendations.append(book_details)
            else:
                # If no rating column, just pick random books they've rated
                logger.info("No rating column found, selecting random books the user has rated")
                random_book_ids = np.random.choice(rated_books, min(n, len(rated_books)), replace=False)
                
                recommendations = []
                for book_id in random_book_ids:
                    book_details = get_book_by_id(book_id)
                    if book_details:
                        recommendations.append(book_details)
        
        # Rank the recommendations
        for i, book in enumerate(recommendations[:n]):
            book['rank'] = i + 1
        
        logger.info(f"Returning {min(n, len(recommendations))} recommendations")
        return recommendations[:n]
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []

def get_similar_books(book_id, n=5):
    """Get books similar to a given book using collaborative filtering"""
    try:
        books_df = get_books_df()
        ratings_df = get_ratings_df()
        model = get_recommender_model()
        
        if books_df.empty:
            logger.error("Books data not found")
            return []
        
        # Get the target book
        target_book = get_book_by_id(book_id)
        if not target_book:
            logger.error(f"Book ID {book_id} not found")
            return []
        
        # Check if ID mapping is needed
        mapping_path = os.path.join(current_dir, 'data', 'processed', 'book_id_mapping.csv')
        mapped_id = book_id
        
        if os.path.exists(mapping_path):
            try:
                mapping_df = pd.read_csv(mapping_path)
                if 'book_id' in mapping_df.columns and 'book_id_encoded' in mapping_df.columns:
                    mapping_row = mapping_df[mapping_df['book_id'] == book_id]
                    if not mapping_row.empty:
                        mapped_id = mapping_row.iloc[0]['book_id_encoded']
                        logger.info(f"Mapped book ID {book_id} to internal ID {mapped_id}")
            except Exception as e:
                logger.warning(f"Error in book ID mapping: {e}")
        
        # Try to use the model for similar books if available
        # Note: Here we're using the collaborative filtering model
        if model is not None:
            try:
                # Check if the model has a get_similar_items method (for item-based collaborative filtering)
                if hasattr(model, 'get_similar_items'):
                    logger.info(f"Using collaborative model to find similar books to {book_id}")
                    similar_item_ids = model.get_similar_items(book_id, n=n*2)
                    
                    # Get book details for the similar items
                    similar_books = []
                    for i, (similar_id, similarity) in enumerate(similar_item_ids[:n]):
                        book_details = get_book_by_id(similar_id)
                        if book_details:
                            book_details['similarity_score'] = float(similarity)
                            book_details['rank'] = i + 1
                            similar_books.append(book_details)
                    
                    if similar_books:
                        return similar_books
                
                # Alternative approach: if no get_similar_items, use the model's similarity matrix
                elif hasattr(model, 'item_similarity_matrix'):
                    logger.info(f"Using item similarity matrix to find similar books to {book_id}")
                    
                    # Get the index of the book in the similarity matrix
                    book_index = None
                    if hasattr(model, 'book_to_idx'):
                        book_index = model.book_to_idx.get(book_id)
                    
                    if book_index is not None:
                        # Get similar items from the similarity matrix
                        similarities = model.item_similarity_matrix[book_index]
                        similar_indices = np.argsort(-similarities)[:n*2]  # Get top n*2 similar items
                        
                        # Map indices back to book IDs
                        similar_book_ids = []
                        if hasattr(model, 'idx_to_book'):
                            similar_book_ids = [model.idx_to_book[idx] for idx in similar_indices 
                                               if idx != book_index and idx in model.idx_to_book]
                        
                        # Get book details
                        similar_books = []
                        for i, similar_id in enumerate(similar_book_ids[:n]):
                            book_details = get_book_by_id(similar_id)
                            if book_details:
                                book_details['similarity_score'] = float(similarities[similar_indices[i]])
                                book_details['rank'] = i + 1
                                similar_books.append(book_details)
                        
                        if similar_books:
                            return similar_books
                
                # Collaborative approach: generate vector representations using user ratings
                logger.info("Using collaborative approach with user rating patterns")
                book_ratings = ratings_df[ratings_df['book_id'] == book_id]
                
                if not book_ratings.empty:
                    # Find users who rated this book
                    users_who_rated = set(book_ratings['user_id'].unique())
                    
                    # Find other books these users rated highly
                    other_books = ratings_df[(ratings_df['user_id'].isin(users_who_rated)) & (ratings_df['book_id'] != book_id)]
                    
                    # Calculate a similarity score based on co-rated books
                    book_counts = other_books['book_id'].value_counts()
                    similar_book_ids = book_counts.index[:n*2].tolist()
                    
                    # Get book details
                    similar_books = []
                    for i, similar_id in enumerate(similar_book_ids[:n]):
                        book_details = get_book_by_id(similar_id)
                        if book_details:
                            # Calculate similarity as proportion of co-ratings
                            similarity = book_counts[similar_id] / len(users_who_rated)
                            book_details['similarity_score'] = float(similarity)
                            book_details['rank'] = i + 1
                            similar_books.append(book_details)
                    
                    if similar_books:
                        return similar_books
            
            except Exception as e:
                logger.error(f"Error using collaborative model for similar books: {e}")
                logger.error(f"Exception details: {str(e)}")
        
        # If we get here, no collaborative approach worked
        logger.warning("All collaborative filtering approaches failed. Returning popular books instead.")
        popular_books = get_popular_books(limit=n)
        
        # Add ranks and similarity scores to popular books
        for i, book in enumerate(popular_books):
            book['rank'] = i + 1
            book['similarity_score'] = 0.5  # Default similarity for popular books
        
        return popular_books
        
    except Exception as e:
        logger.error(f"Error finding similar books: {e}")
        return []

def get_popular_books(limit=6, randomize=True, seed=None):
    """Get popular books based on ratings"""
    try:
        global _POPULAR_BOOKS_CACHE
        
        # Create a cache key based on the parameters
        cache_key = f"popular_books_{limit}_{randomize}_{seed}"
        
        # Check if we have a valid cache entry
        now = datetime.now()
        if cache_key in _POPULAR_BOOKS_CACHE:
            cache_entry = _POPULAR_BOOKS_CACHE[cache_key]
            # Cache is valid for 1 hour
            if (now - cache_entry['timestamp']).total_seconds() < 3600:
                return cache_entry['data']
        
        books_df = get_books_df()
        
        if books_df.empty:
            logger.error("Books data not found")
            return []
        
        # Set seed for reproducibility
        seed_value = seed if seed is not None else int(datetime.now().timestamp())
        np.random.seed(seed_value)
        
        # Get popular books if we have rating data
        if 'average_rating' in books_df.columns and 'ratings_count' in books_df.columns:
            # Calculate a popularity score based on ratings and count
            books_df['popularity_score'] = books_df['average_rating'] * np.log1p(books_df['ratings_count'])
            # Sort by popularity score
            sorted_books = books_df.sort_values('popularity_score', ascending=False)
            
            # Get top books
            top_books = sorted_books.head(min(limit*10, len(sorted_books)))
            
            # Randomly select from top books
            if randomize and len(top_books) > limit:
                selected_books = top_books.sample(n=limit, random_state=seed_value)
            else:
                selected_books = top_books.head(limit)
        else:
            # No ratings data, just random selection
            selected_books = books_df.sample(n=min(limit, len(books_df)), random_state=seed_value)
        
        # Convert to list of dictionaries
        results = []
        for _, row in selected_books.iterrows():
            book_dict = row.to_dict()
            # Clean up NaN values
            for key, value in book_dict.items():
                if pd.isna(value):
                    if key in ['average_rating', 'ratings_count']:
                        book_dict[key] = 0
                    else:
                        book_dict[key] = ''
            results.append(book_dict)
        
        # Cache the results
        _POPULAR_BOOKS_CACHE[cache_key] = {
            'timestamp': now,
            'data': results
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Error getting popular books: {e}")
        return []
