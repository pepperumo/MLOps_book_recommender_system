import os
import pickle
import pandas as pd
import argparse
import numpy as np
import sys
import logging
import traceback
from datetime import datetime
from typing import List, Dict, Union, Optional, Tuple, Any
from pathlib import Path

# Set up logging
log_dir = os.path.join('logs')
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(log_dir, f'predict_model_{timestamp}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('predict_model')

# Import the base classes and functions from our recommender modules
try:
    from src.models.model_utils import BaseRecommender, load_data
    from src.models.train_model import CollaborativeRecommender
except ImportError:
    try:
        from models.model_utils import BaseRecommender, load_data
        from models.train_model import CollaborativeRecommender
    except ImportError:
        import sys
        import os
        # Add the parent directory to the path to ensure we can import the module
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.append(parent_dir)
        try:
            from models.model_utils import BaseRecommender, load_data
            from models.train_model import CollaborativeRecommender
            logger.info("Imported from models directory after adding parent dir to path")
        except ImportError:
            logger.error("Failed to import necessary modules. Please check your installation.")
            sys.exit(1)


def get_book_metadata(book_ids: List[int], data_dir: str = 'data') -> pd.DataFrame:
    """
    Get metadata for books given their IDs.
    
    Parameters
    ----------
    book_ids : List[int]
        List of book IDs to get metadata for
    data_dir : str
        Path to the data directory
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with book metadata for books that have complete metadata
    """
    logger.info(f"Retrieving metadata for {len(book_ids)} books")
    
    result_df = pd.DataFrame()
    
    # First try to get metadata from merged_train.csv
    train_path = os.path.join(data_dir, 'processed', 'merged_train.csv')
    if os.path.exists(train_path):
        try:
            df = pd.read_csv(train_path)
            logger.info(f"Loaded merged_train.csv with shape {df.shape}")
            logger.debug(f"Column dtypes: {df.dtypes}")
            
            # Check that book_id column exists
            if 'book_id' in df.columns:
                # Filter to the books we want
                result_df = df[df['book_id'].isin(book_ids)].drop_duplicates(subset=['book_id'])
                logger.info(f"Found {len(result_df)} books in merged_train.csv")
                
                # Ensure we have required columns (title, author)
                required_cols = ['title', 'authors']
                
                # Check for column variations
                if 'authors' not in result_df.columns and 'author' in result_df.columns:
                    result_df = result_df.rename(columns={'author': 'authors'})
                    
                if 'title' not in result_df.columns and 'book_title' in result_df.columns:
                    result_df = result_df.rename(columns={'book_title': 'title'})
                
                # Check if we have all required columns
                missing_cols = [col for col in required_cols if col not in result_df.columns]
                if missing_cols:
                    logger.warning(f"Missing columns in merged_train.csv: {missing_cols}")
                    # If critical columns are missing from dataset, clear result_df
                    # so we can try with books.csv
                    result_df = pd.DataFrame()
        except Exception as e:
            logger.error(f"Error reading merged_train.csv: {e}")
            logger.debug(traceback.format_exc())
            result_df = pd.DataFrame()
    
    # If we couldn't get all metadata from merged_train.csv, try the original books.csv
    if result_df.empty:
        books_path = os.path.join(data_dir, 'raw', 'books.csv')
        if os.path.exists(books_path):
            try:
                df = pd.read_csv(books_path)
                logger.info(f"Loaded books.csv with shape {df.shape}")
                logger.debug(f"Column dtypes: {df.dtypes}")
                
                # Check that book_id column exists
                if 'book_id' in df.columns:
                    # Filter to the books we want
                    result_df = df[df['book_id'].isin(book_ids)].drop_duplicates(subset=['book_id'])
                    logger.info(f"Found {len(result_df)} books in books.csv")
                    
                    # Handle column variations
                    if 'authors' not in result_df.columns and 'author' in result_df.columns:
                        result_df = result_df.rename(columns={'author': 'authors'})
                        
                    if 'title' not in result_df.columns and 'book_title' in result_df.columns:
                        result_df = result_df.rename(columns={'book_title': 'title'})
            except Exception as e:
                logger.error(f"Error reading books.csv: {e}")
                logger.debug(traceback.format_exc())
    
    # Filter out books that don't have both title and authors
    if not result_df.empty:
        # Check for null/missing values in essential columns
        result_df = result_df.dropna(subset=['title', 'authors'])
        
        # Further filter out books with "Unknown" placeholders that might have slipped through
        result_df = result_df[~((result_df['title'].str.contains('Unknown')) & 
                                (result_df['authors'] == 'Unknown'))]
        
        logger.info(f"After filtering for complete metadata, found {len(result_df)} valid books")
        
        # If we found fewer books than requested, log which ones are missing
        if len(result_df) < len(book_ids):
            missing_ids = set(book_ids) - set(result_df['book_id'])
            logger.warning(f"Could not find metadata for {len(missing_ids)} books: {list(missing_ids)}")
        
        # Save metadata to CSV for analysis
        output_dir = os.path.join(data_dir, 'processed')
        os.makedirs(output_dir, exist_ok=True)
        metadata_file = os.path.join(output_dir, f'book_metadata_{timestamp}.csv')
        result_df.to_csv(metadata_file, index=False)
        logger.info(f"Saved book metadata to {metadata_file}")
    else:
        logger.warning(f"Could not find metadata for any of the {len(book_ids)} requested books")
    
    return result_df


def load_book_id_mapping(data_dir: str = 'data') -> Dict[int, int]:
    """
    Load book ID mapping between original and encoded IDs.
    
    Parameters
    ----------
    data_dir : str
        Base data directory
        
    Returns
    -------
    Dict[int, int]
        Dictionary mapping encoded IDs to original IDs
    """
    mapping_path = os.path.join(data_dir, 'processed', 'book_id_mapping.csv')
    
    if not os.path.exists(mapping_path):
        logger.warning(f"Book ID mapping file not found at {mapping_path}")
        return {}
    
    try:
        mapping_df = pd.read_csv(mapping_path)
        logger.info(f"Loaded book_id_mapping.csv with shape {mapping_df.shape}")
        logger.debug(f"Column dtypes: {mapping_df.dtypes}")
        
        if 'book_id' in mapping_df.columns and 'book_id_encoded' in mapping_df.columns:
            # Create mapping from encoded ID to original ID
            encoded_to_original = {int(row['book_id_encoded']): int(row['book_id']) 
                                  for _, row in mapping_df.iterrows()}
            logger.info(f"Loaded {len(encoded_to_original)} book ID mappings")
            return encoded_to_original
        else:
            logger.warning(f"Book ID mapping file is missing required columns")
            return {}
    except Exception as e:
        logger.error(f"Error loading book ID mapping: {e}")
        logger.debug(traceback.format_exc())
        return {}


def get_popular_books(num_books: int = 10, data_dir: str = 'data', randomize: bool = False, seed: Optional[int] = None) -> List[int]:
    """
    Get the most popular books based on ratings count and average rating.
    
    Parameters
    ----------
    num_books : int
        Number of popular books to return
    data_dir : str
        Path to the data directory
    randomize : bool
        If True, adds some randomization to the popular books selection
    seed : Optional[int]
        Random seed for reproducible randomization
        
    Returns
    -------
    List[int]
        List of book IDs for popular books
    """
    try:
        # Load book metadata
        books_path = os.path.join(data_dir, 'processed', 'books.csv')
        if not os.path.exists(books_path):
            logger.error(f"Books file not found: {books_path}")
            return []
            
        books_df = pd.read_csv(books_path)
        
        # Calculate a popularity score based on ratings count and average rating
        # This creates a balance between highly-rated but less-known books and
        # very popular books with average ratings
        if 'ratings_count' in books_df.columns and 'average_rating' in books_df.columns:
            books_df['popularity_score'] = books_df['ratings_count'] * books_df['average_rating']
            
            # Get a much larger pool of popular books to select from to increase diversity
            popular_pool_size = min(num_books * 20, len(books_df))
            popular_books = books_df.sort_values('popularity_score', ascending=False).head(popular_pool_size)
            
            if randomize and len(popular_books) > num_books:
                # Set random seed if provided
                if seed is not None:
                    np.random.seed(seed)
                
                # Create a list of books to potentially exclude for certain users
                # These are books that might otherwise appear in everyone's recommendations
                top_books_ids = popular_books.head(5)['book_id'].tolist()
                
                # Determine which books to exclude for this specific user
                # Different users will have different exclusions based on their user_id (seed)
                if seed is not None:
                    np.random.seed(seed)
                    # Each user will exclude at least 2 of the top 5 books
                    num_to_exclude = np.random.randint(2, 5)
                    exclude_indices = np.random.choice(5, size=num_to_exclude, replace=False)
                    books_to_exclude = [top_books_ids[i] for i in exclude_indices]
                else:
                    books_to_exclude = []
                
                # Filter out the excluded books for this user
                if books_to_exclude:
                    popular_books = popular_books[~popular_books['book_id'].isin(books_to_exclude)]
                
                # Determine if this user should get any top books at all
                # Use the seed (user_id) to create completely different recommendation patterns
                # This ensures not all users get the same top books
                np.random.seed(seed if seed is not None else 42)
                include_top_books = np.random.random() > 0.4  # 60% chance to include top books
                
                if include_top_books:
                    # Only select a very small number of top books deterministically
                    top_count = max(num_books // 10, 1)  # Reduced from 1/5 to 1/10 of selections
                    top_books = popular_books.head(top_count)
                    
                    # Offset the starting point for remaining books based on seed 
                    # This creates different starting points for different users
                    offset = seed % 15 if seed is not None else 0
                    offset = min(offset * top_count, len(popular_books) - num_books)
                    
                    # Select remaining books randomly from the pool, starting from the offset
                    remaining_pool = popular_books.iloc[top_count + offset:].sample(
                        n=min(num_books - top_count, len(popular_books) - top_count - offset),
                        random_state=seed
                    )
                    
                    # Combine top books with random selections
                    final_selection = pd.concat([top_books, remaining_pool])
                else:
                    # Skip top books entirely for some users
                    # This ensures maximum diversity since some users won't get any top books
                    
                    # Apply different offsets for different users to ensure variety
                    offset = (seed % 8) * num_books if seed is not None else 0
                    offset = min(offset, len(popular_books) - num_books * 2)
                    
                    # Select books randomly from the pool, but skip the very top books
                    # Start selection from a point in the middle of the rankings
                    final_selection = popular_books.iloc[offset:].sample(
                        n=min(num_books, len(popular_books) - offset),
                        random_state=seed
                    )
                
                return final_selection['book_id'].tolist()
            else:
                # Non-randomized selection - just return top N books
                return popular_books['book_id'].head(num_books).tolist()
        else:
            logger.warning("Ratings count or average rating columns not found in books data")
            # Fallback: just return the first N books
            return books_df['book_id'].head(num_books).tolist()
    
    except Exception as e:
        logger.error(f"Error getting popular books: {e}")
        logger.debug(traceback.format_exc())
        return []


def load_recommender_model(model_type: str = 'collaborative', models_dir: str = None) -> BaseRecommender:
    """
    Load recommender model from disk
    
    Args:
        model_type (str): Type of model to load ('collaborative' or 'content')
        models_dir (str): Directory containing the model files (optional)
        
    Returns:
        model: Loaded recommender model
    """
    logger = logging.getLogger('predict_model')
    
    # Determine model directory
    if models_dir is None:
        # Try to find the project root - look for src/models directory structure
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up to src directory
        src_dir = os.path.dirname(current_dir)
        # Go up to project root
        project_root = os.path.dirname(src_dir)
        models_dir = os.path.join(project_root, 'models')
    
    logger.info(f"Loading {model_type} recommender from {models_dir}")
    
    try:
        # Check if models directory exists
        if not os.path.exists(models_dir):
            logger.error(f"Models directory not found: {models_dir}")
            os.makedirs(models_dir, exist_ok=True)
            logger.info(f"Created models directory: {models_dir}")
            raise FileNotFoundError(f"Models directory not found: {models_dir}")
            
        # Check for model files
        model_files = [f for f in os.listdir(models_dir) 
                      if f.startswith(model_type) and f.endswith('.pkl') 
                      and os.path.isfile(os.path.join(models_dir, f))]
        
        if not model_files:
            logger.error(f"No {model_type} model files found in {models_dir}")
            raise FileNotFoundError(f"No {model_type} model files found in {models_dir}")
        
        # Load the newest model file
        model_files.sort(reverse=True)  # Sort by name in descending order
        model_path = os.path.join(models_dir, model_files[0])
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        logger.info(f"Successfully loaded {model_type} recommender model")
        return model
        
    except Exception as e:
        logger.error(f"Error loading {model_type} recommender model: {str(e)}")
        # Return None to indicate failure
        return None


def recommend_for_user(user_id: int, model_type: str = 'collaborative', 
                      num_recommendations: int = 5, data_dir: str = 'data') -> pd.DataFrame:
    """
    Generate book recommendations for a specific user.
    
    Parameters
    ----------
    user_id : int
        ID of the user to generate recommendations for
    model_type : str
        Type of recommender to use (only 'collaborative' is supported)
    num_recommendations : int
        Number of recommendations to generate
    data_dir : str
        Path to the data directory
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with recommendations and metadata
    """
    try:
        # Load the appropriate model
        recommender = load_recommender_model(model_type)
        
        if recommender is None:
            logger.error(f"Failed to load {model_type} recommender model")
            # Fall back to popularity-based recommendations
            logger.info("Falling back to popularity-based recommendations")
            # Use user_id as seed for randomization to diversify recommendations between users
            popular_book_ids = get_popular_books(num_recommendations * 3, data_dir, randomize=True, seed=user_id)
            return get_book_metadata(popular_book_ids, data_dir)
        
        # Get a larger pool of recommendations for the user
        # Fetch more books than needed to ensure diversity (5x instead of 2x)
        logger.info(f"Getting recommendations for user {user_id} using {model_type} model")
        fetch_count = num_recommendations * 5
        book_ids = recommender.recommend_for_user(user_id, n_recommendations=fetch_count)
        
        # Handle case where we get no recommendations (cold start or user not in training data)
        if not book_ids:
            logger.warning(f"No recommendations found for user {user_id}. Using popularity-based recommendations.")
            # Use user_id as seed for randomization to diversify recommendations between users
            book_ids = get_popular_books(fetch_count, data_dir, randomize=True, seed=user_id)
            
            # Still no recommendations? Return empty DataFrame
            if not book_ids:
                logger.error("Failed to generate popularity-based recommendations")
                return pd.DataFrame()
        
        # Mix in some randomness to diversify recommendations across different users
        # Shuffle the recommendations after the top picks to balance quality and diversity
        if len(book_ids) > num_recommendations:
            # Keep the top recommendations (assumed to be most relevant)
            top_count = min(num_recommendations, len(book_ids))
            top_picks = book_ids[:top_count]
            
            # Shuffle the remaining recommendations with a user-specific seed
            remaining = book_ids[top_count:]
            np.random.seed(user_id)  # Use user_id as seed for reproducible randomization
            np.random.shuffle(remaining)
            
            # Combine the top picks with some of the shuffled remaining picks
            random_picks_count = min(num_recommendations * 3, len(remaining))
            book_ids = top_picks + remaining[:random_picks_count]
        
        # Check if we need to map encoded IDs to original IDs
        encoded_to_original = load_book_id_mapping(data_dir)
        if encoded_to_original:
            # Map encoded IDs back to original IDs
            original_book_ids = [encoded_to_original.get(book_id, book_id) for book_id in book_ids]
            logger.info(f"Mapped {len(book_ids)} encoded book IDs to original IDs")
            book_ids = original_book_ids
        
        # Get metadata for the recommended books
        recommendations_df = get_book_metadata(book_ids, data_dir)
        
        # If no books with complete metadata were found, try popularity-based recommendations
        if recommendations_df.empty:
            logger.warning(f"No books with complete metadata found for user {user_id}. Using popularity-based recommendations.")
            popular_book_ids = get_popular_books(num_recommendations * 3, data_dir, randomize=True, seed=user_id)
            recommendations_df = get_book_metadata(popular_book_ids, data_dir)
            
            # If still empty, return empty DataFrame
            if recommendations_df.empty:
                logger.error("Failed to generate popularity-based recommendations with metadata")
                return pd.DataFrame()
        
        # Add rank column for sorting (lower rank = higher recommendation)
        recommendations_df['rank'] = recommendations_df['book_id'].apply(
            lambda x: book_ids.index(x) + 1 if x in book_ids else len(book_ids) + 1
        )
        
        # Sort by rank
        recommendations_df = recommendations_df.sort_values('rank')
        
        # If we don't have enough recommendations yet, get more books
        if len(recommendations_df) < num_recommendations:
            # Calculate how many more books we need
            additional_count = num_recommendations - len(recommendations_df)
            logger.info(f"Need {additional_count} more recommendations to reach requested {num_recommendations}")
            
            # Try to get popular books with randomization based on user_id
            popular_book_ids = get_popular_books(additional_count * 3, data_dir, randomize=True, seed=user_id)
            
            # Filter out books we've already recommended
            existing_ids = set(recommendations_df['book_id'].tolist())
            new_popular_ids = [bid for bid in popular_book_ids if bid not in existing_ids]
            
            # Get metadata for these additional books
            if new_popular_ids:
                additional_df = get_book_metadata(new_popular_ids, data_dir)
                
                if not additional_df.empty:
                    # Add rank starting from where we left off
                    max_rank = recommendations_df['rank'].max() if not recommendations_df.empty else 0
                    additional_df['rank'] = range(max_rank + 1, max_rank + 1 + len(additional_df))
                    
                    # Combine dataframes
                    recommendations_df = pd.concat([recommendations_df, additional_df], ignore_index=True)
                    recommendations_df = recommendations_df.sort_values('rank')
        
        # Ensure we don't return more than requested
        if len(recommendations_df) > num_recommendations:
            recommendations_df = recommendations_df.head(num_recommendations)
        
        # Ensure rank is continuous from 1 to n
        recommendations_df['rank'] = range(1, len(recommendations_df) + 1)
        
        # If we still don't have enough recommendations, log a warning
        if len(recommendations_df) < num_recommendations:
            logger.warning(f"Could only generate {len(recommendations_df)} recommendations for user {user_id}")
        
        return recommendations_df
        
    except Exception as e:
        logger.error(f"Error getting recommendations for user {user_id}: {e}")
        logger.debug(traceback.format_exc())
        # Fall back to popularity-based recommendations
        try:
            logger.info("Attempting to fall back to popularity-based recommendations after error")
            popular_book_ids = get_popular_books(num_recommendations * 2, data_dir, randomize=True, seed=user_id)
            return get_book_metadata(popular_book_ids, data_dir)
        except Exception as e2:
            logger.error(f"Error getting popularity-based recommendations: {e2}")
            return pd.DataFrame()


def recommend_similar_books(book_id: int, model_type: str = 'collaborative',
                           num_recommendations: int = 5, data_dir: str = 'data') -> pd.DataFrame:
    """
    Generate similar book recommendations for a specific book.
    
    Parameters
    ----------
    book_id : int
        ID of the book to find similar books for
    model_type : str
        Type of recommender to use (only 'collaborative' is supported)
    num_recommendations : int
        Number of recommendations to generate
    data_dir : str
        Path to the data directory
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with similar book metadata
    """
    # Only support collaborative model
    if model_type != 'collaborative':
        logger.warning(f"Model type '{model_type}' not supported. Using collaborative model instead.")
        model_type = 'collaborative'
        
    # Load the recommender model
    model = load_recommender_model(model_type=model_type)
    if model is None:
        logger.error(f"Failed to load {model_type} recommender model")
        return pd.DataFrame()
    
    try:
        # Get similar book recommendations
        logger.info(f"Finding books similar to book ID {book_id}")
        similar_book_ids = model.recommend_similar_books(book_id, n=num_recommendations)
        
        if not similar_book_ids:
            logger.warning(f"No similar books found for book ID {book_id}")
            return pd.DataFrame()
            
        # Get metadata for the similar books
        logger.info(f"Found {len(similar_book_ids)} similar books")
        similar_books_df = get_book_metadata(similar_book_ids, data_dir=data_dir)
        
        if similar_books_df.empty:
            logger.warning("Could not retrieve metadata for similar books")
            return pd.DataFrame()
            
        # Add source book information
        source_book_df = get_book_metadata([book_id], data_dir=data_dir)
        if not source_book_df.empty:
            logger.info(f"Source book: {source_book_df.iloc[0]['title']} by {source_book_df.iloc[0]['authors']}")
        
        return similar_books_df
        
    except Exception as e:
        logger.error(f"Error generating similar book recommendations: {e}")
        logger.debug(traceback.format_exc())
        return pd.DataFrame()


def print_recommendations(recommendations_df: pd.DataFrame, header: str = "Recommendations:"):
    """
    Print formatted recommendations with book titles and authors.
    
    Parameters
    ----------
    recommendations_df : pandas.DataFrame
        DataFrame with book metadata
    header : str
        Header text to display before recommendations
    """
    if recommendations_df.empty:
        print("No recommendations found.")
        return
    
    print(f"\n{header}")
    print("-" * 80)
    
    for i, row in recommendations_df.iterrows():
        rank = row.get('rank', i)
        if rank == -1:
            print(f"Source Book: {row['title']} by {row['authors']}")
            print("-" * 80)
        else:
            print(f"{rank + 1}. {row['title']} by {row['authors']}")
    
    print("-" * 80)


def main(args: Optional[List[str]] = None) -> int:
    """
    Main function to run the predict module from command line arguments.
    
    Parameters
    ----------
    args : Optional[List[str]]
        Command line arguments
        
    Returns
    -------
    int
        Exit code
    """
    parser = argparse.ArgumentParser(description='Generate book recommendations')
    
    # Define command line arguments
    parser.add_argument('--user', type=int, help='User ID to generate recommendations for')
    parser.add_argument('--book', type=int, help='Book ID to find similar books for')
    parser.add_argument('--model-type', type=str, default='collaborative', 
                       choices=['collaborative'],
                       help='Recommender model type to use')
    parser.add_argument('--num', type=int, default=5, 
                       help='Number of recommendations to generate')
    parser.add_argument('--model-dir', type=str, default='models', 
                       help='Directory containing trained models')
    parser.add_argument('--data-dir', type=str, default='data', 
                       help='Data directory path')
    parser.add_argument('--demo', action='store_true',
                       help='Run a demonstration of all three model types')
    
    # Parse arguments
    parsed_args = parser.parse_args(args)
    
    try:
        # Set the model directory
        model_dir = parsed_args.model_dir
        
        # If no arguments were provided, run a demo of all three models
        if len(sys.argv) == 1 or parsed_args.demo:
            logger.info("Running demonstration of all three recommendation models")
            
            # Load the processed data to get a valid user ID
            try:
                df = pd.read_csv(os.path.join(parsed_args.data_dir, 'processed', 'merged_train.csv'), encoding='utf-8')
                # Get a random user ID from the data
                user_ids = df['user_id'].unique()
                user_id = np.random.choice(user_ids)
                logger.info(f"Selected random user ID: {user_id} for demonstration")
                
                # Demo collaborative filtering
                logger.info("Collaborative Filtering Recommendations:")
                recommendations_df = recommend_for_user(
                    user_id=user_id,
                    model_type='collaborative',
                    num_recommendations=parsed_args.num,
                    data_dir=parsed_args.data_dir
                )
                print_recommendations(recommendations_df, f"Collaborative Filtering Recommendations for User {user_id}:")
                
                # Demo content-based filtering
                logger.info("Content-Based Filtering Recommendations:")
                recommendations_df = recommend_for_user(
                    user_id=user_id,
                    model_type='content',
                    num_recommendations=parsed_args.num,
                    data_dir=parsed_args.data_dir
                )
                print_recommendations(recommendations_df, f"Content-Based Filtering Recommendations for User {user_id}:")
                
                # Demo hybrid recommendations
                logger.info("Hybrid Recommendations:")
                recommendations_df = recommend_for_user(
                    user_id=user_id,
                    model_type='hybrid',
                    num_recommendations=parsed_args.num,
                    data_dir=parsed_args.data_dir
                )
                print_recommendations(recommendations_df, f"Hybrid Recommendations for User {user_id}:")
                
                # Get a random book ID to demonstrate similar books
                book_ids = df['book_id'].unique()
                book_id = np.random.choice(book_ids)
                logger.info(f"Selected random book ID: {book_id} for similar books demonstration")
                
                # Demo content-based similar books
                logger.info("Finding similar books using Content-Based Filtering:")
                similar_books_df = recommend_similar_books(
                    book_id=book_id,
                    model_type='content',
                    num_recommendations=parsed_args.num,
                    data_dir=parsed_args.data_dir
                )
                print_recommendations(similar_books_df, f"Similar Books to Book ID {book_id} (Content-Based):")
                
                return 0
            
            except Exception as e:
                logger.error(f"Error running demo: {e}")
                logger.debug(traceback.format_exc())
        
        # Handle user recommendations
        if parsed_args.user is not None:
            user_id = parsed_args.user
            logger.info(f"Generating recommendations for user {user_id} using {parsed_args.model_type} model")
            
            # Generate recommendations
            recommendations_df = recommend_for_user(
                user_id=user_id,
                model_type=parsed_args.model_type,
                num_recommendations=parsed_args.num,
                data_dir=parsed_args.data_dir
            )
            
            # Print recommendations
            print_recommendations(recommendations_df, f"Book Recommendations for User {user_id}:")
            
            return 0
            
        # Handle similar book recommendations
        elif parsed_args.book is not None:
            book_id = parsed_args.book
            logger.info(f"Finding similar books for book {book_id} using {parsed_args.model_type} model")
            
            # Find similar books
            similar_books_df = recommend_similar_books(
                book_id=book_id,
                model_type=parsed_args.model_type,
                num_recommendations=parsed_args.num,
                data_dir=parsed_args.data_dir
            )
            
            # Print similar books
            print_recommendations(similar_books_df, f"Similar Books to Book ID {book_id}:")
            
            return 0
            
        else:
            logger.error("Must specify either --user or --book")
            parser.print_help()
            return 1
            
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
