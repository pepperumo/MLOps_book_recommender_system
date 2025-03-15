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
import time

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


def get_book_metadata(book_ids: List[int], data_dir: str = 'data', save_to_csv: bool = False) -> pd.DataFrame:
    """
    Get metadata for books given their IDs.
    
    Parameters
    ----------
    book_ids : List[int]
        List of book IDs to get metadata for
    data_dir : str
        Path to the data directory
    save_to_csv : bool
        Whether to save the metadata to a CSV file
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with book metadata for books that have complete metadata
    """
    logger.info(f"Retrieving metadata for {len(book_ids)} books")
    
    # Check for environment variable for data directory
    env_data_dir = os.environ.get('BOOK_RECOMMENDER_DATA_DIR')
    if env_data_dir:
        data_dir = env_data_dir
        logger.info(f"Using data directory from environment: {data_dir}")
    
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
        
        # Save metadata to CSV only if requested
        if save_to_csv:
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
    # Check for environment variable for data directory
    env_data_dir = os.environ.get('BOOK_RECOMMENDER_DATA_DIR')
    if env_data_dir:
        data_dir = env_data_dir
    
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


def get_popular_books(n: int = 10, data_dir: str = 'data', randomize: bool = False, seed: Optional[int] = None) -> List[int]:
    """
    Get the most popular books based on ratings count and average rating.
    
    Parameters
    ----------
    n : int
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
    # Check for environment variable for data directory
    env_data_dir = os.environ.get('BOOK_RECOMMENDER_DATA_DIR')
    if env_data_dir:
        data_dir = env_data_dir
    
    try:
        # Load book metadata
        books_path = os.path.join(data_dir, 'processed', 'books.csv')
        merged_path = os.path.join(data_dir, 'processed', 'merged.csv')
        
        # First try loading from merged dataset, which is preferred
        try:
            merged_df = pd.read_csv(merged_path)
            # Get unique books with their ratings
            books_df = merged_df.groupby('book_id').agg({
                'average_rating': 'first',
                'ratings_count': 'first',
                'title': 'first',
                'authors': 'first'
            }).reset_index()
            logger.info(f"Loaded book data from merged dataset with {len(books_df)} books")
        except (FileNotFoundError, pd.errors.EmptyDataError):
            logger.warning(f"Merged dataset not found or empty: {merged_path}")
            if os.path.exists(books_path):
                books_df = pd.read_csv(books_path)
                logger.info(f"Loaded book data from books.csv with {len(books_df)} books")
            else:
                raise FileNotFoundError(f"Books file not found: {books_path}")
        
        # Filter for books with at least 4.0 stars average rating
        books_df = books_df[books_df['average_rating'] >= 4.0]
        
        if len(books_df) == 0:
            logger.warning("No books found with 4.0+ star rating. Returning empty list.")
            return []
            
        # Sort by popularity (combination of ratings count and average rating)
        # We give more weight to ratings_count but also consider average_rating
        books_df['popularity_score'] = (
            books_df['ratings_count'] * (books_df['average_rating'] / 5.0)
        )
        
        books_df = books_df.sort_values('popularity_score', ascending=False)
        
        # Get a much larger pool of popular books to select from to increase diversity
        popular_pool_size = min(n * 20, len(books_df))
        popular_books = books_df.head(popular_pool_size)
        
        if randomize and len(popular_books) > n:
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
                top_count = max(n // 10, 1)  # Reduced from 1/5 to 1/10 of selections
                top_books = popular_books.head(top_count)
                
                # Offset the starting point for remaining books based on seed 
                # This creates different starting points for different users
                offset = seed % 15 if seed is not None else 0
                offset = min(offset * top_count, len(popular_books) - n)
                
                # Select remaining books randomly from the pool, starting from the offset
                remaining_pool = popular_books.iloc[top_count + offset:].sample(
                    n=min(n - top_count, len(popular_books) - top_count - offset),
                    random_state=seed
                )
                
                # Combine top books with random selections
                final_selection = pd.concat([top_books, remaining_pool])
            else:
                # Skip top books entirely for some users
                # This ensures maximum diversity since some users won't get any top books
                
                # Apply different offsets for different users to ensure variety
                offset = (seed % 8) * n if seed is not None else 0
                offset = min(offset, len(popular_books) - n * 2)
                
                # Select books randomly from the pool, but skip the very top books
                # Start selection from a point in the middle of the rankings
                final_selection = popular_books.iloc[offset:].sample(
                    n=min(n, len(popular_books) - offset),
                    random_state=seed
                )
            
            return final_selection['book_id'].tolist()
        else:
            # Non-randomized selection - just return top N books
            return popular_books['book_id'].head(n).tolist()
    
    except Exception as e:
        logger.error(f"Error getting popular books: {e}")
        logger.debug(traceback.format_exc())
        return []


def load_recommender_model(model_type: str = 'collaborative', models_dir: str = None) -> BaseRecommender:
    """
    Load recommender model from disk
    
    Args:
        model_type (str): Type of model to load ('collaborative')
        models_dir (str): Directory containing the model files (optional)
        
    Returns:
        model: Loaded recommender model
    """
    logger = logging.getLogger('predict_model')
    
    # Check for environment variable for models directory
    env_models_dir = os.environ.get('BOOK_RECOMMENDER_MODELS_DIR')
    if env_models_dir and models_dir is None:
        models_dir = env_models_dir
        logger.info(f"Using models directory from environment: {models_dir}")
    
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
                      n: int = 5, data_dir: str = 'data') -> pd.DataFrame:
    """
    Generate book recommendations for a specific user.
    
    Parameters
    ----------
    user_id : int
        ID of the user to generate recommendations for
    model_type : str
        Type of recommender to use ('collaborative')
    n : int
        Number of recommendations to generate
    data_dir : str
        Path to the data directory
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with recommendations and metadata
    """
    # Check for environment variable for data directory
    env_data_dir = os.environ.get('BOOK_RECOMMENDER_DATA_DIR')
    if env_data_dir:
        data_dir = env_data_dir
    
    try:
        # Load the appropriate model
        recommender = load_recommender_model(model_type)
        
        if recommender is None:
            logger.error(f"Failed to load {model_type} recommender model")
            # Fall back to popularity-based recommendations
            logger.info("Falling back to popularity-based recommendations")
            popular_book_ids = get_popular_books(n, data_dir, randomize=True, seed=user_id)
            return get_book_metadata(popular_book_ids, data_dir)
        
        # Fetch more books than needed to ensure diversity (5x instead of 2x)
        logger.info(f"Getting recommendations for user {user_id} using {model_type} model")
        fetch_count = n * 5
        book_ids = recommender.recommend_for_user(user_id, n_recommendations=fetch_count)
        
        # Handle case where we get no recommendations (cold start or user not in training data)
        if not book_ids:
            logger.warning(f"No recommendations found for user {user_id} using {model_type} model")
            # Fall back to popularity-based recommendations
            logger.info("Falling back to popularity-based recommendations")
            book_ids = get_popular_books(n, data_dir, randomize=True, seed=user_id)
        
        # --- Add Diversity Enhancement ---
        # If we have more recommendations than needed, add diversity based on user ID
        if len(book_ids) > n:
            # Use the user_id as a seed to create different selection patterns for different users
            # This ensures not all users get the same top recommendations
            np.random.seed(user_id)
            
            # Get unique user preference signal - each user gets a different pattern
            diversity_factor = (user_id % 10) / 10.0  # Value between 0 and 0.9
            
            # Determine how many top recommendations to keep vs diversity picks
            # Users with higher diversity_factor will get more diverse recommendations
            top_count = int(n * (1 - diversity_factor))
            diversity_count = n - top_count
            
            # Always include at least one top recommendation
            top_count = max(1, top_count)
            diversity_count = n - top_count
            
            # Select top recommendations
            top_picks = book_ids[:top_count] if top_count > 0 else []
            
            # Select diverse recommendations from the latter part of the list
            # Skip some obvious picks based on user_id to create different patterns
            skip_offset = (user_id % 5) * diversity_count
            start_index = top_count + skip_offset
            end_index = min(len(book_ids), start_index + diversity_count * 3)
            
            # If we don't have enough books after the skip, wrap around
            if end_index - start_index < diversity_count:
                start_index = top_count
                end_index = min(len(book_ids), start_index + diversity_count * 3)
            
            # Select diverse picks randomly from the available range
            diversity_candidates = book_ids[start_index:end_index]
            diversity_picks = list(np.random.choice(
                diversity_candidates, 
                size=min(diversity_count, len(diversity_candidates)), 
                replace=False
            ))
            
            # Combine picks ensuring no duplicates
            final_book_ids = list(dict.fromkeys(top_picks + diversity_picks))
            
            # If we still need more recommendations, add more from the original list
            if len(final_book_ids) < n:
                remaining_needed = n - len(final_book_ids)
                remaining_candidates = [b for b in book_ids if b not in final_book_ids]
                if remaining_candidates:
                    additional_picks = remaining_candidates[:remaining_needed]
                    final_book_ids.extend(additional_picks)
            
            # Trim to exact count needed
            book_ids = final_book_ids[:n]
            
            logger.info(f"Applied diversity enhancement for user {user_id}: "
                       f"top_count={top_count}, diversity_count={diversity_count}")
        
        # Map encoded book IDs to original book IDs
        book_ids = map_book_ids(book_ids, data_dir)
        
        # Get metadata for recommended books
        metadata_df = get_book_metadata(book_ids, data_dir)
        
        # Check if we have any metadata
        if metadata_df.empty:
            logger.warning(f"Could not find metadata for any of the {len(book_ids)} requested books")
            # If no metadata found, fall back to popularity-based recommendations
            logger.info("Falling back to popularity-based recommendations")
            popular_book_ids = get_popular_books(n, data_dir, randomize=True, seed=user_id)
            metadata_df = get_book_metadata(popular_book_ids, data_dir)
        
        # Add rank information
        metadata_df['rank'] = range(1, len(metadata_df) + 1)
        
        return metadata_df
        
    except Exception as e:
        logger.error(f"Error getting recommendations for user {user_id}: {str(e)}")
        logger.info("Attempting to fall back to popularity-based recommendations after error")
        # Fall back to popularity-based recommendations
        popular_book_ids = get_popular_books(n, data_dir, randomize=True, seed=user_id)
        return get_book_metadata(popular_book_ids, data_dir)


def recommend_similar_books(book_id: int, model_type: str = 'collaborative',
                          n: int = 5, data_dir: str = 'data', save_results: bool = False) -> pd.DataFrame:
    """
    Generate similar book recommendations for a specific book.
    
    Parameters
    ----------
    book_id : int
        ID of the book to find similar books for
    model_type : str
        Type of recommender to use ('collaborative')
    n : int
        Number of recommendations to generate
    data_dir : str
        Path to the data directory
    save_results : bool
        Whether to save the results to a CSV file
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with similar book metadata
    """
    # Only support collaborative model
    if model_type != 'collaborative':
        logger.warning(f"Model type '{model_type}' not supported. Using collaborative model instead.")
        model_type = 'collaborative'
        
    # Check for environment variable for data directory
    env_data_dir = os.environ.get('BOOK_RECOMMENDER_DATA_DIR')
    if env_data_dir:
        data_dir = env_data_dir
    
    # Load the recommender model
    model = load_recommender_model(model_type=model_type)
    if model is None:
        logger.error(f"Failed to load {model_type} recommender model")
        return pd.DataFrame()
    
    try:
        # Get similar book recommendations
        logger.info(f"Finding books similar to book ID {book_id}")
        
        # Set a timeout for this operation to avoid hanging
        start_time = time.time()
        max_execution_time = 5  # Maximum 5 seconds for recommendation
        
        try:
            # Call the model with the 'n' parameter
            similar_book_ids = model.recommend_similar_books(book_id, n=n)
            
            # Check if we're taking too long
            if time.time() - start_time > max_execution_time:
                logger.warning(f"Similar books recommendation taking too long (>{max_execution_time}s), using fallback")
                raise TimeoutError("Recommendation operation timed out")
                
        except Exception as e:
            logger.error(f"Error getting similar books: {e}")
            similar_book_ids = []
        
        # If no similar books found or timeout occurred, provide a fallback 
        if not similar_book_ids:
            logger.warning(f"No similar books found for book ID {book_id}, using fallback approach")
            
            # Get book metadata to understand its genres or categories
            source_book_df = get_book_metadata([book_id], data_dir=data_dir)
            
            if not source_book_df.empty:
                # Get some popular books as fallback but exclude the current book
                fallback_book_ids = get_popular_books(n * 2, data_dir, randomize=True, seed=book_id)
                fallback_book_ids = [b for b in fallback_book_ids if b != book_id][:n]
                
                logger.info(f"Using {len(fallback_book_ids)} popular books as fallbacks for similar books")
                return get_book_metadata(fallback_book_ids, data_dir=data_dir)
            
            return pd.DataFrame()
            
        # Get metadata for the similar books
        logger.info(f"Found {len(similar_book_ids)} similar books")
        similar_books_df = get_book_metadata(similar_book_ids, data_dir=data_dir)
        
        if similar_books_df.empty:
            logger.warning("Could not retrieve metadata for similar books, using fallback")
            # Get popular books as fallback
            fallback_book_ids = get_popular_books(n, data_dir, randomize=True, seed=book_id)
            fallback_book_ids = [b for b in fallback_book_ids if b != book_id][:n]
            return get_book_metadata(fallback_book_ids, data_dir=data_dir)
            
        # Add source book information
        source_book_df = get_book_metadata([book_id], data_dir=data_dir)
        if not source_book_df.empty:
            logger.info(f"Source book: {source_book_df.iloc[0]['title']} by {source_book_df.iloc[0]['authors']}")
        
        if save_results:
            output_dir = os.path.join(data_dir, 'processed')
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            similar_books_file = os.path.join(output_dir, f'similar_books_{book_id}_{timestamp}.csv')
            similar_books_df.to_csv(similar_books_file, index=False)
            logger.info(f"Saved similar books to {similar_books_file}")
        
        return similar_books_df
        
    except Exception as e:
        logger.error(f"Error generating similar book recommendations: {e}")
        logger.info("Falling back to popularity-based recommendations")
        
        # Return popular books as a fallback, but exclude the current book
        fallback_book_ids = get_popular_books(n, data_dir, randomize=True, seed=book_id)
        # Filter out the source book from recommendations
        fallback_book_ids = [b for b in fallback_book_ids if b != book_id][:n]
        return get_book_metadata(fallback_book_ids, data_dir=data_dir)


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
                       help='Run a demonstration of collaborative filtering')
    
    # Parse arguments
    parsed_args = parser.parse_args(args)
    
    try:
        # Set the model directory
        model_dir = parsed_args.model_dir
        
        # If no arguments were provided, run a demo of collaborative filtering
        if len(sys.argv) == 1 or parsed_args.demo:
            logger.info("Running demonstration of collaborative filtering recommendations")
            
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
                    n=parsed_args.num,
                    data_dir=parsed_args.data_dir
                )
                print_recommendations(recommendations_df, f"Collaborative Filtering Recommendations for User {user_id}:")
                
                # Get a random book ID to demonstrate similar books
                book_ids = df['book_id'].unique()
                book_id = np.random.choice(book_ids)
                logger.info(f"Selected random book ID: {book_id} for similar books demonstration")
                
                # Demo collaborative similar books
                logger.info("Finding similar books using Collaborative Filtering:")
                similar_books_df = recommend_similar_books(
                    book_id=book_id,
                    model_type='collaborative',
                    n=parsed_args.num,
                    data_dir=parsed_args.data_dir,
                    save_results=False
                )
                print_recommendations(similar_books_df, f"Similar Books to Book ID {book_id} (Collaborative):")
                
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
                n=parsed_args.num,
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
                n=parsed_args.num,
                data_dir=parsed_args.data_dir,
                save_results=False
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