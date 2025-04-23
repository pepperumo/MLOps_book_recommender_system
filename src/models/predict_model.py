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
import functools

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

# Cache for commonly used data to avoid repeated loading
_MODEL_CACHE = {}
_MAPPING_CACHE = {}
_METADATA_CACHE = {}

# Import the base classes and functions from our recommender modules
try:
    from src.models.model_utils import BaseRecommender
    from src.models.train_model import CollaborativeRecommender
except ImportError:
    try:
        from models.model_utils import BaseRecommender
        from models.train_model import CollaborativeRecommender
    except ImportError:
        # Add the parent directory to the path
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.append(parent_dir)
        try:
            from models.model_utils import BaseRecommender
            from models.train_model import CollaborativeRecommender
            logger.info("Imported from models directory after adding parent dir to path")
        except ImportError:
            logger.error("Failed to import necessary modules. Please check your installation.")
            sys.exit(1)


def get_data_dir(data_dir: str = 'data') -> str:
    """Get the data directory, checking for environment variables first."""
    env_data_dir = os.environ.get('BOOK_RECOMMENDER_DATA_DIR')
    return env_data_dir if env_data_dir else data_dir


@functools.lru_cache(maxsize=32)
def get_book_metadata(book_ids_tuple: Tuple[int, ...], data_dir: str = 'data') -> pd.DataFrame:
    """
    Get metadata for books given their IDs from merged_train.csv.
    Takes a tuple of book_ids for caching purposes.
    
    Parameters
    ----------
    book_ids_tuple : Tuple[int, ...]
        Tuple of book IDs to get metadata for
    data_dir : str
        Path to the data directory
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with book metadata for books that have complete metadata
    """
    book_ids = list(book_ids_tuple)
    logger.info(f"Retrieving metadata for {len(book_ids)} books")
    data_dir = get_data_dir(data_dir)
    
    # Check if we have this metadata in the cache
    cache_key = f"{data_dir}_{'-'.join(map(str, sorted(book_ids)))}"
    if cache_key in _METADATA_CACHE:
        logger.info(f"Using cached metadata for {len(book_ids)} books")
        return _METADATA_CACHE[cache_key]
    
    # Use merged_train.csv directly
    merged_path = os.path.join(data_dir, 'processed', 'merged_train.csv')
    
    if not os.path.exists(merged_path):
        logger.error(f"merged_train.csv not found at {merged_path}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(merged_path)
        logger.info(f"Loaded merged_train.csv with shape {df.shape}")
        
        if 'book_id' in df.columns:
            result_df = df[df['book_id'].isin(book_ids)].drop_duplicates(subset=['book_id'])
            logger.info(f"Found {len(result_df)} books in merged_train.csv")
            
            # Handle column name variations
            if 'authors' not in result_df.columns and 'author' in result_df.columns:
                result_df = result_df.rename(columns={'author': 'authors'})
            if 'title' not in result_df.columns and 'book_title' in result_df.columns:
                result_df = result_df.rename(columns={'book_title': 'title'})
            
            # Ensure required columns exist
            required_cols = ['title', 'authors']
            if not all(col in result_df.columns for col in required_cols):
                logger.warning(f"Missing required columns in merged_train.csv")
                return pd.DataFrame()
                
            # Filter out books with missing or placeholder data
            result_df = result_df.dropna(subset=required_cols)
            result_df = result_df[~((result_df['title'].str.contains('Unknown')) & 
                                    (result_df['authors'] == 'Unknown'))]
            
            logger.info(f"After filtering for complete metadata, found {len(result_df)} valid books")
            
            # Log any missing books
            if len(result_df) < len(book_ids):
                missing_ids = set(book_ids) - set(result_df['book_id'])
                logger.warning(f"Could not find metadata for {len(missing_ids)} books: {list(missing_ids)}")
            
            # Cache the result
            _METADATA_CACHE[cache_key] = result_df
            return result_df
            
    except Exception as e:
        logger.error(f"Error reading merged_train.csv: {e}")
        logger.debug(traceback.format_exc())
    
    return pd.DataFrame()


def load_book_id_mapping(data_dir: str = 'data') -> Dict[int, int]:
    """
    Load book ID mapping with bidirectional mapping for flexibility.
    
    Parameters
    ----------
    data_dir : str
        Base data directory
        
    Returns
    -------
    Dict[int, int]
        Dictionary with bidirectional mapping
    """
    data_dir = get_data_dir(data_dir)
    
    # Check if we have the mapping in cache
    if data_dir in _MAPPING_CACHE:
        return _MAPPING_CACHE[data_dir]
    
    mapping_path = os.path.join(data_dir, 'processed', 'book_id_mapping.csv')
    
    if not os.path.exists(mapping_path):
        logger.warning(f"Book ID mapping file not found at {mapping_path}")
        return {}
    
    try:
        mapping_df = pd.read_csv(mapping_path)
        logger.info(f"Loaded book_id_mapping.csv with shape {mapping_df.shape}")
        
        if 'book_id' in mapping_df.columns and 'book_id_encoded' in mapping_df.columns:
            # Create bidirectional mapping
            mapping = {}
            
            for _, row in mapping_df.iterrows():
                original_id = int(row['book_id'])
                encoded_id = int(row['book_id_encoded'])
                
                # Map both ways: encoded->original and original->original
                mapping[encoded_id] = original_id
                if original_id not in mapping:
                    mapping[original_id] = original_id
            
            logger.info(f"Created bidirectional mapping with {len(mapping)} entries")
            _MAPPING_CACHE[data_dir] = mapping
            return mapping
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
    Simplified version with fewer nested conditions.
    
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
    data_dir = get_data_dir(data_dir)
    
    try:
        # First try loading from merged_train.csv, which is our primary dataset
        merged_path = os.path.join(data_dir, 'processed', 'merged_train.csv')
        
        if not os.path.exists(merged_path):
            logger.error(f"Merged dataset not found: {merged_path}")
            return []
            
        try:
            merged_df = pd.read_csv(merged_path)
            books_df = merged_df.groupby('book_id').agg({
                'average_rating': 'first',
                'ratings_count': 'first',
                'title': 'first',
                'authors': 'first'
            }).reset_index()
            logger.info(f"Loaded book data from merged dataset with {len(books_df)} books")
        except Exception as e:
            logger.error(f"Error loading book data: {e}")
            return []
        
        # Filter for books with at least 4.0 stars average rating
        books_df = books_df[books_df['average_rating'] >= 4.0]
        
        if len(books_df) == 0:
            logger.warning("No books found with 4.0+ star rating")
            return []
            
        # Calculate popularity score and sort
        books_df['popularity_score'] = books_df['ratings_count'] * (books_df['average_rating'] / 5.0)
        books_df = books_df.sort_values('popularity_score', ascending=False)
        
        # Get a pool of popular books
        popular_pool_size = min(n * 20, len(books_df))
        popular_books = books_df.head(popular_pool_size)
        
        # Handle randomization if requested
        if randomize and len(popular_books) > n:
            # Set random seed if provided
            if seed is not None:
                np.random.seed(seed)
                
            # Get semi-random selection of books    
            selected_books = popular_books.sample(n=min(n, len(popular_books)), random_state=seed)
            return selected_books['book_id'].tolist()
        else:
            # Non-randomized - just return top N books
            return popular_books['book_id'].head(n).tolist()
    
    except Exception as e:
        logger.error(f"Error getting popular books: {e}")
        logger.debug(traceback.format_exc())
        return []


def load_recommender_model(model_type: str = 'collaborative', models_dir: str = None) -> BaseRecommender:
    """
    Load recommender model from disk with caching
    
    Parameters
    ----------
    model_type : str
        Type of model to load ('collaborative')
    models_dir : str
        Directory containing the model files (optional)
        
    Returns
    -------
    BaseRecommender
        Loaded recommender model
    """
    # Check model cache first
    cache_key = f"{model_type}_{models_dir}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    
    # Check for environment variable for models directory
    env_models_dir = os.environ.get('BOOK_RECOMMENDER_MODELS_DIR')
    if env_models_dir and models_dir is None:
        models_dir = env_models_dir
    
    # Determine model directory if not provided
    if models_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.dirname(current_dir)
        project_root = os.path.dirname(src_dir)
        models_dir = os.path.join(project_root, 'models')
    
    logger.info(f"Loading {model_type} recommender from {models_dir}")
    
    try:
        # Check if models directory exists
        if not os.path.exists(models_dir):
            logger.error(f"Models directory not found: {models_dir}")
            return None
            
        # Find model files
        model_files = [f for f in os.listdir(models_dir) 
                      if f.startswith(model_type) and f.endswith('.pkl') 
                      and os.path.isfile(os.path.join(models_dir, f))]
        
        if not model_files:
            logger.error(f"No {model_type} model files found in {models_dir}")
            return None
        
        # Get newest model file
        model_files.sort(reverse=True)
        model_path = os.path.join(models_dir, model_files[0])
        
        # Custom unpickler to handle module name changes
        class ModelUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if name == 'CollaborativeRecommender':
                    return CollaborativeRecommender
                return super().find_class(module, name)
        
        # Load the model
        with open(model_path, 'rb') as f:
            model = ModelUnpickler(f).load()
            
        logger.info(f"Successfully loaded {model_type} recommender model")
        _MODEL_CACHE[cache_key] = model
        return model
        
    except Exception as e:
        logger.error(f"Error loading {model_type} recommender model: {str(e)}")
        return None


def map_book_ids(book_ids: List[int], data_dir: str = 'data') -> Tuple[List[int], List[int]]:
    """
    Map book IDs to their original IDs, handling both encoded and original IDs.
    
    Parameters
    ----------
    book_ids : List[int]
        List of book IDs returned by the model
    data_dir : str
        Path to the data directory
    
    Returns
    -------
    Tuple[List[int], List[int]]
        Tuple containing:
        - List of mapped book IDs (with unmapped IDs kept as-is)
        - List of indices for unmapped book IDs
    """
    mapping = load_book_id_mapping(data_dir)
    
    if not mapping:
        logger.warning("No book ID mapping found - using IDs as-is")
        return book_ids, []
    
    # Map IDs and track unmapped indices
    mapped_ids = []
    unmapped_indices = []
    
    for i, book_id in enumerate(book_ids):
        original_id = mapping.get(book_id)
        if original_id is not None:
            mapped_ids.append(original_id)
        else:
            mapped_ids.append(book_id)
            unmapped_indices.append(i)
            logger.warning(f"No mapping found for book ID {book_id} - using as-is")
    
    mapped_count = len(book_ids) - len(unmapped_indices)
    if mapped_count > 0:
        logger.info(f"Mapped {mapped_count} book IDs")
    
    return mapped_ids, unmapped_indices


def fallback_to_popular_books(user_id: int, n: int, data_dir: str) -> pd.DataFrame:
    """Fallback function to provide popular books if standard recommendations fail."""
    logger.info("Falling back to popularity-based recommendations")
    popular_book_ids = get_popular_books(n, data_dir, randomize=True, seed=user_id)
    return get_book_metadata(tuple(popular_book_ids), data_dir)


def recommend_for_user(user_id: int, model_type: str = 'collaborative', 
                      n: int = 5, data_dir: str = 'data') -> pd.DataFrame:
    """
    Generate book recommendations for a specific user.
    Simplified version with less redundancy.

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
    data_dir = get_data_dir(data_dir)

    try:
        # Load the model
        recommender = load_recommender_model(model_type)
        if recommender is None:
            return fallback_to_popular_books(user_id, n, data_dir)

        # Get recommendations from model
        logger.info(f"Getting recommendations for user {user_id}")
        fetch_count = n * 3  # Get extra recommendations to allow for filtering
        book_ids = recommender.recommend_for_user(user_id, n_recommendations=fetch_count)

        # Handle empty results
        if not book_ids or not isinstance(book_ids, list):
            logger.warning(f"No recommendations found for user {user_id}")
            return fallback_to_popular_books(user_id, n, data_dir)

        # Ensure IDs are integers
        book_ids = [int(b) for b in book_ids if str(b).isdigit()]

        # Process and map the recommended book IDs
        mapped_book_ids, unmapped_indices = map_book_ids(book_ids, data_dir)
        
        # Filter out unmapped IDs if we have enough mapped ones
        if unmapped_indices and len(mapped_book_ids) > len(unmapped_indices):
            filtered_book_ids = [mapped_book_ids[i] for i in range(len(mapped_book_ids)) if i not in unmapped_indices]
            
            if len(filtered_book_ids) >= n:
                # Use only properly mapped book IDs
                mapped_book_ids = filtered_book_ids[:n]
            else:
                # Get popular books to fill the gap
                remaining = n - len(filtered_book_ids)
                replacement_ids = get_popular_books(remaining * 2, data_dir, randomize=True, seed=user_id)
                replacement_ids = [b for b in replacement_ids if b not in filtered_book_ids][:remaining]
                mapped_book_ids = filtered_book_ids + replacement_ids
        
        # Ensure we have no more than n books
        mapped_book_ids = mapped_book_ids[:n]
        
        # Get metadata for recommendations
        metadata_df = get_book_metadata(tuple(mapped_book_ids), data_dir)
        
        # Handle missing metadata
        if len(metadata_df) < n:
            found_ids = metadata_df['book_id'].tolist() if not metadata_df.empty else []
            remaining = n - len(found_ids)
            
            if remaining > 0:
                # Get additional books to fill the gap
                additional_ids = get_popular_books(remaining * 2, data_dir, randomize=True, seed=user_id + 100)
                additional_ids = [b for b in additional_ids if b not in found_ids][:remaining]
                
                if additional_ids:
                    additional_df = get_book_metadata(tuple(additional_ids), data_dir)
                    if not additional_df.empty:
                        metadata_df = pd.concat([metadata_df, additional_df]) if not metadata_df.empty else additional_df
                        
        # Final fallback if we still have no results
        if metadata_df.empty:
            return fallback_to_popular_books(user_id, n, data_dir)

        # Add ranking to recommendations
        metadata_df['rank'] = range(len(metadata_df))
        
        return metadata_df

    except Exception as e:
        logger.error(f"Error getting recommendations for user {user_id}: {str(e)}")
        return fallback_to_popular_books(user_id, n, data_dir)


def recommend_similar_books(book_id: int, model_type: str = 'collaborative',
                          n: int = 5, data_dir: str = 'data') -> pd.DataFrame:
    """
    Generate similar book recommendations for a specific book.
    Simplified version with streamlined logic.
    
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
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with similar book metadata
    """
    data_dir = get_data_dir(data_dir)
    
    # Load the model
    model = load_recommender_model(model_type)
    if model is None:
        logger.error(f"Failed to load recommender model")
        return pd.DataFrame()
    
    try:
        # Get similar book recommendations
        logger.info(f"Finding books similar to book ID {book_id}")
        
        # Check if book exists in model
        book_exists = (hasattr(model, 'book_id_to_index') and book_id in model.book_id_to_index) or \
                     (hasattr(model, 'book_ids') and book_id in model.book_ids)
        
        if not book_exists:
            logger.warning(f"Book ID {book_id} not found in model. Using fallback.")
            fallback_ids = get_popular_books(n, data_dir, randomize=True, seed=book_id)
            fallback_ids = [b for b in fallback_ids if b != book_id][:n]
            return get_book_metadata(tuple(fallback_ids), data_dir)
        
        # Get similar books from model
        try:
            fetch_count = n * 3  # Get extra for filtering
            similar_book_ids = model.recommend_similar_books(book_id, n=fetch_count)
        except Exception as e:
            logger.error(f"Error getting similar books: {e}")
            fallback_ids = get_popular_books(n, data_dir, randomize=True, seed=book_id)
            fallback_ids = [b for b in fallback_ids if b != book_id][:n]
            return get_book_metadata(tuple(fallback_ids), data_dir)
        
        # Handle empty results
        if not similar_book_ids:
            logger.warning(f"No similar books found for {book_id}")
            fallback_ids = get_popular_books(n, data_dir, randomize=True, seed=book_id)
            fallback_ids = [b for b in fallback_ids if b != book_id][:n]
            return get_book_metadata(tuple(fallback_ids), data_dir)
            
        # Process book IDs
        similar_book_ids = [int(b) for b in similar_book_ids if str(b).isdigit()]
        
        # Map IDs to handle encoded IDs
        mapped_book_ids, unmapped_indices = map_book_ids(similar_book_ids, data_dir)
        
        # Filter out unmapped IDs if possible
        if unmapped_indices and len(mapped_book_ids) > len(unmapped_indices):
            filtered_ids = [mapped_book_ids[i] for i in range(len(mapped_book_ids)) if i not in unmapped_indices]
            
            if len(filtered_ids) >= n:
                mapped_book_ids = filtered_ids[:n]
            else:
                # Fill the gap with popular books
                remaining = n - len(filtered_ids)
                replacement_ids = get_popular_books(remaining * 2, data_dir, randomize=True, seed=book_id)
                replacement_ids = [b for b in replacement_ids if b not in filtered_ids and b != book_id][:remaining]
                mapped_book_ids = filtered_ids + replacement_ids
        
        # Ensure we have exactly n books
        mapped_book_ids = mapped_book_ids[:n]
            
        # Get metadata
        similar_books_df = get_book_metadata(tuple(mapped_book_ids), data_dir)
        
        # Handle missing metadata
        if len(similar_books_df) < n:
            found_ids = similar_books_df['book_id'].tolist() if not similar_books_df.empty else []
            remaining = n - len(found_ids)
            
            if remaining > 0:
                # Get additional books
                additional_ids = get_popular_books(remaining * 2, data_dir, randomize=True, seed=book_id + 100)
                additional_ids = [b for b in additional_ids if b not in found_ids and b != book_id][:remaining]
                
                if additional_ids:
                    additional_df = get_book_metadata(tuple(additional_ids), data_dir)
                    if not additional_df.empty:
                        similar_books_df = pd.concat([similar_books_df, additional_df]) if not similar_books_df.empty else additional_df
        
        # Final fallback if needed
        if similar_books_df.empty:
            fallback_ids = get_popular_books(n, data_dir, randomize=True, seed=book_id)
            fallback_ids = [b for b in fallback_ids if b != book_id][:n]
            return get_book_metadata(tuple(fallback_ids), data_dir)
            
        # Add source book info for logging
        source_book_df = get_book_metadata(tuple([book_id]), data_dir)
        if not source_book_df.empty:
            logger.info(f"Source book: {source_book_df.iloc[0]['title']} by {source_book_df.iloc[0]['authors']}")
        
        # Add ranking
        similar_books_df['rank'] = range(len(similar_books_df))
        
        return similar_books_df
        
    except Exception as e:
        logger.error(f"Error generating similar book recommendations: {e}")
        fallback_ids = get_popular_books(n, data_dir, randomize=True, seed=book_id)
        fallback_ids = [b for b in fallback_ids if b != book_id][:n]
        return get_book_metadata(tuple(fallback_ids), data_dir)


def print_recommendations(recommendations_df: pd.DataFrame, header: str = "Recommendations:"):
    """Print formatted recommendations with book titles and authors."""
    if recommendations_df.empty:
        print("No recommendations found.")
        return
    
    print(f"\n{header}")
    print("-" * 80)
    
    for i, row in recommendations_df.iterrows():
        rank = row.get('rank', i) + 1  # Display rank as 1-based
        print(f"{rank}. {row['title']} by {row['authors']}")
    
    print("-" * 80)


def main(args: Optional[List[str]] = None) -> int:
    """Main function to run the predict module from command line arguments."""
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
        # Run the demo if requested or no args provided
        if len(sys.argv) == 1 or parsed_args.demo:
            logger.info("Running demonstration of collaborative filtering recommendations")
            
            try:
                # Get a random user and book for the demo
                df = pd.read_csv(os.path.join(parsed_args.data_dir, 'processed', 'merged_train.csv'))
                user_id = np.random.choice(df['user_id'].unique())
                book_id = np.random.choice(df['book_id'].unique())
                
                # Demo user recommendations
                logger.info(f"Selected random user ID: {user_id} for demonstration")
                recommendations_df = recommend_for_user(
                    user_id=user_id,
                    n=parsed_args.num,
                    data_dir=parsed_args.data_dir
                )
                print_recommendations(recommendations_df, f"Recommendations for User {user_id}:")
                
                # Demo similar books
                logger.info(f"Selected random book ID: {book_id} for similar books demonstration")
                similar_books_df = recommend_similar_books(
                    book_id=book_id,
                    n=parsed_args.num,
                    data_dir=parsed_args.data_dir
                )
                print_recommendations(similar_books_df, f"Similar Books to Book ID {book_id}:")
                
                return 0
            
            except Exception as e:
                logger.error(f"Error running demo: {e}")
                logger.debug(traceback.format_exc())
                return 1
        
        # Handle user recommendations
        if parsed_args.user is not None:
            user_id = parsed_args.user
            logger.info(f"Generating recommendations for user {user_id}")
            
            recommendations_df = recommend_for_user(
                user_id=user_id,
                model_type=parsed_args.model_type,
                n=parsed_args.num,
                data_dir=parsed_args.data_dir
            )
            
            print_recommendations(recommendations_df, f"Book Recommendations for User {user_id}:")
            return 0
            
        # Handle similar book recommendations
        elif parsed_args.book is not None:
            book_id = parsed_args.book
            logger.info(f"Finding similar books for book {book_id}")
            
            similar_books_df = recommend_similar_books(
                book_id=book_id,
                model_type=parsed_args.model_type,
                n=parsed_args.num,
                data_dir=parsed_args.data_dir
            )
            
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