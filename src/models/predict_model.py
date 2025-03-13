import os
import pickle
import pandas as pd
import argparse
import numpy as np
import sys
import logging
import traceback
from typing import List, Dict, Union, Optional, Tuple, Any
from datetime import datetime

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
    from src.models.train_model_base import BaseRecommender, load_data
    from src.models.train_model_collaborative import CollaborativeRecommender
    from src.models.train_model_content import ContentBasedRecommender
    from src.models.train_model_hybrid import HybridRecommender
except ImportError:
    try:
        from models.train_model_base import BaseRecommender, load_data
        from models.train_model_collaborative import CollaborativeRecommender
        from models.train_model_content import ContentBasedRecommender
        from models.train_model_hybrid import HybridRecommender
    except ImportError:
        import sys
        import os
        # Add the parent directory to the path to ensure we can import the module
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.append(parent_dir)
        from models.train_model_base import BaseRecommender, load_data
        from models.train_model_collaborative import CollaborativeRecommender
        from models.train_model_content import ContentBasedRecommender
        from models.train_model_hybrid import HybridRecommender
        logger.error("Could not import from src.models. Using relative import.")

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


def load_recommender_model(model_type: str = 'hybrid', model_dir: str = 'models') -> BaseRecommender:
    """
    Load a trained recommender model of the specified type.
    
    Parameters
    ----------
    model_type : str
        Type of recommender to load ('collaborative', 'content', or 'hybrid')
    model_dir : str
        Directory containing saved models
        
    Returns
    -------
    BaseRecommender
        Loaded recommender model
    """
    if model_type == 'collaborative':
        model_path = os.path.join(model_dir, 'collaborative_recommender.pkl')
    elif model_type == 'content':
        model_path = os.path.join(model_dir, 'content_based_recommender.pkl')
    elif model_type == 'hybrid':
        model_path = os.path.join(model_dir, 'hybrid_recommender.pkl')
    else:
        logger.error(f"Unknown model type: {model_type}")
        return None
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        return None
    
    try:
        logger.info(f"Loading {model_type} recommender from {model_path}")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        logger.info(f"Successfully loaded {model_type} recommender model")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.debug(traceback.format_exc())
        return None


def recommend_for_user(user_id: int, model_type: str = 'hybrid', 
                      num_recommendations: int = 5, data_dir: str = 'data') -> pd.DataFrame:
    """
    Generate book recommendations for a specific user.
    
    Parameters
    ----------
    user_id : int
        ID of the user to generate recommendations for
    model_type : str
        Type of recommender to use ('collaborative', 'content', or 'hybrid')
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
            return pd.DataFrame()
        
        # Get recommendations for the user
        logger.info(f"Getting {num_recommendations} recommendations for user {user_id} using {model_type} model")
        book_ids = recommender.recommend_for_user(user_id, n_recommendations=num_recommendations)
        
        if not book_ids:
            logger.warning(f"No recommendations found for user {user_id}")
            return pd.DataFrame()
        
        # Check if we need to map encoded IDs to original IDs
        encoded_to_original = load_book_id_mapping(data_dir)
        if encoded_to_original:
            # Map encoded IDs back to original IDs
            original_book_ids = [encoded_to_original.get(book_id, book_id) for book_id in book_ids]
            logger.info(f"Mapped {len(book_ids)} encoded book IDs to original IDs")
            book_ids = original_book_ids
        
        # Get metadata for the recommended books
        recommendations_df = get_book_metadata(book_ids, data_dir)
        
        # If no books with complete metadata were found, log a warning
        if recommendations_df.empty:
            logger.warning(f"No books with complete metadata found for user {user_id}")
            return pd.DataFrame()
        
        # Add rank column for sorting (lower rank = higher recommendation)
        recommendations_df['rank'] = recommendations_df['book_id'].apply(lambda x: book_ids.index(x) + 1)
        
        # Sort by rank
        recommendations_df = recommendations_df.sort_values('rank')
        
        # Retrieve more books if some were filtered out due to missing metadata
        if len(recommendations_df) < num_recommendations:
            additional_count = num_recommendations - len(recommendations_df)
            logger.info(f"Need {additional_count} more recommendations to reach requested {num_recommendations}")
            
            # Calculate how many more books to request
            # Request 3x what we need since some might be filtered out
            additional_to_request = additional_count * 3
            
            # Get additional recommendations
            additional_book_ids = recommender.recommend_for_user(
                user_id, 
                n_recommendations=num_recommendations + additional_to_request,
                exclude_ids=book_ids  # Exclude already recommended books
            )
            
            if additional_book_ids:
                # Map if needed
                if encoded_to_original:
                    additional_book_ids = [encoded_to_original.get(book_id, book_id) 
                                         for book_id in additional_book_ids]
                
                # Get metadata and add to recommendations
                additional_df = get_book_metadata(additional_book_ids, data_dir)
                
                if not additional_df.empty:
                    # Add rank starting from where we left off
                    max_rank = recommendations_df['rank'].max() if not recommendations_df.empty else 0
                    additional_df['rank'] = range(max_rank + 1, max_rank + 1 + len(additional_df))
                    
                    # Combine dataframes
                    recommendations_df = pd.concat([recommendations_df, additional_df], ignore_index=True)
                    recommendations_df = recommendations_df.sort_values('rank')
                    
                    # Trim to requested number
                    if len(recommendations_df) > num_recommendations:
                        recommendations_df = recommendations_df.head(num_recommendations)
        
        return recommendations_df
        
    except Exception as e:
        logger.error(f"Error getting recommendations for user {user_id}: {e}")
        logger.debug(traceback.format_exc())
        return pd.DataFrame()


def recommend_similar_books(book_id: int, model_type: str = 'content',
                           num_recommendations: int = 5, data_dir: str = 'data') -> pd.DataFrame:
    """
    Generate similar book recommendations for a specific book.
    
    Parameters
    ----------
    book_id : int
        ID of the book to find similar books for
    model_type : str
        Type of recommender to use ('collaborative', 'content', or 'hybrid')
    num_recommendations : int
        Number of recommendations to generate
    data_dir : str
        Path to the data directory
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with similar book metadata
    """
    try:
        # Load the appropriate model
        recommender = load_recommender_model(model_type)
        
        if recommender is None:
            logger.error(f"Failed to load {model_type} recommender model")
            return pd.DataFrame()
        
        # Check if we need to map original ID to encoded ID
        encoded_to_original = load_book_id_mapping(data_dir)
        encoded_book_id = book_id
        
        if encoded_to_original:
            # Get the reverse mapping
            original_to_encoded = {v: k for k, v in encoded_to_original.items()}
            if book_id in original_to_encoded:
                encoded_book_id = original_to_encoded[book_id]
                logger.info(f"Mapped original book ID {book_id} to encoded ID {encoded_book_id}")
        
        # Find similar books
        logger.info(f"Finding similar books for book {book_id} using {model_type} model")
        similar_encoded_ids = recommender.recommend_similar_books(encoded_book_id, n=num_recommendations)
        
        if not similar_encoded_ids:
            logger.warning(f"No similar books found for book {book_id}")
            return pd.DataFrame()
        
        # Map back to original IDs if needed
        if encoded_to_original:
            similar_book_ids = [encoded_to_original.get(book_id, book_id) for book_id in similar_encoded_ids]
        else:
            similar_book_ids = similar_encoded_ids
        
        # Get metadata for the recommended books
        similar_books_df = get_book_metadata(similar_book_ids, data_dir)
        
        # Add metadata for the source book at the top
        source_book_df = get_book_metadata([book_id], data_dir)
        if not source_book_df.empty:
            source_book_df['rank'] = -1  # Rank -1 indicates it's the source book
            similar_books_df['rank'] = list(range(len(similar_books_df)))
            similar_books_df = pd.concat([source_book_df, similar_books_df], ignore_index=True)
        
        return similar_books_df
        
    except Exception as e:
        logger.error(f"Error finding similar books: {e}")
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
    parser.add_argument('--model-type', type=str, default='hybrid', 
                       choices=['collaborative', 'content', 'hybrid'],
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
