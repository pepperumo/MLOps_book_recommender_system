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

# Import the load_data function from train_model
try:
    from src.models.train_model import load_data, BookRecommender
except ImportError:
    logger.error("Could not import from src.models.train_model. Using relative import.")
    try:
        from train_model import load_data, BookRecommender
    except ImportError:
        logger.critical("Failed to import required functions from train_model. This is required for the model to work.")
        raise

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
        DataFrame with book metadata
    """
    logger.info(f"Retrieving metadata for {len(book_ids)} books")
    
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
                else:
                    # If we found all books, return the result
                    if len(result_df) == len(book_ids):
                        return result_df
        except Exception as e:
            logger.error(f"Error reading merged_train.csv: {e}")
            logger.debug(traceback.format_exc())
    
    # If we couldn't get all metadata from merged_train.csv, try the original books.csv
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
                
                # Add missing books from the first DataFrame if available
                if 'train_result_df' in locals() and len(result_df) < len(book_ids):
                    missing_ids = set(book_ids) - set(result_df['book_id'])
                    missing_df = train_result_df[train_result_df['book_id'].isin(missing_ids)]
                    result_df = pd.concat([result_df, missing_df], ignore_index=True)
                    logger.info(f"Combined dataframes to find {len(result_df)} books")
        except Exception as e:
            logger.error(f"Error reading books.csv: {e}")
            logger.debug(traceback.format_exc())
    
    # Create a skeleton DataFrame for books we couldn't find
    if 'result_df' not in locals() or len(result_df) < len(book_ids):
        logger.warning("Creating placeholder entries for books without metadata")
        missing_ids = set(book_ids) - (set(result_df['book_id']) if 'result_df' in locals() else set())
        missing_df = pd.DataFrame({
            'book_id': list(missing_ids),
            'title': [f"Unknown (ID: {book_id})" for book_id in missing_ids],
            'authors': ["Unknown" for _ in missing_ids]
        })
        
        result_df = pd.concat([result_df, missing_df], ignore_index=True) if 'result_df' in locals() else missing_df
    
    # Save metadata to CSV for analysis
    output_dir = os.path.join(data_dir, 'processed')
    os.makedirs(output_dir, exist_ok=True)
    metadata_file = os.path.join(output_dir, f'book_metadata_{timestamp}.csv')
    result_df.to_csv(metadata_file, index=False)
    logger.info(f"Saved book metadata to {metadata_file}")
    
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


def recommend_for_user(user_id: int, model_path: str = 'models/book_recommender.pkl', 
                      num_recommendations: int = 5, strategy: str = 'hybrid', 
                      data_dir: str = 'data') -> pd.DataFrame:
    """
    Generate book recommendations for a specific user.
    
    Parameters
    ----------
    user_id : int
        ID of the user to generate recommendations for
    model_path : str
        Path to the trained recommender model
    num_recommendations : int
        Number of recommendations to generate
    strategy : str
        Recommendation strategy ('collaborative', 'content', or 'hybrid')
    data_dir : str
        Path to the data directory
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with book metadata for recommended books
    """
    logger.info(f"Generating recommendations for user {user_id} using strategy '{strategy}'")
    
    # Load the model
    try:
        with open(model_path, 'rb') as f:
            recommender = pickle.load(f)
        logger.info(f"Loaded recommender model from {model_path}")
    except FileNotFoundError:
        logger.error(f"Model file not found at {model_path}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.debug(traceback.format_exc())
        return pd.DataFrame()
    
    # Load the matrices that weren't included in the pickled model
    try:
        features_dir = os.path.join(data_dir, 'features')
        user_item_matrix, book_feature_matrix, book_similarity_matrix, book_ids, feature_names = load_data(features_dir)
        
        # Set the matrices in the recommender
        recommender.user_item_matrix = user_item_matrix
        recommender.book_feature_matrix = book_feature_matrix
        recommender.book_similarity_matrix = book_similarity_matrix
        recommender.book_ids = book_ids
        recommender.feature_names = feature_names
        
        logger.info(f"Loaded matrices: user-item {user_item_matrix.shape}, book-feature {book_feature_matrix.shape}")
    except Exception as e:
        logger.error(f"Error loading matrices: {e}")
        logger.debug(traceback.format_exc())
        return pd.DataFrame()
    
    # Load book ID mapping
    encoded_to_original = load_book_id_mapping(data_dir)
    
    # Generate recommendations based on the specified strategy
    try:
        if strategy == 'collaborative':
            encoded_book_ids = recommender.recommend_for_user(user_id, n_recommendations=num_recommendations, strategy='collaborative')
        elif strategy == 'content':
            encoded_book_ids = recommender.recommend_for_user(user_id, n_recommendations=num_recommendations, strategy='content')
        else:  # hybrid
            encoded_book_ids = recommender.recommend_for_user(user_id, n_recommendations=num_recommendations, strategy='hybrid')
        
        # Limit to requested number of recommendations (just in case)
        encoded_book_ids = encoded_book_ids[:num_recommendations]
        logger.info(f"Generated {len(encoded_book_ids)} recommendations using {strategy} strategy")
        
        # Convert encoded IDs to original IDs if mapping is available
        if encoded_to_original:
            original_book_ids = [encoded_to_original.get(book_id, book_id) for book_id in encoded_book_ids]
            logger.info(f"Mapped {len(encoded_book_ids)} encoded IDs to original IDs")
        else:
            original_book_ids = encoded_book_ids
            logger.warning("No ID mapping available, using encoded IDs directly")
        
        # Get metadata for recommended books
        book_metadata = get_book_metadata(original_book_ids, data_dir)
        
        # Save recommendations to CSV
        output_dir = os.path.join(data_dir, 'results')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'user_{user_id}_recommendations_{strategy}_{timestamp}.csv')
        book_metadata.to_csv(output_file, index=False)
        logger.info(f"Saved user recommendations to {output_file}")
        
        return book_metadata
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        logger.debug(traceback.format_exc())
        return pd.DataFrame()


def recommend_similar_books(book_id: int, model_path: str = 'models/book_recommender.pkl', 
                           num_recommendations: int = 5, data_dir: str = 'data') -> pd.DataFrame:
    """
    Generate similar book recommendations for a specific book.
    
    Parameters
    ----------
    book_id : int
        ID of the book to find similar books for
    model_path : str
        Path to the trained recommender model
    num_recommendations : int
        Number of recommendations to generate
    data_dir : str
        Path to the data directory
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with similar book metadata
    """
    logger.info(f"Finding similar books to book ID {book_id}")
    
    # Load the model
    try:
        with open(model_path, 'rb') as f:
            recommender = pickle.load(f)
        logger.info(f"Loaded recommender model from {model_path}")
    except FileNotFoundError:
        logger.error(f"Model file not found at {model_path}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.debug(traceback.format_exc())
        return pd.DataFrame()
    
    # Load the matrices that weren't included in the pickled model
    try:
        features_dir = os.path.join(data_dir, 'features')
        user_item_matrix, book_feature_matrix, book_similarity_matrix, book_ids, feature_names = load_data(features_dir)
        
        # Set the matrices in the recommender
        recommender.user_item_matrix = user_item_matrix
        recommender.book_feature_matrix = book_feature_matrix
        recommender.book_similarity_matrix = book_similarity_matrix
        recommender.book_ids = book_ids
        recommender.feature_names = feature_names
        
        logger.info(f"Loaded matrices: user-item {user_item_matrix.shape}, book-feature {book_feature_matrix.shape}")
    except Exception as e:
        logger.error(f"Error loading matrices: {e}")
        logger.debug(traceback.format_exc())
        return pd.DataFrame()
    
    # Load book ID mapping
    encoded_to_original = load_book_id_mapping(data_dir)
    
    # Convert original book ID to encoded ID if mapping is available
    try:
        if encoded_to_original:
            # We need to map in the opposite direction
            original_to_encoded = {v: k for k, v in encoded_to_original.items()}
            encoded_book_id = original_to_encoded.get(book_id, book_id)
            if encoded_book_id != book_id:
                logger.info(f"Mapped original book ID {book_id} to encoded ID {encoded_book_id}")
            else:
                logger.warning(f"Could not find encoded ID for book {book_id}, using as is")
        else:
            encoded_book_id = book_id
            logger.warning("No ID mapping available, using original ID directly")
        
        # Get similar books
        encoded_similar_book_ids = recommender.recommend_similar_books(encoded_book_id, n=num_recommendations)
        
        # Limit to the requested number and remove duplicates
        encoded_similar_book_ids = list(dict.fromkeys(encoded_similar_book_ids))[:num_recommendations]
        logger.info(f"Found {len(encoded_similar_book_ids)} similar books")
        
        # Convert encoded IDs to original IDs if mapping is available
        if encoded_to_original:
            original_similar_book_ids = [encoded_to_original.get(book_id, book_id) 
                                       for book_id in encoded_similar_book_ids]
            logger.info(f"Mapped {len(encoded_similar_book_ids)} encoded IDs to original IDs")
        else:
            original_similar_book_ids = encoded_similar_book_ids
            logger.warning("No ID mapping available, using encoded IDs directly")
        
        # Get metadata for the original book
        original_book_metadata = get_book_metadata([book_id], data_dir)
        
        # Get metadata for recommended books
        similar_books_metadata = get_book_metadata(original_similar_book_ids, data_dir)
        
        # Save recommendations to CSV
        output_dir = os.path.join(data_dir, 'results')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'book_{book_id}_similar_books_{timestamp}.csv')
        similar_books_metadata.to_csv(output_file, index=False)
        logger.info(f"Saved similar books to {output_file}")
        
        return similar_books_metadata
    except Exception as e:
        logger.error(f"Error finding similar books: {e}")
        logger.debug(traceback.format_exc())
        return pd.DataFrame()


def print_recommendations(recommendations_df: pd.DataFrame, header: str = "Recommendations:") -> None:
    """
    Print formatted recommendations with book titles and authors.
    
    Parameters
    ----------
    recommendations_df : pandas.DataFrame
        DataFrame with book metadata
    header : str
        Header text to display before recommendations
    """
    print(f"\n{header}")
    if len(recommendations_df) == 0:
        print("  No recommendations available.")
        return
    
    # Remove any duplicate book_ids, keeping the first occurrence
    recommendations_df = recommendations_df.drop_duplicates(subset=['book_id'])
    
    # Print each book with its metadata
    for idx, book in recommendations_df.iterrows():
        title = book.get('title', f"Unknown (ID: {book['book_id']})")
        authors = book.get('authors', 'Unknown')
        book_id = book['book_id']
        print(f"  * {title} by {authors} (ID: {book_id})")


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
    logger.info("Starting predict_model.py")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate book recommendations')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--user', type=int, help='User ID to generate recommendations for')
    group.add_argument('--book', type=int, help='Book ID to find similar books for')
    parser.add_argument('--num', type=int, default=5, help='Number of recommendations to generate')
    parser.add_argument('--strategy', type=str, choices=['collaborative', 'content', 'hybrid'], 
                        default='hybrid', help='Recommendation strategy for user recommendations')
    parser.add_argument('--model-path', type=str, default='models/book_recommender.pkl', help='Path to the model file')
    parser.add_argument('--data-dir', type=str, default='data', help='Path to the data directory')
    
    args = parser.parse_args(args)
    
    try:
        logger.info(f"Args: {vars(args)}")
        
        # Check if model file exists
        if not os.path.exists(args.model_path):
            logger.error(f"Model file not found at {args.model_path}")
            return 1
        
        # Check if data directory exists
        features_dir = os.path.join(args.data_dir, 'features')
        if not os.path.exists(features_dir):
            logger.error(f"Features directory not found at {features_dir}")
            return 1
        
        print("Loading data...")
        
        if args.user is not None:
            # Generate recommendations for user
            recommendations = recommend_for_user(
                args.user, 
                model_path=args.model_path, 
                num_recommendations=args.num, 
                strategy=args.strategy,
                data_dir=args.data_dir
            )
            
            print_recommendations(recommendations, header=f"Recommendations for User {args.user} (Strategy: {args.strategy}):")
        else:
            # Get book metadata to display what book we're finding similar books for
            book_metadata = get_book_metadata([args.book], args.data_dir)
            if len(book_metadata) > 0:
                book = book_metadata.iloc[0]
                title = book.get('title', f"Unknown (ID: {args.book})")
                authors = book.get('authors', 'Unknown')
                print(f"\nSimilar books to: {title} by {authors} (ID: {args.book})\n")
            
            # Find similar books
            similar_books = recommend_similar_books(
                args.book, 
                model_path=args.model_path, 
                num_recommendations=args.num,
                data_dir=args.data_dir
            )
            
            print_recommendations(similar_books)
        
        logger.info("predict_model.py completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}")
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
