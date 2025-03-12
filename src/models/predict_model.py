import os
import sys
import pickle
import logging
import traceback
import pandas as pd
import argparse
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union
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
        from models.model_utils import BaseRecommender, load_data
        from models.train_model import CollaborativeRecommender
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


def load_recommender_model(model_type: str = 'collaborative', model_dir: str = 'models') -> BaseRecommender:
    """
    Load a trained recommender model with Docker compatibility handling.
    
    Parameters
    ----------
    model_type : str
        Type of recommender to load ('collaborative')
    model_dir : str
        Directory where models are stored
        
    Returns
    -------
    BaseRecommender
        Loaded recommender model
    """
    # Only support collaborative model
    if model_type != 'collaborative':
        logger.warning(f"Only collaborative model is supported. Ignoring model_type: {model_type}")
        model_type = 'collaborative'
    
    try:
        # Ensure we have the models directory
        os.makedirs(model_dir, exist_ok=True)
        
        # First try the new naming convention (model_type.pkl)
        model_path = os.path.join(model_dir, f"{model_type}.pkl")
        
        if not os.path.exists(model_path):
            # Fall back to the old naming convention (model_type_recommender.pkl)
            model_path = os.path.join(model_dir, f"{model_type}_recommender.pkl")
            
            if not os.path.exists(model_path):
                logger.warning(f"Model file not found at {model_path}")
                return None

        # Define a custom unpickler to handle module path differences
        class CustomUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                # Handle cases where the module path might be different in Docker
                if name == 'CollaborativeRecommender':
                    logger.info(f"Finding CollaborativeRecommender class from {module}")
                    # Try different module paths that might exist in Docker
                    possible_modules = [
                        module,
                        'src.models.train_model',
                        'models.train_model',
                        '__main__'
                    ]
                    
                    for mod in possible_modules:
                        try:
                            # Try to import the module and get the class
                            __import__(mod, fromlist=[name])
                            if mod == module:
                                return super().find_class(module, name)
                            else:
                                return super().find_class(mod, name)
                        except (ImportError, AttributeError, ValueError):
                            continue
                    
                    # If we can't find the class in any of the modules, use our imported version
                    logger.warning(f"Could not find {name} in any module, using imported version")
                    return CollaborativeRecommender
                
                # For other classes, try the standard approach
                try:
                    return super().find_class(module, name)
                except (ImportError, AttributeError) as e:
                    logger.warning(f"Error finding class {module}.{name}: {e}")
                    
                    # Handle common module path transformations
                    if module.startswith('__main__'):
                        # Try both src.models and models prefixes
                        possible_new_modules = [
                            module.replace('__main__', 'src.models.train_model'),
                            module.replace('__main__', 'models.train_model')
                        ]
                        
                        for new_module in possible_new_modules:
                            try:
                                return super().find_class(new_module, name)
                            except (ImportError, AttributeError):
                                pass
                    
                    # Last resort: try to find a class with the same name in our current scope
                    if name in globals():
                        logger.info(f"Using globally defined {name} instead of {module}.{name}")
                        return globals()[name]
                    
                    raise  # Re-raise if we can't handle it
            
        logger.info(f"Loading {model_type} model from {model_path} with custom unpickler")
        with open(model_path, 'rb') as f:
            model = CustomUnpickler(f).load()
            
        # Verify that the model is of the right type
        if not isinstance(model, BaseRecommender):
            logger.error(f"Model is not a BaseRecommender instance: {type(model)}")
            return None
            
        logger.info(f"Successfully loaded {model_type} model")
        return model
    except Exception as e:
        logger.error(f"Error loading {model_type} model: {e}")
        logger.debug(traceback.format_exc())
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
        Type of recommender to use ('collaborative')
    num_recommendations : int
        Number of recommendations to generate
    data_dir : str
        Path to the data directory
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with recommendations and metadata
    """
    # Only support collaborative model
    if model_type != 'collaborative':
        logger.warning(f"Only collaborative model is supported. Ignoring model_type: {model_type}")
        model_type = 'collaborative'
    
    try:
        # Load the model
        model_dir = os.path.join(data_dir, '..', 'models')
        model = load_recommender_model(model_type, model_dir=model_dir)
        
        if model is None:
            logger.error(f"Could not load {model_type} model")
            return pd.DataFrame()
        
        logger.info(f"Generating {num_recommendations} recommendations for user {user_id} using {model_type} model")
        
        # Generate recommendations
        start_time = datetime.now()
        book_ids = model.recommend_for_user(user_id, n_recommendations=num_recommendations)
        end_time = datetime.now()
        
        if not book_ids:
            logger.warning(f"No recommendations generated for user {user_id}")
            return pd.DataFrame()
            
        logger.info(f"Generated {len(book_ids)} recommendations in {(end_time - start_time).total_seconds():.2f} seconds")
        
        # Convert book IDs back to original IDs if needed
        book_id_mapping = load_book_id_mapping(data_dir)
        if book_id_mapping:
            original_book_ids = [book_id_mapping.get(book_id, book_id) for book_id in book_ids]
            logger.info(f"Converted {len(original_book_ids)} book IDs from encoded to original IDs")
            book_ids = original_book_ids
        
        # Get book metadata
        recommendations_df = get_book_metadata(book_ids, data_dir)
        
        if recommendations_df.empty:
            logger.warning(f"Could not find metadata for recommended books")
            return pd.DataFrame()
            
        # Add rank information as column
        # Generate a rank dictionary with book_id as key and position as value
        rank_dict = {book_id: i for i, book_id in enumerate(book_ids)}
        
        # Add rank to recommendations DataFrame
        recommendations_df['rank'] = recommendations_df['book_id'].map(rank_dict)
        
        # Sort by rank
        recommendations_df = recommendations_df.sort_values('rank')
        
        # Save the recommendations to CSV for analysis
        if recommendations_df.shape[0] > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(data_dir, 'results')
            os.makedirs(output_dir, exist_ok=True)
            recommendations_file = os.path.join(output_dir, f'user_{user_id}_recommendations_{timestamp}.csv')
            recommendations_df.to_csv(recommendations_file, index=False)
            logger.info(f"Saved recommendations for user {user_id} to {recommendations_file}")
        
        return recommendations_df
        
    except Exception as e:
        logger.error(f"Error generating recommendations for user {user_id}: {e}")
        logger.debug(traceback.format_exc())
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
        Type of recommender to use ('collaborative')
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
        logger.warning(f"Only collaborative model is supported. Ignoring model_type: {model_type}")
        model_type = 'collaborative'
    
    try:
        # Load the model
        model_dir = os.path.join(data_dir, '..', 'models')
        model = load_recommender_model(model_type, model_dir=model_dir)
        
        if model is None:
            logger.error(f"Could not load {model_type} model")
            return pd.DataFrame()
        
        logger.info(f"Finding {num_recommendations} similar books to book {book_id} using {model_type} model")
        
        # Check if we need to convert book_id from original to encoded ID
        book_id_mapping_path = os.path.join(data_dir, 'processed', 'book_id_mapping.csv')
        if os.path.exists(book_id_mapping_path):
            try:
                mapping_df = pd.read_csv(book_id_mapping_path)
                original_to_encoded = dict(zip(mapping_df['book_id'], mapping_df['book_id_encoded']))
                
                if book_id in original_to_encoded:
                    encoded_book_id = original_to_encoded[book_id]
                    logger.info(f"Converted original book ID {book_id} to encoded ID {encoded_book_id}")
                    book_id = encoded_book_id
            except Exception as e:
                logger.error(f"Error converting book ID: {e}")
        
        # Generate recommendations
        start_time = datetime.now()
        similar_book_ids = model.recommend_similar_books(book_id, n=num_recommendations)
        end_time = datetime.now()
        
        if not similar_book_ids:
            logger.warning(f"No similar books found for book {book_id}")
            return pd.DataFrame()
            
        logger.info(f"Found {len(similar_book_ids)} similar books in {(end_time - start_time).total_seconds():.2f} seconds")
        
        # Convert book IDs back to original IDs if needed
        book_id_mapping = load_book_id_mapping(data_dir)
        if book_id_mapping:
            original_book_ids = [book_id_mapping.get(book_id, book_id) for book_id in similar_book_ids]
            logger.info(f"Converted {len(original_book_ids)} book IDs from encoded to original IDs")
            similar_book_ids = original_book_ids
        
        # Get book metadata
        recommendations_df = get_book_metadata(similar_book_ids, data_dir)
        
        if recommendations_df.empty:
            logger.warning(f"Could not find metadata for similar books")
            return pd.DataFrame()
            
        # Add rank information as column
        # Generate a rank dictionary with book_id as key and position as value
        rank_dict = {book_id: i for i, book_id in enumerate(similar_book_ids)}
        
        # Add rank to recommendations DataFrame
        recommendations_df['rank'] = recommendations_df['book_id'].map(rank_dict)
        
        # Sort by rank
        recommendations_df = recommendations_df.sort_values('rank')
        
        # Save the recommendations to CSV for analysis
        if recommendations_df.shape[0] > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(data_dir, 'results')
            os.makedirs(output_dir, exist_ok=True)
            recommendations_file = os.path.join(output_dir, f'book_{book_id}_similar_books_{timestamp}.csv')
            recommendations_df.to_csv(recommendations_file, index=False)
            logger.info(f"Saved similar books for book {book_id} to {recommendations_file}")
        
        return recommendations_df
        
    except Exception as e:
        logger.error(f"Error finding similar books for book {book_id}: {e}")
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
        print("No recommendations available.")
        return
        
    print(f"\n{header}")
    print("-" * 80)
    
    for i, (_, row) in enumerate(recommendations_df.iterrows(), 1):
        title = row.get('title', 'Unknown Title')
        authors = row.get('authors', 'Unknown Author')
        book_id = row.get('book_id', 'Unknown ID')
        print(f"{i}. '{title}' by {authors} (Book ID: {book_id})")
    
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
    
    # Define command-line arguments
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # User recommendations command
    user_parser = subparsers.add_parser('user', help='Generate recommendations for a user')
    user_parser.add_argument('user_id', type=int, help='User ID to generate recommendations for')
    user_parser.add_argument('--num', type=int, default=5, help='Number of recommendations to generate')
    user_parser.add_argument('--model', type=str, choices=['collaborative'], default='collaborative',
                            help='Model type to use for recommendations')
    user_parser.add_argument('--data-dir', type=str, default='data', help='Data directory path')
    
    # Book recommendations command
    book_parser = subparsers.add_parser('book', help='Find similar books')
    book_parser.add_argument('book_id', type=int, help='Book ID to find similar books for')
    book_parser.add_argument('--num', type=int, default=5, help='Number of similar books to find')
    book_parser.add_argument('--model', type=str, choices=['collaborative'], default='collaborative',
                            help='Model type to use for recommendations')
    book_parser.add_argument('--data-dir', type=str, default='data', help='Data directory path')
    
    # Parse arguments
    args = parser.parse_args(args)
    
    # Execute command
    if args.command == 'user':
        try:
            logger.info(f"Generating recommendations for user {args.user_id} using {args.model} model")
            
            recommendations_df = recommend_for_user(
                user_id=args.user_id,
                model_type=args.model,
                num_recommendations=args.num,
                data_dir=args.data_dir
            )
            
            if recommendations_df.empty:
                logger.error(f"No recommendations found for user {args.user_id}")
                print(f"No recommendations found for user {args.user_id}. The user may not exist in the training data.")
                return 1
                
            print_recommendations(
                recommendations_df,
                header=f"Recommendations for User {args.user_id} (Using {args.model} Model):"
            )
            
            return 0
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            logger.debug(traceback.format_exc())
            print(f"Error generating recommendations: {e}")
            return 1
            
    elif args.command == 'book':
        try:
            logger.info(f"Finding similar books to book {args.book_id} using {args.model} model")
            
            similar_books_df = recommend_similar_books(
                book_id=args.book_id,
                model_type=args.model,
                num_recommendations=args.num,
                data_dir=args.data_dir
            )
            
            if similar_books_df.empty:
                logger.error(f"No similar books found for book {args.book_id}")
                print(f"No similar books found for book {args.book_id}. The book may not exist in the training data.")
                return 1
                
            print_recommendations(
                similar_books_df,
                header=f"Books Similar to Book ID {args.book_id} (Using {args.model} Model):"
            )
            
            return 0
            
        except Exception as e:
            logger.error(f"Error finding similar books: {e}")
            logger.debug(traceback.format_exc())
            print(f"Error finding similar books: {e}")
            return 1
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
