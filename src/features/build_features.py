import pandas as pd
import numpy as np
import scipy.sparse as sp
import os
import sys
import logging
import traceback
from typing import Tuple, List, Dict, Optional, Union, Any
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

# Determine project root directory
project_root = Path(__file__).parent.parent.parent.absolute()

# Set up logging
log_dir = os.path.join(project_root, 'logs')
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(log_dir, f'build_features_{timestamp}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('build_features')
logger.info(f"Project root: {project_root}")


def read_ratings(data_dir: str = 'data/processed') -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """
    Read the processed ratings data and create a sparse user-item matrix.
    
    Parameters
    ----------
    data_dir : str
        Directory containing the processed data files
        
    Returns
    -------
    Tuple[sp.csr_matrix, np.ndarray, np.ndarray]
        Tuple containing (user_item_matrix, user_ids, item_ids)
    """
    logger.info(f"Reading ratings data from {data_dir}")
    
    try:
        # Load merged_train.csv which contains ratings
        ratings_file = os.path.join(data_dir, 'merged_train.csv')
        if not os.path.exists(ratings_file):
            logger.error(f"Ratings file not found: {ratings_file}")
            return sp.csr_matrix((0, 0)), np.array([]), np.array([])
            
        logger.info(f"Reading ratings from {ratings_file}")
        ratings_df = pd.read_csv(ratings_file)
        logger.info(f"Loaded ratings dataframe with shape {ratings_df.shape}")
        
        # Check if we have the expected columns
        if 'user_id' not in ratings_df.columns or 'book_id' not in ratings_df.columns:
            logger.error(f"Missing required columns in {ratings_file}")
            return sp.csr_matrix((0, 0)), np.array([]), np.array([])
            
        if 'rating' not in ratings_df.columns:
            # If no rating column, assume implicit ratings (all 1.0)
            logger.warning(f"No rating column found in {ratings_file}, assuming implicit ratings")
            ratings_df['rating'] = 1.0
        
        # Get unique user and book IDs
        user_ids = ratings_df['user_id'].unique()
        book_ids = ratings_df['book_id'].unique()
        
        logger.info(f"Found {len(user_ids)} unique users and {len(book_ids)} unique books")
        
        # Create label encoders for user and book IDs
        user_encoder = LabelEncoder().fit(user_ids)
        book_encoder = LabelEncoder().fit(book_ids)
        
        # Transform IDs to sequential integers starting from 0
        ratings_df['user_id_encoded'] = user_encoder.transform(ratings_df['user_id'])
        ratings_df['book_id_encoded'] = book_encoder.transform(ratings_df['book_id'])
        
        # Save the book ID mapping for later use in predictions
        mapping_df = pd.DataFrame({
            'book_id': book_ids,
            'book_id_encoded': book_encoder.transform(book_ids)
        })
        mapping_path = os.path.join(os.path.dirname(data_dir), 'processed', 'book_id_mapping.csv')
        os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
        mapping_df.to_csv(mapping_path, index=False)
        logger.info(f"Saved book ID mapping to {mapping_path}")
        
        # Also save the user ID mapping
        user_mapping_df = pd.DataFrame({
            'user_id': user_ids,
            'user_id_encoded': user_encoder.transform(user_ids)
        })
        user_mapping_path = os.path.join(os.path.dirname(data_dir), 'processed', 'user_id_mapping.csv')
        user_mapping_df.to_csv(user_mapping_path, index=False)
        logger.info(f"Saved user ID mapping to {user_mapping_path}")
        
        # Map IDs to indices
        rows = ratings_df['user_id_encoded'].values
        cols = ratings_df['book_id_encoded'].values
        data = ratings_df['rating'].values
        
        # Create sparse matrix
        user_item_matrix = sp.csr_matrix((data, (rows, cols)), 
                                         shape=(len(user_ids), len(book_ids)))
        
        logger.info(f"Created user-item matrix with shape {user_item_matrix.shape}")
        
        return user_item_matrix, user_ids, book_ids
        
    except Exception as e:
        logger.error(f"Error reading ratings: {e}")
        logger.debug(traceback.format_exc())
        return sp.csr_matrix((0, 0)), np.array([]), np.array([])


def create_sparse_user_item_matrix(ratings_df: pd.DataFrame) -> Tuple[sp.csr_matrix, Tuple[int, int]]:
    """
    Creates a sparse user-item matrix from ratings data.
    
    Parameters
    ----------
    ratings_df : pd.DataFrame
        DataFrame containing user ratings with encoded IDs
        
    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse matrix where rows are users, columns are books, and values are ratings
    tuple
        (num_users, num_books) dimensions of the matrix
    """
    # Extract user, book, rating data
    user_ids = ratings_df['user_id_encoded'].values
    book_ids = ratings_df['book_id_encoded'].values
    ratings = ratings_df['rating'].values
    
    # Get matrix dimensions
    num_users = ratings_df['user_id_encoded'].max() + 1
    num_books = ratings_df['book_id_encoded'].max() + 1
    
    # Create sparse matrix
    user_item_matrix = sp.csr_matrix((ratings, (user_ids, book_ids)), 
                                     shape=(num_users, num_books))
    
    return user_item_matrix, (num_users, num_books)


def main(data_dir: str = 'data') -> int:
    """
    Main function to build features from processed data.
    
    Parameters
    ----------
    data_dir : str
        Base data directory
        
    Returns
    -------
    int
        Exit code
    """
    try:
        # Convert to absolute paths
        abs_data_dir = os.path.join(project_root, data_dir)
        processed_dir = os.path.join(abs_data_dir, 'processed')
        features_dir = os.path.join(abs_data_dir, 'features')
        
        logger.info(f"Building features from processed data in {processed_dir}")
        
        # Create features directory if it doesn't exist
        os.makedirs(features_dir, exist_ok=True)
        
        # Generate the sparse user-item matrix
        user_item_matrix, user_ids, book_ids = read_ratings(processed_dir)
        
        if user_item_matrix.shape[0] == 0 or user_item_matrix.shape[1] == 0:
            logger.error("Failed to create user-item matrix")
            return 1
            
        # Save the matrix and IDs to the features directory
        from scipy.sparse import save_npz
        save_npz(os.path.join(features_dir, 'user_item_matrix.npz'), user_item_matrix)
        np.save(os.path.join(features_dir, 'user_ids.npy'), user_ids)
        np.save(os.path.join(features_dir, 'book_ids.npy'), book_ids)
        
        logger.info(f"User-item matrix shape: {user_item_matrix.shape}")
        logger.info(f"Number of users: {len(user_ids)}")
        logger.info(f"Number of books: {len(book_ids)}")
        
        # Create a marker file to indicate completion
        with open(os.path.join(features_dir, "features_complete"), 'w') as f:
            f.write(f"Feature building completed at {datetime.now().isoformat()}")
        
        logger.info(f"Features saved to {features_dir}")
        return 0
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Build features for book recommender')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Base directory for data')
    
    args = parser.parse_args()
    
    sys.exit(main(args.data_dir))
