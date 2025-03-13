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

# Set up logging
log_dir = os.path.join('logs')
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
    logger.info(f"Starting build_features.py")
    
    try:
        processed_dir = os.path.join(data_dir, 'processed')
        features_dir = os.path.join(data_dir, 'features')
        
        # Create output directory
        os.makedirs(features_dir, exist_ok=True)
        
        # Read ratings and create user-item matrix
        logger.info("Reading ratings and creating user-item matrix")
        user_item_matrix, user_ids, book_ids = read_ratings(processed_dir)
        
        if user_item_matrix.shape[0] == 0 or user_item_matrix.shape[1] == 0:
            logger.error("Failed to create user-item matrix")
            return 1
        
        # Save user-item matrix
        sp.save_npz(os.path.join(features_dir, 'user_item_matrix.npz'), 
                    user_item_matrix)
        
        # Save book IDs
        np.save(os.path.join(features_dir, 'book_ids.npy'), book_ids)
        
        logger.info(f"Successfully built and saved features to {features_dir}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}")
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Build features for book recommender')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Base directory for data')
    
    args = parser.parse_args()
    
    sys.exit(main(args.data_dir))
