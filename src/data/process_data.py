# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
import sys
import traceback
from typing import Tuple, List, Dict, Optional, Union, Any
from datetime import datetime

# Set up logging
log_dir = os.path.join('logs')
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(log_dir, f'process_data_{timestamp}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('process_data')

def load_book_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    Load book metadata from CSV files.
    
    Parameters
    ----------
    file_path : str
        Path to the directory containing book data files
        
    Returns
    -------
    Optional[pd.DataFrame]
        DataFrame containing book metadata or None if loading fails
    """
    logger.info(f"Loading book data from {file_path}")
    
    try:
        books_path = os.path.join(file_path, 'books.csv')
        if not os.path.exists(books_path):
            logger.error(f"Books file not found: {books_path}")
            logger.info("Please run fetch_hardcover_books.py to generate book data first")
            return None
            
        df = pd.read_csv(books_path)
        logger.info(f"Loaded book data with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading book data: {e}")
        logger.debug(traceback.format_exc())
        return None


def load_ratings_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    Load user ratings data from CSV files.
    
    Parameters
    ----------
    file_path : str
        Path to the directory containing ratings data files
        
    Returns
    -------
    Optional[pd.DataFrame]
        DataFrame containing user ratings or None if loading fails
    """
    logger.info(f"Loading ratings data from {file_path}")
    
    try:
        ratings_path = os.path.join(file_path, 'ratings.csv')
        if not os.path.exists(ratings_path):
            logger.error(f"Ratings file not found: {ratings_path}")
            logger.info("Please run fetch_hardcover_books.py to generate ratings data first")
            return None
            
        df = pd.read_csv(ratings_path)
        logger.info(f"Loaded ratings data with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading ratings data: {e}")
        logger.debug(traceback.format_exc())
        return None


def clean_book_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess book metadata.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing book metadata
        
    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame containing book metadata with only complete records
    """
    logger.info("Cleaning book data")
    
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Log initial count
    logger.info(f"Initial book count: {len(df_clean)}")
    
    # Filter out books with missing critical metadata
    critical_columns = ['title', 'authors']
    initial_count = len(df_clean)
    
    # Drop rows where critical columns are missing
    df_clean = df_clean.dropna(subset=critical_columns)
    
    # Log the number of books removed due to missing metadata
    removed_count = initial_count - len(df_clean)
    logger.info(f"Removed {removed_count} books with missing critical metadata (title or authors)")
    
    # Convert numeric columns
    numeric_cols = ['book_id', 'id', 'best_book_id', 'work_id', 'books_count', 
                     'average_rating', 'ratings_count', 'work_ratings_count', 
                     'work_text_reviews_count', 'ratings_1', 'ratings_2', 
                     'ratings_3', 'ratings_4', 'ratings_5']
    
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Drop rows with missing book_id as this is critical for joining
    df_clean = df_clean.dropna(subset=['book_id'])
    
    # Handle non-critical missing values that can be filled
    df_clean['isbn'] = df_clean['isbn'].fillna('')
    df_clean['isbn13'] = df_clean['isbn13'].fillna(np.nan)
    df_clean['original_publication_year'] = df_clean['original_publication_year'].fillna(np.nan)
    df_clean['original_title'] = df_clean['original_title'].fillna(df_clean['title'])
    df_clean['language_code'] = df_clean['language_code'].fillna('eng')
    df_clean['average_rating'] = df_clean['average_rating'].fillna(0)
    df_clean['ratings_count'] = df_clean['ratings_count'].fillna(0)
    
    # Fill remaining numeric columns with zeros
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(0)
    
    # No need to validate image URLs since they come from Google Books API
    if 'image_url' in df_clean.columns:
        df_clean['image_url'] = df_clean['image_url'].fillna('')
    
    # Ensure book_id is integer type
    df_clean['book_id'] = df_clean['book_id'].astype(int)
    
    logger.info(f"Cleaned book data with shape {df_clean.shape}")
    return df_clean


def clean_ratings_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess user ratings data.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing user ratings
        
    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame containing user ratings
    """
    logger.info("Cleaning ratings data")
    
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Convert columns to appropriate types
    df_clean['user_id'] = pd.to_numeric(df_clean['user_id'], errors='coerce')
    df_clean['book_id'] = pd.to_numeric(df_clean['book_id'], errors='coerce')
    df_clean['rating'] = pd.to_numeric(df_clean['rating'], errors='coerce')
    
    # Drop rows with missing values
    df_clean = df_clean.dropna()
    
    # Keep only ratings within the valid range (1-5)
    df_clean = df_clean[(df_clean['rating'] >= 1) & (df_clean['rating'] <= 5)]
    
    # Ensure user_id and book_id are integers
    df_clean['user_id'] = df_clean['user_id'].astype(int)
    df_clean['book_id'] = df_clean['book_id'].astype(int)
    
    # Log the unique counts
    logger.info(f"Number of unique users: {df_clean['user_id'].nunique()}")
    logger.info(f"Number of unique books: {df_clean['book_id'].nunique()}")
    logger.info(f"Number of ratings: {len(df_clean)}")
    
    logger.info(f"Cleaned ratings data with shape {df_clean.shape}")
    return df_clean


def merge_and_prepare_data(books_df: pd.DataFrame, ratings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge book and ratings data, and prepare for model training.
    
    Parameters
    ----------
    books_df : pd.DataFrame
        DataFrame containing book metadata
    ratings_df : pd.DataFrame
        DataFrame containing user ratings
        
    Returns
    -------
    pd.DataFrame
        Merged DataFrame
    """
    logger.info("Merging and preparing data")
    
    # Make sure book_id is of the same type in both dataframes
    books_df['book_id'] = books_df['book_id'].astype(int)
    ratings_df['book_id'] = ratings_df['book_id'].astype(int)
    
    # Merge ratings with book metadata
    merged_df = pd.merge(ratings_df, books_df, on='book_id', how='inner')
    
    # Report merging results
    logger.info(f"Number of unique users in merged data: {merged_df['user_id'].nunique()}")
    logger.info(f"Number of unique books in merged data: {merged_df['book_id'].nunique()}")
    logger.info(f"Number of ratings in merged data: {len(merged_df)}")
    
    # Check for any unexpected data loss
    if len(merged_df) < len(ratings_df):
        logger.warning(f"Some ratings were lost in the merge! Original: {len(ratings_df)}, After merge: {len(merged_df)}")
        
        # Find which book_ids in ratings don't exist in books
        missing_book_ids = set(ratings_df['book_id']) - set(books_df['book_id'])
        logger.warning(f"Number of book_ids in ratings that don't exist in books: {len(missing_book_ids)}")
        if len(missing_book_ids) > 0:
            logger.warning(f"Sample of missing book_ids: {list(missing_book_ids)[:5]}")
    
    logger.info(f"Merged data shape: {merged_df.shape}")
    return merged_df


def train_test_split(df: pd.DataFrame, test_size: float = 0.2, 
                    random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and test sets.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to split
    test_size : float
        Fraction of data to use for testing
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df)
    """
    logger.info(f"Splitting data with test_size={test_size}, random_state={random_state}")
    
    if len(df) == 0:
        logger.warning("Empty DataFrame provided for splitting")
        return pd.DataFrame(columns=df.columns), pd.DataFrame(columns=df.columns)
    
    # Shuffle the DataFrame
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Calculate split point
    split_idx = int(len(df_shuffled) * (1 - test_size))
    
    # Split the data
    train_df = df_shuffled.iloc[:split_idx].reset_index(drop=True)
    test_df = df_shuffled.iloc[split_idx:].reset_index(drop=True)
    
    logger.info(f"Training set shape: {train_df.shape}")
    logger.info(f"Test set shape: {test_df.shape}")
    
    return train_df, test_df


def main(input_filepath: str = 'data/raw', output_filepath: str = 'data/processed') -> int:
    """
    Main function to load, clean, merge and save data.
    
    Parameters
    ----------
    input_filepath : str
        Path to the directory containing input data files
    output_filepath : str
        Directory to save processed data
    
    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure)
    """
    logger.info('Making final data set from raw data.')
    
    # Load books data
    books_df = load_book_data(input_filepath)
    
    # If no books data, exit
    if books_df is None or len(books_df) == 0:
        logger.error("No books data available. Please generate books data first.")
        return 1
    
    # Load ratings data
    ratings_df = load_ratings_data(input_filepath)
    
    # If no ratings data, exit
    if ratings_df is None or len(ratings_df) == 0:
        logger.error("No ratings data available. Please generate ratings data first.")
        return 1
    
    # Clean data
    books_df = clean_book_data(books_df)
    ratings_df = clean_ratings_data(ratings_df)
    
    # Merge the data
    merged_df = merge_and_prepare_data(books_df, ratings_df)
    
    # Split into train and test
    train_df, test_df = train_test_split(merged_df, test_size=0.2, random_state=42)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_filepath, exist_ok=True)
    
    # Save processed data
    books_df.to_csv(os.path.join(output_filepath, 'books.csv'), index=False)
    ratings_df.to_csv(os.path.join(output_filepath, 'ratings.csv'), index=False)
    merged_df.to_csv(os.path.join(output_filepath, 'merged.csv'), index=False)
    train_df.to_csv(os.path.join(output_filepath, 'merged_train.csv'), index=False)
    test_df.to_csv(os.path.join(output_filepath, 'merged_test.csv'), index=False)
    
    logger.info('Data processing completed successfully.')
    return 0


@click.command()
@click.option('--input-filepath', type=click.Path(exists=True), default='data/raw',
              help='Path to the directory containing input data files')
@click.option('--output-filepath', type=click.Path(), default='data/processed',
              help='Directory to save processed data')
def cli(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    sys.exit(main(input_filepath, output_filepath))


if __name__ == "__main__":
    # If run directly, use default paths without requiring command line arguments
    sys.exit(main())
