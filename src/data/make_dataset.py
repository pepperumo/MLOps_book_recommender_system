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
log_filename = os.path.join(log_dir, f'make_dataset_{timestamp}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('make_dataset')


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
            return None
            
        df = pd.read_csv(books_path)
        logger.info(f"Loaded book data with shape {df.shape}")
        
        # Check required column
        if 'book_id' not in df.columns:
            logger.error("Missing required 'book_id' column in books file")
            return None
        
        # Log data info
        for col in df.columns:
            non_null = df[col].count()
            logger.info(f"Column '{col}': {non_null}/{len(df)} non-null values " +
                      f"({100 * non_null / len(df):.1f}%)")
        
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
            return None
            
        df = pd.read_csv(ratings_path)
        logger.info(f"Loaded ratings data with shape {df.shape}")
        
        # Check required columns
        required_cols = ['user_id', 'book_id', 'rating']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns in ratings file: {missing_cols}")
            return None
            
        # Log data info
        logger.info(f"Ratings range: {df['rating'].min()} to {df['rating'].max()}")
        logger.info(f"Number of unique users: {df['user_id'].nunique()}")
        logger.info(f"Number of unique books: {df['book_id'].nunique()}")
        
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
        Cleaned DataFrame containing book metadata
    """
    logger.info("Cleaning book data")
    
    try:
        # Make a copy to avoid modifying the original
        df_clean = df.copy()
        
        # Convert book_id to integer if possible
        try:
            df_clean['book_id'] = df_clean['book_id'].astype(int)
            logger.info("Converted book_id to integer type")
        except Exception as e:
            logger.warning(f"Could not convert book_id to integer: {e}")
        
        # Check for and remove duplicates
        dupe_count = df_clean.duplicated(subset=['book_id']).sum()
        if dupe_count > 0:
            logger.warning(f"Found {dupe_count} duplicate book_ids")
            df_clean = df_clean.drop_duplicates(subset=['book_id'])
            logger.info(f"Removed {dupe_count} duplicate books")
        
        # Fill missing values for important columns
        for col, fill_value in [
            ('title', 'Unknown Title'),
            ('authors', 'Unknown Author'),
            ('average_rating', 0.0),
            ('ratings_count', 0)
        ]:
            if col in df_clean.columns:
                null_count = df_clean[col].isnull().sum()
                if null_count > 0:
                    logger.warning(f"Filling {null_count} missing values in '{col}' with '{fill_value}'")
                    df_clean[col] = df_clean[col].fillna(fill_value)
        
        logger.info(f"Cleaned book data shape: {df_clean.shape}")
        return df_clean
        
    except Exception as e:
        logger.error(f"Error cleaning book data: {e}")
        logger.debug(traceback.format_exc())
        return df  # Return original data if cleaning fails


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
    
    try:
        # Make a copy to avoid modifying the original
        df_clean = df.copy()
        
        # Convert IDs to integers if possible
        for id_col in ['user_id', 'book_id']:
            if id_col in df_clean.columns:
                try:
                    df_clean[id_col] = df_clean[id_col].astype(int)
                    logger.info(f"Converted {id_col} to integer type")
                except Exception as e:
                    logger.warning(f"Could not convert {id_col} to integer: {e}")
        
        # Check for and remove duplicates
        dupe_count = df_clean.duplicated().sum()
        if dupe_count > 0:
            logger.warning(f"Found {dupe_count} duplicate rows in ratings data")
            df_clean = df_clean.drop_duplicates()
            logger.info(f"Removed {dupe_count} duplicate rows")
        
        # Check for and remove ratings outside valid range (assuming 1-5)
        valid_range = (1, 5)
        invalid_count = df_clean[(df_clean['rating'] < valid_range[0]) | 
                               (df_clean['rating'] > valid_range[1])].shape[0]
        
        if invalid_count > 0:
            logger.warning(f"Found {invalid_count} ratings outside valid range {valid_range}")
            df_clean = df_clean[(df_clean['rating'] >= valid_range[0]) & 
                             (df_clean['rating'] <= valid_range[1])]
            logger.info(f"Removed {invalid_count} invalid ratings")
            
        logger.info(f"Cleaned ratings data shape: {df_clean.shape}")
        return df_clean
        
    except Exception as e:
        logger.error(f"Error cleaning ratings data: {e}")
        logger.debug(traceback.format_exc())
        return df  # Return original data if cleaning fails


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
    logger.info("Merging book and ratings data")
    
    try:
        # Merge on book_id
        merged_df = pd.merge(ratings_df, books_df, on='book_id', how='inner')
        
        # Log merge results
        ratings_count_before = len(ratings_df)
        ratings_count_after = len(merged_df)
        
        logger.info(f"Merge results: {ratings_count_after}/{ratings_count_before} ratings retained " +
                  f"({100 * ratings_count_after / ratings_count_before:.1f}%)")
        logger.info(f"Merged data shape: {merged_df.shape}")
        
        # Create a summary of the merged data
        summary = {
            'num_users': merged_df['user_id'].nunique(),
            'num_books': merged_df['book_id'].nunique(),
            'num_ratings': len(merged_df),
            'avg_rating': merged_df['rating'].mean(),
            'rating_density': len(merged_df) / (merged_df['user_id'].nunique() * merged_df['book_id'].nunique()),
            'timestamp': timestamp
        }
        
        # Save the summary
        summary_df = pd.DataFrame([summary])
        output_dir = os.path.join('data', 'processed')
        os.makedirs(output_dir, exist_ok=True)
        summary_file = os.path.join(output_dir, f'data_summary_{timestamp}.csv')
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"Saved data summary to {summary_file}")
        
        return merged_df
        
    except Exception as e:
        logger.error(f"Error merging data: {e}")
        logger.debug(traceback.format_exc())
        # Create an empty DataFrame with the expected columns
        columns = list(set(ratings_df.columns) | set(books_df.columns))
        return pd.DataFrame(columns=columns)


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
    
    try:
        # Simple random split
        np.random.seed(random_state)
        msk = np.random.rand(len(df)) < (1 - test_size)
        train_df = df[msk]
        test_df = df[~msk]
        
        logger.info(f"Split result: train={len(train_df)} samples, test={len(test_df)} samples")
        
        # Verify that split preserves users and books
        train_users = set(train_df['user_id'].unique())
        test_users = set(test_df['user_id'].unique())
        train_books = set(train_df['book_id'].unique())
        test_books = set(test_df['book_id'].unique())
        
        user_overlap = len(train_users & test_users)
        book_overlap = len(train_books & test_books)
        
        logger.info(f"User overlap: {user_overlap}/{len(test_users)} test users also in train " +
                  f"({100 * user_overlap / len(test_users):.1f}%)")
        logger.info(f"Book overlap: {book_overlap}/{len(test_books)} test books also in train " +
                  f"({100 * book_overlap / len(test_books):.1f}%)")
        
        return train_df, test_df
        
    except Exception as e:
        logger.error(f"Error splitting data: {e}")
        logger.debug(traceback.format_exc())
        # If splitting fails, return the original data as train and an empty DataFrame as test
        return df, pd.DataFrame(columns=df.columns)


def main(input_filepath: str, output_filepath: str) -> int:
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
        Exit code
    """
    logger.info(f"Starting make_dataset.py with input_filepath={input_filepath}, output_filepath={output_filepath}")
    
    try:
        # Create output directory
        os.makedirs(output_filepath, exist_ok=True)
        
        # Step 1: Load data
        logger.info("Step 1: Loading data")
        books_df = load_book_data(input_filepath)
        if books_df is None:
            logger.error("Failed to load book data, exiting")
            return 1
            
        ratings_df = load_ratings_data(input_filepath)
        if ratings_df is None:
            logger.error("Failed to load ratings data, exiting")
            return 1
        
        # Step 2: Clean data
        logger.info("Step 2: Cleaning data")
        books_df = clean_book_data(books_df)
        ratings_df = clean_ratings_data(ratings_df)
        
        # Step 3: Merge data
        logger.info("Step 3: Merging data")
        merged_df = merge_and_prepare_data(books_df, ratings_df)
        
        if len(merged_df) == 0:
            logger.error("Merged data is empty, exiting")
            return 1
        
        # Step 4: Split data
        logger.info("Step 4: Splitting data")
        train_df, test_df = train_test_split(merged_df)
        
        # Save train and test data
        train_file = os.path.join(output_filepath, 'merged_train.csv')
        test_file = os.path.join(output_filepath, 'merged_test.csv')
        
        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)
        
        logger.info(f"Saved train data to {train_file} with shape {train_df.shape}")
        logger.info(f"Saved test data to {test_file} with shape {test_df.shape}")
        
        # Save standalone clean data files for reference
        clean_ratings_file = os.path.join(output_filepath, 'clean_ratings.csv')
        clean_books_file = os.path.join(output_filepath, 'clean_books.csv')
        
        ratings_df.to_csv(clean_ratings_file, index=False)
        books_df.to_csv(clean_books_file, index=False)
        
        logger.info(f"Saved clean ratings to {clean_ratings_file} with shape {ratings_df.shape}")
        logger.info(f"Saved clean books to {clean_books_file} with shape {books_df.shape}")
        
        logger.info("make_dataset.py completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}")
        logger.debug(traceback.format_exc())
        return 1


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def cli(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    sys.exit(main(input_filepath, output_filepath))


if __name__ == "__main__":
    cli()
