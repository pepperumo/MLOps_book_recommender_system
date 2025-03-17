"""
Test prediction functionality for collaborative filtering book recommender model.

This script runs comprehensive tests for the collaborative filtering recommendation model.

Tests include:
- User recommendations
- Performance analysis
- Coverage analysis
- Cold-start behavior
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
import time
import json
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional, Union
import matplotlib.pyplot as plt
from tabulate import tabulate

# Add the project root to the Python path so we can import modules correctly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Configure logging
log_dir = os.path.join('logs')
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(log_dir, f'test_model_{timestamp}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('test_model')

# Import recommender functions and classes
try:
    from src.models.model_utils import BaseRecommender, load_data
    from src.models.train_model import CollaborativeRecommender
    from src.models.predict_model import (
        recommend_for_user, 
        recommend_similar_books,
        load_recommender_model,
        load_book_id_mapping
    )
except ImportError:
    try:
        from models.model_utils import BaseRecommender, load_data
        from models.train_model import CollaborativeRecommender
        from models.predict_model import (
            recommend_for_user, 
            recommend_similar_books,
            load_recommender_model,
            load_book_id_mapping
        )
    except ImportError:
        import sys
        import os
        # Add the parent directory to the path to ensure we can import the modules
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.append(parent_dir)
        try:
            from models.model_utils import BaseRecommender, load_data
            from models.train_model import CollaborativeRecommender
            from models.predict_model import (
                recommend_for_user, 
                recommend_similar_books,
                load_recommender_model,
                load_book_id_mapping
            )
            logger.info("Imported from models directory after adding parent dir to path")
        except ImportError:
            logger.error("Failed to import necessary modules. Please check your installation.")
            sys.exit(1)

# Define model type - collaborative only
MODEL_TYPE = 'collaborative'

def load_test_data(data_dir: str = 'data/processed') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load test data for prediction tests.
    
    Parameters
    ----------
    data_dir : str
        Directory containing processed data files
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Tuple of (users_df, books_df)
    """
    try:
        # Load a sample of test data
        ratings_path = os.path.join(data_dir, 'merged_train.csv')
        
        logger.info(f"Loading ratings data from {ratings_path}")
        ratings_df = pd.read_csv(ratings_path, encoding='utf-8')
        
        # Get unique users and books
        users_df = pd.DataFrame({'user_id': ratings_df['user_id'].unique()})
        books_df = ratings_df[['book_id', 'title', 'authors']].drop_duplicates()
        
        logger.info(f"Loaded {len(users_df)} users and {len(books_df)} books")
        
        return users_df, books_df
    
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        return pd.DataFrame(), pd.DataFrame()

def test_user_recommendations(
    user_ids: List[int], 
    n: int = 5,
    data_dir: str = 'data'
) -> Dict[str, Any]:
    """
    Test user recommendations for multiple users with the collaborative filtering model.
    
    Parameters
    ----------
    user_ids : List[int]
        List of user IDs to test
    n : int
        Number of recommendations to generate
    data_dir : str
        Data directory path
        
    Returns
    -------
    Dict[str, Any]
        Dictionary of test results
    """
    model_results = {
        'user_results': {},
        'timing': [],
        'recommendation_count': [],
        'recommendation_overlap': []
    }
    
    logger.info(f"Testing collaborative recommendations for {len(user_ids)} users")
    
    for user_id in user_ids:
        start_time = time.time()
        
        # Get recommendations for this user
        recommendations_df = recommend_for_user(
            user_id=user_id,
            model_type=MODEL_TYPE,
            n=n,
            data_dir=data_dir
        )
        
        # Calculate timing
        duration = time.time() - start_time
        
        # Store results
        model_results['user_results'][user_id] = {
            'recommendations': recommendations_df['book_id'].tolist() if not recommendations_df.empty else [],
            'count': len(recommendations_df),
            'timing': duration
        }
        
        model_results['timing'].append(duration)
        model_results['recommendation_count'].append(len(recommendations_df))
    
    # Calculate overlap between users' recommendations
    if len(user_ids) > 1:
        overlaps = []
        for i, user1 in enumerate(user_ids[:-1]):
            for user2 in user_ids[i+1:]:
                recs1 = model_results['user_results'][user1]['recommendations']
                recs2 = model_results['user_results'][user2]['recommendations']
                
                if recs1 and recs2:
                    set1 = set(recs1)
                    set2 = set(recs2)
                    
                    if set1 and set2:
                        overlap = len(set1.intersection(set2)) / len(set1.union(set2))
                        overlaps.append(overlap)
        
        if overlaps:
            model_results['recommendation_overlap'] = overlaps
    
    return model_results

def test_similar_books(
    book_ids: List[int], 
    n: int = 5,
    data_dir: str = 'data'
) -> Dict[str, Any]:
    """
    Test similar book recommendations for multiple books with the collaborative filtering model.
    
    Parameters
    ----------
    book_ids : List[int]
        List of book IDs to test
    n : int
        Number of recommendations to generate
    data_dir : str
        Data directory path
        
    Returns
    -------
    Dict[str, Any]
        Dictionary of test results
    """
    model_results = {
        'book_results': {},
        'timing': [],
        'recommendation_count': []
    }
    
    logger.info(f"Testing collaborative similar book recommendations for {len(book_ids)} books")
    
    for book_id in book_ids:
        start_time = time.time()
        
        # Get recommendations for this book
        similar_books_df = recommend_similar_books(
            book_id=book_id,
            model_type=MODEL_TYPE,
            n=n,
            data_dir=data_dir
        )
        
        # Calculate timing
        duration = time.time() - start_time
        
        # Store results
        model_results['book_results'][book_id] = {
            'recommendations': similar_books_df['book_id'].tolist() if not similar_books_df.empty else [],
            'count': len(similar_books_df),
            'timing': duration
        }
        
        model_results['timing'].append(duration)
        model_results['recommendation_count'].append(len(similar_books_df))
    
    return model_results

def test_cold_start(
    n: int = 5,
    data_dir: str = 'data'
) -> Dict[str, Any]:
    """
    Test how the collaborative model handles cold-start users (users with few or no ratings).
    
    Parameters
    ----------
    n : int
        Number of recommendations to generate
    data_dir : str
        Data directory path
        
    Returns
    -------
    Dict[str, Any]
        Dictionary of test results
    """
    # Create a non-existent user ID that's not in the dataset
    # This is a cold-start scenario
    ratings_path = os.path.join(data_dir, 'processed', 'merged_train.csv')
    df = pd.read_csv(ratings_path, encoding='utf-8')
    max_user_id = df['user_id'].max()
    cold_start_user_id = max_user_id + 100  # Make sure it's not in the dataset
    
    logger.info(f"Testing cold-start with user ID {cold_start_user_id}")
    logger.info(f"Testing collaborative model for cold-start")
    
    start_time = time.time()
    
    # Get recommendations for this cold-start user
    recommendations_df = recommend_for_user(
        user_id=cold_start_user_id,
        model_type=MODEL_TYPE,
        n=n,
        data_dir=data_dir
    )
    
    # Calculate timing
    duration = time.time() - start_time
    
    # Store results
    results = {
        'recommendations': recommendations_df.to_dict() if not recommendations_df.empty else {},
        'count': len(recommendations_df),
        'timing': duration
    }
    
    return results

def summarize_results(
    user_results: Dict[str, Any],
    similar_book_results: Dict[str, Any],
    cold_start_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Summarize test results for reporting.
    
    Parameters
    ----------
    user_results : Dict[str, Any]
        Results from user recommendation tests
    similar_book_results : Dict[str, Any]
        Results from similar book recommendation tests
    cold_start_results : Dict[str, Any]
        Results from cold-start tests
        
    Returns
    -------
    Dict[str, Any]
        Summarized results
    """
    summary = {}
    
    # Process user recommendation results
    avg_timing = np.mean(user_results['timing']) if user_results['timing'] else 0
    avg_count = np.mean(user_results['recommendation_count']) if user_results['recommendation_count'] else 0
    avg_overlap = np.mean(user_results['recommendation_overlap']) if user_results.get('recommendation_overlap') else 0
    
    summary['avg_user_recommendation_time'] = avg_timing
    summary['avg_user_recommendations_count'] = avg_count
    summary['avg_user_recommendation_overlap'] = avg_overlap
    
    # Process similar book results
    avg_timing = np.mean(similar_book_results['timing']) if similar_book_results['timing'] else 0
    avg_count = np.mean(similar_book_results['recommendation_count']) if similar_book_results['recommendation_count'] else 0
    
    summary['avg_similar_book_time'] = avg_timing
    summary['avg_similar_book_count'] = avg_count
    
    # Process cold-start results
    summary['cold_start_time'] = cold_start_results['timing']
    summary['cold_start_count'] = cold_start_results['count']
    
    return summary

def display_results(summary: Dict[str, Any]) -> None:
    """
    Display test results in a formatted table.
    
    Parameters
    ----------
    summary : Dict[str, Any]
        Summary of test results
    """
    # Create table for display
    headers = ['Metric', 'Collaborative']
    
    metrics = [
        'avg_user_recommendation_time',
        'avg_user_recommendations_count',
        'avg_user_recommendation_overlap',
        'avg_similar_book_time',
        'avg_similar_book_count',
        'cold_start_time',
        'cold_start_count'
    ]
    
    metric_display_names = {
        'avg_user_recommendation_time': 'Avg. User Recommendation Time (s)',
        'avg_user_recommendations_count': 'Avg. User Recommendations Count',
        'avg_user_recommendation_overlap': 'Avg. User Recommendation Overlap',
        'avg_similar_book_time': 'Avg. Similar Book Time (s)',
        'avg_similar_book_count': 'Avg. Similar Book Count',
        'cold_start_time': 'Cold Start Time (s)',
        'cold_start_count': 'Cold Start Recommendation Count'
    }
    
    # Build table rows
    rows = []
    for metric in metrics:
        display_name = metric_display_names.get(metric, metric)
        row = [display_name, f"{summary.get(metric, 'N/A'):.4f}" if isinstance(summary.get(metric), (int, float)) else "N/A"]
        rows.append(row)
    
    # Print the table
    print("\n" + tabulate(rows, headers=headers, tablefmt="grid"))
    logger.info("Results table generated")


def generate_visualization(summary: Dict[str, Any], output_dir: str = 'figures') -> None:
    """
    Generate visualizations for test results.
    
    Parameters
    ----------
    summary : Dict[str, Dict[str, Any]]
        Summary of test results
    output_dir : str
        Output directory for figures
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style for plots
    plt.style.use('ggplot')
    
    # Create a bar chart for timing metrics
    timing_metrics = [
        'avg_user_recommendation_time',
        'avg_similar_book_time',
        'cold_start_time'
    ]
    
    metric_display_names = {
        'avg_user_recommendation_time': 'User Recommendations',
        'avg_similar_book_time': 'Similar Books',
        'cold_start_time': 'Cold Start'
    }
    
    timing_values = [summary.get(metric, 0) for metric in timing_metrics]
    
    plt.figure(figsize=(10, 6))
    plt.bar([metric_display_names.get(metric, metric) for metric in timing_metrics], 
            timing_values, color='royalblue')
    plt.title('Response Time Comparison (seconds)', fontsize=14)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'response_time_comparison.png'), dpi=300)
    plt.close()
    
    # Create a bar chart for recommendation count metrics
    count_metrics = [
        'avg_user_recommendations_count',
        'avg_similar_book_count',
        'cold_start_count'
    ]
    
    count_display_names = {
        'avg_user_recommendations_count': 'User Recommendations',
        'avg_similar_book_count': 'Similar Books',
        'cold_start_count': 'Cold Start'
    }
    
    count_values = [summary.get(metric, 0) for metric in count_metrics]
    
    plt.figure(figsize=(10, 6))
    plt.bar([count_display_names.get(metric, metric) for metric in count_metrics], 
            count_values, color='forestgreen')
    plt.title('Recommendation Count Comparison', fontsize=14)
    plt.ylabel('Average Number of Recommendations', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'recommendation_count_comparison.png'), dpi=300)
    plt.close()
    
    logger.info(f"Visualizations saved to {output_dir}")

def main() -> int:
    """
    Run comprehensive prediction tests on the collaborative recommender model.
    """
    try:
        logger.info("Starting prediction tests for collaborative filtering model")
        
        # Create results directory
        results_dir = os.path.join('data', 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Load test data
        users_df, books_df = load_test_data()
        
        if users_df.empty or books_df.empty:
            logger.error("Failed to load test data")
            return 1
        
        # Sample users and books for testing
        sample_size = 5
        user_sample = np.random.choice(users_df['user_id'].values, 
                                       size=min(sample_size, len(users_df)), 
                                       replace=False)
        book_sample = np.random.choice(books_df['book_id'].values, 
                                       size=min(sample_size, len(books_df)), 
                                       replace=False)
        
        # Convert numpy int64 to regular Python integers
        user_sample = [int(user_id) for user_id in user_sample]
        book_sample = [int(book_id) for book_id in book_sample]
        
        # Run tests
        user_results = test_user_recommendations(user_ids=user_sample)
        similar_book_results = test_similar_books(book_ids=book_sample)
        cold_start_results = test_cold_start()
        
        # Summarize and display results
        summary = summarize_results(user_results, similar_book_results, cold_start_results)
        display_results(summary)
        
        # Generate visualizations
        generate_visualization(summary)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error running prediction tests: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
