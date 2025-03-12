"""
Test prediction functionality for book recommender models.

This script runs comprehensive tests for all three recommendation models:
1. Collaborative filtering
2. Content-based filtering 
3. Hybrid recommender

Tests include:
- User recommendations
- Similar book recommendations
- Performance comparisons
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

# Configure logging
log_dir = os.path.join('logs')
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(log_dir, f'test_prediction_{timestamp}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('test_prediction')

# Import recommender functions
try:
    from src.models.predict_model import (
        recommend_for_user, 
        recommend_similar_books,
        load_recommender_model,
        load_book_id_mapping
    )
except ImportError:
    try:
        from models.predict_model import (
            recommend_for_user, 
            recommend_similar_books,
            load_recommender_model,
            load_book_id_mapping
        )
    except ImportError:
        # Add parent directory to path
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.append(parent_dir)
        from models.predict_model import (
            recommend_for_user, 
            recommend_similar_books,
            load_recommender_model,
            load_book_id_mapping
        )
        logger.warning("Using relative imports for prediction functions")

# Define model types
MODEL_TYPES = ['collaborative', 'content', 'hybrid']

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
    model_types: List[str] = MODEL_TYPES,
    num_recommendations: int = 5,
    data_dir: str = 'data'
) -> Dict[str, Any]:
    """
    Test user recommendations for multiple users across all model types.
    
    Parameters
    ----------
    user_ids : List[int]
        List of user IDs to test
    model_types : List[str]
        List of model types to test
    num_recommendations : int
        Number of recommendations to generate
    data_dir : str
        Data directory path
        
    Returns
    -------
    Dict[str, Any]
        Dictionary of test results
    """
    results = {}
    
    for model_type in model_types:
        logger.info(f"Testing {model_type} recommendations for {len(user_ids)} users")
        
        model_results = {
            'user_results': {},
            'timing': [],
            'recommendation_count': [],
            'recommendation_overlap': []
        }
        
        for user_id in user_ids:
            start_time = time.time()
            
            # Get recommendations for this user
            recommendations_df = recommend_for_user(
                user_id=user_id,
                model_type=model_type,
                num_recommendations=num_recommendations,
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
        
        results[model_type] = model_results
    
    return results

def test_similar_books(
    book_ids: List[int], 
    model_types: List[str] = MODEL_TYPES,
    num_recommendations: int = 5,
    data_dir: str = 'data'
) -> Dict[str, Any]:
    """
    Test similar book recommendations for multiple books across all model types.
    
    Parameters
    ----------
    book_ids : List[int]
        List of book IDs to test
    model_types : List[str]
        List of model types to test
    num_recommendations : int
        Number of recommendations to generate
    data_dir : str
        Data directory path
        
    Returns
    -------
    Dict[str, Any]
        Dictionary of test results
    """
    results = {}
    
    for model_type in model_types:
        logger.info(f"Testing {model_type} similar book recommendations for {len(book_ids)} books")
        
        # Skip collaborative for similar books as it may not be implemented
        if model_type == 'collaborative':
            logger.info("Skipping collaborative model for similar books test")
            continue
        
        model_results = {
            'book_results': {},
            'timing': [],
            'recommendation_count': []
        }
        
        for book_id in book_ids:
            start_time = time.time()
            
            # Get recommendations for this book
            similar_books_df = recommend_similar_books(
                book_id=book_id,
                model_type=model_type,
                num_recommendations=num_recommendations,
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
        
        results[model_type] = model_results
    
    return results

def test_cold_start(
    model_types: List[str] = MODEL_TYPES,
    num_recommendations: int = 5,
    data_dir: str = 'data'
) -> Dict[str, Any]:
    """
    Test how models handle cold-start users (users with few or no ratings).
    
    Parameters
    ----------
    model_types : List[str]
        List of model types to test
    num_recommendations : int
        Number of recommendations to generate
    data_dir : str
        Data directory path
        
    Returns
    -------
    Dict[str, Any]
        Dictionary of test results
    """
    results = {}
    
    # Create a non-existent user ID that's not in the dataset
    # This is a cold-start scenario
    ratings_path = os.path.join(data_dir, 'processed', 'merged_train.csv')
    df = pd.read_csv(ratings_path, encoding='utf-8')
    max_user_id = df['user_id'].max()
    cold_start_user_id = max_user_id + 100  # Make sure it's not in the dataset
    
    logger.info(f"Testing cold-start with user ID {cold_start_user_id}")
    
    for model_type in model_types:
        logger.info(f"Testing {model_type} model for cold-start")
        
        start_time = time.time()
        
        # Get recommendations for this cold-start user
        recommendations_df = recommend_for_user(
            user_id=cold_start_user_id,
            model_type=model_type,
            num_recommendations=num_recommendations,
            data_dir=data_dir
        )
        
        # Calculate timing
        duration = time.time() - start_time
        
        # Store results
        results[model_type] = {
            'recommendations': recommendations_df.to_dict() if not recommendations_df.empty else {},
            'count': len(recommendations_df),
            'timing': duration
        }
    
    return results

def summarize_results(
    user_results: Dict[str, Any],
    similar_book_results: Dict[str, Any],
    cold_start_results: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
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
    Dict[str, Dict[str, Any]]
        Summarized results
    """
    summary = {}
    
    # Process user recommendation results
    for model_type, results in user_results.items():
        if model_type not in summary:
            summary[model_type] = {}
        
        avg_timing = np.mean(results['timing']) if results['timing'] else 0
        avg_count = np.mean(results['recommendation_count']) if results['recommendation_count'] else 0
        avg_overlap = np.mean(results['recommendation_overlap']) if results.get('recommendation_overlap') else 0
        
        summary[model_type]['avg_user_recommendation_time'] = avg_timing
        summary[model_type]['avg_user_recommendations_count'] = avg_count
        summary[model_type]['avg_user_recommendation_overlap'] = avg_overlap
    
    # Process similar book results
    for model_type, results in similar_book_results.items():
        if model_type not in summary:
            summary[model_type] = {}
        
        avg_timing = np.mean(results['timing']) if results['timing'] else 0
        avg_count = np.mean(results['recommendation_count']) if results['recommendation_count'] else 0
        
        summary[model_type]['avg_similar_book_time'] = avg_timing
        summary[model_type]['avg_similar_book_count'] = avg_count
    
    # Process cold-start results
    for model_type, results in cold_start_results.items():
        if model_type not in summary:
            summary[model_type] = {}
        
        summary[model_type]['cold_start_time'] = results['timing']
        summary[model_type]['cold_start_count'] = results['count']
    
    return summary

def display_results(summary: Dict[str, Dict[str, Any]]) -> None:
    """
    Display test results in a formatted table.
    
    Parameters
    ----------
    summary : Dict[str, Dict[str, Any]]
        Summary of test results
    """
    # Create table for display
    headers = ['Metric'] + list(summary.keys())
    
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
        'cold_start_time': 'Cold Start Recommendation Time (s)',
        'cold_start_count': 'Cold Start Recommendation Count'
    }
    
    rows = []
    
    for metric in metrics:
        row = [metric_display_names.get(metric, metric)]
        
        for model_type in summary.keys():
            value = summary[model_type].get(metric, 'N/A')
            if isinstance(value, float):
                value = f"{value:.3f}"
            row.append(value)
        
        rows.append(row)
    
    # Display table
    print("\n" + "=" * 80)
    print("RECOMMENDATION MODEL TEST RESULTS")
    print("=" * 80)
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    print("=" * 80 + "\n")
    
    # Save results to JSON
    results_dir = os.path.join('data', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, f'prediction_test_results_{timestamp}.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, default=str)
    
    logger.info(f"Results saved to data/results/prediction_test_results_{timestamp}.json")

def generate_visualization(summary: Dict[str, Dict[str, Any]], output_dir: str = 'figures') -> None:
    """
    Generate visualizations for test results.
    
    Parameters
    ----------
    summary : Dict[str, Dict[str, Any]]
        Summary of test results
    output_dir : str
        Output directory for figures
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define metrics to plot
    time_metrics = [
        ('avg_user_recommendation_time', 'User Recommendation Time (s)'),
        ('avg_similar_book_time', 'Similar Book Recommendation Time (s)'),
        ('cold_start_time', 'Cold Start Recommendation Time (s)')
    ]
    
    count_metrics = [
        ('avg_user_recommendations_count', 'User Recommendation Count'),
        ('avg_similar_book_count', 'Similar Book Recommendation Count'),
        ('cold_start_count', 'Cold Start Recommendation Count')
    ]
    
    model_types = list(summary.keys())
    
    # Create time comparison plot
    plt.figure(figsize=(10, 6))
    
    bar_width = 0.25
    bar_positions = np.arange(len(time_metrics))
    
    for i, model_type in enumerate(model_types):
        values = []
        for metric, _ in time_metrics:
            value = summary[model_type].get(metric, 0)
            values.append(value if isinstance(value, (int, float)) else 0)
        
        plt.bar(
            bar_positions + i * bar_width, 
            values, 
            width=bar_width, 
            label=model_type.capitalize()
        )
    
    plt.xlabel('Metric')
    plt.ylabel('Time (seconds)')
    plt.title('Recommendation Time Comparison')
    plt.xticks(bar_positions + bar_width, [label for _, label in time_metrics])
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, f'time_comparison_{timestamp}.png'))
    logger.info(f"Time comparison plot saved to {output_dir}/time_comparison_{timestamp}.png")
    
    # Create count comparison plot
    plt.figure(figsize=(10, 6))
    
    bar_positions = np.arange(len(count_metrics))
    
    for i, model_type in enumerate(model_types):
        values = []
        for metric, _ in count_metrics:
            value = summary[model_type].get(metric, 0)
            values.append(value if isinstance(value, (int, float)) else 0)
        
        plt.bar(
            bar_positions + i * bar_width, 
            values, 
            width=bar_width, 
            label=model_type.capitalize()
        )
    
    plt.xlabel('Metric')
    plt.ylabel('Count')
    plt.title('Recommendation Count Comparison')
    plt.xticks(bar_positions + bar_width, [label for _, label in count_metrics])
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, f'count_comparison_{timestamp}.png'))
    logger.info(f"Count comparison plot saved to {output_dir}/count_comparison_{timestamp}.png")

def main():
    """
    Run comprehensive prediction tests on all recommendation models.
    """
    logger.info("Starting prediction tests")
    
    # Load test data
    users_df, books_df = load_test_data()
    
    if users_df.empty or books_df.empty:
        logger.error("Failed to load test data")
        return 1
    
    # Select a sample of users and books for testing
    sample_user_ids = np.random.choice(users_df['user_id'].values, min(10, len(users_df)), replace=False).tolist()
    sample_book_ids = np.random.choice(books_df['book_id'].values, min(10, len(books_df)), replace=False).tolist()
    
    logger.info(f"Selected {len(sample_user_ids)} users and {len(sample_book_ids)} books for testing")
    
    # Run tests
    logger.info("Testing user recommendations")
    user_results = test_user_recommendations(sample_user_ids)
    
    logger.info("Testing similar book recommendations")
    similar_book_results = test_similar_books(sample_book_ids)
    
    logger.info("Testing cold-start scenario")
    cold_start_results = test_cold_start()
    
    # Summarize results
    summary = summarize_results(user_results, similar_book_results, cold_start_results)
    
    # Display results
    display_results(summary)
    
    # Generate visualizations
    generate_visualization(summary)
    
    logger.info("Testing complete")
    return 0

if __name__ == "__main__":
    sys.exit(main())
