"""
Model evaluation module for book recommender systems.

This module provides functions for evaluating recommendation models.
"""
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

# Set up logging
logger = logging.getLogger('train_model')

def evaluate_recommender(recommender, test_df: pd.DataFrame, 
                         strategies: List[str], 
                         k_values: List[int] = [5, 10, 20],
                         sample_ratio: float = 0.1) -> Dict[str, Dict[str, float]]:
    """
    Evaluate a recommender model using precision@k and recall@k.
    
    Parameters
    ----------
    recommender : BaseRecommender
        The recommender model to evaluate
    test_df : pandas.DataFrame
        Test data with user_id and book_id columns
    strategies : list
        List of strategies to evaluate ('collaborative', 'content', 'hybrid')
    k_values : list, optional
        List of k values to evaluate (default: [5, 10, 20])
    sample_ratio : float, optional
        Ratio of test users to sample for evaluation (default: 0.1 = 10%)
        
    Returns
    -------
    dict
        Evaluation results for each strategy
    """
    results = {}
    
    # Verify test data has required columns
    if 'user_id' not in test_df.columns or 'book_id' not in test_df.columns:
        logger.error("Test data must have user_id and book_id columns")
        return {}
    
    # Group test data by user
    test_users = test_df.groupby('user_id')['book_id'].apply(list).to_dict()
    
    # Sample a subset of users to speed up evaluation
    user_ids = list(test_users.keys())
    n_sample = max(1, int(len(user_ids) * sample_ratio))
    sampled_user_ids = np.random.choice(user_ids, size=n_sample, replace=False)
    logger.info(f"Evaluating on {n_sample} users ({sample_ratio*100:.1f}% of {len(user_ids)} total users)")
    
    # For each strategy requested
    for strategy in strategies:
        # Skip if recommender doesn't support this strategy
        if not hasattr(recommender, f'recommend_for_user_{strategy}') and not hasattr(recommender, 'recommend_for_user'):
            logger.warning(f"Recommender does not support strategy: {strategy}")
            continue
            
        results[strategy] = {}
        
        # Calculate precision and recall for each k value
        for k in k_values:
            precisions = []
            recalls = []
            
            for user_id in sampled_user_ids:
                true_books = test_users[user_id]
                
                # Skip users not in training data
                if user_id not in recommender.user_ids:
                    continue
                
                # Get recommendations for this user based on strategy
                try:
                    if hasattr(recommender, f'recommend_for_user_{strategy}'):
                        # Strategy-specific recommendation method
                        recs = getattr(recommender, f'recommend_for_user_{strategy}')(user_id, n_recommendations=k)
                    else:
                        # Generic recommendation method
                        recs = recommender.recommend_for_user(user_id, n_recommendations=k)
                    
                    # Calculate precision and recall
                    n_relevant = len(set(recs) & set(true_books))
                    
                    precision = n_relevant / k if k > 0 else 0
                    recall = n_relevant / len(true_books) if len(true_books) > 0 else 0
                    
                    precisions.append(precision)
                    recalls.append(recall)
                except Exception as e:
                    logger.error(f"Error generating recommendations for user {user_id}: {e}")
                    continue
            
            # Average precision and recall
            if precisions:
                results[strategy][f'precision@{k}'] = float(np.mean(precisions))
                results[strategy][f'recall@{k}'] = float(np.mean(recalls))
    
    return results

def save_evaluation_results(evaluation_results: Dict[str, Dict[str, float]], 
                           results_dir: str = 'data/results') -> str:
    """
    Save evaluation results to a CSV file.
    
    Parameters
    ----------
    evaluation_results : dict
        Evaluation results to save
    results_dir : str, optional
        Directory to save results to (default: 'data/results')
        
    Returns
    -------
    str
        Path to the saved results file
    """
    try:
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Get timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(results_dir, f'evaluation_results_{timestamp}.csv')
        
        # Convert to DataFrame for easier saving
        results_df = pd.DataFrame()
        
        for strategy, metrics in evaluation_results.items():
            for metric, value in metrics.items():
                results_df.loc[strategy, metric] = value
        
        # Save to CSV
        results_df.to_csv(results_file)
        logger.info(f"Saved evaluation results to {results_file}")
        
        return results_file
    except Exception as e:
        logger.error(f"Error saving evaluation results: {e}")
        return ""

def run_evaluation(recommender, test_file: str = 'merged_test.csv', 
                  data_dir: str = 'data/processed',
                  strategies: List[str] = None, 
                  sample_ratio: float = 0.1) -> Dict[str, Dict[str, float]]:
    """
    Run evaluation on a recommender model.
    
    Parameters
    ----------
    recommender : BaseRecommender
        The recommender model to evaluate
    test_file : str, optional
        Name of the test file (default: 'merged_test.csv')
    data_dir : str, optional
        Directory containing the test file (default: 'data/processed')
    strategies : list, optional
        List of strategies to evaluate (default: all supported strategies)
    sample_ratio : float, optional
        Ratio of test users to sample for evaluation (default: 0.1 = 10%)
        
    Returns
    -------
    dict
        Evaluation results
    """
    test_path = os.path.join(data_dir, test_file)
    
    if not os.path.exists(test_path):
        logger.error(f"Test file not found: {test_path}")
        return {}
    
    try:
        # Load test data
        test_df = pd.read_csv(test_path)
        logger.info(f"Loaded test data with shape {test_df.shape}")
        
        # If strategies not specified, try to infer from recommender
        if strategies is None:
            # Determine recommender type from class name
            class_name = recommender.__class__.__name__.lower()
            if 'collaborative' in class_name:
                strategies = ['collaborative']
            elif 'content' in class_name:
                strategies = ['content']
            elif 'hybrid' in class_name:
                strategies = ['hybrid']
            else:
                strategies = ['collaborative', 'content', 'hybrid']
        
        # Evaluate the model
        evaluation_results = evaluate_recommender(
            recommender=recommender,
            test_df=test_df,
            strategies=strategies,
            sample_ratio=sample_ratio
        )
        
        # Save evaluation results
        if evaluation_results:
            save_evaluation_results(evaluation_results)
        
        return evaluation_results
    
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        return {}
