"""
Evaluation functionality for book recommender models.

This module provides functions to evaluate the performance of different recommendation models,
calculate metrics, and save evaluation results.
"""
import os
import sys
import pickle
import logging
import numpy as np
import pandas as pd
import glob
import traceback
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

# Add the project root to the Python path so we can import modules correctly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import from model_utils
try:
    from src.models.model_utils import BaseRecommender
except ImportError:
    try:
        from models.model_utils import BaseRecommender
    except ImportError:
        import sys
        import os
        # Add the parent directory to the path to ensure we can import the module
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.append(parent_dir)
        from models.model_utils import BaseRecommender

# Set up logging
log_dir = os.path.join('logs')
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(log_dir, f'evaluate_model_{timestamp}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('evaluate_model')


def evaluate_recommender(recommender: Any, test_df: pd.DataFrame, 
                         k_values: List[int] = [10],
                         sample_size: int = 50) -> Dict[str, float]:
    """
    Evaluate a recommender model using precision@k and recall@k.
    
    Parameters
    ----------
    recommender : BaseRecommender
        The recommender model to evaluate
    test_df : pandas.DataFrame
        Test data with user_id and book_id columns
    k_values : list, optional
        List of k values to evaluate (default: [10])
    sample_size : int, optional
        Number of test users to sample for evaluation (default: 50)
        
    Returns
    -------
    dict
        Evaluation results
    """
    results = {}
    
    # Verify test data has required columns
    if 'user_id' not in test_df.columns or 'book_id' not in test_df.columns:
        logger.error("Test data must have user_id and book_id columns")
        return {}
    
    # Group test data by user
    test_users = test_df.groupby('user_id')['book_id'].apply(list).to_dict()
    
    # Sample a fixed number of users to speed up evaluation
    user_ids = list(test_users.keys())
    n_sample = min(sample_size, len(user_ids))  # Cap at the number of available users
    sampled_user_ids = np.random.choice(user_ids, size=n_sample, replace=False)
    logger.info(f"Evaluating on {n_sample} users (out of {len(user_ids)} total users)")
    
    # Calculate precision and recall for each k value
    for k in k_values:
        precisions = []
        recalls = []
        
        for user_id in sampled_user_ids:
            true_books = test_users[user_id]
            
            # Skip users not in training data
            try:
                # Check if user exists in the model's user mapping
                if hasattr(recommender, 'user_ids') and user_id not in recommender.user_ids:
                    continue
                
                # Get recommendations for this user
                try:
                    # Different models might have different interfaces
                    if hasattr(recommender, 'recommend_for_user'):
                        recs = recommender.recommend_for_user(user_id, k)
                    elif hasattr(recommender, 'recommend_items'):
                        recs = recommender.recommend_items(user_id, k)
                    elif hasattr(recommender, 'recommend'):
                        recs = recommender.recommend(user_id, k)
                    else:
                        # Default to the common predict method
                        recs = [item for item, _ in recommender.predict(user_id, k)]
                except Exception as e:
                    logger.error(f"Error generating recommendations for user {user_id}: {e}")
                    continue
                
                # Calculate precision and recall
                n_relevant = len(set(recs) & set(true_books))
                
                precision = n_relevant / k if k > 0 else 0
                recall = n_relevant / len(true_books) if len(true_books) > 0 else 0
                
                precisions.append(precision)
                recalls.append(recall)
            except Exception as e:
                logger.error(f"Error evaluating user {user_id}: {e}")
                continue
        
        # Average precision and recall
        if precisions:
            results[f'precision@{k}'] = float(np.mean(precisions))
            results[f'recall@{k}'] = float(np.mean(recalls))
    
    return results


def save_evaluation_results(evaluation_results: Dict[str, float], 
                           results_dir: str = 'data/results',
                           model_name: str = 'collaborative') -> str:
    """
    Save evaluation results to a CSV file.
    
    Parameters
    ----------
    evaluation_results : dict
        Evaluation results to save
    results_dir : str, optional
        Directory to save results to (default: 'data/results')
    model_name : str, optional
        Name of the model being evaluated (default: 'collaborative')
        
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
        
        for metric, value in evaluation_results.items():
            results_df.loc[model_name, metric] = value
        
        # Save to CSV
        results_df.to_csv(results_file)
        logger.info(f"Saved evaluation results to {results_file}")
        
        return results_file
    except Exception as e:
        logger.error(f"Error saving evaluation results: {e}")
        return ""


def run_evaluation(recommender, test_file: str = 'merged_test.csv', 
                  data_dir: str = 'data/processed',
                  model_name: str = 'collaborative',
                  sample_size: int = 50) -> Dict[str, float]:
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
    model_name : str, optional
        Name of the model being evaluated (default: 'collaborative')
    sample_size : int, optional
        Number of test users to sample for evaluation (default: 50)
        
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
        
        # Evaluate the model
        logger.info(f"Evaluating {model_name} model")
        evaluation_results = evaluate_recommender(
            recommender=recommender,
            test_df=test_df,
            k_values=[10],
            sample_size=sample_size
        )
        
        # Save the results
        results_dir = os.path.join(os.path.dirname(data_dir), 'results')
        save_evaluation_results(evaluation_results, results_dir, model_name)
        
        # Log evaluation results
        for metric, value in evaluation_results.items():
            logger.info(f"{model_name.capitalize()} model {metric}: {value:.4f}")
        
        return evaluation_results
        
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        logger.debug(traceback.format_exc())
        return {}


def find_latest_model(models_dir: str = 'models', model_prefix: str = 'collaborative') -> str:
    """
    Find the latest model file in the models directory.
    
    Parameters
    ----------
    models_dir : str
        Directory containing model files
    model_prefix : str
        Prefix of the model file to find
        
    Returns
    -------
    str
        Path to the latest model file, or empty string if none found
    """
    try:
        # Find all model files with the given prefix
        model_files = glob.glob(os.path.join(models_dir, f"{model_prefix}*.pkl"))
        
        if not model_files:
            logger.error(f"No model files found in {models_dir} with prefix {model_prefix}")
            return ""
        
        # Sort by modification time (newest first)
        model_files.sort(key=os.path.getmtime, reverse=True)
        
        latest_model = model_files[0]
        logger.info(f"Found latest model: {latest_model}")
        
        return latest_model
    
    except Exception as e:
        logger.error(f"Error finding latest model: {e}")
        return ""


if __name__ == "__main__":
    import argparse
    
    # Add the project root to the Python path so we can import modules correctly
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    # Import here to avoid circular import errors
    from src.models.train_model import CollaborativeRecommender
    
    parser = argparse.ArgumentParser(description='Evaluate a trained book recommender model')
    parser.add_argument('--model-path', type=str, 
                       help='Path to the trained model file (.pkl). If not provided, the latest collaborative model will be used.')
    parser.add_argument('--test-file', type=str, default='merged_test.csv',
                       help='Name of the test data file')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                       help='Directory containing the test data')
    parser.add_argument('--sample-size', type=int, default=50,
                       help='Number of users to sample for evaluation')
    parser.add_argument('--output-dir', type=str, default='data/results',
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    try:
        # If model path not provided, find the latest collaborative model
        model_path = args.model_path
        if not model_path:
            model_path = find_latest_model(models_dir='models', model_prefix='collaborative')
            
            if not model_path:
                logger.error("No model found and no model path provided. Please train a model first.")
                sys.exit(1)
        
        # Load the model
        logger.info(f"Loading model from {model_path}")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Determine model name from file path
        model_name = os.path.basename(model_path).split('.')[0]
        
        # Run evaluation
        logger.info(f"Evaluating {model_name} model")
        results = run_evaluation(
            recommender=model,
            test_file=args.test_file,
            data_dir=args.data_dir,
            model_name=model_name,
            sample_size=args.sample_size
        )
        
        # Save results with consistent naming
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(args.output_dir, f'evaluation_results_{timestamp}.csv')
        
        # Convert to DataFrame for easier saving
        results_df = pd.DataFrame()
        for metric, value in results.items():
            results_df.loc[model_name, metric] = value
        
        # Save to CSV
        results_df.to_csv(results_file)
        logger.info(f"Saved evaluation results to {results_file}")
        
        # Print results to console
        print("\nEvaluation Results:")
        print("-------------------")
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")
        print()
        
    except Exception as e:
        logger.error(f"Error in evaluation: {e}")
        logger.debug(traceback.format_exc())
