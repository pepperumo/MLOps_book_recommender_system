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
import random

# Add the project directory to the path so we can import modules properly
try:
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if project_dir not in sys.path:
        sys.path.append(project_dir)
except:
    pass

# Import MLflow utilities
try:
    import mlflow
    import dagshub
    from src.models.mlflow_utils import (
        setup_mlflow, log_params_from_model, log_metrics_safely,
        log_model_version_as_tag, log_precision_recall_curve, get_dagshub_url
    )
    MLFLOW_AVAILABLE = True
except ImportError as e:
    print(f"Warning: MLflow integration not available: {e}")
    MLFLOW_AVAILABLE = False

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
    try:
        # Initialize MLflow if available
        if MLFLOW_AVAILABLE:
            setup_mlflow(repo_owner='pepperumo', repo_name='MLOps_book_recommender_system')
            
        # Default k values if not provided
        k_values = [5, 10, 20]
            
        # Use model params for sample size if available and not explicitly provided
        if hasattr(recommender, 'params') and 'eval_sample_size' in recommender.params and sample_size is None:
            sample_size = recommender.params['eval_sample_size']
            
        # Load test data if not provided
        test_file_path = os.path.join(data_dir, test_file)
        if not os.path.exists(test_file_path):
            logger.error(f"Test file not found: {test_file_path}")
            return {}
        
        logger.info(f"Loading test data from {test_file_path}")
        test_df = pd.read_csv(test_file_path)
        
        # Evaluate the model
        logger.info(f"Evaluating model with k values: {k_values}")
        results = evaluate_recommender(recommender, test_df, k_values=k_values, sample_size=sample_size)
        
        # Log the evaluation metrics with MLflow
        if MLFLOW_AVAILABLE:
            try:
                with mlflow.start_run(run_name=f"{model_name}_evaluation"):
                    # Set up model version tags if available
                    if hasattr(recommender, 'params'):
                        model_version = recommender.params.get("model_version", model_name)
                        config_file = recommender.params.get("config_file", None)
                        log_model_version_as_tag(model_version, config_file)
                    
                    # Log model parameters
                    log_params_from_model(recommender)
                    
                    # Log evaluation metrics
                    log_metrics_safely(results)
                    
                    # Create and log precision-recall data
                    precision_metrics = {k: v for k, v in results.items() if k.startswith("precision")}
                    recall_metrics = {k: v for k, v in results.items() if k.startswith("recall")}
                    if precision_metrics and recall_metrics:
                        log_precision_recall_curve(precision_metrics, recall_metrics, k_values)
                    
                    # Log additional info
                    mlflow.set_tag("model_type", model_name)
                    mlflow.set_tag("evaluation_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    mlflow.set_tag("test_file", test_file)
                    mlflow.set_tag("sample_size", str(sample_size))
                    
                    # Print direct link to MLflow run
                    print(f"\nMLflow run URL: {get_dagshub_url()}")
            except Exception as e:
                logger.warning(f"Error during MLflow logging: {e}")
                logger.warning(traceback.format_exc())
        
        # Save to local filesystem
        save_evaluation_results(results, model_name=model_name)
        
        # Print nicely formatted results by k value
        print("\nEvaluation Results:")
        print("=" * 50)
        
        for k in k_values:
            print(f"\nResults for k={k}:")
            print("-" * 20)
            precision_key = f"precision@{k}"
            recall_key = f"recall@{k}"
            
            if precision_key in results:
                print(f"Precision@{k}: {results[precision_key]:.4f}")
            if recall_key in results:
                print(f"Recall@{k}:    {results[recall_key]:.4f}")
        
        return results
    
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        logger.error(traceback.format_exc())
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
    parser.add_argument('--disable-mlflow', action='store_true',
                       help='Disable MLflow tracking')
    
    args = parser.parse_args()
    
    try:
        # Set up MLflow tracking
        if not args.disable_mlflow:
            setup_mlflow(repo_owner='pepperumo', repo_name='MLOps_book_recommender_system')
            
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
        
        # Organize metrics by k value for better readability
        metrics_by_k = {}
        for metric, value in results.items():
            # Extract k value from metric name (e.g., 'precision@10' -> 10)
            if '@' in metric:
                metric_type, k = metric.split('@')
                k = int(k)
                if k not in metrics_by_k:
                    metrics_by_k[k] = {}
                metrics_by_k[k][metric_type] = value
        
        # Print metrics organized by k value
        print(f"{'k':>2} | {'Precision':>10} | {'Recall':>10}")
        print("-" * 35)
        for k in sorted(metrics_by_k.keys()):
            precision = metrics_by_k[k].get('precision', 0)
            recall = metrics_by_k[k].get('recall', 0)
            print(f"{k:>2} | {precision:>10.4f} | {recall:>10.4f}")
        
        # Also print raw metrics for backwards compatibility
        print("\nRaw Metrics:")
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")
        print()
        
    except Exception as e:
        logger.error(f"Error in evaluation: {e}")
        logger.debug(traceback.format_exc())
