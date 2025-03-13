"""
Book recommender system based on collaborative filtering.

This module contains the implementation of the book recommender system,
including the CollaborativeRecommender implementation and main training functionality.
"""
import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.sparse import load_npz, save_npz
from sklearn.neighbors import NearestNeighbors
import pickle
import logging
import os
import sys
import json
import time
import traceback
import argparse
import yaml
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any

# Add the project root to the Python path so we can import modules correctly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import MLflow utilities
try:
    import mlflow
    import dagshub
    from src.models.mlflow_utils import (
        setup_mlflow, log_params_from_model, log_metrics_safely,
        log_model_version_as_tag, get_dagshub_url
    )
    MLFLOW_AVAILABLE = True
except ImportError as e:
    print(f"Warning: MLflow integration not available: {e}")
    MLFLOW_AVAILABLE = False

# Import from model_utils
try:
    from src.models.model_utils import BaseRecommender, load_data
except ImportError:
    try:
        from models.model_utils import BaseRecommender, load_data
    except ImportError:
        import sys
        import os
        # Add the parent directory to the path to ensure we can import the module
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.append(parent_dir)
        from models.model_utils import BaseRecommender, load_data

# Set up logging
log_dir = os.path.join('logs')
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(log_dir, f'train_model_{timestamp}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('train_model')


def load_config(config_path="config/model_params.yaml"):
    """
    Load model configuration from YAML file
    
    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file
        
    Returns
    -------
    dict
        Configuration dictionary with model parameters
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        logger.info("Using default configuration")
        return {
            "collaborative": {
                "n_neighbors": 20,
                "max_rated_items": 50,
                "similarity_metric": "cosine",
                "algorithm": "brute",
                "n_jobs": -1,
                "k_values": [5, 10, 20],
                "eval_sample_size": 50
            },
            "data": {
                "features_dir": "data/features",
                "output_dir": "models"
            }
        }


###################################################
# Collaborative Recommender - main model class     #
###################################################

class CollaborativeRecommender(BaseRecommender):
    """A book recommender system based on collaborative filtering.
    
    This class uses user-item interactions to recommend books based on similarity
    between users and items.
    """
    
    def __init__(self, 
                 user_item_matrix: Optional[sp.csr_matrix] = None,
                 book_ids: Optional[np.ndarray] = None,
                 n_neighbors: int = 20,
                 max_rated_items: int = 50,
                 similarity_metric: str = "cosine",
                 algorithm: str = "brute",
                 n_jobs: int = -1):
        """
        Initialize the collaborative filtering recommender system.
        
        Parameters
        ----------
        user_item_matrix : scipy.sparse.csr_matrix, optional
            Sparse matrix of user-item interactions
        book_ids : array-like, optional
            Array of book IDs corresponding to the matrices
        n_neighbors : int, optional
            Number of neighbors to consider for recommendations
        max_rated_items : int, optional
            Maximum number of user-rated items to consider when generating recommendations
        similarity_metric : str, optional
            Metric to use for similarity calculation (e.g., 'cosine', 'euclidean')
        algorithm : str, optional
            Algorithm for nearest neighbors search (e.g., 'brute', 'kd_tree')
        n_jobs : int, optional
            Number of jobs to run in parallel. -1 means using all processors
        """
        super().__init__()
        self.user_item_matrix = user_item_matrix
        self.book_ids = book_ids
        self.n_neighbors = n_neighbors
        self.max_rated_items = max_rated_items
        self.similarity_metric = similarity_metric
        self.algorithm = algorithm
        self.n_jobs = n_jobs
        self.item_nn_model = None
        self.item_similarity_matrix = None
        self.book_id_to_index = {}
        
        # Store hyperparameters in a params dictionary for MLflow tracking
        self.params = {
            "n_neighbors": n_neighbors,
            "max_rated_items": max_rated_items,
            "model_type": "collaborative",
            "similarity_metric": similarity_metric,
            "algorithm": algorithm,
            "n_jobs": n_jobs,
            "training_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if self.book_ids is not None:
            # Create mapping from book ID to matrix index
            self.book_id_to_index = {int(book_id): i for i, book_id in enumerate(self.book_ids)}
    
    def fit(self):
        """
        Train the collaborative filtering model.
        
        Builds nearest neighbors model for item-based collaborative filtering
        and pre-computes item-item similarity matrix.
        
        Returns
        -------
        self
        """
        logger.info("Training item-based collaborative filtering model")
        
        if self.user_item_matrix is None or self.book_ids is None:
            logger.error("Cannot train model: missing user_item_matrix or book_ids")
            return self
        
        # Train nearest neighbors model
        logger.info(f"Building nearest neighbors model with n_neighbors={self.n_neighbors}")
        
        # Convert to CSR for efficient row slicing
        self.user_item_matrix = sp.csr_matrix(self.user_item_matrix)
        
        # Track matrix shape in params
        self.params["user_item_matrix_shape"] = f"{self.user_item_matrix.shape[0]}x{self.user_item_matrix.shape[1]}"
        self.params["num_users"] = self.user_item_matrix.shape[0]
        self.params["num_items"] = self.user_item_matrix.shape[1]
        self.params["matrix_density"] = float(self.user_item_matrix.nnz / (self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1]))
        
        # Build item-item similarity model
        logger.info("Building item similarity model")
        self.item_nn_model = NearestNeighbors(
            metric=self.similarity_metric,
            algorithm=self.algorithm,
            n_neighbors=self.n_neighbors + 1,  # +1 because it will include the item itself
            n_jobs=self.n_jobs  # Use all available cores
        )
        self.item_nn_model.fit(self.user_item_matrix.T.toarray())
        
        # Pre-compute item similarity matrix
        logger.info("Pre-computing item similarity matrix")
        num_items = self.user_item_matrix.shape[1]
        
        # For large datasets, compute similarities in batches
        batch_size = 1000
        self.item_similarity_matrix = {}
        
        for i in range(0, num_items, batch_size):
            batch_end = min(i + batch_size, num_items)
            batch_items = list(range(i, batch_end))
            
            # If batch is empty, skip
            if not batch_items:
                continue
                
            # Get item vectors for this batch
            item_vectors = self.user_item_matrix.T[batch_items].toarray()
            
            # Find k+1 nearest neighbors for each item (including itself)
            distances, indices = self.item_nn_model.kneighbors(
                item_vectors, 
                n_neighbors=self.n_neighbors + 1
            )
            
            # Store similarities (1 - distance) for each item
            for j, (item_idx, item_distances, item_indices) in enumerate(zip(batch_items, distances, indices)):
                # Skip the first result (the item itself)
                self.item_similarity_matrix[item_idx] = {
                    'indices': item_indices[1:],  # Skip the item itself (first element)
                    'similarities': 1 - item_distances[1:]  # Convert distances to similarities
                }
        
        logger.info("Model training completed")
        return self
        
    def recommend_for_user(self, user_id: int, n_recommendations: int = 10) -> List[int]:
        """
        Generate book recommendations for a user based on collaborative filtering.
        
        Parameters
        ----------
        user_id : int
            User ID to generate recommendations for
        n_recommendations : int, optional
            Number of recommendations to generate
            
        Returns
        -------
        list
            List of recommended book IDs
        """
        # Convert user_id to matrix index if needed
        user_idx = user_id
        if hasattr(self, 'user_id_to_index'):
            if user_id not in self.user_id_to_index:
                logger.warning(f"User {user_id} not found in training data")
                return []
            user_idx = self.user_id_to_index[user_id]
        
        # Get user profile
        if user_idx >= self.user_item_matrix.shape[0]:
            logger.warning(f"User index {user_idx} out of bounds")
            return []
            
        user_profile = self.user_item_matrix[user_idx].toarray().flatten()
        
        # If user has no ratings, use popularity-based recommendations
        if np.sum(user_profile) == 0:
            logger.warning(f"User {user_id} has no ratings. Using popularity-based recommendations.")
            # Fallback to overall popularity
            item_popularity = self.user_item_matrix.sum(axis=0).A1
            top_items = np.argsort(item_popularity)[::-1][:n_recommendations]
            return [int(self.book_ids[i]) for i in top_items]
            
        # Get indices of books the user has already interacted with
        already_interacted = np.where(user_profile > 0)[0]
        
        # Limit the number of user-rated items we consider to improve performance
        if len(already_interacted) > self.max_rated_items:
            # Sort by rating and take top rated items
            top_ratings_idx = np.argsort(user_profile[already_interacted])[::-1][:self.max_rated_items]
            already_interacted = already_interacted[top_ratings_idx]
        
        # Convert to set for faster lookups
        already_interacted_set = set(already_interacted)
        
        # Use user profile to generate recommendations
        candidate_scores = {}
        
        for book_idx in already_interacted:
            # Skip if this book doesn't have pre-computed similarities
            if book_idx not in self.item_similarity_matrix:
                continue
                
            # Get pre-computed similar items
            similar_items = self.item_similarity_matrix[book_idx]
            item_indices = similar_items['indices']
            item_similarities = similar_items['similarities']
            
            # Score similar books based on similarity
            for idx, sim in zip(item_indices, item_similarities):
                if idx not in already_interacted_set:
                    if idx not in candidate_scores:
                        candidate_scores[idx] = 0
                    # Weight similarity by user's rating
                    candidate_scores[idx] += sim * user_profile[book_idx]
        
        # Sort candidates by score
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top N book IDs
        recommended_indices = [idx for idx, _ in sorted_candidates[:n_recommendations]]
        recommended_book_ids = [int(self.book_ids[i]) for i in recommended_indices]
        
        return recommended_book_ids
    
    def recommend_similar_books(self, book_id: int, n: int = 10):
        """
        Recommend books similar to a given book based on collaborative filtering.
        
        Parameters
        ----------
        book_id : int
            Book ID to find similar books for
        n : int, optional
            Number of similar books to recommend
            
        Returns
        -------
        list
            List of book IDs similar to the given book
        """
        if self.item_nn_model is None:
            logger.error("Model not trained. Call fit() before finding similar books.")
            return []
        
        try:
            # Convert book_id to matrix index
            if book_id not in self.book_id_to_index:
                logger.warning(f"Book ID {book_id} not found in the model. Returning empty recommendations.")
                return []
                
            book_idx = self.book_id_to_index[book_id]
            
            # Get the book's feature vector
            book_vector = self.user_item_matrix.T[book_idx].toarray().reshape(1, -1)
            
            # Find similar books using the nearest neighbors model
            distances, indices = self.item_nn_model.kneighbors(
                book_vector,
                n_neighbors=n+1  # +1 because it will include the book itself
            )
            
            # Skip the first item (which is the book itself)
            similar_indices = indices.flatten()[1:n+1]
            
            # Convert indices to book IDs
            similar_book_ids = [int(self.book_ids[idx]) for idx in similar_indices]
            
            return similar_book_ids
            
        except Exception as e:
            logger.error(f"Error finding similar books for book {book_id}: {str(e)}")
            logger.error(traceback.format_exc())
            return []


###################################################
# Main Training Function                           #
###################################################

def train_model(config_path="config/model_params.yaml", model_version="collaborative"):
    """
    Train the collaborative filtering model using parameters from config file.
    
    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file
    model_version : str
        Which model configuration to use from the YAML file
        
    Returns
    -------
    CollaborativeRecommender
        Trained collaborative filtering model
    """
    try:
        # Load configuration
        config = load_config(config_path)
        model_config = config.get(model_version, config.get("collaborative", {}))
        data_config = config.get("data", {})
        
        # Extract parameters
        n_neighbors = model_config.get("n_neighbors", 20)
        max_rated_items = model_config.get("max_rated_items", 50)
        similarity_metric = model_config.get("similarity_metric", "cosine")
        algorithm = model_config.get("algorithm", "brute")
        n_jobs = model_config.get("n_jobs", -1)
        features_dir = data_config.get("features_dir", "data/features")
        output_dir = data_config.get("output_dir", "models")
        
        # Log configuration
        logger.info(f"Training model with version: {model_version}")
        logger.info(f"Parameters: n_neighbors={n_neighbors}, max_rated_items={max_rated_items}")
        logger.info(f"Algorithm: {algorithm}, Similarity: {similarity_metric}")
        
        # Load data
        logger.info("Loading data for model training...")
        user_item_matrix, book_ids = load_data(features_dir=features_dir)
        
        if user_item_matrix is None or book_ids is None:
            logger.error("Failed to load required data. Please check the data files.")
            return None
        
        # Create and train the collaborative model
        logger.info("Creating and training collaborative filtering model...")
        collaborative_model = CollaborativeRecommender(
            user_item_matrix=user_item_matrix,
            book_ids=book_ids,
            n_neighbors=n_neighbors,
            max_rated_items=max_rated_items,
            similarity_metric=similarity_metric,
            algorithm=algorithm,
            n_jobs=n_jobs
        )
        
        # Update params with configuration info
        collaborative_model.params.update({
            "model_version": model_version,
            "config_file": config_path
        })
        
        collaborative_model.fit()
        
        # Save the trained model
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, f'{model_version}.pkl')
        
        logger.info(f"Saving trained model to {model_path}")
        collaborative_model.save(model_path)
        
        return collaborative_model
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        logger.error(traceback.format_exc())
        return None


def main():
    """
    Main function for the model training script.
    """
    parser = argparse.ArgumentParser(description='Train collaborative filtering model for book recommendations')
    parser.add_argument('--config', type=str, default='config/model_params.yaml',
                        help='Path to the configuration YAML file')
    parser.add_argument('--model-version', type=str, default='collaborative',
                        help='Model version/configuration to use from the YAML file')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Override the output directory for model saving')
    parser.add_argument('--disable-mlflow', action='store_true',
                        help='Disable MLflow logging')
    
    args = parser.parse_args()
    
    # Override MLFLOW_AVAILABLE if requested
    global MLFLOW_AVAILABLE
    if args.disable_mlflow:
        MLFLOW_AVAILABLE = False
    
    try:
        # Set up MLflow if available
        if MLFLOW_AVAILABLE:
            try:
                # Setup MLflow
                setup_mlflow(experiment_name=f"collaborative_model_{args.model_version}")
                logger.info(f"MLflow tracking is enabled. Visit {get_dagshub_url()} to view experiments.")
                
                with mlflow.start_run(run_name=f"{args.model_version}_training"):
                    # Train the model
                    model = train_model(
                        config_path=args.config,
                        model_version=args.model_version
                    )
                    
                    if model:
                        # Log parameters from model
                        log_params_from_model(model)
                        
                        # Log model artifacts
                        model_path = os.path.join('models', f"{args.model_version}.pkl")
                        mlflow.log_artifact(model_path, "model")
                        
                        # Save model config for reproducibility
                        model_config_path = os.path.join('models', f"{args.model_version}_config.json")
                        os.makedirs(os.path.dirname(model_config_path), exist_ok=True)
                        
                        with open(model_config_path, 'w') as f:
                            json.dump(model.params, f, indent=2)
                            
                        mlflow.log_artifact(model_config_path, "model_config")
                        
                        # Log model version as a tag
                        log_model_version_as_tag(args.model_version)
            except Exception as e:
                logger.warning(f"Error setting up MLflow: {e}")
                MLFLOW_AVAILABLE = False
                model = train_model(
                    config_path=args.config,
                    model_version=args.model_version
                )
        else:
            # Train without MLflow
            model = train_model(
                config_path=args.config,
                model_version=args.model_version
            )
        
        if model:
            # Print success message
            print("\nTraining completed successfully!")
            print(f"\nTrained model saved to: models/{args.model_version}.pkl")
            print("\nTo evaluate this model, run:")
            print(f"python src/models/evaluate_model.py --model-path models/{args.model_version}.pkl")
            return 0
        else:
            logger.error("Model training failed.")
            return 1
    
    except Exception as e:
        logger.error(f"Error in main: {e}")
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
