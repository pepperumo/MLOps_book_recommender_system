"""
Book recommender system based on collaborative filtering.

This module contains the implementation of the book recommender system,
including the CollaborativeRecommender implementation and main training functionality.
"""
import argparse
import json
import logging
import numpy as np
import os
import pandas as pd
import pickle
import scipy.sparse as sp
import sys
import warnings
import yaml
import traceback
from pathlib import Path
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional, Union

# Import local modules
from src.models.model_utils import load_data, BaseRecommender

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('train_model')

# Determine project root directory
project_root = Path(__file__).parent.parent.parent.absolute()
logger.info(f"Project root: {project_root}")

# Add the project root to the Python path so we can import modules correctly
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import MLflow utilities
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    import dagshub
    from src.models.mlflow_utils import (
        setup_mlflow, log_params_from_model, log_metrics_safely,
        log_model_version_as_tag, get_dagshub_url
    )
    MLFLOW_AVAILABLE = True
except ImportError as e:
    print(f"Warning: MLflow integration not available: {e}")
    MLFLOW_AVAILABLE = False

# Set up logging
log_dir = os.path.join(project_root, 'logs')
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

    
    def fit(self, config_path=None):
        """
        Train the collaborative filtering model with optional configuration file.
        
        Args:
            config_path (str, optional): Path to configuration file.
        
        Returns:
            CollaborativeRecommender: Trained model
        """
        # Load configuration
        config = load_config(config_path)
        
        # Extract parameters
        model_config = config.get("model", {})
        core_config = config.get("core", {})
        data_config = config.get("data", {})
        
        # Get model parameters from config
        model_version = core_config.get("model_version", "collaborative")
        n_neighbors = model_config.get("n_neighbors", 20)
        max_rated_items = model_config.get("max_rated_items", 50)
        similarity_metric = model_config.get("similarity_metric", "cosine")
        algorithm = model_config.get("algorithm", "brute")
        n_jobs = model_config.get("n_jobs", -1)
        features_dir = data_config.get("features_dir", os.path.join(project_root, "data/features"))
        output_dir = data_config.get("output_dir", os.path.join(project_root, "models"))
        
        # Log configuration
        logger.info(f"Training model with version: {model_version}")
        logger.info(f"Parameters: n_neighbors={n_neighbors}, max_rated_items={max_rated_items}")
        logger.info(f"Algorithm: {algorithm}, Similarity: {similarity_metric}")
        
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
        Generate recommendations for a specific user.
        
        Args:
            user_id: ID of the user
            n_recommendations: Number of recommendations to generate
            
        Returns:
            list: List of recommended book IDs
        """
        if self.item_nn_model is None:
            logger.error("Model not trained. Call fit() before making recommendations.")
            return []
        
        try:
            # Get the user's vector (list of ratings)
            user_idx = user_id  # In our case, user_id is the index
            if user_idx >= self.user_item_matrix.shape[0]:
                logger.warning(f"User ID {user_id} not found in training data")
                return []
                
            user_vector = self.user_item_matrix[user_idx].toarray().reshape(-1)
            
            # Find books the user has already rated
            rated_indices = np.where(user_vector > 0)[0]
            
            if len(rated_indices) == 0:
                logger.warning(f"User {user_id} has no ratings in the training data")
                return []
                
            # Create a dictionary to store scores for candidate books
            candidate_scores = defaultdict(float)
            
            # For each rated book, find similar books and score them
            for item_idx in rated_indices:
                if item_idx not in self.item_similarity_matrix:
                    continue
                    
                # Get similarity data for this item
                sim_data = self.item_similarity_matrix[item_idx]
                similar_indices = sim_data['indices']
                similarities = sim_data['similarities']
                
                # User's rating for this item
                user_rating = user_vector[item_idx]
                
                # For each similar item, update its score
                for j, sim_idx in enumerate(similar_indices):
                    if sim_idx not in rated_indices:  # Only consider unrated items
                        # Weight similarity by user's rating
                        candidate_scores[sim_idx] += similarities[j] * user_rating
            
            # Convert scores to a list of (item_id, score) tuples and sort
            scored_candidates = [(item, score) for item, score in candidate_scores.items()]
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Return top N recommendations
            top_items = [item for item, score in scored_candidates[:n_recommendations]]
            return top_items
                
        except Exception as e:
            logger.error(f"Error generating recommendations for user {user_id}: {e}")
            return []


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
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                import yaml
                config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {config_path}")
                return config
        else:
            logger.warning(f"Config file not found at {config_path}")
            return {}
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        return {}
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
        features_dir = data_config.get("features_dir", os.path.join(project_root, "data/features"))
        output_dir = data_config.get("output_dir", os.path.join(project_root, "models"))
        
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
        
        collaborative_model.fit(config_path=config_path)
        
        # Save the trained model
        # Convert relative path to absolute path using project_root
        abs_output_dir = os.path.join(project_root, output_dir)
        os.makedirs(abs_output_dir, exist_ok=True)
        model_path = os.path.join(abs_output_dir, f'{model_version}.pkl')
        
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
                setup_mlflow(repo_owner='pepperumo', repo_name='MLOps_book_recommender_system')
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
                        model_path = os.path.join(project_root, 'models', f"{args.model_version}.pkl")
                        mlflow.log_artifact(model_path, "model")
                        
                        # Save model config for reproducibility
                        model_config_path = os.path.join(project_root, 'models', f"{args.model_version}_config.json")
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

