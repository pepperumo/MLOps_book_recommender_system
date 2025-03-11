"""
Base classes and utilities for book recommender models.

This module provides the foundation for different recommendation strategies.
"""
import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.sparse import load_npz, save_npz
from sklearn.neighbors import NearestNeighbors
import pickle
import os
import json
import sys
import logging
import traceback
from pathlib import Path
import time
from typing import List, Dict, Tuple, Optional, Union, Any
from datetime import datetime
import argparse

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


class BaseRecommender:
    """
    Base class for book recommendation systems.
    
    This class defines the interface that all recommendation strategies should implement.
    """
    
    def __init__(self, 
                 user_item_matrix: Optional[sp.csr_matrix] = None,
                 book_ids: Optional[np.ndarray] = None,
                 n_neighbors: int = 20):
        """
        Initialize the base recommender system.
        
        Parameters
        ----------
        user_item_matrix : scipy.sparse.csr_matrix, optional
            Sparse matrix of user-item interactions
        book_ids : array-like, optional
            Array of book IDs corresponding to the matrices
        n_neighbors : int, optional
            Number of neighbors to consider for recommendations
        """
        self.user_item_matrix = user_item_matrix
        self.book_ids = book_ids
        self.n_neighbors = n_neighbors
        
        if self.user_item_matrix is not None:
            self.user_ids = set(range(self.user_item_matrix.shape[0]))
            logger.info(f"Initialized BaseRecommender with {len(self.user_ids)} users and {self.user_item_matrix.shape[1]} books")
        else:
            self.user_ids = set()
            logger.warning("Initialized BaseRecommender without user-item matrix")
    
    def fit(self) -> 'BaseRecommender':
        """
        Train the recommendation model.
        
        This method should be implemented by subclasses.
        
        Returns
        -------
        self
        """
        raise NotImplementedError("Subclasses must implement fit method")
    
    def recommend_for_user(self, user_id: int, n_recommendations: int = 10) -> List[int]:
        """
        Generate book recommendations for a user.
        
        This method should be implemented by subclasses.
        
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
        raise NotImplementedError("Subclasses must implement recommend_for_user method")
    
    def recommend_similar_books(self, book_id: int, n: int = 10) -> List[int]:
        """
        Recommend books similar to a given book.
        
        This method should be implemented by subclasses.
        
        Parameters
        ----------
        book_id : int
            Book ID to find similar books for
        n : int, optional
            Number of similar books to recommend
            
        Returns
        -------
        list
            List of recommended book IDs
        """
        raise NotImplementedError("Subclasses must implement recommend_similar_books method")
    
    def save(self, model_dir: str = 'models', model_name: str = None) -> bool:
        """
        Save the trained model to a file.
        
        Parameters
        ----------
        model_dir : str, optional
            Directory to save the model
        model_name : str, optional
            Name of the model file, without extension
            
        Returns
        -------
        bool
            True if the model was saved successfully, False otherwise
        """
        try:
            logger.info(f"Saving model to {model_dir}")
            
            # Create directory if it doesn't exist
            os.makedirs(model_dir, exist_ok=True)
            
            if model_name is None:
                model_name = self.__class__.__name__.lower()
            
            # Save the model using pickle
            model_path = os.path.join(model_dir, f'{model_name}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(self, f)
            
            logger.info(f"Model saved successfully to {model_path}")
            
            # Save minimal metadata about the model in JSON format
            metadata = {
                'model_type': self.__class__.__name__,
                'user_count': len(self.user_ids) if self.user_item_matrix is not None else 0,
                'book_count': self.user_item_matrix.shape[1] if self.user_item_matrix is not None else 0,
                'n_neighbors': self.n_neighbors,
                'timestamp': timestamp
            }
            
            # Save metadata file
            with open(os.path.join(model_dir, f'{model_name}_metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=4)
                
            logger.info(f"Model metadata saved successfully to {model_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            logger.debug(traceback.format_exc())
            return False
    
    def _get_top_n_recommendations(self, recommendations_dict: Dict[int, float], n: int) -> List[int]:
        """
        Get the top N recommendations from a dictionary of scores.
        
        Parameters
        ----------
        recommendations_dict : dict
            Dictionary of {book_id: score} pairs
        n : int
            Number of recommendations to return
            
        Returns
        -------
        list
            List of book IDs ordered by score
        """
        if not recommendations_dict:
            return []
            
        # Sort by score and get top n
        top_n = sorted(recommendations_dict.items(), key=lambda x: x[1], reverse=True)[:n]
        
        # Return only book IDs
        recommendations = [int(book_id) for book_id, score in top_n]
        
        return recommendations


def load_data(features_dir: str = 'data/features') -> Tuple[Optional[sp.csr_matrix], Optional[sp.csr_matrix], Optional[sp.csr_matrix], Optional[np.ndarray], Optional[List[str]]]:
    """
    Load data for training the recommender model.
    
    Parameters
    ----------
    features_dir : str
        Directory containing feature files
    
    Returns
    -------
    tuple
        (user_item_matrix, book_feature_matrix, book_similarity_matrix, book_ids, feature_names)
    """
    try:
        logger.info(f"Loading data from {features_dir}")
        
        # Load user-item matrix
        user_item_path = os.path.join(features_dir, 'user_item_matrix.npz')
        if os.path.exists(user_item_path):
            user_item_matrix = load_npz(user_item_path)
            logger.info(f"Loaded user-item matrix with shape {user_item_matrix.shape}")
        else:
            logger.error(f"User-item matrix not found at {user_item_path}")
            return None, None, None, None, None
        
        # Load book feature matrix
        book_feature_path = os.path.join(features_dir, 'book_feature_matrix.npz')
        if os.path.exists(book_feature_path):
            book_feature_matrix = load_npz(book_feature_path)
            logger.info(f"Loaded book feature matrix with shape {book_feature_matrix.shape}")
        else:
            logger.warning(f"Book feature matrix not found at {book_feature_path}")
            book_feature_matrix = None
        
        # Load book similarity matrix (if it exists)
        book_sim_path = os.path.join(features_dir, 'book_similarity_matrix.npz')
        if os.path.exists(book_sim_path):
            book_similarity_matrix = load_npz(book_sim_path)
            logger.info(f"Loaded book similarity matrix with shape {book_similarity_matrix.shape}")
        else:
            logger.info(f"Book similarity matrix not found at {book_sim_path}, will calculate it")
            book_similarity_matrix = None
        
        # Load book IDs
        book_ids_path = os.path.join(features_dir, 'book_ids.npy')
        if os.path.exists(book_ids_path):
            book_ids = np.load(book_ids_path)
            logger.info(f"Loaded {len(book_ids)} book IDs")
        else:
            logger.warning(f"Book IDs not found at {book_ids_path}")
            book_ids = None
        
        # Load feature names
        feature_names_path = os.path.join(features_dir, 'feature_names.txt')
        if os.path.exists(feature_names_path):
            with open(feature_names_path, 'r') as f:
                feature_names = [line.strip() for line in f.readlines()]
            logger.info(f"Loaded {len(feature_names)} feature names")
        else:
            logger.warning(f"Feature names not found at {feature_names_path}")
            feature_names = None
        
        # Save load status as CSV
        load_status = {
            'user_item_matrix': user_item_matrix is not None,
            'book_feature_matrix': book_feature_matrix is not None,
            'book_similarity_matrix': book_similarity_matrix is not None,
            'book_ids': book_ids is not None,
            'feature_names': feature_names is not None,
            'timestamp': timestamp
        }
        
        if any(load_status.values()):
            results_dir = os.path.join('data', 'results')
            os.makedirs(results_dir, exist_ok=True)
            pd.DataFrame([load_status]).to_csv(
                os.path.join(results_dir, f'data_load_status_{timestamp}.csv'), 
                index=False
            )
        
        return user_item_matrix, book_feature_matrix, book_similarity_matrix, book_ids, feature_names
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        logger.debug(traceback.format_exc())
        return None, None, None, None, None


def evaluate_model_with_test_data(recommender, test_file: str = 'merged_test.csv', 
                                 data_dir: str = 'data/processed'):
    """
    Evaluate the model with test data.
    
    Parameters
    ----------
    recommender : BaseRecommender
        The trained recommender model
    test_file : str
        Name of the test file
    data_dir : str
        Directory containing the test file
    
    Returns
    -------
    dict
        Evaluation results for different strategies and k values
    """
    test_path = os.path.join(data_dir, test_file)
    
    if not os.path.exists(test_path):
        logger.error(f"Test file not found: {test_path}")
        return {}
    
    try:
        # Load test data
        test_df = pd.read_csv(test_path)
        logger.info(f"Loaded test data with shape {test_df.shape}")
        
        # Make sure we have user_id and book_id columns
        if 'user_id' not in test_df.columns or 'book_id' not in test_df.columns:
            logger.error("Test data must have user_id and book_id columns")
            return {}
        
        # Get unique users and books
        test_users = test_df['user_id'].unique()
        logger.info(f"Test data contains {len(test_users)} unique users")
        
        # Evaluate the model using its evaluate method if it has one
        try:
            # Try different k values for precision@k and recall@k
            k_values = [5, 10, 20]
            strategies = ['collaborative', 'content', 'hybrid']
            
            try:
                # Assuming recommender has an evaluate method
                if hasattr(recommender, 'evaluate'):
                    evaluation_results = recommender.evaluate(test_df, k_values, strategies)
                    
                    # Save evaluation results
                    results_dir = os.path.join('data', 'results')
                    os.makedirs(results_dir, exist_ok=True)
                    
                    # Convert to DataFrame for easier saving
                    results_df = pd.DataFrame()
                    
                    for strategy, metrics in evaluation_results.items():
                        for metric, value in metrics.items():
                            results_df.loc[strategy, metric] = value
                    
                    # Save to CSV
                    results_df.to_csv(os.path.join(results_dir, f'evaluation_results_{timestamp}.csv'))
                    logger.info(f"Saved evaluation results to {results_dir}")
                    
                    return evaluation_results
                else:
                    logger.warning("Recommender does not have an evaluate method")
                    return {}
            except Exception as e:
                logger.error(f"Error evaluating model: {e}")
                return {}
        
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            logger.debug(traceback.format_exc())
            return {}
    
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        logger.debug(traceback.format_exc())
        return {}
