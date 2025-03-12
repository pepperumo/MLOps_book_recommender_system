"""
Utility classes and functions for book recommender models.

This module contains the implementation of base classes and utility functions
used by the book recommender system, including the BaseRecommender class and
data loading functionality.
"""
import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.sparse import load_npz, save_npz
import pickle
import logging
import os
import sys
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union, Any

# Set up logging
logger = logging.getLogger('model_utils')

class BaseRecommender:
    """Base class for recommender systems.
    
    This class provides the foundation for all recommender models.
    It handles common operations like loading and saving models.
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
            Array of book IDs corresponding to the columns of the matrix
        n_neighbors : int, optional
            Number of neighbors to consider for recommendations
        """
        self.user_item_matrix = user_item_matrix
        self.book_ids = book_ids
        self.n_neighbors = n_neighbors
        
    def fit(self):
        """
        Train the recommender model.
        
        This method should be implemented by subclasses.
        
        Returns
        -------
        self
        """
        raise NotImplementedError("Subclasses must implement fit()")
    
    def recommend_for_user(self, user_id, n_recommendations=10):
        """
        Generate recommendations for a user.
        
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
        raise NotImplementedError("Subclasses must implement recommend_for_user()")
    
    def recommend_similar_books(self, book_id, n=10):
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
            List of book IDs similar to the given book
        """
        raise NotImplementedError("Subclasses must implement recommend_similar_books()")
    
    def save(self, filepath):
        """
        Save the model to disk.
        
        Parameters
        ----------
        filepath : str
            Path to save the model to
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            logger.info(f"Model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    @classmethod
    def load(cls, filepath):
        """
        Load a model from disk.
        
        Parameters
        ----------
        filepath : str
            Path to load the model from
            
        Returns
        -------
        BaseRecommender
            Loaded model
        """
        try:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Model loaded from {filepath}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None


def load_data(features_dir: str = 'data/features') -> Tuple[sp.csr_matrix, np.ndarray]:
    """
    Load data for training the recommender model.
    
    Parameters
    ----------
    features_dir : str
        Directory containing feature files
    
    Returns
    -------
    tuple
        (user_item_matrix, book_ids)
    """
    try:
        # Create the features directory if it doesn't exist
        os.makedirs(features_dir, exist_ok=True)
        
        # Load the user-item matrix
        matrix_path = os.path.join(features_dir, 'user_item_matrix.npz')
        
        if os.path.exists(matrix_path):
            logger.info(f"Loading user-item matrix from {matrix_path}")
            user_item_matrix = load_npz(matrix_path)
        else:
            logger.error(f"User-item matrix not found at {matrix_path}")
            return None, None
            
        # Load the book IDs
        book_ids_path = os.path.join(features_dir, 'book_ids.npy')
        
        if os.path.exists(book_ids_path):
            logger.info(f"Loading book IDs from {book_ids_path}")
            book_ids = np.load(book_ids_path)
        else:
            logger.error(f"Book IDs not found at {book_ids_path}")
            return user_item_matrix, None
                
        return user_item_matrix, book_ids
            
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None, None
