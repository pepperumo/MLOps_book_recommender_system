"""
This module contains the class definitions required to load the pickled collaborative filtering model.
It's a self-contained version of the model classes so that the Streamlit app can work
without external dependencies when deployed to platforms like Hugging Face.
"""

import os
import sys
import pickle
import logging
import numpy as np
import pandas as pd
import scipy.sparse as sp
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('pickle_loader')

# Base class definition
class BaseRecommender:
    """Base class for recommender systems."""
    
    def __init__(self):
        """Initialize the base recommender class."""
        self.params = {}
        
    def fit(self, *args, **kwargs):
        """Train the model with the given data."""
        return self
        
    def save(self, path):
        """Save the model to a file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
                
    @classmethod
    def load(cls, path):
        """Load the model from a file."""
        if not os.path.exists(path):
            logger.error(f"Model file not found: {path}")
            return None
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None

# Collaborative recommender class
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
            "n_jobs": n_jobs
        }
        
        if self.book_ids is not None:
            # Create mapping from book ID to matrix index
            self.book_id_to_index = {int(book_id): i for i, book_id in enumerate(self.book_ids)}
    
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
            top_items = [int(self.book_ids[item]) for item, score in scored_candidates[:n_recommendations]]
            return top_items
                
        except Exception as e:
            logger.error(f"Error generating recommendations for user {user_id}: {e}")
            return []
        
    def recommend_similar_books(self, book_id: int, n: int = 10) -> List[int]:
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
            return []

# Create dummy predict method for backward compatibility
def predict(self, user_id, book_id):
    """Predict rating for a user-book pair."""
    return 4.0

# Add the predict method to the CollaborativeRecommender class
CollaborativeRecommender.predict = predict

# Special unpickler class to handle class references
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'CollaborativeRecommender':
            return CollaborativeRecommender
        elif name == 'BaseRecommender':
            return BaseRecommender
        return super().find_class(module, name)

def load_model(model_path):
    """Load a pickled model with the correct class definitions."""
    try:
        logger.info(f"Loading model from {model_path}")
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return None
            
        # Use the custom unpickler instead of the standard pickle.load
        with open(model_path, 'rb') as f:
            unpickler = CustomUnpickler(f)
            model = unpickler.load()
            
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None
        
# Import traceback to provide better error messages
import traceback