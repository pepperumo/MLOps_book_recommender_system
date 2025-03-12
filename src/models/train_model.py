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
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any

# Add the project root to the Python path so we can import modules correctly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

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
                 n_neighbors: int = 20):
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
        """
        super().__init__(user_item_matrix, book_ids, n_neighbors)
        self.item_nn_model = None
        self.book_id_to_index = {}
        
        if self.book_ids is not None:
            # Create mapping from book ID to matrix index
            self.book_id_to_index = {int(book_id): i for i, book_id in enumerate(self.book_ids)}
    
    def fit(self):
        """
        Train the collaborative filtering model.
        
        Builds nearest neighbors model for item-based collaborative filtering.
        
        Returns
        -------
        self
        """
        if self.user_item_matrix is None:
            logger.error("Cannot train model: user_item_matrix is not initialized")
            return self
        
        try:
            start_time = time.time()
            logger.info("Training collaborative filtering model...")
            
            # Build item-item similarity model
            # Transpose matrix to compute item-item similarity
            item_item_matrix = self.user_item_matrix.T.tocsr()
            
            # Save the matrix shape for logging
            n_books, n_users = item_item_matrix.shape
            logger.info(f"User-item matrix shape: {self.user_item_matrix.shape} (users x books)")
            logger.info(f"Item-item matrix shape: {item_item_matrix.shape} (books x users)")
            
            # Train NearestNeighbors model for item similarity
            self.item_nn_model = NearestNeighbors(
                n_neighbors=min(self.n_neighbors + 1, n_books),  # +1 because it will include the item itself
                metric='cosine',
                algorithm='brute',
                n_jobs=-1
            ).fit(item_item_matrix)
            
            end_time = time.time()
            logger.info(f"Model training completed in {end_time - start_time:.2f} seconds")
            
            return self
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            logger.error(traceback.format_exc())
            return self
    
    def recommend_for_user(self, user_id: int, n_recommendations: int = 10):
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
        if self.item_nn_model is None:
            logger.error("Model not trained. Call fit() before making recommendations.")
            return []
            
        if self.user_item_matrix is None:
            logger.error("Cannot make recommendations: user_item_matrix is not initialized")
            return []
            
        try:
            # Check if user exists in the matrix
            user_idx = None
            # Get all users in user_item_matrix
            user_indices = np.arange(self.user_item_matrix.shape[0])
            
            if user_id in user_indices:
                user_idx = user_id
            else:
                logger.warning(f"User ID {user_id} not found in the matrix. Returning empty recommendations.")
                return []
                
            # Get user's profile (list of books they've rated)
            user_profile = self.user_item_matrix[user_idx].toarray().flatten()
            
            # Check if user has rated any books
            if user_profile.sum() == 0:
                logger.warning(f"User {user_id} has no ratings. Using popularity-based recommendations.")
                # Fallback to overall popularity
                item_popularity = self.user_item_matrix.sum(axis=0).A1
                top_items = np.argsort(item_popularity)[::-1][:n_recommendations]
                return [int(self.book_ids[i]) for i in top_items]
                
            # Get indices of books the user has already interacted with
            already_interacted = set(np.where(user_profile > 0)[0])
            
            # Use user profile to generate recommendations
            # Start with all books user has rated
            candidate_scores = {}
            
            for book_idx in already_interacted:
                # Use our item-item similarity model to find similar books
                distances, indices = self.item_nn_model.kneighbors(
                    self.user_item_matrix.T[book_idx].toarray().reshape(1, -1),
                    n_neighbors=self.n_neighbors
                )
                
                # Convert distances to similarities (1 - distance)
                similarities = 1 - distances.flatten()
                
                # Score similar books based on similarity
                for sim, idx in zip(similarities, indices.flatten()):
                    if idx not in already_interacted:
                        if idx not in candidate_scores:
                            candidate_scores[idx] = 0
                        # Weight similarity by user's rating
                        candidate_scores[idx] += sim * user_profile[book_idx]
            
            # Sort candidates by score
            sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Return top N book IDs
            recommended_indices = [idx for idx, _ in sorted_candidates[:n_recommendations]]
            recommended_book_ids = [int(self.book_ids[idx]) for idx in recommended_indices]
            
            return recommended_book_ids
            
        except Exception as e:
            logger.error(f"Error generating recommendations for user {user_id}: {str(e)}")
            logger.error(traceback.format_exc())
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


###################################################
# Main Training Function                           #
###################################################

def train_model(eval_model=True):
    """
    Train the collaborative filtering model.
    
    Parameters
    ----------
    eval_model : bool
        Whether to evaluate the model after training
        
    Returns
    -------
    tuple
        (collaborative_model, evaluation_results)
    """
    try:
        # Load data
        logger.info("Loading data for model training...")
        user_item_matrix, book_ids = load_data(features_dir='data/features')
        
        if user_item_matrix is None or book_ids is None:
            logger.error("Failed to load required data. Please check the data files.")
            return None, {}
        
        # Create and train the collaborative model
        logger.info("Creating and training collaborative filtering model...")
        collaborative_model = CollaborativeRecommender(
            user_item_matrix=user_item_matrix,
            book_ids=book_ids,
            n_neighbors=20
        )
        
        collaborative_model.fit()
        
        # Save the trained model
        model_dir = 'models'
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'collaborative.pkl')
        
        logger.info(f"Saving trained model to {model_path}")
        collaborative_model.save(model_path)
        
        # Evaluate the model if requested
        evaluation_results = {}
        if eval_model:
            logger.info("Evaluating model...")
            
            # Import here to avoid circular imports
            from src.models.evaluate_model import run_evaluation
            
            evaluation_results = run_evaluation(collaborative_model)
            
        return collaborative_model, evaluation_results
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        logger.error(traceback.format_exc())
        return None, {}


###################################################
# Command-line interface                           #
###################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a book recommender model')
    parser.add_argument('--eval', action='store_true', help='Evaluate the model after training')
    parser.add_argument('--features-dir', type=str, default='data/features',
                        help='Directory containing feature files')
    
    args = parser.parse_args()
    
    # Train the model
    model, results = train_model(eval_model=args.eval)
    
    if model is not None:
        logger.info("Model training completed successfully.")
        
        if args.eval and results:
            logger.info(f"Evaluation results: {results}")
    else:
        logger.error("Model training failed.")
        sys.exit(1)
