"""
Collaborative filtering implementation for book recommender system.

This module implements a recommender system based on collaborative filtering.
"""
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
import logging
import os
import pickle
from typing import List, Dict, Tuple, Optional, Union, Any
from datetime import datetime
import argparse

from .train_model_base import BaseRecommender, load_data, evaluate_model_with_test_data

logger = logging.getLogger('train_model')


class CollaborativeRecommender(BaseRecommender):
    """
    A book recommender system based on collaborative filtering.
    
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
    
    def fit(self) -> 'CollaborativeRecommender':
        """
        Train the collaborative filtering model.
        
        Builds nearest neighbors model for item-based collaborative filtering.
        
        Returns
        -------
        self
        """
        if self.user_item_matrix is None:
            logger.error("Cannot train model: user-item matrix is missing")
            return self
            
        start_time = datetime.now()
        logger.info("Training collaborative filtering model...")
        
        try:
            # Calculate item vectors by transposing the user-item matrix
            item_vecs = self.user_item_matrix.T.tocsr()
            
            # Train nearest neighbors model on item vectors
            self.item_nn_model = NearestNeighbors(
                n_neighbors=self.n_neighbors + 1,  # +1 because the item itself will be included
                metric='cosine',
                algorithm='brute'
            )
            self.item_nn_model.fit(item_vecs)
            
            training_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Collaborative filtering model trained in {training_time:.2f} seconds")
            
            return self
            
        except Exception as e:
            logger.error(f"Error training collaborative filtering model: {e}")
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
        if self.user_item_matrix is None or self.item_nn_model is None:
            logger.error("Cannot generate recommendations: model not trained")
            return []
            
        if user_id not in self.user_ids:
            logger.warning(f"User {user_id} not found in training data")
            return []
            
        try:
            # Get books that the user has already rated
            user_row = self.user_item_matrix[user_id].toarray().flatten()
            rated_indices = np.where(user_row > 0)[0]
            
            if len(rated_indices) == 0:
                logger.warning(f"User {user_id} has not rated any books")
                return []
                
            # Get top rated books for this user
            top_rated_indices = sorted(
                [(i, user_row[i]) for i in rated_indices],
                key=lambda x: x[1],
                reverse=True
            )[:20]  # Consider only top 20 rated books
            
            recommendations = {}
            
            # For each book the user has rated, find similar books
            for idx, rating in top_rated_indices:
                # Find nearest neighbors to this book
                distances, indices = self.item_nn_model.kneighbors(
                    self.user_item_matrix.T[idx].toarray().reshape(1, -1),
                    n_neighbors=self.n_neighbors + 1
                )
                
                # Skip the first result (it's the book itself)
                similar_indices = indices.flatten()[1:]
                similar_scores = 1 - distances.flatten()[1:]  # Convert distance to similarity
                
                # Weight by user's rating
                weighted_scores = similar_scores * rating
                
                # Add to recommendations
                for i, score in zip(similar_indices, weighted_scores):
                    book_id = self.book_ids[i]
                    if book_id not in recommendations:
                        recommendations[book_id] = 0
                    recommendations[book_id] += score
                    
            # Remove books that the user has already rated
            rated_book_ids = {self.book_ids[i] for i in rated_indices}
            for book_id in rated_book_ids:
                if book_id in recommendations:
                    del recommendations[book_id]
                    
            return self._get_top_n_recommendations(recommendations, n_recommendations)
            
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
            List of recommended book IDs
        """
        if self.item_nn_model is None:
            logger.error("Cannot find similar books: model not trained")
            return []
            
        try:
            # Get matrix index for the book ID
            if book_id not in self.book_id_to_index:
                logger.warning(f"Book {book_id} not found in training data")
                return []
                
            book_idx = self.book_id_to_index[book_id]
            
            # Find nearest neighbors to this book
            distances, indices = self.item_nn_model.kneighbors(
                self.user_item_matrix.T[book_idx].toarray().reshape(1, -1),
                n_neighbors=n + 1
            )
            
            # Skip the first result (it's the book itself)
            similar_indices = indices.flatten()[1:n+1]
            
            # Convert to book IDs
            similar_book_ids = [int(self.book_ids[i]) for i in similar_indices]
            
            return similar_book_ids
            
        except Exception as e:
            logger.error(f"Error finding similar books for book {book_id}: {e}")
            return []
    
    def evaluate(self, test_df, k_values, strategies):
        """
        Evaluate the collaborative filtering model using precision@k and recall@k.
        
        Parameters
        ----------
        test_df : pandas.DataFrame
            Test data with user_id and book_id columns
        k_values : list
            List of k values to evaluate
        strategies : list
            List of strategies to evaluate
            
        Returns
        -------
        dict
            Evaluation results
        """
        # We only support collaborative filtering strategy
        if 'collaborative' not in strategies:
            return {}
            
        results = {'collaborative': {}}
        
        # Group test data by user
        test_users = test_df.groupby('user_id')['book_id'].apply(list).to_dict()
        
        # Calculate precision and recall for each k value
        for k in k_values:
            precisions = []
            recalls = []
            
            for user_id, true_books in test_users.items():
                # Skip users not in training data
                if user_id not in self.user_ids:
                    continue
                    
                # Get recommendations for this user
                recs = self.recommend_for_user(user_id, n_recommendations=k)
                
                # Calculate precision and recall
                n_relevant = len(set(recs) & set(true_books))
                
                precision = n_relevant / k if k > 0 else 0
                recall = n_relevant / len(true_books) if len(true_books) > 0 else 0
                
                precisions.append(precision)
                recalls.append(recall)
            
            # Average precision and recall and convert to regular Python float
            if precisions:
                results['collaborative'][f'precision@{k}'] = float(np.mean(precisions))
                results['collaborative'][f'recall@{k}'] = float(np.mean(recalls))
        
        return results


def train_model():
    """
    Train the collaborative filtering recommender model.
    
    Returns
    -------
    CollaborativeRecommender
        Trained recommender model
    """
    try:
        # Load data
        logger.info("Loading data for collaborative filtering model")
        user_item_matrix, _, _, book_ids, _ = load_data()
        
        if user_item_matrix is None or book_ids is None:
            logger.error("Could not load required data for collaborative filtering")
            return None
            
        # Create and train recommender
        recommender = CollaborativeRecommender(
            user_item_matrix=user_item_matrix,
            book_ids=book_ids
        )
        
        # Train the model
        recommender.fit()
        
        # Save the trained model
        model_dir = os.path.join('models')
        recommender.save(model_dir=model_dir, model_name='collaborative_recommender')
        
        return recommender
        
    except Exception as e:
        logger.error(f"Error training collaborative filtering model: {e}")
        return None


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description='Train and evaluate collaborative filtering book recommender model')
        parser.add_argument('--eval', action='store_true', help='Evaluate the model after training')
        parser.add_argument('--model-dir', type=str, default='models', help='Directory to save the model')
        args = parser.parse_args()
        
        # Train the model
        recommender = train_model()
        
        if recommender is None:
            logger.error("Failed to train collaborative filtering model")
            sys.exit(1)
            
        if args.eval:
            # Evaluate the model
            logger.info("Evaluating collaborative filtering model")
            results = evaluate_model_with_test_data(recommender)
            logger.info(f"Evaluation results: {results}")
            
        logger.info("Done!")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
