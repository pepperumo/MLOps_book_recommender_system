"""
Hybrid recommender implementation for book recommender system.

This module implements a hybrid recommender system that combines collaborative
filtering and content-based filtering approaches.
"""
import pandas as pd
import numpy as np
import scipy.sparse as sp
import logging
import os
import sys
import traceback
from typing import List, Dict, Tuple, Optional, Union, Any
from datetime import datetime
import argparse
import pickle

from .train_model_base import BaseRecommender, load_data, evaluate_model_with_test_data
from .train_model_collaborative import CollaborativeRecommender
from .train_model_content import ContentBasedRecommender

logger = logging.getLogger('train_model')


class HybridRecommender(BaseRecommender):
    """
    A hybrid book recommender system.
    
    This class combines collaborative filtering and content-based filtering
    to provide recommendations.
    """
    
    def __init__(self, 
                 collaborative_recommender: Optional[CollaborativeRecommender] = None,
                 content_based_recommender: Optional[ContentBasedRecommender] = None,
                 collaborative_weight: float = 0.7,
                 n_neighbors: int = 20):
        """
        Initialize the hybrid recommender system.
        
        Parameters
        ----------
        collaborative_recommender : CollaborativeRecommender, optional
            Trained collaborative filtering recommender
        content_based_recommender : ContentBasedRecommender, optional
            Trained content-based filtering recommender
        collaborative_weight : float, optional
            Weight given to collaborative filtering recommendations (between 0 and 1)
        n_neighbors : int, optional
            Number of neighbors to consider for recommendations
        """
        # Use data from the collaborative recommender for the base class
        user_item_matrix = None
        book_ids = None
        
        if collaborative_recommender is not None:
            user_item_matrix = collaborative_recommender.user_item_matrix
            book_ids = collaborative_recommender.book_ids
        elif content_based_recommender is not None:
            user_item_matrix = content_based_recommender.user_item_matrix
            book_ids = content_based_recommender.book_ids
            
        super().__init__(user_item_matrix, book_ids, n_neighbors)
        
        self.collaborative_recommender = collaborative_recommender
        self.content_based_recommender = content_based_recommender
        self.collaborative_weight = max(0, min(1, collaborative_weight))  # Clamp between 0 and 1
        self.content_weight = 1 - self.collaborative_weight
        
        logger.info(f"Initialized hybrid recommender with collaborative weight {self.collaborative_weight} "
                   f"and content weight {self.content_weight}")
    
    def fit(self) -> 'HybridRecommender':
        """
        Train the hybrid recommender model.
        
        Ensures that both component recommenders are trained.
        
        Returns
        -------
        self
        """
        start_time = datetime.now()
        logger.info("Training hybrid recommender model...")
        
        try:
            # Train collaborative recommender if available
            if self.collaborative_recommender is not None and self.collaborative_recommender.item_nn_model is None:
                logger.info("Training collaborative filtering component...")
                self.collaborative_recommender.fit()
                
            # Train content-based recommender if available
            if self.content_based_recommender is not None and self.content_based_recommender.book_similarity_matrix is None:
                logger.info("Training content-based filtering component...")
                self.content_based_recommender.fit()
            
            training_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Hybrid recommender model trained in {training_time:.2f} seconds")
            
            return self
            
        except Exception as e:
            logger.error(f"Error training hybrid recommender model: {e}")
            return self
    
    def recommend_for_user(self, user_id: int, n_recommendations: int = 10) -> List[int]:
        """
        Generate book recommendations for a user using a hybrid approach.
        
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
        if self.user_item_matrix is None:
            logger.error("Cannot generate recommendations: model not initialized")
            return []
            
        if user_id not in self.user_ids:
            logger.warning(f"User {user_id} not found in training data")
            return []
            
        try:
            all_recommendations = {}
            
            # Get recommendations from collaborative filtering
            if self.collaborative_recommender is not None and self.collaborative_weight > 0:
                collab_recs = self.collaborative_recommender.recommend_for_user(
                    user_id, 
                    n_recommendations=n_recommendations * 2  # Get more to have better candidates for merging
                )
                
                # Add to combined recommendations with collaborative weight
                for i, book_id in enumerate(collab_recs):
                    score = 1.0 - (i / len(collab_recs))  # Score based on rank, decreasing
                    all_recommendations[book_id] = self.collaborative_weight * score
            
            # Get recommendations from content-based filtering
            if self.content_based_recommender is not None and self.content_weight > 0:
                content_recs = self.content_based_recommender.recommend_for_user(
                    user_id, 
                    n_recommendations=n_recommendations * 2  # Get more to have better candidates for merging
                )
                
                # Add to combined recommendations with content weight
                for i, book_id in enumerate(content_recs):
                    score = 1.0 - (i / len(content_recs))  # Score based on rank, decreasing
                    if book_id not in all_recommendations:
                        all_recommendations[book_id] = 0
                    all_recommendations[book_id] += self.content_weight * score
            
            # Return top recommendations
            return self._get_top_n_recommendations(all_recommendations, n_recommendations)
            
        except Exception as e:
            logger.error(f"Error generating hybrid recommendations for user {user_id}: {e}")
            return []
    
    def recommend_similar_books(self, book_id: int, n: int = 10) -> List[int]:
        """
        Recommend books similar to a given book using a hybrid approach.
        
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
        try:
            all_recommendations = {}
            
            # Get recommendations from collaborative filtering
            if self.collaborative_recommender is not None and self.collaborative_weight > 0:
                collab_recs = self.collaborative_recommender.recommend_similar_books(
                    book_id, 
                    n=n * 2  # Get more to have better candidates for merging
                )
                
                # Add to combined recommendations with collaborative weight
                for i, similar_id in enumerate(collab_recs):
                    score = 1.0 - (i / len(collab_recs))  # Score based on rank, decreasing
                    all_recommendations[similar_id] = self.collaborative_weight * score
            
            # Get recommendations from content-based filtering
            if self.content_based_recommender is not None and self.content_weight > 0:
                content_recs = self.content_based_recommender.recommend_similar_books(
                    book_id, 
                    n=n * 2  # Get more to have better candidates for merging
                )
                
                # Add to combined recommendations with content weight
                for i, similar_id in enumerate(content_recs):
                    score = 1.0 - (i / len(content_recs))  # Score based on rank, decreasing
                    if similar_id not in all_recommendations:
                        all_recommendations[similar_id] = 0
                    all_recommendations[similar_id] += self.content_weight * score
            
            # Return top recommendations
            return self._get_top_n_recommendations(all_recommendations, n)
            
        except Exception as e:
            logger.error(f"Error generating hybrid similar books for book {book_id}: {e}")
            return []
    
    def evaluate(self, test_df, k_values, strategies):
        """
        Evaluate the hybrid recommender model using precision@k and recall@k.
        
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
        # We only support hybrid strategy
        if 'hybrid' not in strategies:
            return {}
            
        results = {'hybrid': {}}
        
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
                results['hybrid'][f'precision@{k}'] = float(np.mean(precisions))
                results['hybrid'][f'recall@{k}'] = float(np.mean(recalls))
        
        return results
    
    def save(self, model_dir: str = 'models', model_name: str = None) -> bool:
        """
        Save the hybrid model.
        
        Parameters
        ----------
        model_dir : str, optional
            Directory to save the model
        model_name : str, optional
            Name of the model file, without extension
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        if model_name is None:
            model_name = 'hybrid_recommender'
            
        # Save component models
        if self.collaborative_recommender is not None:
            self.collaborative_recommender.save(model_dir, 'collaborative_recommender')
            
        if self.content_based_recommender is not None:
            self.content_based_recommender.save(model_dir, 'content_based_recommender')
            
        # Save the hybrid model
        return super().save(model_dir, model_name)


def train_model(collaborative_weight: float = 0.7):
    """
    Train the hybrid recommender model.
    
    Parameters
    ----------
    collaborative_weight : float, optional
        Weight to give to collaborative filtering recommendations
        
    Returns
    -------
    HybridRecommender
        Trained recommender model
    """
    try:
        # Load or train collaborative recommender
        collab_model_path = os.path.join('models', 'collaborative_recommender.pkl')
        content_model_path = os.path.join('models', 'content_based_recommender.pkl')
        
        collaborative_recommender = None
        content_based_recommender = None
        
        # Try to load collaborative recommender
        if os.path.exists(collab_model_path):
            logger.info(f"Loading collaborative recommender from {collab_model_path}")
            try:
                with open(collab_model_path, 'rb') as f:
                    collaborative_recommender = pickle.load(f)
                logger.info("Collaborative recommender loaded successfully")
            except Exception as e:
                logger.error(f"Error loading collaborative recommender: {e}")
                # We'll train it later if needed
        
        # Try to load content-based recommender
        if os.path.exists(content_model_path):
            logger.info(f"Loading content-based recommender from {content_model_path}")
            try:
                with open(content_model_path, 'rb') as f:
                    content_based_recommender = pickle.load(f)
                logger.info("Content-based recommender loaded successfully")
            except Exception as e:
                logger.error(f"Error loading content-based recommender: {e}")
                # We'll train it later if needed
        
        # If we couldn't load the collaborative recommender, train it
        if collaborative_recommender is None:
            logger.info("Training new collaborative recommender")
            from .train_model_collaborative import train_model as train_collab
            collaborative_recommender = train_collab()
            
        # If we couldn't load the content-based recommender, train it
        if content_based_recommender is None:
            logger.info("Training new content-based recommender")
            from .train_model_content import train_model as train_content
            content_based_recommender = train_content()
        
        # Create and train hybrid recommender
        recommender = HybridRecommender(
            collaborative_recommender=collaborative_recommender,
            content_based_recommender=content_based_recommender,
            collaborative_weight=collaborative_weight
        )
        
        # Train (mostly just ensures component models are trained)
        recommender.fit()
        
        # Save the trained model
        model_dir = os.path.join('models')
        recommender.save(model_dir=model_dir, model_name='hybrid_recommender')
        
        return recommender
        
    except Exception as e:
        logger.error(f"Error training hybrid recommender model: {e}")
        return None


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description='Train and evaluate hybrid book recommender model')
        parser.add_argument('--eval', action='store_true', help='Evaluate the model after training')
        parser.add_argument('--model-dir', type=str, default='models', help='Directory to save the model')
        parser.add_argument('--collaborative-weight', type=float, default=0.7, 
                            help='Weight to give collaborative filtering (between 0 and 1)')
        args = parser.parse_args()
        
        # Train the model
        recommender = train_model(collaborative_weight=args.collaborative_weight)
        
        if recommender is None:
            logger.error("Failed to train hybrid recommender model")
            sys.exit(1)
            
        if args.eval:
            # Evaluate the model
            logger.info("Evaluating hybrid recommender model")
            results = evaluate_model_with_test_data(recommender)
            logger.info(f"Evaluation results: {results}")
            
        logger.info("Done!")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
