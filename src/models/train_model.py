import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.sparse import load_npz, save_npz
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
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


class BookRecommender:
    """
    A book recommender system that combines collaborative filtering and content-based approaches.
    
    This class loads pre-computed sparse matrices for user-item interactions and book features,
    and provides methods to recommend books for users based on different strategies.
    """
    
    def __init__(self, 
                 user_item_matrix: Optional[sp.csr_matrix] = None,
                 book_feature_matrix: Optional[sp.csr_matrix] = None,
                 book_similarity_matrix: Optional[sp.csr_matrix] = None,
                 book_ids: Optional[np.ndarray] = None,
                 feature_names: Optional[List[str]] = None,
                 n_neighbors: int = 20):
        """
        Initialize the recommender system.
        
        Parameters
        ----------
        user_item_matrix : scipy.sparse.csr_matrix
            Sparse matrix of user-item interactions
        book_feature_matrix : scipy.sparse.csr_matrix
            Sparse matrix of book features
        book_similarity_matrix : scipy.sparse.csr_matrix
            Pre-computed similarity matrix between books
        book_ids : array-like
            Array of book IDs corresponding to the matrices
        feature_names : list
            List of feature names
        n_neighbors : int
            Number of neighbors to consider for recommendations
        """
        try:
            self.user_item_matrix = user_item_matrix
            self.book_feature_matrix = book_feature_matrix
            self.book_similarity_matrix = book_similarity_matrix
            self.book_ids = book_ids
            self.feature_names = feature_names
            self.n_neighbors = n_neighbors
            
            # Models
            self.item_nn_model = None
            self.content_nn_model = None
            
            # Store user IDs for quick lookup
            if self.user_item_matrix is not None:
                self.user_ids = set(range(self.user_item_matrix.shape[0]))
                logger.info(f"Initialized BookRecommender with {len(self.user_ids)} users and {self.user_item_matrix.shape[1]} books")
            else:
                self.user_ids = set()
                logger.warning("Initialized BookRecommender without user-item matrix")
        except Exception as e:
            logger.error(f"Error initializing BookRecommender: {e}")
            logger.debug(traceback.format_exc())
            raise
    
    def fit(self) -> 'BookRecommender':
        """
        Train recommendation models.
        
        This fits:
        1. An item-based collaborative filtering model
        2. A content-based model using book features
        
        Returns
        -------
        self
        """
        try:
            logger.info("Training item-based collaborative filtering model...")
            start_time = time.time()
            
            # For item-based collaborative filtering, we use the transpose of user-item matrix
            # This way, each row represents a book, and we find similar books
            if self.book_similarity_matrix is not None:
                # If we already have a pre-computed similarity matrix, we don't need to fit a model
                logger.info("Using pre-computed book similarity matrix")
                self.item_nn_model = None
            else:
                # If we don't have a pre-computed similarity matrix, we fit a NearestNeighbors model
                # We use the transpose of the user-item matrix, so each row is a book
                if self.user_item_matrix is None:
                    logger.warning("Cannot train item-based model: user_item_matrix is None")
                else:
                    item_vecs = self.user_item_matrix.T.tocsr()
                    self.item_nn_model = NearestNeighbors(n_neighbors=self.n_neighbors + 1, 
                                                          metric='cosine', 
                                                          algorithm='brute')
                    self.item_nn_model.fit(item_vecs)
                    logger.info(f"Trained item-based model with shape {item_vecs.shape}")
            
            logger.info(f"Item-based model trained in {time.time() - start_time:.2f} seconds")
            
            # For content-based filtering, we use the book feature matrix
            logger.info("Training content-based model...")
            start_time = time.time()
            
            if self.book_feature_matrix is not None:
                self.content_nn_model = NearestNeighbors(n_neighbors=self.n_neighbors, 
                                                         metric='cosine', 
                                                         algorithm='brute')
                self.content_nn_model.fit(self.book_feature_matrix)
                logger.info(f"Content-based model trained in {time.time() - start_time:.2f} seconds")
                logger.info(f"Trained content-based model with shape {self.book_feature_matrix.shape}")
            else:
                logger.warning("No book feature matrix provided, skipping content-based model")
                
            return self
        except Exception as e:
            logger.error(f"Error training models: {e}")
            logger.debug(traceback.format_exc())
            return self
    
    def recommend_for_user(self, 
                          user_id: int, 
                          n_recommendations: int = 10, 
                          strategy: str = 'hybrid', 
                          alpha: float = 0.5) -> List[int]:
        """
        Generate book recommendations for a user.
        
        Parameters
        ----------
        user_id : int
            User ID to generate recommendations for
        n_recommendations : int, optional
            Number of recommendations to generate
        strategy : str, optional
            Recommendation strategy. Can be 'collaborative', 'content', or 'hybrid'
        alpha : float, optional
            Weight for hybrid recommendations (0.0-1.0). Higher values prioritize collaborative filtering.
            
        Returns
        -------
        list
            List of recommended book IDs
        """
        try:
            logger.info(f"Generating recommendations for user {user_id} using strategy '{strategy}'")
            
            if strategy == 'collaborative':
                if user_id in self.user_ids:  # Check if user exists in training data
                    logger.info(f"Using collaborative filtering for user {user_id}")
                    return self._collaborative_recommendations(user_id, n_recommendations)
                else:
                    # Fall back to content-based recommendations
                    logger.warning(f"User {user_id} not found in training data, falling back to content-based filtering")
                    return self._content_based_recommendations(user_id, n_recommendations)
            elif strategy == 'content':
                logger.info(f"Using content-based filtering for user {user_id}")
                return self._content_based_recommendations(user_id, n_recommendations)
            elif strategy == 'hybrid':
                if user_id in self.user_ids:  # Check if user exists in training data
                    logger.info(f"Using hybrid filtering for user {user_id} with alpha={alpha}")
                    return self._hybrid_recommendations(user_id, n_recommendations, alpha)
                else:
                    # Fall back to content-based recommendations
                    logger.warning(f"User {user_id} not found in training data, falling back to content-based filtering")
                    return self._content_based_recommendations(user_id, n_recommendations)
            else:
                logger.error(f"Unknown recommendation strategy: {strategy}")
                raise ValueError(f"Unknown recommendation strategy: {strategy}")
        except Exception as e:
            logger.error(f"Error generating recommendations for user {user_id}: {e}")
            logger.debug(traceback.format_exc())
            return []
    
    def _collaborative_recommendations(self, user_id: int, n: int) -> List[int]:
        """
        Generate collaborative filtering recommendations.
        
        Parameters
        ----------
        user_id : int
            User ID to generate recommendations for
        n : int
            Number of recommendations to generate
            
        Returns
        -------
        list
            List of recommended book IDs
        """
        if self.user_item_matrix is None or self.book_similarity_matrix is None:
            return []
            
        # Check if user is in our dataset
        if user_id >= self.user_item_matrix.shape[0]:
            # For users not in the training set, we can't use collaborative filtering
            return self._content_based_recommendations(user_id, n)
            
        # Get user's ratings
        user_ratings = self.user_item_matrix[user_id, :].toarray().flatten()
        
        # If user has no ratings, fall back to content-based
        if np.sum(user_ratings) == 0:
            return self._content_based_recommendations(user_id, n)
        
        # Initialize dictionary to store recommendations
        recommendations = {}
        
        # Number of books in the user-item matrix
        n_books = self.user_item_matrix.shape[1]
        
        # Only consider books that are in both matrices
        valid_books = min(n_books, self.book_similarity_matrix.shape[0])
        
        # For each book
        for i in range(valid_books):
            # Skip books the user has already rated
            if user_ratings[i] > 0:
                continue
            
            # Calculate recommendation score
            # Get similarity scores only for books that exist in both matrices
            for j in range(valid_books):
                # Only consider books the user has rated
                if user_ratings[j] > 0:
                    # Get similarity between book i and book j
                    try:
                        sim_score = self.book_similarity_matrix[i, j]
                        
                        # Add weighted similarity score
                        if i not in recommendations:
                            recommendations[i] = 0
                        recommendations[i] += sim_score * user_ratings[j]
                    except IndexError:
                        # Skip if index is out of range
                        continue
        
        # Convert book indices to book IDs
        if self.book_ids is not None:
            recommendations_with_ids = {}
            for book_idx, score in recommendations.items():
                if book_idx < len(self.book_ids):
                    book_id = self.book_ids[book_idx]
                    recommendations_with_ids[book_id] = score
            recommendations = recommendations_with_ids
        
        # Get top N recommendations
        return self._get_top_n_recommendations(recommendations, n)

    def _content_based_recommendations(self, user_id: int, n: int) -> List[int]:
        """
        Generate content-based recommendations.
        
        Parameters
        ----------
        user_id : int
            User ID to generate recommendations for
        n : int
            Number of recommendations to generate
            
        Returns
        -------
        list
            List of recommended book IDs
        """
        if self.content_nn_model is None or self.book_feature_matrix is None:
            return []
        
        # Check if user is in our dataset
        if user_id >= self.user_item_matrix.shape[0]:
            # For users not in the training set, return popular items
            return self._get_popular_recommendations(n)
            
        # Get user's ratings
        user_ratings = self.user_item_matrix[user_id, :].toarray().flatten()
        
        # If user has no ratings, return popular items
        if np.sum(user_ratings) == 0:
            return self._get_popular_recommendations(n)
        
        # Initialize dictionary to store recommendations
        recommendations = {}
        
        # Number of books in the user-item matrix
        n_books = self.user_item_matrix.shape[1]
        
        # Only consider books that are in both matrices
        valid_books = min(n_books, self.book_feature_matrix.shape[0])
        
        # For each book the user has rated
        for i in range(valid_books):
            if user_ratings[i] > 0:
                try:
                    # Find similar books based on features
                    distances, indices = self.content_nn_model.kneighbors(
                        self.book_feature_matrix[i].toarray().reshape(1, -1),
                        n_neighbors=min(n+1, self.book_feature_matrix.shape[0])  # Ensure we don't request more neighbors than available
                    )
                    
                    # Convert distances to similarities (1 - distance)
                    similarities = 1 - distances.flatten()
                    
                    # The user's rating for this book
                    rating = user_ratings[i]
                    
                    # Update recommendations
                    for idx, sim in zip(indices.flatten()[1:], similarities[1:]):
                        if idx < valid_books and user_ratings[idx] == 0:  # Only recommend unrated books
                            if idx not in recommendations:
                                recommendations[idx] = 0
                            recommendations[idx] += sim * rating
                except Exception as e:
                    continue  # Skip this book if there's an error
        
        # If no content-based recommendations could be generated, fall back to popular items
        if not recommendations:
            return self._get_popular_recommendations(n)
            
        # Convert book indices to book IDs
        if self.book_ids is not None:
            recommendations_with_ids = {}
            for book_idx, score in recommendations.items():
                if book_idx < len(self.book_ids):
                    book_id = self.book_ids[book_idx]
                    recommendations_with_ids[book_id] = score
            recommendations = recommendations_with_ids
        
        # Get top N recommendations
        return self._get_top_n_recommendations(recommendations, n)
        
    def _get_popular_recommendations(self, n: int) -> List[int]:
        """
        Generate recommendations based on popularity.
        
        Parameters
        ----------
        n : int
            Number of recommendations to generate
            
        Returns
        -------
        list
            List of recommended book IDs
        """
        if self.user_item_matrix is None:
            return []
            
        # Sum ratings for each book to get popularity
        popularity = np.asarray(self.user_item_matrix.sum(axis=0)).flatten()
        
        # Get indices of top N popular books
        popular_indices = np.argsort(popularity)[-n:][::-1]
        
        # Convert indices to book IDs if available
        if self.book_ids is not None:
            popular_ids = [int(self.book_ids[idx]) for idx in popular_indices if idx < len(self.book_ids)]
        else:
            popular_ids = [int(idx) for idx in popular_indices]
            
        return popular_ids

    def _hybrid_recommendations(self, user_id: int, n: int, alpha: float) -> List[int]:
        """
        Generate hybrid recommendations.
        
        Parameters
        ----------
        user_id : int
            User ID to generate recommendations for
        n : int
            Number of recommendations to generate
        alpha : float
            Weight for hybrid recommendations (0.0 to 1.0), where:
            - 0.0: Only content-based
            - 1.0: Only collaborative filtering
            
        Returns
        -------
        list
            List of recommended book IDs
        """
        # Get recommendations from both methods
        collab_recs = self._collaborative_recommendations(user_id, 2*n)  # Get more to ensure enough for hybrid
        content_recs = self._content_based_recommendations(user_id, 2*n)  # Get more to ensure enough for hybrid
        
        # If one method doesn't produce recommendations, return results from the other
        if not collab_recs:
            return content_recs[:n]
        if not content_recs:
            return collab_recs[:n]
            
        # Create a score dictionary to merge results
        hybrid_scores = {}
        
        # Add collaborative filtering recommendations with weight alpha
        for i, book_id in enumerate(collab_recs):
            # Higher ranked items get higher scores - use inverse of position
            score = (len(collab_recs) - i) / len(collab_recs)
            hybrid_scores[int(book_id)] = score * alpha
            
        # Add content-based recommendations with weight (1-alpha)
        for i, book_id in enumerate(content_recs):
            # Higher ranked items get higher scores - use inverse of position
            score = (len(content_recs) - i) / len(content_recs)
            
            book_id = int(book_id)
            if book_id in hybrid_scores:
                hybrid_scores[book_id] += score * (1 - alpha)
            else:
                hybrid_scores[book_id] = score * (1 - alpha)
                
        # Get top recommendations based on combined scores
        return [int(book_id) for book_id in sorted(hybrid_scores.keys(), key=lambda x: hybrid_scores[x], reverse=True)[:n]]

    def recommend_similar_books(self, book_id: int, n: int = 10) -> List[int]:
        """
        Recommend books similar to a given book.
        
        Parameters
        ----------
        book_id : int
            Book ID to find similar books for
        n : int
            Number of similar books to recommend
            
        Returns
        -------
        list
            List of recommended book IDs
        """
        # Convert original book ID to encoded ID if necessary
        if self.book_ids is not None:
            try:
                encoded_book_id = np.where(self.book_ids == book_id)[0][0]
            except IndexError:
                print(f"Book ID {book_id} not found in the dataset")
                return []
        else:
            encoded_book_id = book_id
        
        # Get similar books using content-based features
        if self.book_feature_matrix is not None and self.content_nn_model is not None:
            # Find similar books based on features
            distances, indices = self.content_nn_model.kneighbors(
                self.book_feature_matrix[encoded_book_id].toarray().reshape(1, -1),
                n_neighbors=n+1  # +1 because the book itself will be included
            )
            
            # Convert distances to similarities (1 - distance)
            similarities = 1 - distances.flatten()
            
            # Remove the book itself from recommendations
            similar_books = []
            for i, idx in enumerate(indices.flatten()):
                if idx != encoded_book_id:
                    similar_books.append(idx)
                if len(similar_books) >= n:
                    break
        # If content-based not available, use collaborative filtering
        elif self.book_similarity_matrix is not None:
            # Use pre-computed similarity matrix
            sim_scores = self.book_similarity_matrix[encoded_book_id].toarray().flatten()
            # Get the indices of the top n similar books (excluding the book itself)
            similar_indices = np.argsort(sim_scores)[::-1]
            similar_books = [idx for idx in similar_indices if idx != encoded_book_id][:n]
        else:
            # Use item-based collaborative filtering with nearest neighbors
            if self.item_nn_model is not None:
                item_vecs = self.user_item_matrix.T.tocsr()
                distances, indices = self.item_nn_model.kneighbors(
                    item_vecs[encoded_book_id].reshape(1, -1),
                    n_neighbors=n+1  # +1 because the book itself will be included
                )
                # Remove the book itself from recommendations
                similar_books = [idx for idx in indices.flatten() if idx != encoded_book_id][:n]
            else:
                print("No models available for finding similar books")
                return []
        
        # Convert encoded book IDs to original book IDs
        if self.book_ids is not None:
            recommendations = [int(self.book_ids[book_id]) for book_id in similar_books]
        else:
            # Convert to Python int type to avoid np.int64 in output
            recommendations = [int(book_id) for book_id in similar_books]
        
        return recommendations
    
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
        
    def evaluate(self, test_df: pd.DataFrame, k_values: List[int] = [5, 10, 20], strategies: List[str] = ['collaborative', 'content', 'hybrid']) -> Dict[str, Dict[str, float]]:
        """
        Evaluate the recommender system on test data.
        
        Parameters
        ----------
        test_df : pandas.DataFrame
            Test data containing user-item interactions
        k_values : list
            List of k values for precision@k and recall@k metrics
        strategies : list
            List of recommendation strategies to evaluate
            
        Returns
        -------
        dict
            Dictionary of evaluation metrics
        """
        if test_df is None or len(test_df) == 0:
            print("No test data provided for evaluation")
            return {}
        
        # Extract user-item interactions from test data
        user_item_pairs = test_df[['user_id', 'book_id']].values
        
        # Prepare results dictionary
        results = {strategy: {} for strategy in strategies}
        
        # Convert original book IDs to encoded IDs if needed
        if self.book_ids is not None:
            book_id_to_idx = {book_id: idx for idx, book_id in enumerate(self.book_ids)}
        else:
            book_id_to_idx = None
            
        # Group test data by user
        user_groups = test_df.groupby('user_id')
        
        # Keep track of users we can evaluate
        evaluated_users = 0
        skipped_users = 0
        errors = 0
        users_not_found = 0
        
        # Process each user in the test set
        for user_id, group in user_groups:
            # Get the books this user has rated in the test set
            user_test_books = group['book_id'].values
            
            # Skip users with too few ratings
            if len(user_test_books) < 2:
                skipped_users += 1
                continue
            
            # Get recommendations for each strategy
            for strategy in strategies:
                try:
                    # Get recommendations for this user
                    recs = self.recommend_for_user(user_id, n_recommendations=max(k_values), strategy=strategy)
                    
                    # Skip if no recommendations (user might not be in training set)
                    if not recs:
                        users_not_found += 1
                        continue
                    
                    # Calculate precision and recall at different k values
                    for k in k_values:
                        # Consider only top-k recommendations
                        top_k_recs = set(recs[:k])
                        
                        # Calculate precision@k: proportion of recommended items that are relevant
                        precision = len(top_k_recs.intersection(user_test_books)) / len(top_k_recs) if len(top_k_recs) > 0 else 0
                        
                        # Calculate recall@k: proportion of relevant items that are recommended
                        recall = len(top_k_recs.intersection(user_test_books)) / len(user_test_books) if len(user_test_books) > 0 else 0
                        
                        # Store metrics
                        if f'precision@{k}' not in results[strategy]:
                            results[strategy][f'precision@{k}'] = []
                        if f'recall@{k}' not in results[strategy]:
                            results[strategy][f'recall@{k}'] = []
                            
                        results[strategy][f'precision@{k}'].append(precision)
                        results[strategy][f'recall@{k}'].append(recall)
                    
                    evaluated_users += 1
                except Exception as e:
                    # Skip users that cause errors
                    errors += 1
                    if errors < 10:  # Only print first 10 errors to avoid flooding console
                        print(f"Error evaluating user {user_id} with strategy {strategy}: {e}")
                    continue
        
        # Print summary of users not found in training data
        if users_not_found > 0:
            print(f"{users_not_found} users were not found in training data and used content-based fallback")
        
        print(f"Evaluated {evaluated_users} users, skipped {skipped_users} users with insufficient data, encountered {errors} errors")
        
        # Calculate average metrics
        for strategy in strategies:
            for metric in list(results[strategy].keys()):
                if results[strategy][metric]:
                    results[strategy][metric] = sum(results[strategy][metric]) / len(results[strategy][metric])
                else:
                    results[strategy][metric] = 0.0
        
        return results
    
    def save(self, model_dir: str = 'models') -> bool:
        """
        Save the trained model to files.
        
        Parameters
        ----------
        model_dir : str
            Directory to save the model
            
        Returns
        -------
        bool
            True if the model was saved successfully, False otherwise
        """
        try:
            logger.info(f"Saving model to {model_dir}")
            
            # Create directory if it doesn't exist
            os.makedirs(model_dir, exist_ok=True)
            
            # Save the model using pickle
            model_path = os.path.join(model_dir, 'book_recommender.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(self, f)
                
            # Save also individual matrices for potential future use without loading the entire model
            if self.user_item_matrix is not None:
                save_npz(os.path.join(model_dir, 'user_item_matrix.npz'), self.user_item_matrix)
                logger.info(f"Saved user-item matrix with shape {self.user_item_matrix.shape}")
                
            if self.book_feature_matrix is not None:
                save_npz(os.path.join(model_dir, 'book_feature_matrix.npz'), self.book_feature_matrix)
                logger.info(f"Saved book feature matrix with shape {self.book_feature_matrix.shape}")
                
            if self.book_similarity_matrix is not None:
                save_npz(os.path.join(model_dir, 'book_similarity_matrix.npz'), self.book_similarity_matrix)
                logger.info(f"Saved book similarity matrix with shape {self.book_similarity_matrix.shape}")
                
            if self.book_ids is not None:
                np.save(os.path.join(model_dir, 'book_ids.npy'), self.book_ids)
                logger.info(f"Saved {len(self.book_ids)} book IDs")
                
            if self.feature_names is not None:
                with open(os.path.join(model_dir, 'feature_names.txt'), 'w') as f:
                    for name in self.feature_names:
                        f.write(f"{name}\n")
                logger.info(f"Saved {len(self.feature_names)} feature names")
                
            # Save metadata about the model
            metadata = {
                'user_count': len(self.user_ids) if self.user_item_matrix is not None else 0,
                'book_count': self.user_item_matrix.shape[1] if self.user_item_matrix is not None else 0,
                'feature_count': len(self.feature_names) if self.feature_names is not None else 0,
                'n_neighbors': self.n_neighbors,
                'timestamp': timestamp
            }
            
            with open(os.path.join(model_dir, 'model_metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=4)
                
            # Save metadata as CSV for easier reading in Excel/other tools
            results_dir = os.path.join('data', 'results')
            os.makedirs(results_dir, exist_ok=True)
            pd.DataFrame([metadata]).to_csv(
                os.path.join(results_dir, f'model_metadata_{timestamp}.csv'), 
                index=False
            )
                
            logger.info(f"Model saved successfully to {model_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            logger.debug(traceback.format_exc())
            return False


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


def train_model() -> Optional[BookRecommender]:
    """
    Train the book recommender model.
    
    Returns
    -------
    BookRecommender
        Trained recommender model
    """
    try:
        logger.info("Starting model training")
        
        # Load data
        user_item_matrix, book_feature_matrix, book_similarity_matrix, book_ids, feature_names = load_data()
        
        if user_item_matrix is None:
            logger.error("Failed to load user-item matrix, cannot train model")
            return None
        
        # Initialize and train the recommender
        recommender = BookRecommender(
            user_item_matrix=user_item_matrix,
            book_feature_matrix=book_feature_matrix,
            book_similarity_matrix=book_similarity_matrix,
            book_ids=book_ids,
            feature_names=feature_names
        )
        
        # Fit the model
        recommender.fit()
        
        # Save the model
        model_dir = os.path.join('models')
        success = recommender.save(model_dir)
        
        if success:
            logger.info("Model training completed successfully")
        else:
            logger.warning("Model trained but not saved successfully")
            
        return recommender
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        logger.debug(traceback.format_exc())
        return None


def evaluate_model_with_test_data(recommender: BookRecommender, test_file: str = 'merged_test.csv', data_dir: str = 'data/processed') -> Dict[str, Dict[str, float]]:
    """
    Evaluate the model with test data.
    
    Parameters
    ----------
    recommender : BookRecommender
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
    try:
        test_path = os.path.join(data_dir, test_file)
        logger.info(f"Evaluating model with test data from {test_path}")
        
        if not os.path.exists(test_path):
            logger.error(f"Test file not found: {test_path}")
            return {}
        
        # Load test data
        test_df = pd.read_csv(test_path)
        logger.info(f"Loaded test data with shape {test_df.shape}")
        
        # Evaluate with different strategies and k values
        results = recommender.evaluate(test_df)
        
        # Save results to CSV
        results_dir = os.path.join('data', 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Convert nested dict to DataFrame
        results_df = pd.DataFrame()
        for strategy, metrics in results.items():
            strategy_df = pd.DataFrame([metrics])
            strategy_df['strategy'] = strategy
            results_df = pd.concat([results_df, strategy_df])
        
        # Save to CSV
        csv_path = os.path.join(results_dir, f'evaluation_results_{timestamp}.csv')
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Saved evaluation results to {csv_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        logger.debug(traceback.format_exc())
        return {}


def evaluate_model(recommender: BookRecommender, test_users: int = 5, n_recommendations: int = 5) -> None:
    """
    Evaluate the recommender model by generating recommendations for a few test users.
    
    Parameters
    ----------
    recommender : BookRecommender
        The trained recommender model
    test_users : int
        Number of test users to generate recommendations for
    n_recommendations : int
        Number of recommendations to generate per user
    """
    try:
        logger.info(f"Generating sample recommendations for {test_users} test users")
        
        # If we don't have a recommender or no users, return
        if recommender is None or not recommender.user_ids:
            logger.error("Cannot generate recommendations: recommender is None or has no users")
            return
        
        # Get a few random users
        all_users = list(recommender.user_ids)
        if test_users > len(all_users):
            test_users = len(all_users)
            
        import random
        random.seed(42)  # For reproducibility
        sample_users = random.sample(all_users, test_users)
        
        # Store results for all strategies and users
        all_recommendations = []
        
        # Generate recommendations for each user with different strategies
        for user_id in sample_users:
            user_recommendations = {'user_id': user_id}
            
            # Generate collaborative recommendations
            try:
                collab_recs = recommender.recommend_for_user(
                    user_id, n_recommendations=n_recommendations, strategy='collaborative')
                user_recommendations['collaborative'] = collab_recs
                logger.info(f"Generated {len(collab_recs)} collaborative recommendations for user {user_id}")
            except Exception as e:
                logger.error(f"Error generating collaborative recommendations for user {user_id}: {e}")
                user_recommendations['collaborative'] = []
            
            # Generate content-based recommendations
            try:
                content_recs = recommender.recommend_for_user(
                    user_id, n_recommendations=n_recommendations, strategy='content')
                user_recommendations['content'] = content_recs
                logger.info(f"Generated {len(content_recs)} content-based recommendations for user {user_id}")
            except Exception as e:
                logger.error(f"Error generating content-based recommendations for user {user_id}: {e}")
                user_recommendations['content'] = []
            
            # Generate hybrid recommendations
            try:
                hybrid_recs = recommender.recommend_for_user(
                    user_id, n_recommendations=n_recommendations, strategy='hybrid')
                user_recommendations['hybrid'] = hybrid_recs
                logger.info(f"Generated {len(hybrid_recs)} hybrid recommendations for user {user_id}")
            except Exception as e:
                logger.error(f"Error generating hybrid recommendations for user {user_id}: {e}")
                user_recommendations['hybrid'] = []
            
            all_recommendations.append(user_recommendations)
        
        # Save recommendations to CSV
        results_dir = os.path.join('data', 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Format the recommendations for CSV output
        formatted_recommendations = []
        for user_recs in all_recommendations:
            user_id = user_recs['user_id']
            
            for strategy, recs in user_recs.items():
                if strategy == 'user_id':
                    continue
                
                for i, book_id in enumerate(recs):
                    formatted_recommendations.append({
                        'user_id': user_id,
                        'strategy': strategy,
                        'rank': i + 1,
                        'book_id': book_id
                    })
        
        # Save to CSV
        if formatted_recommendations:
            rec_df = pd.DataFrame(formatted_recommendations)
            csv_path = os.path.join(results_dir, f'sample_recommendations_{timestamp}.csv')
            rec_df.to_csv(csv_path, index=False)
            logger.info(f"Saved sample recommendations to {csv_path}")
        
    except Exception as e:
        logger.error(f"Error in evaluate_model: {e}")
        logger.debug(traceback.format_exc())


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description='Train and evaluate book recommender model')
        parser.add_argument('--features-dir', type=str, default='data/features',
                            help='Directory containing feature matrices')
        parser.add_argument('--model-dir', type=str, default='models',
                            help='Directory to save the trained model')
        parser.add_argument('--evaluate', action='store_true',
                            help='Evaluate the model after training')
        parser.add_argument('--test-file', type=str, default='merged_test.csv',
                            help='Name of the test file (if evaluating)')
        parser.add_argument('--test-dir', type=str, default='data/processed',
                            help='Directory containing the test file (if evaluating)')
        
        args = parser.parse_args()
        
        logger.info(f"Starting train_model.py with arguments: {args}")
        
        # Train the model
        recommender = train_model()
        
        if recommender is None:
            logger.error("Failed to train model, exiting")
            sys.exit(1)
        
        # Evaluate the model if requested
        if args.evaluate:
            # Generate sample recommendations for a few users
            evaluate_model(recommender)
            
            # Evaluate with test data if available
            results = evaluate_model_with_test_data(
                recommender, 
                test_file=args.test_file, 
                data_dir=args.test_dir
            )
            
            if not results:
                logger.warning("No evaluation results were produced")
            
        logger.info("train_model.py completed successfully")
        
    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)
