import pandas as pd
import numpy as np
import scipy.sparse as sp
import os
import sys
import logging
import traceback
from typing import Tuple, List, Dict, Optional, Union, Any
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Set up logging
log_dir = os.path.join('logs')
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(log_dir, f'build_features_{timestamp}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('build_features')


def read_ratings(data_dir: str = 'data/processed') -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """
    Read the processed ratings data and create a sparse user-item matrix.
    
    Parameters
    ----------
    data_dir : str
        Directory containing the processed data files
        
    Returns
    -------
    Tuple[sp.csr_matrix, np.ndarray, np.ndarray]
        Tuple containing (user_item_matrix, user_ids, item_ids)
    """
    logger.info(f"Reading ratings data from {data_dir}")
    
    try:
        # Load merged_train.csv which contains ratings
        ratings_file = os.path.join(data_dir, 'merged_train.csv')
        if not os.path.exists(ratings_file):
            logger.error(f"Ratings file not found: {ratings_file}")
            return sp.csr_matrix((0, 0)), np.array([]), np.array([])
            
        logger.info(f"Reading ratings from {ratings_file}")
        ratings_df = pd.read_csv(ratings_file)
        logger.info(f"Loaded ratings dataframe with shape {ratings_df.shape}")
        
        # Check if we have the expected columns
        if 'user_id' not in ratings_df.columns or 'book_id' not in ratings_df.columns:
            logger.error(f"Missing required columns in {ratings_file}")
            return sp.csr_matrix((0, 0)), np.array([]), np.array([])
            
        if 'rating' not in ratings_df.columns:
            # If no rating column, assume implicit ratings (all 1.0)
            logger.warning(f"No rating column found in {ratings_file}, assuming implicit ratings")
            ratings_df['rating'] = 1.0
        
        # Get unique user and book IDs
        user_ids = ratings_df['user_id'].unique()
        book_ids = ratings_df['book_id'].unique()
        
        logger.info(f"Found {len(user_ids)} unique users and {len(book_ids)} unique books")
        
        # Create label encoders for user and book IDs
        user_encoder = LabelEncoder().fit(user_ids)
        book_encoder = LabelEncoder().fit(book_ids)
        
        # Transform IDs to sequential integers starting from 0
        ratings_df['user_id_encoded'] = user_encoder.transform(ratings_df['user_id'])
        ratings_df['book_id_encoded'] = book_encoder.transform(ratings_df['book_id'])
        
        # Save the book ID mapping for later use in predictions
        mapping_df = pd.DataFrame({
            'book_id': book_ids,
            'book_id_encoded': book_encoder.transform(book_ids)
        })
        mapping_path = os.path.join(os.path.dirname(data_dir), 'processed', 'book_id_mapping.csv')
        os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
        mapping_df.to_csv(mapping_path, index=False)
        logger.info(f"Saved book ID mapping to {mapping_path}")
        
        # Also save the user ID mapping
        user_mapping_df = pd.DataFrame({
            'user_id': user_ids,
            'user_id_encoded': user_encoder.transform(user_ids)
        })
        user_mapping_path = os.path.join(os.path.dirname(data_dir), 'processed', 'user_id_mapping.csv')
        user_mapping_df.to_csv(user_mapping_path, index=False)
        logger.info(f"Saved user ID mapping to {user_mapping_path}")
        
        # Map IDs to indices
        rows = ratings_df['user_id_encoded'].values
        cols = ratings_df['book_id_encoded'].values
        ratings = ratings_df['rating'].values
        
        # Create sparse matrix
        user_item_matrix = sp.csr_matrix((ratings, (rows, cols)), 
                                         shape=(len(user_ids), len(book_ids)))
        
        logger.info(f"Created user-item matrix with shape {user_item_matrix.shape}")
        return user_item_matrix, user_ids, book_ids
        
    except Exception as e:
        logger.error(f"Error reading ratings data: {e}")
        logger.debug(traceback.format_exc())
        return sp.csr_matrix((0, 0)), np.array([]), np.array([])


def create_sparse_user_item_matrix(ratings_df):
    """
    Creates a sparse user-item matrix from ratings data.
    
    Parameters
    ----------
    ratings_df : pd.DataFrame
        DataFrame containing user ratings with encoded IDs
        
    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse matrix where rows are users, columns are books, and values are ratings
    tuple
        (num_users, num_books) dimensions of the matrix
    """
    # Check that we have the encoded columns
    if 'user_id_encoded' not in ratings_df.columns or 'book_id_encoded' not in ratings_df.columns:
        raise ValueError("Ratings DataFrame must contain encoded user and book IDs")
    
    # Get dimensions
    num_users = ratings_df['user_id_encoded'].max() + 1
    num_books = ratings_df['book_id_encoded'].max() + 1
    
    # Create sparse matrix in COO format (easy to construct)
    user_indices = ratings_df['user_id_encoded'].values
    book_indices = ratings_df['book_id_encoded'].values
    
    if 'rating' in ratings_df.columns:
        rating_values = ratings_df['rating'].values
    else:
        # If no ratings, use binary interactions (1 = interaction)
        rating_values = np.ones(len(ratings_df))
    
    # Create sparse matrix
    matrix = sp.coo_matrix(
        (rating_values, (user_indices, book_indices)),
        shape=(num_users, num_books)
    )
    
    # Convert to CSR format for efficient row operations
    return matrix.tocsr(), (num_users, num_books)


def extract_book_features(data_dir: str = 'data/processed', 
                         min_df: int = 5, 
                         max_df: float = 0.8) -> Tuple[sp.csr_matrix, List[str], np.ndarray]:
    """
    Extract features from book metadata.
    
    Parameters
    ----------
    data_dir : str
        Directory containing the processed data files
    min_df : int
        Minimum document frequency for TF-IDF
    max_df : float
        Maximum document frequency for TF-IDF
        
    Returns
    -------
    Tuple[sp.csr_matrix, List[str], np.ndarray]
        Tuple containing (book_feature_matrix, feature_names, book_ids)
    """
    logger.info(f"Extracting book features from {data_dir}")
    
    try:
        # Load merged_train.csv which should contain book metadata
        book_file = os.path.join(data_dir, 'merged_train.csv')
        if not os.path.exists(book_file):
            logger.error(f"Book file not found: {book_file}")
            return sp.csr_matrix((0, 0)), [], np.array([])
        
        df = pd.read_csv(book_file)
        logger.info(f"Loaded book data with shape {df.shape}")
        
        # Check for required book ID column
        if 'book_id' not in df.columns:
            logger.error("book_id column not found in book data")
            return sp.csr_matrix((0, 0)), [], np.array([])
        
        # Drop duplicates to get unique books
        df = df.drop_duplicates(subset=['book_id'])
        logger.info(f"Found {len(df)} unique books after removing duplicates")
        
        # Get book IDs
        book_ids = df['book_id'].values
        
        # Prepare text features
        features = []
        feature_names = []
        
        # Add features based on what columns are available
        if 'title' in df.columns:
            df['title'] = df['title'].fillna('').astype(str)
            features.append(df['title'])
            feature_names.append('title')
        
        if 'authors' in df.columns:
            df['authors'] = df['authors'].fillna('').astype(str)
            features.append(df['authors'])
            feature_names.append('authors')
            
        if 'original_title' in df.columns:
            df['original_title'] = df['original_title'].fillna('').astype(str)
            features.append(df['original_title'])
            feature_names.append('original_title')
            
        if len(features) == 0:
            logger.warning("No text features found for TF-IDF extraction")
            return sp.csr_matrix((len(df), 0)), [], book_ids
        
        # Combine features into a single text field
        df['text_features'] = ''
        for feature in features:
            df['text_features'] += ' ' + feature
        
        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df, 
                                    stop_words='english')
        
        tfidf_matrix = vectorizer.fit_transform(df['text_features'])
        tfidf_feature_names = vectorizer.get_feature_names_out()
        
        logger.info(f"Created book feature matrix with shape {tfidf_matrix.shape}")
        logger.info(f"Extracted {len(tfidf_feature_names)} TF-IDF features")
        
        # Save vectorizer vocabulary as CSV for reference
        vocab_df = pd.DataFrame({
            'term': list(vectorizer.vocabulary_.keys()),
            'index': list(vectorizer.vocabulary_.values())
        }).sort_values('index')
        
        output_dir = os.path.join('data', 'features')
        os.makedirs(output_dir, exist_ok=True)
        vocab_file = os.path.join(output_dir, f'tfidf_vocabulary_{timestamp}.csv')
        vocab_df.to_csv(vocab_file, index=False, encoding='utf-8')
        logger.info(f"Saved TF-IDF vocabulary to {vocab_file}")
        
        return tfidf_matrix, tfidf_feature_names.tolist(), book_ids
    
    except Exception as e:
        logger.error(f"Error extracting book features: {e}")
        logger.debug(traceback.format_exc())
        return sp.csr_matrix((0, 0)), [], np.array([])


def read_books(books_csv, data_dir="data/processed"):
    """
    Read book data from a CSV file and create a sparse feature matrix.
    
    Parameters
    ----------
    books_csv : str
        Name of the CSV file containing book data
    data_dir : str
        Directory containing the CSV file
        
    Returns
    -------
    tuple
        (book_ids, sparse feature matrix, feature names)
    """
    file_path = os.path.join(data_dir, books_csv)
    
    try:
        books_df = pd.read_csv(file_path)
        print(f"Read {len(books_df)} books from {file_path}")
        
        # Ensure book_id is present
        if 'book_id' not in books_df.columns:
            print(f"Error: book_id column missing from {file_path}")
            return None, None, None
            
        # Convert book_id to integer if not already
        if books_df['book_id'].dtype != 'int64':
            books_df['book_id'] = books_df['book_id'].astype(int)
        
        # Sort by book_id for consistent feature ordering
        books_df = books_df.sort_values('book_id')
        
        # Get book IDs as numpy array
        book_ids = books_df['book_id'].values
        
        # Initialize for feature collection
        feature_matrices = []
        feature_names = []
        
        # Use a single approach for text features - word counts
        if 'authors' in books_df.columns:
            # Fill missing authors
            books_df['authors'] = books_df['authors'].fillna('Unknown')
            
            # Simple author word count feature
            from collections import Counter
            all_authors = Counter()
            for authors in books_df['authors']:
                all_authors.update(authors.split())
            
            # Keep top 100 author words
            top_authors = [author for author, _ in all_authors.most_common(100)]
            
            # Create a simple binary feature matrix for authors
            author_matrix = np.zeros((len(books_df), len(top_authors)))
            
            for i, authors in enumerate(books_df['authors']):
                for j, author in enumerate(top_authors):
                    if author in authors:
                        author_matrix[i, j] = 1
            
            # Convert to sparse
            author_features = sp.csr_matrix(author_matrix)
            feature_matrices.append(author_features)
            feature_names.extend([f'author_{author}' for author in top_authors])
        
        # Language code - just use one-hot encoding
        if 'language_code' in books_df.columns:
            # Fill missing
            books_df['language_code'] = books_df['language_code'].fillna('unknown')
            
            # Get the top languages
            top_languages = books_df['language_code'].value_counts().head(20).index.tolist()
            
            # Create one-hot encoding matrix
            language_matrix = np.zeros((len(books_df), len(top_languages)))
            
            for i, lang in enumerate(books_df['language_code']):
                if lang in top_languages:
                    j = top_languages.index(lang)
                    language_matrix[i, j] = 1
            
            # Convert to sparse
            language_features = sp.csr_matrix(language_matrix)
            feature_matrices.append(language_features)
            feature_names.extend([f'language_{lang}' for lang in top_languages])
        
        # Numeric features - simple normalization
        numeric_cols = ['average_rating', 'ratings_count']
        for col in numeric_cols:
            if col in books_df.columns:
                # Fill missing
                books_df[col] = books_df[col].fillna(0)
                
                # Simple min-max scaling
                values = books_df[col].values
                min_val = values.min()
                max_val = values.max()
                
                # Avoid division by zero
                if max_val > min_val:
                    normalized = (values - min_val) / (max_val - min_val)
                else:
                    normalized = np.zeros_like(values)
                
                # Convert to sparse matrix
                col_features = sp.csr_matrix(normalized.reshape(-1, 1))
                feature_matrices.append(col_features)
                feature_names.append(col)
        
        # Combine all features
        if feature_matrices:
            # Combine all features into a single matrix
            combined_features = sp.hstack(feature_matrices).tocsr()
            print(f"Created feature matrix with {combined_features.shape[1]} features")
            return book_ids, combined_features, feature_names
        else:
            # Create a simple placeholder feature if no features could be extracted
            placeholder = sp.csr_matrix(np.ones((len(book_ids), 1)))
            return book_ids, placeholder, ['placeholder']
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error reading book data: {e}")
        return None, None, None


def create_user_feature_matrix(user_item_matrix, book_feature_matrix):
    """
    Creates a user-feature matrix by aggregating book features weighted by ratings.
    
    Parameters
    ----------
    user_item_matrix : scipy.sparse.csr_matrix
        Sparse matrix of user-book ratings
    book_feature_matrix : scipy.sparse.csr_matrix
        Sparse matrix of book features
        
    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse matrix of user features
    """
    # Normalize the user-item matrix by row (each user's ratings)
    # Add a small value to avoid division by zero
    row_sums = user_item_matrix.sum(axis=1).A.flatten()
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    
    # Normalize user-item matrix (convert from CSR to CSC and back for efficiency)
    user_weights = sp.diags(1.0 / row_sums).dot(user_item_matrix)
    
    # Compute weighted average of book features for each user
    # This multiplies the user weights by the book features
    user_features = user_weights.dot(book_feature_matrix)
    
    return user_features


def calculate_book_similarity_matrix(book_feature_matrix):
    """
    Calculates a book similarity matrix based on book features using cosine similarity.
    
    Parameters
    ----------
    book_feature_matrix : scipy.sparse.csr_matrix
        Sparse matrix of book features
        
    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse similarity matrix where each element is the cosine similarity between books
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Normalize the feature matrix for cosine similarity
    norms = sp.linalg.norm(book_feature_matrix, axis=1)
    norms[norms == 0] = 1  # Avoid division by zero
    
    # Normalize rows to unit length
    normalized_features = sp.diags(1.0 / norms).dot(book_feature_matrix)
    
    # Calculate cosine similarity - for large matrices, we calculate in chunks
    # to avoid memory issues
    chunk_size = 1000  # Adjust based on your memory constraints
    n_books = book_feature_matrix.shape[0]
    similarity_matrix = sp.lil_matrix((n_books, n_books))
    
    for i in range(0, n_books, chunk_size):
        end = min(i + chunk_size, n_books)
        chunk = normalized_features[i:end]
        
        # Calculate similarity between this chunk and all books
        chunk_sim = chunk.dot(normalized_features.T).toarray()
        similarity_matrix[i:end] = chunk_sim
    
    # Convert to CSR format for efficient storage and operations
    return similarity_matrix.tocsr()


def compute_book_similarity(book_features):
    """
    Compute book similarity matrix.
    
    Parameters
    ----------
    book_features : scipy.sparse.csr_matrix
        Sparse matrix of book features
        
    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse similarity matrix where each element is the cosine similarity between books
    """
    return cosine_similarity(book_features, dense_output=False)


def calculate_book_similarity(book_feature_matrix: sp.csr_matrix) -> sp.csr_matrix:
    """
    Calculate similarity between books based on feature matrix.
    
    Parameters
    ----------
    book_feature_matrix : sp.csr_matrix
        Matrix of book features
        
    Returns
    -------
    sp.csr_matrix
        Book similarity matrix
    """
    logger.info("Calculating book similarity matrix")
    
    try:
        # Normalize the feature matrix
        norms = np.sqrt(book_feature_matrix.multiply(book_feature_matrix).sum(axis=1))
        row_norms = np.asarray(norms).flatten()
        
        # Avoid division by zero
        row_norms[row_norms == 0] = 1.0
        
        # Normalize
        book_feature_matrix_normalized = book_feature_matrix.multiply(
            1.0 / row_norms.reshape(-1, 1))
        
        # Calculate cosine similarity
        similarity = book_feature_matrix_normalized.dot(book_feature_matrix_normalized.T)
        
        logger.info(f"Created book similarity matrix with shape {similarity.shape}")
        return similarity
    
    except Exception as e:
        logger.error(f"Error calculating book similarity: {e}")
        logger.debug(traceback.format_exc())
        
        # Return empty matrix on error
        n = book_feature_matrix.shape[0] if hasattr(book_feature_matrix, 'shape') else 0
        return sp.csr_matrix((n, n))


def main(data_dir: str = 'data', min_df: int = 5, max_df: float = 0.8) -> int:
    """
    Main function to build features from processed data.
    
    Parameters
    ----------
    data_dir : str
        Base data directory
    min_df : int
        Minimum document frequency for TF-IDF
    max_df : float
        Maximum document frequency for TF-IDF
        
    Returns
    -------
    int
        Exit code
    """
    logger.info(f"Starting build_features.py with min_df={min_df}, max_df={max_df}")
    
    try:
        processed_dir = os.path.join(data_dir, 'processed')
        features_dir = os.path.join(data_dir, 'features')
        
        # Create output directory
        os.makedirs(features_dir, exist_ok=True)
        
        # Step 1: Read ratings and create user-item matrix
        logger.info("Step 1: Reading ratings and creating user-item matrix")
        user_item_matrix, user_ids, book_ids = read_ratings(processed_dir)
        
        if user_item_matrix.shape[0] == 0 or user_item_matrix.shape[1] == 0:
            logger.error("Failed to create user-item matrix")
            return 1
        
        # Save user-item matrix
        sp.save_npz(os.path.join(features_dir, 'user_item_matrix.npz'), 
                    user_item_matrix)
        
        # Step 2: Extract book features
        logger.info("Step 2: Extracting book features")
        book_feature_matrix, feature_names, book_ids_from_features = extract_book_features(
            processed_dir, min_df, max_df)
        
        if book_feature_matrix.shape[0] == 0:
            logger.error("Failed to extract book features")
            return 1
        
        # Save book feature matrix
        sp.save_npz(os.path.join(features_dir, 'book_feature_matrix.npz'),
                   book_feature_matrix)
        
        # Save feature names
        with open(os.path.join(features_dir, 'feature_names.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(feature_names))
        
        # Save book IDs if they match between ratings and features
        if len(book_ids) == book_feature_matrix.shape[0]:
            np.save(os.path.join(features_dir, 'book_ids.npy'), book_ids)
        else:
            logger.warning(f"Book IDs mismatch: {len(book_ids)} from ratings vs "
                          f"{book_feature_matrix.shape[0]} from features")
            np.save(os.path.join(features_dir, 'book_ids.npy'), book_ids_from_features)
        
        # Step 3: Calculate book similarity
        logger.info("Step 3: Calculating book similarity matrix")
        book_similarity_matrix = calculate_book_similarity(book_feature_matrix)
        
        # Save book similarity matrix
        sp.save_npz(os.path.join(features_dir, 'book_similarity_matrix.npz'),
                   book_similarity_matrix)
        
        logger.info(f"Successfully built and saved features to {features_dir}")
        
        # Create summary report
        summary = {
            'user_item_matrix_shape': user_item_matrix.shape,
            'book_feature_matrix_shape': book_feature_matrix.shape,
            'book_similarity_matrix_shape': book_similarity_matrix.shape,
            'num_users': len(user_ids),
            'num_books': len(book_ids),
            'num_features': len(feature_names),
            'min_df': min_df,
            'max_df': max_df,
            'timestamp': timestamp
        }
        
        summary_df = pd.DataFrame([summary])
        summary_file = os.path.join(features_dir, f'features_summary_{timestamp}.csv')
        summary_df.to_csv(summary_file, index=False, encoding='utf-8')
        logger.info(f"Saved features summary to {summary_file}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}")
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Build features for book recommender')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Base data directory')
    parser.add_argument('--min-df', type=int, default=5,
                      help='Minimum document frequency for TF-IDF')
    parser.add_argument('--max-df', type=float, default=0.8,
                      help='Maximum document frequency for TF-IDF')
    
    args = parser.parse_args()
    
    sys.exit(main(
        data_dir=args.data_dir,
        min_df=args.min_df,
        max_df=args.max_df
    ))
