# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import requests
import logging
import time
import random
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import click
import math

# Set up logging
log_dir = os.path.join('logs')
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(log_dir, f'retrieve_raw_data_{timestamp}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('retrieve_raw_data')

# Hardcover API URL
HARDCOVER_API_URL = "https://api.hardcover.app/v1/graphql"

# Hardcoded API key (for convenience)
DEFAULT_API_KEY = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ2ZXJzaW9uIjoiNyIsImlkIjozMDg4OCwiYXBwbGljYXRpb25JZCI6MiwibG9nZ2VkSW4iOnRydWUsInN1YiI6IjMwODg4IiwiaWF0IjoxNzQxNzI1OTAxLCJleHAiOjE3NDQxNDU0MDEsImh0dHBzOi8vaGFzdXJhLmlvL2p3dC9jbGFpbXMiOnsieC1oYXN1cmEtYWxsb3dlZC1yb2xlcyI6WyJ1c2VyIl0sIngtaGFzdXJhLWRlZmF1bHQtcm9sZSI6InVzZXIiLCJ4LWhhc3VyYS1yb2xlIjoidXNlciIsIlgtaGFzdXJhLXVzZXItaWQiOiIzMDg4OCJ9fQ.fBA1LweEx6yWDV-vlw0LSo9o3nHmljvOLSgN39fhgWM"

def execute_query(query: str, api_key: str, variables: Optional[Dict] = None) -> Dict:
    """Execute a GraphQL query against the Hardcover API"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": api_key
    }
    
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    
    try:
        logger.info(f"Sending request to Hardcover API")
        response = requests.post(HARDCOVER_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"API request failed: {e}")
        return {"errors": [{"message": str(e)}]}

def get_books(api_key: str, limit: int = 500) -> List[Dict]:
    """
    Retrieve books from Hardcover API and construct a list of book dicts.
    Only books with valid images are included in the result.
    
    Args:
        api_key: The API key for Hardcover API
        limit: The total number of books to retrieve
        
    Returns:
        A list of book dictionaries with valid images
    """
    batch_size = 50  # API limitation - retrieve in batches of 50 max
    num_batches = math.ceil(limit / batch_size)
    
    books_with_images = []
    
    for batch in range(1, num_batches + 1):
        logger.info(f"Fetching batch {batch}/{num_batches}")
        
        # Use only the fields we know are supported
        query = f"""
        query {{
          books(limit: {batch_size}) {{
            id
            title
            cached_image
            default_cover_edition_id
            slug
            authors {{
              name
            }}
            description
            published
            genres {{
              name
            }}
            avg_rating
            ratings_count
            isbn
            isbn13
            language_code
            num_pages
            publisher
          }}
        }}
        """
        
        logger.info("Sending request to Hardcover API")
        
        # Use the execute_query function that was working previously
        result = execute_query(query, api_key)
        
        if "errors" in result:
            # If there's an error with the complex query, try a simpler one with fewer fields
            logger.warning(f"Error with extended fields query: {result.get('errors', 'Unknown error')}")
            logger.info("Falling back to basic fields query")
            
            # Fallback to minimal query that we know works
            query = f"""
            query {{
              books(limit: {batch_size}) {{
                id
                title
                cached_image
                default_cover_edition_id
                slug
              }}
            }}
            """
            
            result = execute_query(query, api_key)
        
        if "errors" in result:
            logger.error(f"Error fetching books: {result.get('errors', 'Unknown error')}")
            # Continue with next batch instead of returning empty
            time.sleep(1)  # Add delay to avoid rate limiting
            continue
        
        try:
            batch_books = result["data"]["books"]
            
            # Only keep books that have valid image URLs from the API
            valid_books = []
            for book in batch_books:
                has_valid_image = (
                    book.get('cached_image') is not None and 
                    (
                        (isinstance(book['cached_image'], dict) and 'url' in book['cached_image']) or
                        (isinstance(book['cached_image'], str) and book['cached_image'].startswith('http'))
                    )
                )
                if has_valid_image:
                    valid_books.append(book)
                else:
                    logger.info(f"Skipping book '{book.get('title', 'Unknown')}' (ID: {book.get('id', 'Unknown')}) due to missing image")
            
            logger.info(f"Retrieved {len(batch_books)} books in this batch, {len(valid_books)} have valid images")
            books_with_images.extend(valid_books)
            
            # If we've reached our limit, stop fetching
            if len(books_with_images) >= limit:
                books_with_images = books_with_images[:limit]
                break
            
            # Add delay to avoid rate limiting
            if batch < num_batches:
                time.sleep(1)
                
        except (KeyError, TypeError) as e:
            logger.error(f"Error parsing book results in batch {batch}: {e}")
    
    logger.info(f"Total books retrieved with valid images: {len(books_with_images)}")
    return books_with_images

def supplement_book_data(books: List[Dict]) -> List[Dict]:
    """Add missing fields to match the structure of the original CSV"""
    logger.info("Supplementing book data with missing fields")
    
    supplemented_books = []
    
    # Keep track of data sources for reporting
    data_sources = {
        "real": set(),         # Fields directly from API
        "synthetic": set(),    # Fields generated synthetically
        "mixed": set()         # Fields where some books use real data, others synthetic
    }
    
    # Common genres for randomly assigning to books when not available from API
    genres = ["Fantasy", "Science Fiction", "Mystery", "Romance", "Thriller", 
              "Historical Fiction", "Young Adult", "Literary Fiction", "Horror", 
              "Non-fiction", "Biography", "Self-help"]
    
    for i, book in enumerate(books):
        book_id = i + 1
        
        # Extract real data from the API response when available
        # For each field, use API data if available, otherwise generate synthetic data
        
        # AUTHOR HANDLING
        # Try to get real authors from the API
        if 'authors' in book and book['authors'] and isinstance(book['authors'], list):
            authors = [author.get('name', '') for author in book['authors'] if author.get('name')]
            author_string = ", ".join(authors) if authors else None
            data_sources["real"].add("authors")
        else:
            author_string = None
            data_sources["synthetic"].add("authors")
            
        # Generate synthetic authors only if real ones aren't available
        if not author_string:
            num_authors = random.randint(1, 3)
            authors = []
            for _ in range(num_authors):
                first = random.choice(["John", "Sarah", "Michael", "Emily", "David", "Jennifer", "Robert", "Michelle", "James", "Linda"])
                last = random.choice(["Smith", "Johnson", "Brown", "Davis", "Miller", "Wilson", "Moore", "Taylor", "Anderson", "Thomas"])
                authors.append(f"{first} {last}")
            author_string = ", ".join(authors)
        
        # YEAR/PUBLICATION DATE HANDLING
        # Try to get real publication year from the API
        year = None
        published_date = None
        if 'published' in book and book['published']:
            try:
                # Published might be in various formats, try to extract a year
                published_str = str(book['published'])
                # Try to find a 4-digit year in the string
                import re
                year_match = re.search(r'\b(19|20)\d{2}\b', published_str)
                if year_match:
                    year = int(year_match.group(0))
                    published_date = published_str
                    data_sources["real"].add("publication_year")
            except (ValueError, TypeError):
                pass
                
        # Generate synthetic publication info if not available from API
        if not year:
            year = random.randint(1900, 2024)
            data_sources["synthetic"].add("publication_year")
            
        if not published_date:
            month = random.randint(1, 12)
            day = random.randint(1, 28)  # Using 28 to be safe for all months
            published_date = f"{month}/{day}/{year}"
            data_sources["synthetic"].add("published_date")
        
        # RATINGS HANDLING
        # Try to get real rating data from the API
        avg_rating = None
        ratings_count = None
        
        if 'avg_rating' in book and book['avg_rating'] is not None:
            try:
                avg_rating = float(book['avg_rating'])
                data_sources["real"].add("avg_rating")
            except (ValueError, TypeError):
                pass
                
        if 'ratings_count' in book and book['ratings_count'] is not None:
            try:
                ratings_count = int(book['ratings_count'])
                data_sources["real"].add("ratings_count")
            except (ValueError, TypeError):
                pass
                
        # Generate synthetic ratings if not available from API
        if avg_rating is None:
            avg_rating = round(random.uniform(1.0, 5.0), 2)
            data_sources["synthetic"].add("avg_rating")
            
        if ratings_count is None:
            ratings_count = random.randint(10, 10000)
            data_sources["synthetic"].add("ratings_count")
        
        # Calculate individual rating counts (always synthetic since the API doesn't provide this breakdown)
        if avg_rating >= 4.5:
            dist = [0.01, 0.04, 0.10, 0.25, 0.60]
        elif avg_rating >= 4.0:
            dist = [0.02, 0.08, 0.15, 0.35, 0.40]
        elif avg_rating >= 3.5:
            dist = [0.05, 0.10, 0.25, 0.40, 0.20]
        elif avg_rating >= 3.0:
            dist = [0.10, 0.15, 0.40, 0.25, 0.10]
        else:
            dist = [0.30, 0.25, 0.20, 0.15, 0.10]
        
        rating_counts = [int(ratings_count * p) for p in dist]
        data_sources["synthetic"].add("rating_distribution")
        
        # Ensure the total matches the ratings count
        diff = ratings_count - sum(rating_counts)
        if diff > 0:
            rating_counts[-1] += diff
        
        # GENRES HANDLING
        # Try to get real genres from the API
        if 'genres' in book and book['genres'] and isinstance(book['genres'], list):
            book_genres = [genre.get('name', '') for genre in book['genres'] if genre.get('name')]
            genre_string = ", ".join(book_genres) if book_genres else None
            data_sources["real"].add("genres")
        else:
            genre_string = None
            data_sources["synthetic"].add("genres")
            
        # Generate synthetic genres if not available from API
        if not genre_string:
            num_genres = random.randint(1, 3)
            book_genres = random.sample(genres, num_genres)
            genre_string = ", ".join(book_genres)
        
        # ISBN HANDLING
        # Try to get real ISBN from the API
        isbn = book.get('isbn')
        isbn13 = book.get('isbn13')
        
        if isbn:
            data_sources["real"].add("isbn")
        else:
            data_sources["synthetic"].add("isbn")
            
        if isbn13:
            data_sources["real"].add("isbn13")
        else:
            data_sources["synthetic"].add("isbn13")
        
        # Generate synthetic ISBNs if not available from API
        if not isbn:
            isbn = ''.join([str(random.randint(0, 9)) for _ in range(10)])
        if not isbn13:
            isbn13 = ''.join([str(random.randint(0, 9)) for _ in range(13)])
        
        # LANGUAGE HANDLING
        # Try to get real language code from the API
        language_code = book.get('language_code')
        
        if language_code:
            data_sources["real"].add("language_code")
        else:
            language_code = 'en'  # Default to English
            data_sources["synthetic"].add("language_code")
        
        # PAGE COUNT HANDLING
        # Try to get real page count from the API
        num_pages = None
        if 'num_pages' in book and book['num_pages'] is not None:
            try:
                num_pages = int(book['num_pages'])
                data_sources["real"].add("num_pages")
            except (ValueError, TypeError):
                pass
                
        # Generate synthetic page count if not available from API
        if not num_pages:
            num_pages = random.randint(100, 800)
            data_sources["synthetic"].add("num_pages")
        
        # PUBLISHER HANDLING
        # Try to get real publisher from the API
        publisher = book.get('publisher')
        
        if publisher:
            data_sources["real"].add("publisher")
        else:
            data_sources["synthetic"].add("publisher")
        
        # Generate synthetic publisher if not available from API
        if not publisher:
            publisher = random.choice(["Penguin Random House", "HarperCollins", "Simon & Schuster", 
                                       "Hachette Book Group", "Macmillan Publishers", "Scholastic",
                                       "Wiley", "Oxford University Press", "Cambridge University Press"])
        
        # DESCRIPTION HANDLING
        # Try to get real description from the API
        description = book.get('description')
        
        if description:
            data_sources["real"].add("description")
        else:
            data_sources["synthetic"].add("description")
        
        # Generate synthetic description if not available from API
        if not description:
            description = f"This is a fictional description for the book '{book['title']}' by {author_string}."
        
        # Get image URLs - use cached_image from API if available
        image_url = None
        small_image_url = None
        
        # The cached_image field has the full-sized image
        if 'cached_image' in book and book['cached_image']:
            # Handle cached_image field which may be a JSON string or dictionary
            try:
                # If it's already a dictionary (parsed JSON)
                if isinstance(book['cached_image'], dict) and 'url' in book['cached_image']:
                    image_url = book['cached_image']['url']
                    # For small image URL, use the same URL but add width parameter if not already there
                    if '?' in image_url:
                        small_image_url = f"{image_url}&width=200"
                    else:
                        small_image_url = f"{image_url}?width=200"
                # If it's a string, try to use it directly
                elif isinstance(book['cached_image'], str) and book['cached_image'].startswith('http'):
                    image_url = book['cached_image']
                    # For small image URL, add width parameter
                    if '?' in image_url:
                        small_image_url = f"{image_url}&width=200"
                    else:
                        small_image_url = f"{image_url}?width=200"
            except (TypeError, ValueError) as e:
                logger.warning(f"Could not parse cached_image for book {book['id']}: {e}")
        
        # Fallback to placeholders if no images available from API
        if not image_url:
            # Use placeholder.com which provides reliable generic image placeholders
            image_url = f"https://via.placeholder.com/800x1200/E8E8E8/333333?text=Book+{book_id}"
        if not small_image_url:
            small_image_url = f"https://via.placeholder.com/400x600/E8E8E8/333333?text=Book+{book_id}"
        
        # Build the supplemented book record
        supplemented_book = {
            "book_id": book_id,
            "goodreads_book_id": book.get("id", str(book_id)),  # Use API ID if available
            "best_book_id": book.get("id", str(book_id)),  # Use API ID if available
            "work_id": book.get("id", str(book_id)),  # Use API ID if available
            "books_count": random.randint(1, 10),  # Always synthetic
            "isbn": isbn,
            "isbn13": isbn13,
            "authors": author_string,
            "original_publication_year": year,
            "original_title": book.get("title", "Unknown Title"),  # Use API title if available
            "title": book.get("title", "Unknown Title"),  # Use API title if available
            "language_code": language_code,
            "average_rating": avg_rating,
            "ratings_count": ratings_count,
            "work_ratings_count": ratings_count,  # Same as ratings_count
            "work_text_reviews_count": int(ratings_count * 0.3),  # Synthetic (30% of ratings leave reviews)
            "ratings_1": rating_counts[0],
            "ratings_2": rating_counts[1],
            "ratings_3": rating_counts[2],
            "ratings_4": rating_counts[3],
            "ratings_5": rating_counts[4],
            "image_url": image_url,
            "small_image_url": small_image_url,
            "num_pages": num_pages,
            "publisher": publisher,
            "publication_day": random.randint(1, 28),  # Always synthetic
            "publication_month": random.randint(1, 12),  # Always synthetic
            "publication_year": year,
            "average_rating_rounded": round(avg_rating),
            "book_description": description,
            "genres": genre_string,
            "published_date": published_date
        }
        
        supplemented_books.append(supplemented_book)
    
    # Log which fields are real vs. synthetic
    logger.info("=== DATA SOURCE SUMMARY ===")
    logger.info(f"Real data from API: {', '.join(sorted(data_sources['real']))}")
    logger.info(f"Synthetic data (generated): {', '.join(sorted(data_sources['synthetic']))}")
    if data_sources['mixed']:
        logger.info(f"Mixed (some books real, some synthetic): {', '.join(sorted(data_sources['mixed']))}")
    
    return supplemented_books

def generate_ratings(books_df: pd.DataFrame, num_users: int = 1000, sparsity_factor: float = 0.20) -> pd.DataFrame:
    """Generate ratings data to match the structure of the original ratings.csv with realistic sparsity
    
    Args:
        books_df: DataFrame containing book data
        num_users: Number of user IDs to generate
        sparsity_factor: Fraction of books each user rates on average (0.20 = 20%)
    
    Returns:
        DataFrame with book_id, user_id, and rating columns
    """
    logger.info(f"Generating ratings data for {num_users} users with sparsity factor {sparsity_factor}")
    
    ratings_data = []
    book_ids = books_df["book_id"].tolist()
    total_books = len(book_ids)
    
    # For each user, determine how many and which books they will rate
    for user_id in range(1, num_users + 1):
        # Users typically follow a power law distribution - some rate many books, most rate few
        # Generate a random number that follows roughly a power law distribution
        if random.random() < 0.05:  # 5% of users are "power users"
            # Power users rate between 5% and 20% of books
            user_rating_probability = random.uniform(0.05, 0.2)
        else:  # 95% of users rate very few books
            # Regular users rate between 0.1% and 5% of books
            user_rating_probability = random.uniform(0.001, 0.05)
            
        # Calculate how many books this user will rate
        num_to_rate = max(1, int(total_books * user_rating_probability))
        num_to_rate = min(num_to_rate, total_books)  # Can't rate more books than exist
        
        # Select random books for this user to rate
        if num_to_rate > 0:
            user_books = random.sample(book_ids, num_to_rate)
            
            # For each book, generate a rating with a bias toward higher ratings
            # (most users rate things they like)
            for book_id in user_books:
                # Biased random rating - more likely to be 4 or 5 than 1 or 2
                rating_weights = [0.05, 0.1, 0.2, 0.35, 0.3]  # Probabilities of ratings 1-5
                rating = random.choices([1, 2, 3, 4, 5], weights=rating_weights)[0]
                
                # Add to ratings data
                ratings_data.append({
                    "book_id": book_id,
                    "user_id": user_id,
                    "rating": rating
                })
    
    # Create DataFrame
    ratings_df = pd.DataFrame(ratings_data)
    logger.info(f"Generated {len(ratings_df)} ratings from {num_users} users")
    
    return ratings_df

def fetch_and_process_data(api_key: str, limit: int = 50, append: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch books from Hardcover API and process into the required format"""
    # Output directory and path
    output_dir = os.path.join('data', 'raw')
    books_path = os.path.join(output_dir, 'books.csv')
    ratings_path = os.path.join(output_dir, 'ratings.csv')
    
    # Check if we should load existing data first (for append mode)
    existing_books_df = pd.DataFrame()
    existing_ratings_df = pd.DataFrame()
    max_book_id = 0
    max_user_id = 0
    
    if append and os.path.exists(books_path):
        try:
            existing_books_df = pd.read_csv(books_path)
            logger.info(f"Loaded {len(existing_books_df)} existing books")
            
            if len(existing_books_df) > 0:
                max_book_id = existing_books_df['book_id'].max()
                logger.info(f"Max existing book_id: {max_book_id}")
        except Exception as e:
            logger.error(f"Error loading existing books data: {e}")
    
    if append and os.path.exists(ratings_path):
        try:
            existing_ratings_df = pd.read_csv(ratings_path)
            logger.info(f"Loaded {len(existing_ratings_df)} existing ratings")
            
            if len(existing_ratings_df) > 0:
                max_user_id = existing_ratings_df['user_id'].max()
                logger.info(f"Max existing user_id: {max_user_id}")
        except Exception as e:
            logger.error(f"Error loading existing ratings data: {e}")
    
    # Fetch books from API
    raw_books = get_books(api_key, limit)
    
    if not raw_books:
        logger.error("Failed to retrieve any books from the API")
        return pd.DataFrame(), pd.DataFrame()
    
    # If we couldn't fetch enough books with valid images, we'll adjust the limit
    actual_limit = len(raw_books)
    if actual_limit < limit:
        logger.warning(f"Could only retrieve {actual_limit} books with valid images instead of the requested {limit}")
        # Adjust limit to the actual number of books we have
        limit = actual_limit
    
    # Supplement with additional fields to match original format
    supplemented_books = supplement_book_data(raw_books)
    
    # Create books DataFrame
    books_df = pd.DataFrame(supplemented_books)
    
    # Adjust IDs if appending to existing data
    if append and len(existing_books_df) > 0:
        logger.info(f"Adjusting IDs for {len(books_df)} new books to avoid conflicts")
        books_df['book_id'] = books_df['book_id'].apply(lambda x: x + max_book_id)
        books_df['id'] = books_df['id'].apply(lambda x: x + max_book_id)
        books_df['best_book_id'] = books_df['best_book_id'].apply(lambda x: x + max_book_id)
        books_df['work_id'] = books_df['work_id'].apply(lambda x: x + max_book_id)
    
    # Generate ratings for the new books
    num_users = 2000  # Generate 2000 users
    ratings_df = generate_ratings(books_df, num_users=num_users, sparsity_factor=0.03)
    
    # Adjust user IDs if appending to existing data
    if append and max_user_id > 0:
        logger.info(f"Adjusting user IDs for {len(ratings_df)} new ratings to avoid conflicts")
        ratings_df['user_id'] = ratings_df['user_id'].apply(lambda x: x + max_user_id)
    
    # Combine with existing data if appending
    if append:
        if not existing_books_df.empty:
            logger.info(f"Concatenating {len(books_df)} new books with {len(existing_books_df)} existing books")
            books_df = pd.concat([existing_books_df, books_df], ignore_index=True)
        
        if not existing_ratings_df.empty:
            logger.info(f"Concatenating {len(ratings_df)} new ratings with {len(existing_ratings_df)} existing ratings")
            ratings_df = pd.concat([existing_ratings_df, ratings_df], ignore_index=True)
    
    # Remove duplicates
    logger.info(f"Removing duplicates from {len(books_df)} books")
    original_book_count = len(books_df)
    books_df = books_df.drop_duplicates(subset=['title', 'authors'], keep='first')
    logger.info(f"Removed {original_book_count - len(books_df)} duplicate books")
    
    # Make sure all books are in English
    books_df['language_code'] = 'eng'
    
    # Save books to CSV
    os.makedirs(output_dir, exist_ok=True)
    books_df.to_csv(books_path, index=False)
    logger.info(f"Saved {len(books_df)} books to {books_path}")
    
    # Remove any ratings for books that no longer exist (after deduplication)
    valid_book_ids = set(books_df['book_id'].unique())
    original_ratings_count = len(ratings_df)
    ratings_df = ratings_df[ratings_df['book_id'].isin(valid_book_ids)]
    logger.info(f"Removed {original_ratings_count - len(ratings_df)} ratings for non-existent books")
    
    # Remove duplicate ratings (same user rating the same book multiple times)
    ratings_df = ratings_df.drop_duplicates(subset=['user_id', 'book_id'], keep='last')
    logger.info(f"After removing duplicate ratings: {len(ratings_df)} ratings")
    
    # Save ratings to CSV
    ratings_df.to_csv(ratings_path, index=False)
    logger.info(f"Saved {len(ratings_df)} ratings to {ratings_path}")
    
    return books_df, ratings_df

@click.command()
@click.argument('output_filepath', type=click.Path(), default='data/raw')
@click.option('--limit', default=500, help='Number of books to fetch')
@click.option('--append/--no-append', default=True, help='Append to existing data')
@click.option('--api-key', help='Hardcover API key (Bearer token)')
def main(output_filepath, limit, append, api_key=None):
    """Fetch book data from Hardcover API and save to CSV"""
    # Get API key
    if not api_key:
        api_key = os.environ.get("HARDCOVER_API_KEY")
        if not api_key:
            # Use the hardcoded API key if no other key is provided
            logger.info("Using default hardcoded API key")
            api_key = DEFAULT_API_KEY
    
    logger.info(f"Starting data retrieval, output_filepath: {output_filepath}")
    
    try:
        books_df, ratings_df = fetch_and_process_data(
            api_key=api_key,
            limit=limit,
            append=append
        )
        
        if len(books_df) == 0:
            logger.error("Failed to retrieve and process book data")
            return 1
            
        logger.info("Data retrieval and processing complete")
    except Exception as e:
        logger.error(f"Error in data retrieval: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    # Find .env file, if it exists
    load_dotenv(find_dotenv())
    
    # Call the main function with default values without requiring args
    # Default: 500 books, append=False, output='data/raw'
    main.callback(output_filepath='data/raw', limit=500, append=False, api_key=None)
