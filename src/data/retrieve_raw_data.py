"""
Script to retrieve book data from Google Books API and create a dataset for the recommender system
"""
import os
import json
import random
import pandas as pd
import requests
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import itertools
import numpy as np
import click

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("retrieve_from_google")

# Google Books API constants
GOOGLE_API_KEY = "AIzaSyAfBBcsfRQWfwSn9csJJjqoCxqPRZmSDOE"
BASE_URL = "https://www.googleapis.com/books/v1/volumes"
BATCH_SIZE = 40  # Google Books API allows up to 40 results per page
RATE_LIMIT_DELAY = 1  # Seconds to wait between API calls to avoid rate limiting


def get_books(limit: int = 500) -> List[Dict]:
    """
    Retrieve book data from Google Books API
    
    Args:
        limit: Maximum number of books to retrieve
        
    Returns:
        List of book dictionaries with data from the API
    """
    logger.info(f"Retrieving up to {limit} books from Google Books API")
    books = []
    
    # Statistics for logging
    total_books_retrieved = 0
    total_books_with_images = 0
    total_books_without_images = 0
    
    # Use a mix of popular search terms to get a diverse set of books
    search_terms = [
        "subject:fiction",
        "subject:fantasy",
        "subject:mystery",
        "subject:romance",
        "subject:science fiction",
        "subject:thriller",
        "subject:historical fiction",
        "subject:horror",
        "subject:adventure",
        "subject:classics",
        "subject:young adult",
        "subject:crime",
        "subject:biography",
        "subject:history",
        "subject:self-help",
        "subject:science",
        "subject:psychology",
        "subject:business",
        "subject:comics",
        "subject:poetry",
        "subject:art",
        "subject:philosophy",
        "subject:travel",
        "subject:religion",
        "subject:memoir",
        "subject:children",
        "inauthor:stephen king",
        "inauthor:j.k. rowling",
        "inauthor:agatha christie",
        "inauthor:james patterson",
        "inauthor:dan brown",
        "inauthor:john grisham",
        "inauthor:george r.r. martin",
        "inauthor:paulo coelho",
        "inauthor:haruki murakami",
        "inauthor:jane austen",
        "inauthor:mark twain"
    ]
    
    # Multiple language filters to increase diversity
    language_filters = ["&langRestrict=en", "&langRestrict=es", "&langRestrict=fr", "&langRestrict=de", "&langRestrict=it", "&langRestrict=zh"]
    
    # Use different search strategies
    batch_num = 0
    max_attempts = limit * 3  # Safety limit on attempts
    attempts = 0
    
    # First try all search terms without additional filters
    for search_term in search_terms:
        if len(books) >= limit:
            break
            
        if attempts >= max_attempts:
            logger.warning(f"Reached maximum attempt limit of {max_attempts}")
            break
            
        batch_num += 1
        attempts += 1
        
        logger.info(f"Attempt {attempts}: Fetching books for '{search_term}'")
        
        params = {
            "q": search_term,
            "maxResults": BATCH_SIZE,
            "key": GOOGLE_API_KEY,
            "printType": "books",
            "orderBy": "relevance"
        }
        
        # Try getting books with this search term
        batch_books = fetch_batch(params, batch_num, total_books_retrieved, total_books_with_images, total_books_without_images)
        
        # Update statistics
        total_books_retrieved += batch_books["total_retrieved"]
        total_books_with_images += batch_books["with_images"]
        total_books_without_images += batch_books["without_images"]
        
        # Add books
        books.extend(batch_books["books"])
    
    # If we still need more books, try with language filters
    if len(books) < limit:
        for search_term in search_terms:
            for lang_filter in language_filters:
                if len(books) >= limit:
                    break
                    
                if attempts >= max_attempts:
                    logger.warning(f"Reached maximum attempt limit of {max_attempts}")
                    break
                    
                batch_num += 1
                attempts += 1
                
                lang_code = lang_filter.split('=')[1]
                logger.info(f"Attempt {attempts}: Fetching books for '{search_term}' in language {lang_code}")
                
                # Construct full query with language filter
                full_query = f"{search_term}{lang_filter}"
                
                params = {
                    "q": search_term,
                    "maxResults": BATCH_SIZE,
                    "key": GOOGLE_API_KEY,
                    "langRestrict": lang_code,
                    "printType": "books",
                    "orderBy": "relevance"
                }
                
                # Try getting books with this search term and language
                batch_books = fetch_batch(params, batch_num, total_books_retrieved, total_books_with_images, total_books_without_images)
                
                # Update statistics
                total_books_retrieved += batch_books["total_retrieved"]
                total_books_with_images += batch_books["with_images"]
                total_books_without_images += batch_books["without_images"]
                
                # Add books
                books.extend(batch_books["books"])
    
    logger.info(f"Total books retrieved: {total_books_retrieved}")
    logger.info(f"Total books with images: {total_books_with_images}")
    logger.info(f"Total books without images: {total_books_without_images}")
    
    if total_books_retrieved > 0:
        image_percentage = total_books_with_images / total_books_retrieved * 100
        logger.info(f"Percentage of books with images: {image_percentage:.2f}%")
    
    logger.info(f"Final book count with valid images: {len(books)}")
    
    # Limit to requested number
    if len(books) > limit:
        books = books[:limit]
        logger.info(f"Trimmed to {limit} books as requested")
    elif len(books) < limit:
        logger.warning(f"Could only retrieve {len(books)} books with valid images instead of the requested {limit}")
        logger.info("To get more books, try increasing the limit or adding more search terms")
    
    return books


def fetch_batch(params, batch_num, total_retrieved, total_with_images, total_without_images):
    """
    Fetch a batch of books from the Google Books API
    
    Args:
        params: API request parameters
        batch_num: Current batch number
        total_retrieved: Running total of retrieved books
        total_with_images: Running total of books with images
        total_without_images: Running total of books without images
        
    Returns:
        Dictionary with batch statistics and books
    """
    result = {
        "books": [],
        "total_retrieved": 0,
        "with_images": 0,
        "without_images": 0
    }
    
    response = requests.get(BASE_URL, params=params)
    
    if response.status_code == 200:
        data = response.json()
        
        if "items" in data:
            batch_books = data["items"]
            result["total_retrieved"] = len(batch_books)
            logger.info(f"Retrieved {len(batch_books)} books in batch {batch_num}")
            
            # Process each book to extract relevant data
            batch_books_with_images = 0
            batch_books_without_images = 0
            
            for book in batch_books:
                # Skip books without volume info
                volume_info = book.get("volumeInfo", {})
                if not volume_info:
                    continue
                
                # Check if book has images
                if "imageLinks" not in volume_info:
                    title = volume_info.get('title', 'Unknown')
                    book_id = book.get('id', 'Unknown')
                    logger.info(f"Skipping book '{title}' (ID: {book_id}) due to missing image")
                    batch_books_without_images += 1
                    continue
                
                # Add the book to our collection
                result["books"].append(book)
                batch_books_with_images += 1
            
            result["with_images"] = batch_books_with_images
            result["without_images"] = batch_books_without_images
            
            logger.info(f"Batch {batch_num} statistics: {batch_books_with_images} books with images, {batch_books_without_images} books without images")
        else:
            logger.warning(f"No items found in batch {batch_num}")
    else:
        logger.error(f"Error fetching batch {batch_num}: {response.status_code}")
        logger.error(response.text)
        # Add delay to avoid rate limiting
        time.sleep(RATE_LIMIT_DELAY * 2)
    
    # Add delay to avoid rate limiting
    time.sleep(RATE_LIMIT_DELAY)
    
    return result


def extract_book_data(books: List[Dict]) -> List[Dict]:
    """
    Extract relevant fields from the Google Books API response
    
    Args:
        books: List of book dictionaries from the API
        
    Returns:
        List of dictionaries with standardized book data
    """
    logger.info("Extracting standardized book data")
    
    standardized_books = []
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
        volume_info = book.get("volumeInfo", {})
        
        # Basic book information
        title = volume_info.get("title", f"Unknown Book {book_id}")
        
        # Extract image URL
        image_links = volume_info.get("imageLinks", {})
        image_url = None
        # Try to get the largest available image
        for size in ["extraLarge", "large", "medium", "small", "thumbnail", "smallThumbnail"]:
            if size in image_links:
                image_url = image_links[size]
                break
        
        # If no image URL was found, skip this book
        if not image_url:
            continue
            
        # Standardize image URL to https
        if image_url.startswith("http:"):
            image_url = "https" + image_url[4:]
            
        # AUTHOR HANDLING
        authors = volume_info.get("authors", [])
        
        if authors:
            author_string = ", ".join(authors)
            data_sources["real"].add("authors")
        else:
            # Generate synthetic authors
            num_authors = random.randint(1, 3)
            synthetic_authors = []
            for _ in range(num_authors):
                first = random.choice(["John", "Sarah", "Michael", "Emily", "David", "Jennifer", "Robert", "Michelle", "James", "Linda"])
                last = random.choice(["Smith", "Johnson", "Brown", "Davis", "Miller", "Wilson", "Moore", "Taylor", "Anderson", "Thomas"])
                synthetic_authors.append(f"{first} {last}")
            author_string = ", ".join(synthetic_authors)
            data_sources["synthetic"].add("authors")
        
        # YEAR/PUBLICATION DATE HANDLING
        published_date = volume_info.get("publishedDate")
        
        if published_date:
            # Try to extract just the year from the date
            try:
                import re
                year_match = re.search(r'\b(19|20)\d{2}\b', published_date)
                if year_match:
                    year = int(year_match.group(0))
                else:
                    year = int(published_date[:4])  # Assume format starts with year
                data_sources["real"].add("publication_year")
            except (ValueError, TypeError):
                year = random.randint(1900, 2024)
                data_sources["synthetic"].add("publication_year")
        else:
            year = random.randint(1900, 2024)
            published_date = f"{random.randint(1, 12)}/{random.randint(1, 28)}/{year}"
            data_sources["synthetic"].add("publication_year")
            data_sources["synthetic"].add("published_date")
        
        # RATINGS HANDLING
        avg_rating = volume_info.get("averageRating")
        ratings_count = volume_info.get("ratingsCount")
        
        if avg_rating is not None:
            data_sources["real"].add("avg_rating")
        else:
            avg_rating = round(random.uniform(1.0, 5.0), 2)
            data_sources["synthetic"].add("avg_rating")
            
        if ratings_count is not None:
            data_sources["real"].add("ratings_count")
        else:
            ratings_count = random.randint(10, 10000)
            data_sources["synthetic"].add("ratings_count")
        
        # Calculate individual rating counts
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
        categories = volume_info.get("categories", [])
        
        if categories:
            genre_string = ", ".join(categories)
            data_sources["real"].add("genres")
        else:
            num_genres = random.randint(1, 3)
            book_genres = random.sample(genres, num_genres)
            genre_string = ", ".join(book_genres)
            data_sources["synthetic"].add("genres")
        
        # ISBN HANDLING
        industry_identifiers = volume_info.get("industryIdentifiers", [])
        isbn = ""
        isbn13 = ""
        
        for identifier in industry_identifiers:
            id_type = identifier.get("type", "")
            if id_type == "ISBN_10":
                isbn = identifier.get("identifier", "")
                data_sources["real"].add("isbn")
            elif id_type == "ISBN_13":
                isbn13 = identifier.get("identifier", "")
                data_sources["real"].add("isbn13")
        
        # Generate synthetic ISBNs if not available from API
        if not isbn:
            isbn = ''.join([str(random.randint(0, 9)) for _ in range(10)])
            data_sources["synthetic"].add("isbn")
        if not isbn13:
            isbn13 = ''.join([str(random.randint(0, 9)) for _ in range(13)])
            data_sources["synthetic"].add("isbn13")
        
        # LANGUAGE HANDLING
        language_code = volume_info.get("language")
        
        if language_code:
            data_sources["real"].add("language_code")
        else:
            language_code = 'en'  # Default to English
            data_sources["synthetic"].add("language_code")
        
        # PAGE COUNT HANDLING
        num_pages = volume_info.get("pageCount")
        
        if num_pages:
            data_sources["real"].add("num_pages")
        else:
            num_pages = random.randint(100, 800)
            data_sources["synthetic"].add("num_pages")
        
        # PUBLISHER HANDLING
        publisher = volume_info.get("publisher")
        
        if publisher:
            data_sources["real"].add("publisher")
        else:
            publisher = random.choice(["Penguin Random House", "HarperCollins", "Simon & Schuster", 
                                       "Hachette Book Group", "Macmillan Publishers", "Scholastic",
                                       "Wiley", "Oxford University Press", "Cambridge University Press"])
            data_sources["synthetic"].add("publisher")
        
        # DESCRIPTION HANDLING
        description = volume_info.get("description")
        
        if description:
            # Truncate very long descriptions
            if len(description) > 1000:
                description = description[:997] + "..."
            data_sources["real"].add("description")
        else:
            description = f"This is a fictional description for the book '{title}' by {author_string}."
            data_sources["synthetic"].add("description")
        
        # Create the standardized book dictionary
        standardized_book = {
            "book_id": book_id,
            "title": title,
            "authors": author_string,
            "average_rating": float(avg_rating),
            "isbn": isbn,
            "isbn13": isbn13,
            "language_code": language_code,
            "num_pages": int(num_pages) if num_pages else 0,
            "ratings_count": int(ratings_count) if ratings_count else 0,
            "ratings_1": rating_counts[0],
            "ratings_2": rating_counts[1],
            "ratings_3": rating_counts[2],
            "ratings_4": rating_counts[3],
            "ratings_5": rating_counts[4],
            "published_year": year,
            "original_publication_year": year,  # Using the same year for simplicity
            "original_title": title,
            "image_url": image_url,
            "publisher": publisher,
            "description": description,
            "genres": genre_string,
        }
        
        standardized_books.append(standardized_book)
    
    # Log which fields are real vs. synthetic
    logger.info("=== DATA SOURCE SUMMARY ===")
    logger.info(f"Real data from API: {', '.join(sorted(data_sources['real']))}")
    logger.info(f"Synthetic data (generated): {', '.join(sorted(data_sources['synthetic']))}")
    if data_sources['mixed']:
        logger.info(f"Mixed (some books real, some synthetic): {', '.join(sorted(data_sources['mixed']))}")
    
    return standardized_books


def generate_ratings(books_df: pd.DataFrame, num_users: int = 1000, sparsity_factor: float = 0.20) -> pd.DataFrame:
    """
    Generate synthetic ratings data for books
    
    Args:
        books_df: DataFrame containing book data
        num_users: Number of synthetic users to generate
        sparsity_factor: Controls how many books each user rates (lower = more ratings)
        
    Returns:
        DataFrame containing user ratings
    """
    logger.info(f"Generating ratings data for {num_users} users with sparsity factor {sparsity_factor}")
    
    # Get the list of book IDs
    book_ids = books_df["book_id"].tolist()
    
    # Get the most popular books based on average rating and ratings count
    # These will be used to ensure each user has some ratings for popular books
    if "ratings_count" in books_df.columns and "average_rating" in books_df.columns:
        popular_books = books_df.sort_values(by=["ratings_count", "average_rating"], ascending=False)
        popular_book_ids = popular_books.head(50)["book_id"].tolist()
    else:
        # Fallback if ratings data isn't available
        popular_book_ids = book_ids[:50]
    
    ratings = []
    
    for user_id in range(1, num_users + 1):
        # Ensure each user rates at least 3 books, regardless of sparsity
        min_ratings = 3
        
        # Calculate total ratings for this user, ensuring at least min_ratings
        n_ratings = max(min_ratings, int(len(book_ids) * (1 - sparsity_factor) * random.uniform(0.5, 1.5)))
        
        # Make sure each user rates some popular books
        # This ensures better collaborative filtering quality
        num_popular_to_rate = min(min_ratings, len(popular_book_ids))
        popular_to_rate = random.sample(popular_book_ids, num_popular_to_rate)
        
        # Then select remaining random books to meet the total required ratings
        remaining_books = [bid for bid in book_ids if bid not in popular_to_rate]
        remaining_to_rate = random.sample(remaining_books, 
                                         min(n_ratings - num_popular_to_rate, len(remaining_books)))
        
        # Combine popular and random books
        books_to_rate = popular_to_rate + remaining_to_rate
        
        for book_id in books_to_rate:
            # Get book information to influence the rating
            book = books_df[books_df["book_id"] == book_id].iloc[0]
            avg_rating = book.get("average_rating", 3.5)  # Default to 3.5 if not available
            
            # Generate a rating with some randomness but influenced by the average rating
            # This creates more realistic rating distributions
            rating_bias = (avg_rating - 3) * 0.5  # Bias towards the book's actual rating
            
            # Add randomness with more weight to 4 and 5 star ratings
            weights = [0.05, 0.1, 0.2, 0.3, 0.35]  # Probability of 1-5 stars
            
            # Adjust weights based on the book's average rating
            if avg_rating >= 4.0:
                weights = [0.02, 0.08, 0.15, 0.35, 0.4]
            elif avg_rating <= 3.0:
                weights = [0.15, 0.25, 0.3, 0.2, 0.1]
                
            rating_value = random.choices([1, 2, 3, 4, 5], weights=weights)[0]
            
            ratings.append({
                "user_id": user_id,
                "book_id": book_id,
                "rating": rating_value
            })
    
    # Convert to DataFrame
    ratings_df = pd.DataFrame(ratings)
    
    # Verify that each user has at least min_ratings
    user_rating_counts = ratings_df.groupby('user_id').size()
    min_user_ratings = user_rating_counts.min()
    logger.info(f"Generated {len(ratings_df)} ratings from {num_users} users")
    logger.info(f"Minimum ratings per user: {min_user_ratings}")
    
    return ratings_df


def remove_duplicates(books_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate books based on title and author
    
    Args:
        books_df: DataFrame containing book data
        
    Returns:
        DataFrame with duplicates removed
    """
    logger.info(f"Removing duplicates from {len(books_df)} books")
    
    # Create a new column with lowercase title+author for deduplication
    books_df["dedup_key"] = books_df["title"].str.lower() + "|" + books_df["authors"].str.lower()
    
    # Keep the first occurrence of each title+author combination
    books_df_unique = books_df.drop_duplicates(subset=["dedup_key"])
    
    # Remove the temporary column
    books_df_unique = books_df_unique.drop(columns=["dedup_key"])
    
    logger.info(f"Removed {len(books_df) - len(books_df_unique)} duplicate books")
    
    return books_df_unique


@click.command()
@click.option("--limit", default=500, help="Number of books to retrieve")
@click.option("--num-users", default=2000, help="Number of users to generate")
@click.option("--sparsity", default=0.03, help="Sparsity factor for ratings (lower = more ratings)")
@click.option("--output-filepath", default="data/raw", help="Output directory for the dataset")
def main(limit: int, num_users: int, sparsity: float, output_filepath: str):
    """
    Main function to retrieve data from Google Books API and save to CSV files
    """
    logger.info(f"Starting data retrieval, output_filepath: {output_filepath}")
    
    # Create output directory if it doesn't exist
    Path(output_filepath).mkdir(parents=True, exist_ok=True)
    
    # Retrieve books from Google Books API
    books = get_books(limit=limit)
    
    # Extract and standardize book data
    standardized_books = extract_book_data(books)
    
    # Convert to DataFrame
    books_df = pd.DataFrame(standardized_books)
    
    # Remove duplicate books
    books_df = remove_duplicates(books_df)
    
    # Generate ratings
    ratings_df = generate_ratings(books_df, num_users=num_users, sparsity_factor=sparsity)
    
    # Check for and remove ratings for non-existent books (due to duplicates being removed)
    valid_book_ids = set(books_df["book_id"].tolist())
    original_ratings_count = len(ratings_df)
    ratings_df = ratings_df[ratings_df["book_id"].isin(valid_book_ids)]
    
    if len(ratings_df) < original_ratings_count:
        logger.info(f"Removed {original_ratings_count - len(ratings_df)} ratings for non-existent books")
    
    # Remove duplicate ratings (same user rating the same book multiple times)
    original_ratings_count = len(ratings_df)
    ratings_df = ratings_df.drop_duplicates(subset=["user_id", "book_id"])
    
    if len(ratings_df) < original_ratings_count:
        logger.info(f"After removing duplicate ratings: {len(ratings_df)} ratings")
    
    # Save the data to CSV files
    books_filepath = os.path.join(output_filepath, "books.csv")
    ratings_filepath = os.path.join(output_filepath, "ratings.csv")
    
    books_df.to_csv(books_filepath, index=False)
    logger.info(f"Saved {len(books_df)} books to {books_filepath}")
    
    ratings_df.to_csv(ratings_filepath, index=False)
    logger.info(f"Saved {len(ratings_df)} ratings to {ratings_filepath}")
    
    logger.info("Data retrieval and processing complete")


if __name__ == "__main__":
    main()
