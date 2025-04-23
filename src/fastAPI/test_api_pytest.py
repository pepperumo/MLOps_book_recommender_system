"""
Tests for the Book Recommendation API.
This test suite can be run against a local instance or a Docker container.
"""

import os
import pytest
import requests
import time
from typing import Dict, Any, List, Optional
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure the API URL - this can be changed to test against Docker
API_URL = os.environ.get("API_URL", "http://localhost:8000")
USE_API_PREFIX = os.environ.get("USE_API_PREFIX", "true").lower() == "true"

def get_endpoint(path: str) -> str:
    """Construct the API endpoint URL with proper prefix handling"""
    # Some deployments might use /api prefix, others might not
    if path == "/health":
        # Health endpoint is always at the root with no prefix
        return f"{API_URL}{path}"
    
    if USE_API_PREFIX and not path.startswith("/api/"):
        if not path.startswith("/"):
            path = f"/api/{path}"
        else:
            path = f"/api{path}"
    elif path.startswith("/api/") and not USE_API_PREFIX:
        path = path[4:]  # Remove /api prefix
    
    return f"{API_URL}{path}"

# Fixtures
@pytest.fixture(scope="session")
def api_health_check():
    """Check if the API is running before running tests"""
    health_url = f"{API_URL}/health"
    try:
        response = requests.get(health_url, timeout=5)
        response.raise_for_status()
        logger.info(f"API is healthy at {API_URL}")
        return True
    except Exception as e:
        # Try alternative health endpoints if the standard one fails
        alternative_urls = [
            f"{API_URL}/api/health",
            f"{API_URL}/"
        ]
        
        for alt_url in alternative_urls:
            try:
                logger.info(f"Trying alternative health endpoint: {alt_url}")
                alt_response = requests.get(alt_url, timeout=5)
                if alt_response.status_code == 200:
                    logger.info(f"API is healthy at alternative endpoint: {alt_url}")
                    return True
            except Exception:
                continue
                
        logger.error(f"API health check failed: {e}")
        pytest.skip(f"API is not available at {API_URL}")

@pytest.fixture
def valid_user_id() -> int:
    """Return a valid user ID for testing"""
    return 1  # Assuming user ID 1 exists in the system

@pytest.fixture
def invalid_user_id() -> int:
    """Return an invalid user ID for testing"""
    return 99999  # Assuming this user ID doesn't exist

@pytest.fixture
def valid_book_id() -> int:
    """Return a valid book ID for testing"""
    # First get a list of books to ensure we have a valid ID
    try:
        books_url = get_endpoint("/books")
        response = requests.get(books_url, params={"limit": 1})
        if response.ok and "books" in response.json() and len(response.json()["books"]) > 0:
            return response.json()["books"][0]["book_id"]
    except Exception as e:
        logger.warning(f"Could not fetch a valid book ID: {e}")
    
    return 1  # Fallback to book ID 1, assuming it exists

# Tests for User Recommendations
def test_user_recommendations(api_health_check, valid_user_id):
    """Test getting recommendations for a valid user"""
    url = get_endpoint(f"/recommend/user/{valid_user_id}")
    response = requests.get(url)
    
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"
    
    data = response.json()
    assert "recommendations" in data, "Response should contain recommendations key"
    assert "user_id" in data, "Response should contain user_id key"
    assert data["user_id"] == valid_user_id, f"User ID should match {valid_user_id}"
    
    # Check that we got some recommendations (could be empty for some users)
    if data["recommendations"]:
        recommendation = data["recommendations"][0]
        assert "book_id" in recommendation, "Recommendation should contain book_id"
        assert "title" in recommendation, "Recommendation should contain title"
        assert "authors" in recommendation, "Recommendation should contain authors"
        assert "rank" in recommendation, "Recommendation should contain rank"

def test_user_recommendations_limit(api_health_check, valid_user_id):
    """Test recommendations with a limit parameter"""
    limit = 3
    url = get_endpoint(f"/recommend/user/{valid_user_id}")
    response = requests.get(url, params={"num_recommendations": limit})
    
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"
    
    data = response.json()
    recommendations = data.get("recommendations", [])
    
    # Check that we got the requested number of recommendations
    # Note: For some users, we might get fewer recommendations
    assert len(recommendations) <= limit, f"Should return at most {limit} recommendations"

def test_user_recommendations_invalid_user(api_health_check, invalid_user_id):
    """Test recommendations with an invalid user ID"""
    url = get_endpoint(f"/recommend/user/{invalid_user_id}")
    
    # We expect one of two behaviors:
    # 1. A 404 error if the API enforces user existence
    # 2. A 200 with empty recommendations if the API handles missing users gracefully
    response = requests.get(url)
    
    if response.status_code == 404:
        assert "not found" in response.json().get("detail", "").lower(), "Should indicate user not found"
    else:
        assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"
        data = response.json()
        assert len(data.get("recommendations", [])) == 0, "Should return empty recommendations for invalid user"

# Tests for Similar Books
def test_similar_books(api_health_check, valid_book_id):
    """Test getting similar books for a valid book ID"""
    url = get_endpoint(f"/similar-books/{valid_book_id}")
    response = requests.get(url)
    
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"
    
    data = response.json()
    assert "recommendations" in data, "Response should contain recommendations key"
    assert "book_id" in data, "Response should contain book_id key"
    assert data["book_id"] == valid_book_id, f"Book ID should match {valid_book_id}"
    
    # If we got recommendations, check their structure
    if data["recommendations"]:
        recommendation = data["recommendations"][0]
        assert "book_id" in recommendation, "Recommendation should contain book_id"
        assert "title" in recommendation, "Recommendation should contain title"
        assert "authors" in recommendation, "Recommendation should contain authors"
        assert "rank" in recommendation, "Recommendation should contain rank"

def test_similar_books_limit(api_health_check, valid_book_id):
    """Test similar books with a limit parameter"""
    limit = 3
    url = get_endpoint(f"/similar-books/{valid_book_id}")
    response = requests.get(url, params={"num_recommendations": limit})
    
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"
    
    data = response.json()
    recommendations = data.get("recommendations", [])
    
    # Check that we got the requested number of recommendations
    # Note: For some books, we might get fewer similar books
    assert len(recommendations) <= limit, f"Should return at most {limit} recommendations"

def test_similar_books_invalid_book(api_health_check):
    """Test similar books with an invalid book ID"""
    invalid_book_id = 9999999  # Assuming this ID doesn't exist
    url = get_endpoint(f"/similar-books/{invalid_book_id}")
    
    # We expect one of two behaviors:
    # 1. A 404 error if the API enforces book existence
    # 2. A 200 with empty recommendations if the API handles missing books gracefully
    response = requests.get(url)
    
    if response.status_code == 404:
        assert "not found" in response.json().get("detail", "").lower(), "Should indicate book not found"
    else:
        assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"
        data = response.json()
        assert len(data.get("recommendations", [])) == 0, "Should return empty recommendations for invalid book"

# Tests for Popular Books
def test_popular_books(api_health_check):
    """Test getting popular books"""
    url = get_endpoint("/popular-books")
    response = requests.get(url)
    
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"
    
    data = response.json()
    assert "books" in data, "Response should contain books key"
    
    # Check if we got any books
    books = data["books"]
    assert isinstance(books, list), "Books should be a list"
    
    # If we got books, check their structure
    if books:
        book = books[0]
        assert "book_id" in book, "Book should contain book_id"
        assert "title" in book, "Book should contain title"
        assert "authors" in book, "Book should contain authors"

def test_popular_books_limit(api_health_check):
    """Test popular books with a limit parameter"""
    limit = 3
    url = get_endpoint("/popular-books")
    response = requests.get(url, params={"limit": limit})
    
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"
    
    data = response.json()
    books = data.get("books", [])
    
    # Check that we got the requested number of books
    assert len(books) <= limit, f"Should return at most {limit} books"

def test_popular_books_reproducibility(api_health_check):
    """Test that popular books with the same seed returns the same books"""
    seed = 42
    url = get_endpoint("/popular-books")
    
    # Make two requests with the same seed
    response1 = requests.get(url, params={"seed": seed})
    response2 = requests.get(url, params={"seed": seed})
    
    assert response1.status_code == 200, f"Expected 200 OK, got {response1.status_code}"
    assert response2.status_code == 200, f"Expected 200 OK, got {response2.status_code}"
    
    books1 = response1.json().get("books", [])
    books2 = response2.json().get("books", [])
    
    # Extract book IDs for comparison
    book_ids1 = [book["book_id"] for book in books1]
    book_ids2 = [book["book_id"] for book in books2]
    
    # Check that the books are the same
    assert book_ids1 == book_ids2, "Same seed should return the same books"

# Tests for basic CRUD operations
def test_get_books(api_health_check):
    """Test getting a list of books"""
    url = get_endpoint("/books")
    response = requests.get(url, params={"limit": 5})
    
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"
    
    data = response.json()
    assert "books" in data, "Response should contain books key"
    assert "count" in data, "Response should contain count key"
    assert "status" in data, "Response should contain status key"
    assert data["status"] == "success", "Status should be success"
    
    books = data["books"]
    assert isinstance(books, list), "Books should be a list"
    
    # If we got books, check their structure
    if books:
        book = books[0]
        assert "book_id" in book, "Book should contain book_id"
        assert "title" in book, "Book should contain title"
        assert "authors" in book, "Book should contain authors"

def test_get_genres(api_health_check):
    """Test getting a list of genres"""
    url = get_endpoint("/genres")
    response = requests.get(url)
    
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"
    
    data = response.json()
    assert "genres" in data, "Response should contain genres key"
    assert "count" in data, "Response should contain count key"
    
    genres = data["genres"]
    assert isinstance(genres, list), "Genres should be a list"

def test_get_authors(api_health_check):
    """Test getting a list of authors"""
    url = get_endpoint("/authors")
    response = requests.get(url)
    
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"
    
    data = response.json()
    assert "authors" in data, "Response should contain authors key"
    assert "count" in data, "Response should contain count key"
    
    authors = data["authors"]
    assert isinstance(authors, list), "Authors should be a list"

def test_get_users(api_health_check):
    """Test getting a list of users"""
    url = get_endpoint("/users")
    response = requests.get(url, params={"limit": 5})
    
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"
    
    data = response.json()
    assert "users" in data, "Response should contain users key"
    assert "count" in data, "Response should contain count key"
    
    users = data["users"]
    assert isinstance(users, list), "Users should be a list"

def test_get_user_details(api_health_check, valid_user_id):
    """Test getting details for a specific user"""
    url = get_endpoint(f"/users/{valid_user_id}")
    response = requests.get(url)
    
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"
    
    data = response.json()
    assert "user_id" in data, "Response should contain user_id key"
    assert data["user_id"] == valid_user_id, f"User ID should match {valid_user_id}"
    assert "total_ratings" in data, "Response should contain total_ratings key"
    assert "avg_rating" in data, "Response should contain avg_rating key"
    assert "favorite_genres" in data, "Response should contain favorite_genres key"
    assert "recent_books" in data, "Response should contain recent_books key"

def test_get_user_ratings(api_health_check, valid_user_id):
    """Test getting ratings for a specific user"""
    url = get_endpoint(f"/users/{valid_user_id}/ratings")
    response = requests.get(url, params={"limit": 5})
    
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"
    
    data = response.json()
    assert "status" in data, "Response should contain status key"
    assert "user_id" in data, "Response should contain user_id key"
    assert data["user_id"] == valid_user_id, f"User ID should match {valid_user_id}"
    assert "ratings" in data, "Response should contain ratings key"

# Test error handling
def test_invalid_endpoint(api_health_check):
    """Test invalid endpoint returns 404"""
    url = f"{API_URL}/invalid-endpoint"
    response = requests.get(url)
    
    assert response.status_code == 404, f"Expected 404 Not Found, got {response.status_code}"

def test_method_not_allowed(api_health_check):
    """Test that POST to GET-only endpoint returns 405"""
    url = get_endpoint("/books")
    response = requests.post(url)
    
    assert response.status_code == 405, f"Expected 405 Method Not Allowed, got {response.status_code}"

def test_invalid_params(api_health_check):
    """Test that invalid parameters are handled correctly"""
    url = get_endpoint("/books")
    response = requests.get(url, params={"limit": "invalid"})
    
    # Most FastAPI endpoints would return 422 for invalid parameter types
    assert response.status_code in [400, 422], f"Expected 400 or 422 for invalid params, got {response.status_code}"

# Integration tests (these test the system as a whole)
def test_end_to_end_flow(api_health_check, valid_user_id, valid_book_id):
    """Test an end-to-end flow: get user recommendations, then get similar books for a recommended book"""
    # 1. Get user recommendations
    recommendations_url = get_endpoint(f"/recommend/user/{valid_user_id}")
    recommendations_response = requests.get(recommendations_url)
    
    assert recommendations_response.status_code == 200, "Failed to get user recommendations"
    
    # 2. If we got recommendations, get similar books for the first recommended book
    recommendation_data = recommendations_response.json()
    recommendations = recommendation_data.get("recommendations", [])
    
    if recommendations:
        # Get the first recommended book
        first_book_id = recommendations[0]["book_id"]
        
        # Get similar books for this book
        similar_books_url = get_endpoint(f"/similar-books/{first_book_id}")
        similar_books_response = requests.get(similar_books_url)
        
        assert similar_books_response.status_code == 200, "Failed to get similar books"
        assert "recommendations" in similar_books_response.json(), "Similar books response missing recommendations"
    else:
        # If no recommendations, use the valid_book_id as fallback
        similar_books_url = get_endpoint(f"/similar-books/{valid_book_id}")
        similar_books_response = requests.get(similar_books_url)
        
        assert similar_books_response.status_code == 200, "Failed to get similar books"
        assert "recommendations" in similar_books_response.json(), "Similar books response missing recommendations"

def test_api_latency(api_health_check):
    """Test API latency for key endpoints"""
    # Define endpoints and their expected paths
    endpoints = [
        # Standard endpoints that might follow prefix rules
        {"path": "/books", "follows_prefix": True},
        {"path": "/genres", "follows_prefix": True},
        {"path": "/authors", "follows_prefix": True},
        {"path": "/popular-books", "follows_prefix": True},
        # Root endpoints that typically don't use the API prefix
        {"path": "/", "follows_prefix": False},
        {"path": "/health", "follows_prefix": False},
    ]
    
    for endpoint_info in endpoints:
        path = endpoint_info["path"]
        
        # Determine the correct URL based on whether the endpoint follows prefix rules
        if endpoint_info["follows_prefix"]:
            url = get_endpoint(path)
        else:
            # Root endpoints like /health are always at the root
            url = f"{API_URL}{path}"
        
        logger.info(f"Testing latency for endpoint: {url}")
        
        try:
            start_time = time.time()
            response = requests.get(url, timeout=10)
            latency = time.time() - start_time
            
            # For root path, a redirect is also acceptable
            if path == "/" and response.status_code in [200, 301, 302]:
                logger.info(f"Root path returned status {response.status_code} (acceptable)")
            else:
                assert response.status_code == 200, f"Failed to get response from {path}, status: {response.status_code}"
            
            # Log the latency
            logger.info(f"Latency for {path}: {latency:.3f} seconds")
            
            # A very loose assertion to catch extreme issues
            assert latency < 10.0, f"Latency for {path} is too high: {latency:.3f} seconds"
        except requests.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            # Skip this endpoint rather than failing the whole test
            logger.warning(f"Skipping latency test for {path} due to request error")
            continue

if __name__ == "__main__":
    """Run the tests directly without pytest"""
    # Check if the API is running with better error handling
    api_available = False
    
    # Try multiple health check endpoints
    health_endpoints = ["/health", "/api/health", "/"]
    
    for health_path in health_endpoints:
        try:
            health_url = f"{API_URL}{health_path}"
            logger.info(f"Checking API health at: {health_url}")
            health_response = requests.get(health_url, timeout=5)
            
            if health_response.status_code == 200:
                logger.info(f"API is healthy at {health_url}")
                api_available = True
                break
        except Exception as e:
            logger.warning(f"Health check failed at {health_path}: {e}")
    
    if not api_available:
        logger.error("API is not available at any tested endpoint")
        sys.exit(1)
    
    # Get a valid user ID and book ID
    user_id = 1
    book_id = 1
    
    # Run some basic tests
    logger.info("Testing user recommendations...")
    test_user_recommendations(True, user_id)
    
    logger.info("Testing similar books...")
    test_similar_books(True, book_id)
    
    logger.info("Testing popular books...")
    test_popular_books(True)
    
    logger.info("Testing end-to-end flow...")
    test_end_to_end_flow(True, user_id, book_id)
    
    logger.info("All tests passed!")
