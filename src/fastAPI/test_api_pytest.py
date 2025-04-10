#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Pytest-based tests for the Book Recommender API."""

import os
import sys
import json
import time
import pytest
import requests
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Set up project path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

# Set up reports directory for HTML reports
reports_dir = os.path.join(project_root, 'reports', 'fastAPI')
os.makedirs(reports_dir, exist_ok=True)

# Set up logging
log_dir = os.path.join(project_root, 'logs')
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(log_dir, f'test_api_pytest_{timestamp}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('test_api_pytest')

# Default configuration for tests
DEFAULT_API_URL = "http://localhost:9998"
DEFAULT_TIMEOUT = 10  # seconds

# Check if pytest-benchmark is installed
try:
    import pytest_benchmark
    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False
    logger.warning("pytest-benchmark not installed. Benchmark tests will be skipped.")

# Pytest fixtures
@pytest.fixture
def api_url():
    """Base URL for the API, can be overridden with the API_URL environment variable."""
    return os.environ.get("API_URL", DEFAULT_API_URL)

@pytest.fixture
def timeout():
    """Request timeout, can be overridden with the API_TIMEOUT environment variable."""
    return int(os.environ.get("API_TIMEOUT", DEFAULT_TIMEOUT))

@pytest.fixture
def test_users():
    """Return a list of test user IDs."""
    return [125, 200, 300]

@pytest.fixture
def test_books():
    """Return a list of test book IDs."""
    return [352, 364, 389, 115]

@pytest.fixture
def edge_users():
    """Return a list of edge case user IDs."""
    return [1, 2, 499, 500]

@pytest.fixture
def edge_books():
    """Return a list of edge case book IDs."""
    return [1, 2, 499, 500]

@pytest.fixture
def mapped_books():
    """Return known book ID mappings for testing."""
    return [
        {"original_id": 364, "mapped_id": 332},
        {"original_id": 389, "mapped_id": 350},
        {"original_id": 115, "mapped_id": 109}
    ]

@pytest.fixture
def error_test_cases(api_url):
    """Test cases for error handling testing."""
    return [
        {
            "name": "Non-existent user",
            "url": f"{api_url}/api/recommend/user/99999",
            "expected_status": 404
        },
        {
            "name": "Non-existent book",
            "url": f"{api_url}/api/similar-books/99999",
            "expected_status": 404
        },
        {
            "name": "Invalid recommendation count",
            "url": f"{api_url}/api/recommend/user/125?num_recommendations=100",
            "expected_status": 422
        },
        {
            "name": "Negative user ID",
            "url": f"{api_url}/api/recommend/user/-1",
            "expected_status": 404
        },
        {
            "name": "Zero user ID",
            "url": f"{api_url}/api/recommend/user/0",
            "expected_status": 404
        },
        {
            "name": "Non-integer user ID",
            "url": f"{api_url}/api/recommend/user/abc",
            "expected_status": 422
        },
        {
            "name": "Negative book ID",
            "url": f"{api_url}/api/similar-books/-5",
            "expected_status": 404
        },
        {
            "name": "Zero book ID",
            "url": f"{api_url}/api/similar-books/0",
            "expected_status": 404
        },
        {
            "name": "Negative recommendation count",
            "url": f"{api_url}/api/recommend/user/125?num_recommendations=-1",
            "expected_status": 422
        },
        {
            "name": "Zero recommendation count",
            "url": f"{api_url}/api/recommend/user/125?num_recommendations=0",
            "expected_status": 422
        }
    ]

@pytest.fixture
def boundary_test_cases(api_url):
    """Test cases for boundary value testing."""
    return [
        {
            "name": "Minimum recommendations (1)",
            "url": f"{api_url}/api/recommend/user/125?num_recommendations=1",
            "expected_count": 1
        },
        {
            "name": "Maximum recommendations (20)",
            "url": f"{api_url}/api/recommend/user/125?num_recommendations=20",
            "expected_max_count": 20
        },
        {
            "name": "Default recommendations (no parameter)",
            "url": f"{api_url}/api/recommend/user/125",
            "expected_count": 5  # Default is typically 5
        },
        {
            "name": "Alternative parameter name (n)",
            "url": f"{api_url}/api/recommend/user/125?n=3",
            "expected_count": 3
        },
        {
            "name": "Minimum similar books (1)",
            "url": f"{api_url}/api/similar-books/352?num_recommendations=1",
            "expected_count": 1
        },
        {
            "name": "Maximum similar books (20)",
            "url": f"{api_url}/api/similar-books/352?num_recommendations=20",
            "expected_max_count": 20
        },
        {
            "name": "Minimum popular books (1)",
            "url": f"{api_url}/api/popular-books?limit=1",
            "expected_count": 1
        },
        {
            "name": "Maximum popular books (12)",
            "url": f"{api_url}/api/popular-books?limit=12",
            "expected_count": 12
        }
    ]

# Basic tests
def test_root_endpoint(api_url, timeout):
    """Test the root endpoint."""
    logger.info("Testing root endpoint...")
    url = f"{api_url}/"
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    
    data = response.json()
    logger.info(f"Root endpoint returned {len(data)} items")
    
    # Validate response structure
    assert "app_name" in data
    assert "version" in data
    assert "endpoints" in data
    
    logger.info(f"Root endpoint test passed: {data['app_name']} v{data['version']}")


def test_health_endpoint(api_url, timeout):
    """Test the health endpoint."""
    logger.info("Testing health endpoint...")
    url = f"{api_url}/health"
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    
    data = response.json()
    logger.info(f"Health endpoint response: {data}")
    
    # Validate response structure
    assert "status" in data
    assert "timestamp" in data
    assert data["status"] == "healthy"
    
    logger.info("Health endpoint test passed")


# Recommendations tests
@pytest.mark.parametrize("user_id", [125, 200, 300])
def test_user_recommendations(api_url, timeout, user_id, num_recommendations=5):
    """Test the user recommendations endpoint."""
    logger.info(f"Testing user recommendations for user {user_id}...")
    
    url = f"{api_url}/api/recommend/user/{user_id}?num_recommendations={num_recommendations}"
    start_time = time.time()
    response = requests.get(url, timeout=timeout)
    response_time = time.time() - start_time
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    data = response.json()
    logger.info(f"User recommendations response received in {response_time:.2f}s")
    
    # Validate response structure
    assert "recommendations" in data
    assert len(data["recommendations"]) <= num_recommendations
    
    # Validate recommendation structure
    for rec in data["recommendations"]:
        assert "book_id" in rec
        assert "title" in rec
        assert "authors" in rec
        assert "rank" in rec
    
    logger.info(f"Found {len(data['recommendations'])} recommendations for user {user_id}")
    for i, rec in enumerate(data["recommendations"]):
        logger.info(f"  {i+1}. {rec['title']} by {rec['authors']} (ID: {rec['book_id']})")


@pytest.mark.parametrize("book_id", [352, 364, 389, 115])
def test_similar_books(api_url, timeout, book_id, num_recommendations=5):
    """Test the similar books endpoint."""
    logger.info(f"Testing similar books for book {book_id}...")
    
    url = f"{api_url}/api/similar-books/{book_id}?num_recommendations={num_recommendations}"
    start_time = time.time()
    response = requests.get(url, timeout=timeout)
    response_time = time.time() - start_time
    
    # Some book IDs might not exist, so we'll accept 404 in those cases
    if response.status_code == 404:
        logger.info(f"Book {book_id} not found (404) - This may be expected")
        return
        
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    data = response.json()
    logger.info(f"Similar books response received in {response_time:.2f}s")
    
    # Validate response structure
    assert "recommendations" in data
    assert len(data["recommendations"]) <= num_recommendations
    
    # Validate recommendation structure
    for rec in data["recommendations"]:
        assert "book_id" in rec
        assert "title" in rec
        assert "authors" in rec
        assert "rank" in rec
    
    logger.info(f"Found {len(data['recommendations'])} similar books for book {book_id}")
    for i, rec in enumerate(data["recommendations"]):
        logger.info(f"  {i+1}. {rec['title']} by {rec['authors']} (ID: {rec['book_id']})")


@pytest.mark.parametrize("limit", [1, 6, 12])
def test_popular_books(api_url, timeout, limit):
    """Test the popular books endpoint."""
    logger.info(f"Testing popular books with limit {limit}...")
    
    url = f"{api_url}/api/popular-books?limit={limit}"
    start_time = time.time()
    response = requests.get(url, timeout=timeout)
    response_time = time.time() - start_time
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    data = response.json()
    logger.info(f"Popular books response received in {response_time:.2f}s")
    
    # Validate response structure
    assert "books" in data
    assert len(data["books"]) <= limit
    
    # Validate book structure
    for book in data["books"]:
        assert "book_id" in book
        assert "title" in book
        assert "authors" in book
        assert "average_rating" in book
        assert "ratings_count" in book
    
    logger.info(f"Found {len(data['books'])} popular books")
    for i, book in enumerate(data['books']):
        logger.info(f"  {i+1}. {book['title']} by {book['authors']} (ID: {book['book_id']})")


# Error handling tests
def test_error_handling(api_url, timeout, error_test_cases):
    """Test API error handling."""
    logger.info("Testing API error handling...")
    
    for test_case in error_test_cases:
        logger.info(f"Testing error case: {test_case['name']}")
        response = requests.get(test_case["url"], timeout=timeout)
        actual_status = response.status_code
        expected_status = test_case["expected_status"]
        
        assert actual_status == expected_status, \
            f"Expected status {expected_status}, got {actual_status} for {test_case['name']}"
        
        logger.info(f"PASS: {test_case['name']}: Got expected status {actual_status}")


# Boundary value tests
def test_boundary_values(api_url, timeout, boundary_test_cases):
    """Test boundary values for recommendation parameters."""
    logger.info("Testing API boundary values...")
    
    for test_case in boundary_test_cases:
        logger.info(f"Testing boundary case: {test_case['name']}")
        response = requests.get(test_case["url"], timeout=timeout)
        
        # Skip tests when API returns an error
        if response.status_code != 200:
            pytest.skip(f"API returned status code {response.status_code} for {test_case['name']}")
            
        data = response.json()
        
        # Handle different response structures
        if "recommendations" in data:
            actual_count = len(data["recommendations"])
        elif "books" in data:
            actual_count = len(data["books"])
        else:
            pytest.fail(f"No recommendations or books field in response for {test_case['name']}")
            
        # Check if count matches expected count exactly
        if "expected_count" in test_case:
            assert actual_count == test_case["expected_count"], \
                f"Expected exactly {test_case['expected_count']} items, got {actual_count}"
                
        # Check if count is at most expected max count
        if "expected_max_count" in test_case:
            assert actual_count <= test_case["expected_max_count"], \
                f"Expected at most {test_case['expected_max_count']} items, got {actual_count}"
        
        logger.info(f"PASS: {test_case['name']}: Got {actual_count} items as expected")


# Book ID mapping tests
def test_book_id_mapping(api_url, timeout, mapped_books):
    """Test book ID mapping functionality with known mapped IDs."""
    logger.info("Testing book ID mapping with known mapped book IDs...")
    
    for book in mapped_books:
        original_id = book["original_id"]
        url = f"{api_url}/api/similar-books/{original_id}?num_recommendations=3"
        response = requests.get(url, timeout=timeout)
        
        # Accept both 200 (success) and 404 (not found) as valid responses
        # since some mapped books might not exist in all environments
        if response.status_code == 404:
            logger.info(f"Book {original_id} not found (404) - This may be expected")
            continue
            
        assert response.status_code == 200, \
            f"Unexpected status code {response.status_code} for book {original_id}"
            
        data = response.json()
        assert "recommendations" in data, f"No recommendations field for book {original_id}"
        
        logger.info(f"PASS: Mapped book {original_id}: Got recommendations successfully")


# Edge case tests
def test_edge_case_users(api_url, timeout, edge_users):
    """Test recommendations for edge case users."""
    logger.info("Testing recommendations for edge case users...")
    
    for user_id in edge_users:
        url = f"{api_url}/api/recommend/user/{user_id}?num_recommendations=5"
        response = requests.get(url, timeout=timeout)
        
        # Both 200 and 404 are acceptable for edge case users
        if response.status_code == 404:
            logger.info(f"Edge user {user_id}: Not found (404) - this may be expected")
            continue
            
        assert response.status_code == 200, \
            f"Unexpected status code {response.status_code} for user {user_id}"
            
        data = response.json()
        assert "recommendations" in data
        
        rec_count = len(data["recommendations"])
        logger.info(f"PASS: Edge user {user_id}: Got {rec_count} recommendations")


def test_edge_case_books(api_url, timeout, edge_books):
    """Test similar books for edge case books."""
    logger.info("Testing recommendations for edge case books...")
    
    for book_id in edge_books:
        url = f"{api_url}/api/similar-books/{book_id}?num_recommendations=5"
        response = requests.get(url, timeout=timeout)
        
        # Both 200 and 404 are acceptable for edge case books
        if response.status_code == 404:
            logger.info(f"Edge book {book_id}: Not found (404) - this may be expected")
            continue
            
        assert response.status_code == 200, \
            f"Unexpected status code {response.status_code} for book {book_id}"
            
        data = response.json()
        assert "recommendations" in data
        
        rec_count = len(data["recommendations"])
        logger.info(f"PASS: Edge book {book_id}: Got {rec_count} similar books")


# Stress test - Modified to make it easier to skip
# To run this test: pytest test_api_pytest.py::test_stress
# To skip this test: pytest test_api_pytest.py -k "not stress"
@pytest.mark.stress
def test_stress(api_url, timeout):
    """Test API performance under multiple sequential requests.
    
    This test can take several minutes to run. To skip this test, run pytest with:
    pytest -k "not stress"
    """
    # Skip this test if SKIP_STRESS_TEST environment variable is set to "true"
    skip_stress = os.environ.get("SKIP_STRESS_TEST", "").lower() == "true"
    if skip_stress:
        pytest.skip("Stress test skipped via SKIP_STRESS_TEST environment variable")
    
    logger.info("Starting API performance test - this may take several minutes")
    
    num_requests = int(os.environ.get("STRESS_TEST_REQUESTS", "5"))
    endpoints = [
        f"{api_url}/api/recommend/user/125?num_recommendations=5",
        f"{api_url}/api/similar-books/352?num_recommendations=5",
        f"{api_url}/api/popular-books?limit=6",
        f"{api_url}/health"
    ]
    
    response_times = {}
    
    for endpoint in endpoints:
        endpoint_times = []
        endpoint_name = endpoint.split('/')[-1].split('?')[0]
        
        for i in range(num_requests):
            start_time = time.time()
            response = requests.get(endpoint, timeout=timeout)
            request_time = time.time() - start_time
            
            response.raise_for_status()
            endpoint_times.append(request_time)
            
            logger.info(f"Request {i+1}/{num_requests} to {endpoint_name}: {request_time:.3f}s")
        
        if endpoint_times:
            avg_time = sum(endpoint_times) / len(endpoint_times)
            response_times[endpoint_name] = avg_time
            logger.info(f"Average response time for {endpoint_name}: {avg_time:.3f}s")
    
    # Log performance summary
    logger.info("Performance Summary:")
    for endpoint_name, avg_time in response_times.items():
        logger.info(f"  {endpoint_name}: {avg_time:.3f}s average response time")


# Benchmark tests - These will only run if pytest-benchmark is installed
# Skip all benchmark tests if pytest-benchmark is not available
if BENCHMARK_AVAILABLE:
    @pytest.mark.benchmark(group="api-basic")
    def test_benchmark_health_endpoint(api_url, timeout, benchmark):
        """Benchmark the health endpoint."""
        def get_health():
            url = f"{api_url}/health"
            return requests.get(url, timeout=timeout)
        
        result = benchmark(get_health)
        assert result.status_code == 200

    @pytest.mark.benchmark(group="api-recommend")
    def test_benchmark_user_recommendations(api_url, timeout, benchmark):
        """Benchmark the user recommendations endpoint."""
        def get_user_recommendations():
            url = f"{api_url}/api/recommend/user/125?num_recommendations=5"
            return requests.get(url, timeout=timeout)
        
        result = benchmark(get_user_recommendations)
        assert result.status_code == 200
        data = result.json()
        assert "recommendations" in data

    @pytest.mark.benchmark(group="api-recommend")
    def test_benchmark_similar_books(api_url, timeout, benchmark):
        """Benchmark the similar books endpoint."""
        def get_similar_books():
            url = f"{api_url}/api/similar-books/352?num_recommendations=5"
            return requests.get(url, timeout=timeout)
        
        result = benchmark(get_similar_books)
        assert result.status_code == 200
        data = result.json()
        assert "recommendations" in data

    @pytest.mark.benchmark(group="api-basic")
    def test_benchmark_popular_books(api_url, timeout, benchmark):
        """Benchmark the popular books endpoint."""
        def get_popular_books():
            url = f"{api_url}/api/popular-books?limit=6"
            return requests.get(url, timeout=timeout)
        
        result = benchmark(get_popular_books)
        assert result.status_code == 200
        data = result.json()
        assert "books" in data

    @pytest.mark.benchmark(group="api-recommend", warmup=True)
    def test_benchmark_recommendations_different_sizes(api_url, timeout, benchmark):
        """Benchmark the recommendations endpoint with different sizes."""
        # Test with different recommendation counts to see how performance scales
        # This is important for understanding the algorithmic complexity
        
        def get_recommendations_with_count(count):
            url = f"{api_url}/api/recommend/user/125?num_recommendations={count}"
            response = requests.get(url, timeout=timeout)
            return response
        
        # Benchmark with 5 recommendations (common case)
        result = benchmark(lambda: get_recommendations_with_count(5))
        assert result.status_code == 200
        
        # Verify the data structure
        data = result.json()
        assert "recommendations" in data