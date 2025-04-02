#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test script for the Book Recommender API."""

import os
import sys
import json
import time
import requests
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Set up project path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

# Set up logging
log_dir = os.path.join(project_root, 'logs')
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(log_dir, f'test_api_{timestamp}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('test_api')

# Configuration
DEFAULT_API_URL = "http://localhost:9999"
DEFAULT_TIMEOUT = 10  # seconds

def test_root_endpoint(base_url: str = DEFAULT_API_URL, timeout: int = DEFAULT_TIMEOUT) -> bool:
    """Test the root endpoint."""
    logger.info("Testing root endpoint...")
    try:
        url = f"{base_url}/"
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        
        data = response.json()
        logger.info(f"Root endpoint returned {len(data)} items")
        
        # Validate response structure
        assert "app_name" in data
        assert "version" in data
        assert "endpoints" in data
        
        logger.info(f"Root endpoint test passed: {data['app_name']} v{data['version']}")
        return True
    except Exception as e:
        logger.error(f"Root endpoint test failed: {e}")
        return False

def test_health_endpoint(base_url: str = DEFAULT_API_URL, timeout: int = DEFAULT_TIMEOUT) -> bool:
    """Test the health endpoint."""
    logger.info("Testing health endpoint...")
    try:
        url = f"{base_url}/health"
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        
        data = response.json()
        logger.info(f"Health endpoint response: {data}")
        
        # Validate response structure
        assert "status" in data
        assert "timestamp" in data
        assert data["status"] == "healthy"
        
        logger.info("Health endpoint test passed")
        return True
    except Exception as e:
        logger.error(f"Health endpoint test failed: {e}")
        return False

def test_user_recommendations(user_id: int = 125, 
                           num_recommendations: int = 5,
                           base_url: str = DEFAULT_API_URL, 
                           timeout: int = DEFAULT_TIMEOUT) -> bool:
    """Test the user recommendations endpoint."""
    logger.info(f"Testing user recommendations for user {user_id}...")
    try:
        url = f"{base_url}/recommend/user/{user_id}?num_recommendations={num_recommendations}"
        start_time = time.time()
        response = requests.get(url, timeout=timeout)
        response_time = time.time() - start_time
        response.raise_for_status()
        
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
        
        logger.info(f"User recommendations test passed for user {user_id}")
        return True
    except Exception as e:
        logger.error(f"User recommendations test failed for user {user_id}: {e}")
        return False

def test_similar_books(book_id: int = 352, 
                     num_recommendations: int = 5,
                     base_url: str = DEFAULT_API_URL, 
                     timeout: int = DEFAULT_TIMEOUT) -> bool:
    """Test the similar books endpoint."""
    logger.info(f"Testing similar books for book {book_id}...")
    try:
        url = f"{base_url}/similar-books/{book_id}?num_recommendations={num_recommendations}"
        start_time = time.time()
        response = requests.get(url, timeout=timeout)
        response_time = time.time() - start_time
        response.raise_for_status()
        
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
        
        logger.info(f"Similar books test passed for book {book_id}")
        return True
    except Exception as e:
        logger.error(f"Similar books test failed for book {book_id}: {e}")
        return False

def test_error_cases(base_url: str = DEFAULT_API_URL, timeout: int = DEFAULT_TIMEOUT) -> bool:
    """Test API error handling."""
    test_cases = [
        {
            "name": "Non-existent user",
            "url": f"{base_url}/recommend/user/99999",
            "expected_status": 404
        },
        {
            "name": "Non-existent book",
            "url": f"{base_url}/similar-books/99999",
            "expected_status": 404
        },
        {
            "name": "Invalid recommendation count",
            "url": f"{base_url}/recommend/user/125?num_recommendations=100",
            "expected_status": 422
        },
        {
            "name": "Negative user ID",
            "url": f"{base_url}/recommend/user/-1",
            "expected_status": 404
        },
        {
            "name": "Zero user ID",
            "url": f"{base_url}/recommend/user/0",
            "expected_status": 404
        },
        {
            "name": "Non-integer user ID",
            "url": f"{base_url}/recommend/user/abc",
            "expected_status": 422
        },
        {
            "name": "Negative book ID",
            "url": f"{base_url}/similar-books/-5",
            "expected_status": 404
        },
        {
            "name": "Zero book ID",
            "url": f"{base_url}/similar-books/0",
            "expected_status": 404
        },
        {
            "name": "Negative recommendation count",
            "url": f"{base_url}/recommend/user/125?num_recommendations=-1",
            "expected_status": 422
        },
        {
            "name": "Zero recommendation count",
            "url": f"{base_url}/recommend/user/125?num_recommendations=0",
            "expected_status": 422
        }
    ]
    
    logger.info("Testing API error handling...")
    all_passed = True
    
    for test_case in test_cases:
        try:
            response = requests.get(test_case["url"], timeout=timeout)
            actual_status = response.status_code
            expected_status = test_case["expected_status"]
            
            if actual_status == expected_status:
                logger.info(f"PASS: {test_case['name']}: Got expected status {actual_status}")
            else:
                logger.warning(f"FAIL: {test_case['name']}: Expected status {expected_status}, got {actual_status}")
                all_passed = False
                
        except Exception as e:
            logger.error(f"FAIL: {test_case['name']} test failed with error: {e}")
            all_passed = False
    
    if all_passed:
        logger.info("All error handling tests passed")
    else:
        logger.warning("Some error handling tests failed")
    
    return all_passed

def test_boundary_values(base_url: str = DEFAULT_API_URL, timeout: int = DEFAULT_TIMEOUT) -> bool:
    """Test boundary values for recommendation parameters."""
    test_cases = [
        {
            "name": "Minimum recommendations (1)",
            "url": f"{base_url}/recommend/user/125?num_recommendations=1",
            "expected_count": 1
        },
        {
            "name": "Maximum recommendations (20)",
            "url": f"{base_url}/recommend/user/125?num_recommendations=20",
            "expected_max_count": 20
        },
        {
            "name": "Default recommendations (no parameter)",
            "url": f"{base_url}/recommend/user/125",
            "expected_count": 5  # Default is typically 5
        },
        {
            "name": "Alternative parameter name (n)",
            "url": f"{base_url}/recommend/user/125?n=3",
            "expected_count": 3
        },
        {
            "name": "Minimum similar books (1)",
            "url": f"{base_url}/similar-books/352?num_recommendations=1",
            "expected_count": 1
        },
        {
            "name": "Maximum similar books (20)",
            "url": f"{base_url}/similar-books/352?num_recommendations=20",
            "expected_max_count": 20
        }
    ]
    
    logger.info("Testing API boundary values...")
    all_passed = True
    
    for test_case in test_cases:
        try:
            response = requests.get(test_case["url"], timeout=timeout)
            response.raise_for_status()
            data = response.json()
            
            # Check if recommendations exist
            if "recommendations" not in data:
                logger.warning(f"FAIL: {test_case['name']}: No recommendations field in response")
                all_passed = False
                continue
                
            actual_count = len(data["recommendations"])
            
            # Check if count matches expected count exactly
            if "expected_count" in test_case and actual_count != test_case["expected_count"]:
                logger.warning(f"FAIL: {test_case['name']}: Expected exactly {test_case['expected_count']} recommendations, got {actual_count}")
                all_passed = False
            # Check if count is at most expected max count
            elif "expected_max_count" in test_case and actual_count > test_case["expected_max_count"]:
                logger.warning(f"FAIL: {test_case['name']}: Expected at most {test_case['expected_max_count']} recommendations, got {actual_count}")
                all_passed = False
            else:
                logger.info(f"PASS: {test_case['name']}: Got {actual_count} recommendations as expected")
                
        except Exception as e:
            logger.error(f"FAIL: {test_case['name']} test failed with error: {e}")
            all_passed = False
    
    if all_passed:
        logger.info("All boundary value tests passed")
    else:
        logger.warning("Some boundary value tests failed")
    
    return all_passed

def test_stress(base_url: str = DEFAULT_API_URL, timeout: int = DEFAULT_TIMEOUT, num_requests: int = 10) -> bool:
    """Test API performance under multiple sequential requests."""
    logger.info(f"Testing API performance with {num_requests} sequential requests...")
    endpoints = [
        f"{base_url}/recommend/user/125?num_recommendations=5",
        f"{base_url}/similar-books/352?num_recommendations=5",
        f"{base_url}/health"
    ]
    
    all_passed = True
    response_times = []
    
    for endpoint in endpoints:
        endpoint_times = []
        endpoint_name = endpoint.split('/')[-1].split('?')[0]
        
        for i in range(num_requests):
            try:
                start_time = time.time()
                response = requests.get(endpoint, timeout=timeout)
                request_time = time.time() - start_time
                
                response.raise_for_status()
                endpoint_times.append(request_time)
                
                logger.info(f"Request {i+1}/{num_requests} to {endpoint_name}: {request_time:.3f}s")
                
            except Exception as e:
                logger.error(f"Request {i+1}/{num_requests} to {endpoint_name} failed: {e}")
                all_passed = False
        
        if endpoint_times:
            avg_time = sum(endpoint_times) / len(endpoint_times)
            response_times.append((endpoint_name, avg_time))
            logger.info(f"Average response time for {endpoint_name}: {avg_time:.3f}s")
    
    if all_passed:
        logger.info("All stress test requests completed successfully")
    else:
        logger.warning("Some stress test requests failed")
    
    # Log performance summary
    logger.info("Performance Summary:")
    for endpoint, avg_time in response_times:
        logger.info(f"  {endpoint}: {avg_time:.3f}s average response time")
    
    return all_passed

def test_edge_users_and_books(base_url: str = DEFAULT_API_URL, timeout: int = DEFAULT_TIMEOUT) -> bool:
    """Test recommendations for edge case users and books."""
    # These are assumed to be valid but possibly edge cases
    # For example, users with very few ratings or books with few interactions
    edge_users = [1, 2, 499, 500]  # First and last users in the system
    edge_books = [1, 2, 499, 500]  # First and last books in the system
    
    logger.info("Testing recommendations for edge case users and books...")
    all_passed = True
    
    # Test user recommendations
    for user_id in edge_users:
        try:
            url = f"{base_url}/recommend/user/{user_id}?num_recommendations=5"
            response = requests.get(url, timeout=timeout)
            
            if response.status_code == 200:
                data = response.json()
                rec_count = len(data.get("recommendations", []))
                logger.info(f"PASS: Edge user {user_id}: Got {rec_count} recommendations")
            elif response.status_code == 404:
                # This might be expected for some edge users
                logger.info(f"PASS: Edge user {user_id}: Not found (404) - this may be expected")
            else:
                logger.warning(f"FAIL: Edge user {user_id}: Unexpected status code {response.status_code}")
                all_passed = False
                
        except Exception as e:
            logger.error(f"FAIL: Edge user {user_id} test failed with error: {e}")
            all_passed = False
    
    # Test similar books
    for book_id in edge_books:
        try:
            url = f"{base_url}/similar-books/{book_id}?num_recommendations=5"
            response = requests.get(url, timeout=timeout)
            
            if response.status_code == 200:
                data = response.json()
                rec_count = len(data.get("recommendations", []))
                logger.info(f"PASS: Edge book {book_id}: Got {rec_count} similar books")
            elif response.status_code == 404:
                # This might be expected for some edge books
                logger.info(f"PASS: Edge book {book_id}: Not found (404) - this may be expected")
            else:
                logger.warning(f"FAIL: Edge book {book_id}: Unexpected status code {response.status_code}")
                all_passed = False
                
        except Exception as e:
            logger.error(f"FAIL: Edge book {book_id} test failed with error: {e}")
            all_passed = False
    
    if all_passed:
        logger.info("All edge user and book tests passed")
    else:
        logger.warning("Some edge user and book tests failed")
    
    return all_passed

def run_all_tests(base_url: str = DEFAULT_API_URL, timeout: int = DEFAULT_TIMEOUT) -> Dict[str, bool]:
    """Run all API tests and return results."""
    test_results = {
        "root_endpoint": test_root_endpoint(base_url, timeout),
        "health_endpoint": test_health_endpoint(base_url, timeout),
        "user_recommendations": test_user_recommendations(125, 5, base_url, timeout),
        "similar_books": test_similar_books(352, 5, base_url, timeout),
        "error_handling": test_error_cases(base_url, timeout),
        "boundary_values": test_boundary_values(base_url, timeout),
        "stress_test": test_stress(base_url, timeout, 5),
        "edge_users_and_books": test_edge_users_and_books(base_url, timeout)
    }
    
    # Additional user recommendation tests
    for user_id in [125, 200, 300]:
        test_results[f"user_{user_id}_recommendations"] = test_user_recommendations(user_id, 5, base_url, timeout)
    
    # Additional similar books tests
    for book_id in [352, 200, 100]:
        test_results[f"book_{book_id}_similar"] = test_similar_books(book_id, 5, base_url, timeout)
    
    return test_results

def main(args: Optional[List[str]] = None) -> int:
    """Main function to run the API tests."""
    parser = argparse.ArgumentParser(description='Test the Book Recommender API')
    parser.add_argument('--url', type=str, default=DEFAULT_API_URL,
                        help=f'Base URL of the API (default: {DEFAULT_API_URL})')
    parser.add_argument('--timeout', type=int, default=DEFAULT_TIMEOUT,
                        help=f'Request timeout in seconds (default: {DEFAULT_TIMEOUT})')
    parser.add_argument('--test', type=str, 
                        choices=['all', 'root', 'health', 'user', 'similar', 'errors', 
                                'boundary', 'stress', 'edge'],
                        default='all', help='Specific test to run (default: all)')
    parser.add_argument('--user-id', type=int, default=125,
                        help='User ID to test recommendations for (default: 125)')
    parser.add_argument('--book-id', type=int, default=352,
                        help='Book ID to test similar books for (default: 352)')
    parser.add_argument('--num-requests', type=int, default=5,
                        help='Number of requests for stress testing (default: 5)')
    
    args = parser.parse_args(args)
    
    logger.info(f"Starting API tests against {args.url} with timeout {args.timeout}s")
    
    try:
        if args.test == 'all':
            test_results = run_all_tests(args.url, args.timeout)
            
            # Print test summary
            print("\nTest Results Summary:")
            print("=====================")
            all_passed = True
            for test_name, result in test_results.items():
                status = "PASS: PASSED" if result else "FAIL: FAILED"
                print(f"{test_name}: {status}")
                all_passed = all_passed and result
            
            # Save results to a JSON file
            results_dir = os.path.join(project_root, 'results')
            os.makedirs(results_dir, exist_ok=True)
            results_file = os.path.join(results_dir, f'api_test_results_{timestamp}.json')
            with open(results_file, 'w') as f:
                json.dump(test_results, f, indent=2)
            logger.info(f"Test results saved to {results_file}")
            
            return 0 if all_passed else 1
        elif args.test == 'root':
            return 0 if test_root_endpoint(args.url, args.timeout) else 1
        elif args.test == 'health':
            return 0 if test_health_endpoint(args.url, args.timeout) else 1
        elif args.test == 'user':
            return 0 if test_user_recommendations(args.user_id, 5, args.url, args.timeout) else 1
        elif args.test == 'similar':
            return 0 if test_similar_books(args.book_id, 5, args.url, args.timeout) else 1
        elif args.test == 'errors':
            return 0 if test_error_cases(args.url, args.timeout) else 1
        elif args.test == 'boundary':
            return 0 if test_boundary_values(args.url, args.timeout) else 1
        elif args.test == 'stress':
            return 0 if test_stress(args.url, args.timeout, args.num_requests) else 1
        elif args.test == 'edge':
            return 0 if test_edge_users_and_books(args.url, args.timeout) else 1
        else:
            logger.error(f"Unknown test: {args.test}")
            return 1
    except Exception as e:
        logger.error(f"Error running tests: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
