#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the book recommender API.
This script tests all endpoints and functionality of the API.
"""

import os
import sys
import json
import time
import logging
import unittest
import requests
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('api_test')

# API configuration
API_BASE_URL = "http://localhost:8000"
TEST_BOOK_IDS = [3, 5, 8, 10, 24, 27]  # Some book IDs we saw in previous responses
TEST_USER_IDS = [1, 2, 3, 10, 20, 30]  # A range of user IDs to test

# Test all model types
MODEL_TYPES = ["collaborative", "content", "hybrid"]

class BookRecommenderApiTest(unittest.TestCase):
    """Test cases for the Book Recommender API"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests"""
        logger.info("Starting API tests")
        # Check if API is running
        cls._check_api_availability()

    @classmethod
    def tearDownClass(cls):
        """Tear down test fixtures after running tests"""
        logger.info("All tests completed")

    @classmethod
    def _check_api_availability(cls):
        """Check if the API is available before running tests"""
        max_retries = 5
        for i in range(max_retries):
            try:
                response = requests.get(f"{API_BASE_URL}/health")
                if response.status_code == 200:
                    logger.info("API is available")
                    return
            except requests.exceptions.ConnectionError:
                pass
            logger.warning(f"API not available, retrying in 2 seconds (attempt {i+1}/{max_retries})")
            time.sleep(2)
        raise ConnectionError("Could not connect to the API. Make sure it's running.")

    def _make_request(self, endpoint: str, expected_status: int = 200) -> Dict[str, Any]:
        """Make a request to the API and validate the response"""
        url = f"{API_BASE_URL}/{endpoint}"
        logger.info(f"Making request to {url}")
        try:
            response = requests.get(url)
            self.assertEqual(response.status_code, expected_status, 
                             f"Expected status {expected_status}, got {response.status_code} for {url}")
            if expected_status == 200:
                return response.json()
            return {}
        except requests.exceptions.ConnectionError:
            self.fail(f"Connection error when accessing {url}")
        except json.JSONDecodeError:
            self.fail(f"Could not parse JSON response from {url}")

    def test_root_endpoint(self):
        """Test the root endpoint"""
        logger.info("Testing root endpoint")
        data = self._make_request("")
        self.assertIn("app_name", data)
        self.assertIn("version", data)
        self.assertIn("endpoints", data)
        endpoints = data["endpoints"]
        self.assertTrue(isinstance(endpoints, list))
        self.assertGreater(len(endpoints), 0)

    def test_health_endpoint(self):
        """Test the health endpoint"""
        logger.info("Testing health endpoint")
        data = self._make_request("health")
        self.assertEqual(data["status"], "healthy")
        self.assertIn("timestamp", data)

    def test_user_recommendations(self):
        """Test user recommendations endpoint for different users and model types"""
        for user_id in TEST_USER_IDS:
            for model_type in MODEL_TYPES:
                logger.info(f"Testing user recommendations for user {user_id} with model {model_type}")
                try:
                    data = self._make_request(f"recommend/user/{user_id}?model_type={model_type}&num_recommendations=3")
                    self.assertIn("recommendations", data)
                    recommendations = data["recommendations"]
                    
                    # Check if we have recommendations
                    if not recommendations:
                        logger.warning(f"No recommendations for user {user_id} with model {model_type}")
                    else:
                        # Validate recommendation structure
                        for rec in recommendations:
                            self.assertIn("book_id", rec)
                            self.assertIn("title", rec)
                            self.assertIn("authors", rec)
                            self.assertIn("rank", rec)
                except AssertionError as e:
                    # Some users might not have recommendations, which is okay
                    if "404" in str(e):
                        logger.warning(f"No recommendations found for user {user_id} with model {model_type}")
                    else:
                        raise

    def test_similar_books(self):
        """Test similar books endpoint for different books and model types"""
        for book_id in TEST_BOOK_IDS:
            for model_type in MODEL_TYPES:
                logger.info(f"Testing similar books for book {book_id} with model {model_type}")
                try:
                    data = self._make_request(f"similar-books/{book_id}?model_type={model_type}&num_recommendations=3")
                    self.assertIn("recommendations", data)
                    recommendations = data["recommendations"]
                    
                    # Check if we have recommendations
                    if not recommendations:
                        logger.warning(f"No similar books for book {book_id} with model {model_type}")
                    else:
                        # Validate recommendation structure
                        for rec in recommendations:
                            self.assertIn("book_id", rec)
                            self.assertIn("title", rec)
                            self.assertIn("authors", rec)
                            self.assertIn("rank", rec)
                            # Make sure the original book is not in the recommendations
                            self.assertNotEqual(rec["book_id"], book_id)
                except AssertionError as e:
                    # Some books might not have similar books, which is okay
                    if "404" in str(e):
                        logger.warning(f"No similar books found for book {book_id} with model {model_type}")
                    else:
                        raise

    def test_invalid_user(self):
        """Test with invalid user ID"""
        logger.info("Testing with invalid user ID")
        self._make_request("recommend/user/999999?num_recommendations=3", expected_status=404)

    def test_invalid_book(self):
        """Test with invalid book ID"""
        logger.info("Testing with invalid book ID")
        self._make_request("similar-books/999999?num_recommendations=3", expected_status=404)

    def test_invalid_path(self):
        """Test with an invalid path"""
        logger.info("Testing with invalid path")
        self._make_request("not-a-valid-endpoint", expected_status=404)

    def test_invalid_parameters(self):
        """Test with invalid parameters"""
        logger.info("Testing with invalid parameters")
        self._make_request("recommend/user/1?num_recommendations=100", expected_status=422)
        self._make_request("recommend/user/1?model_type=invalid", expected_status=422)

    def test_performance(self):
        """Test API performance by measuring response times"""
        logger.info("Testing API performance")
        endpoints = [
            "health",
            "recommend/user/1?num_recommendations=5",
            "similar-books/3?num_recommendations=5"
        ]
        
        for endpoint in endpoints:
            start_time = time.time()
            self._make_request(endpoint)
            end_time = time.time()
            elapsed = end_time - start_time
            logger.info(f"Endpoint {endpoint} response time: {elapsed:.2f} seconds")
            # Most API calls should be reasonably fast
            self.assertLess(elapsed, 5.0, f"Endpoint {endpoint} took too long to respond: {elapsed:.2f} seconds")

if __name__ == "__main__":
    # Enable PYTHONPATH to find modules
    if "PYTHONPATH" not in os.environ:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        os.environ["PYTHONPATH"] = project_root
        logger.info(f"Set PYTHONPATH to {project_root}")
    
    # Run the tests
    unittest.main()
