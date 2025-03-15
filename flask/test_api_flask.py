#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test script for Flask API endpoints."""

import os
import sys
import time
import unittest
import requests
import json
from pathlib import Path
from datetime import datetime

# Add project root to path for module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Base URL for the Flask API
API_BASE_URL = "http://localhost:5000/api"

def test_api():
    """Test Flask API endpoints."""
    print("\n=== Testing Flask API Endpoints ===\n")
    
    # Test root endpoint (new)
    print("Testing / endpoint...")
    try:
        response = requests.get("http://localhost:5000/")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Root endpoint: {response.status_code} - App name: {data.get('app_name')}")
            print(f"  Available endpoints: {len(data.get('endpoints', []))}")
        else:
            print(f"❌ Root endpoint failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Root endpoint request failed: {str(e)}")
    
    # Test health check endpoint
    print("\nTesting /health endpoint...")
    try:
        response = requests.get("http://localhost:5000/health")
        if response.status_code == 200:
            print(f"✅ Health check endpoint: {response.status_code} - {response.json()}")
        else:
            print(f"❌ Health check endpoint failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Health check request failed: {str(e)}")
    
    # Test books endpoint
    print("\nTesting /api/books endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/books", params={"limit": 3})
        if response.status_code == 200:
            data = response.json()
            books = data.get("books", [])
            print(f"✅ Books endpoint: {response.status_code} - Got {len(books)} books out of {data.get('total', 0)} total")
            if books:
                print("  First book:", json.dumps(books[0], indent=2))
        else:
            print(f"❌ Books endpoint failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Books request failed: {str(e)}")
    
    # Test genres endpoint
    print("\nTesting /api/genres endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/genres")
        if response.status_code == 200:
            genres = response.json().get("genres", [])
            print(f"✅ Genres endpoint: {response.status_code} - Got {len(genres)} genres")
            if genres:
                print("  Sample genres:", genres[:5])
        else:
            print(f"❌ Genres endpoint failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Genres request failed: {str(e)}")
    
    # Test authors endpoint
    print("\nTesting /api/authors endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/authors")
        if response.status_code == 200:
            authors = response.json().get("authors", [])
            print(f"✅ Authors endpoint: {response.status_code} - Got {len(authors)} authors")
            if authors:
                print("  Sample authors:", authors[:5])
        else:
            print(f"❌ Authors endpoint failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Authors request failed: {str(e)}")
    
    # Test users endpoint (new)
    print("\nTesting /api/users endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/users", params={"limit": 5})
        if response.status_code == 200:
            users = response.json().get("users", [])
            print(f"✅ Users endpoint: {response.status_code} - Got {len(users)} users")
            if users:
                print("  Sample user IDs:", users[:5])
        else:
            print(f"❌ Users endpoint failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Users request failed: {str(e)}")
    
    # Test ratings endpoint (new)
    print("\nTesting /api/ratings endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/ratings", params={"limit": 5})
        if response.status_code == 200:
            data = response.json()
            ratings = data.get("ratings", [])
            print(f"✅ Ratings endpoint: {response.status_code} - Got {len(ratings)} ratings out of {data.get('total', 0)} total")
            if ratings:
                print("  Sample rating:", json.dumps(ratings[0], indent=2))
        else:
            print(f"❌ Ratings endpoint failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Ratings request failed: {str(e)}")
    
    print("\n=== Testing Recommendation Endpoints ===\n")
    
    # Test user recommendations endpoint with various user IDs
    user_ids = [1, 15, 42]
    for user_id in user_ids:
        print(f"\nTesting /api/recommend/user/{user_id} endpoint...")
        try:
            response = requests.get(
                f"{API_BASE_URL}/recommend/user/{user_id}", 
                params={"include_images": "true", "num_recommendations": 3}
            )
            if response.status_code == 200:
                recommendations = response.json().get("recommendations", [])
                print(f"✅ User recommendations endpoint: {response.status_code} - Got {len(recommendations)} recommendations for user {user_id}")
                if recommendations:
                    print("  First recommendation:", json.dumps(recommendations[0], indent=2))
            else:
                print(f"❌ User recommendations endpoint failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ User recommendations request failed: {str(e)}")
    
    # Test similar books endpoint with various book IDs
    book_ids = [1, 100, 1000]
    for book_id in book_ids:
        print(f"\nTesting /api/similar-books/{book_id} endpoint...")
        try:
            response = requests.get(
                f"{API_BASE_URL}/similar-books/{book_id}", 
                params={"include_images": "true", "num_recommendations": 3}
            )
            if response.status_code == 200:
                similar_books = response.json().get("similar_books", [])
                print(f"✅ Similar books endpoint: {response.status_code} - Got {len(similar_books)} similar books for book ID {book_id}")
                if similar_books:
                    print("  First similar book:", json.dumps(similar_books[0], indent=2))
            else:
                print(f"❌ Similar books endpoint failed for book ID {book_id}: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Similar books request failed: {str(e)}")
    
    # Test user ratings endpoint (new)
    print("\n=== Testing User Ratings Endpoints ===\n")
    for user_id in [1, 15, 42]:
        print(f"\nTesting /api/users/{user_id}/ratings endpoint...")
        try:
            response = requests.get(
                f"{API_BASE_URL}/users/{user_id}/ratings", 
                params={"limit": 3, "include_books": "true"}
            )
            if response.status_code == 200:
                data = response.json()
                ratings = data.get("ratings", [])
                print(f"✅ User ratings endpoint: {response.status_code} - Got {len(ratings)} ratings for user {user_id} out of {data.get('total', 0)} total")
                if ratings and "book" in ratings[0]:
                    book = ratings[0]["book"]
                    print(f"  User rated '{book.get('title', 'Unknown')}' as {ratings[0].get('rating', 'N/A')}")
            else:
                print(f"❌ User ratings endpoint failed for user ID {user_id}: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ User ratings request failed: {str(e)}")
    
    # Test book ratings endpoint (new)
    print("\n=== Testing Book Ratings Endpoints ===\n")
    for book_id in [1, 100, 369]:
        print(f"\nTesting /api/books/{book_id}/ratings endpoint...")
        try:
            response = requests.get(
                f"{API_BASE_URL}/books/{book_id}/ratings", 
                params={"limit": 3}
            )
            if response.status_code == 200:
                data = response.json()
                ratings = data.get("ratings", [])
                print(f"✅ Book ratings endpoint: {response.status_code} - Got {len(ratings)} ratings for book {book_id} out of {data.get('total', 0)} total")
                print(f"  Average rating: {data.get('average_rating', 'N/A')}")
                if data.get('rating_distribution'):
                    print(f"  Rating distribution: {data.get('rating_distribution')}")
            else:
                print(f"❌ Book ratings endpoint failed for book ID {book_id}: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Book ratings request failed: {str(e)}")
    
    print("\n=== Flask API Testing Complete ===")
    print(f"Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    # Check if Flask API is running
    try:
        response = requests.get("http://localhost:5000/health", timeout=2)
        if response.status_code == 200:
            print("Flask API is running. Starting tests...")
            print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            test_api()
        else:
            print(f"Flask API returned status code {response.status_code}. Make sure it's running correctly.")
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to Flask API at http://localhost:5000/health")
        print("Make sure the Flask API is running before executing this test script.")
        print("Start it with: python flask/backend/app.py")
    except Exception as e:
        print(f"Error checking API availability: {str(e)}")
