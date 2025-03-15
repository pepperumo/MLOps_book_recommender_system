"""
Test script to verify compatibility between Flask API and FastAPI endpoints.
This script tests the newly updated Flask API endpoints to ensure they match
the FastAPI structure and response format.
"""

import requests
import json
import argparse
from datetime import datetime

def test_health_endpoint(flask_url):
    """Test the health check endpoint"""
    print("\n===== Testing Health Check Endpoint =====")
    
    response = requests.get(f"{flask_url}/health")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Success! Health check response: {json.dumps(data, indent=2)}")
        
        # Verify it has the expected fields - for Flask API we need service and status
        if "service" in data and "status" in data:
            print("✓ Response format matches API structure")
        else:
            print("✗ Response format does not match API structure")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def test_root_endpoint(flask_url):
    """Test the root endpoint"""
    print("\n===== Testing Root Endpoint =====")
    
    response = requests.get(flask_url)
    
    if response.status_code == 200:
        data = response.json()
        print(f"Success! Root endpoint response includes {len(data.get('endpoints', []))} endpoints")
        
        # Verify it has the expected fields
        if "app_name" in data and "version" in data and "endpoints" in data:
            print("✓ Response format matches FastAPI structure")
        else:
            print("✗ Response format does not match FastAPI structure")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def test_books_endpoint(flask_url):
    """Test the books endpoint"""
    print("\n===== Testing Books Endpoint =====")
    
    response = requests.get(f"{flask_url}/api/books", params={"limit": 5})
    
    if response.status_code == 200:
        data = response.json()
        print(f"Success! Received {data.get('count', 0)} books")
        
        # Verify it has the expected fields
        if "books" in data and "count" in data and "total" in data:
            print("✓ Response format matches FastAPI structure")
            
            # Print a sample book
            if data["books"]:
                print(f"Sample book: {json.dumps(data['books'][0], indent=2)}")
        else:
            print("✗ Response format does not match FastAPI structure")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def test_user_recommendations(flask_url, user_id=1):
    """Test the user recommendations endpoint"""
    print(f"\n===== Testing User Recommendations Endpoint (User ID: {user_id}) =====")
    
    response = requests.get(
        f"{flask_url}/api/recommend/user/{user_id}",
        params={
            "num_recommendations": 5,
            "include_images": "true"
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        
        # Verify it has the expected fields
        if "recommendations" in data:
            print(f"Success! Received {len(data['recommendations'])} recommendations")
            print("✓ Response format matches FastAPI structure")
            
            # Print a sample recommendation
            if data["recommendations"]:
                print(f"Sample recommendation: {json.dumps(data['recommendations'][0], indent=2)}")
        else:
            print("✗ Response format does not match FastAPI structure")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def test_similar_books(flask_url, book_id=369):
    """Test the similar books endpoint"""
    print(f"\n===== Testing Similar Books Endpoint (Book ID: {book_id}) =====")
    
    response = requests.get(
        f"{flask_url}/api/similar-books/{book_id}",
        params={
            "num_recommendations": 5,
            "include_images": "true"
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        
        # Verify it has the expected fields
        if "similar_books" in data:
            print(f"Success! Received {len(data['similar_books'])} similar books")
            print("✓ Response format matches FastAPI structure")
            
            # Print a sample similar book
            if data["similar_books"]:
                print(f"Sample similar book: {json.dumps(data['similar_books'][0], indent=2)}")
        else:
            print("✗ Response format does not match FastAPI structure")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def test_ratings_endpoint(flask_url):
    """Test the ratings endpoint"""
    print("\n===== Testing Ratings Endpoint =====")
    
    response = requests.get(f"{flask_url}/api/ratings", params={"limit": 5})
    
    if response.status_code == 200:
        data = response.json()
        print(f"Success! Received {data.get('count', 0)} ratings out of {data.get('total', 0)} total")
        
        # Verify it has the expected fields
        if "ratings" in data and "count" in data and "total" in data:
            print("✓ Response format matches expected structure")
            
            # Print a sample rating
            if data["ratings"]:
                print(f"Sample rating: {json.dumps(data['ratings'][0], indent=2)}")
        else:
            print("✗ Response format does not match expected structure")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def test_user_ratings(flask_url, user_id=1):
    """Test the user ratings endpoint"""
    print(f"\n===== Testing User Ratings Endpoint (User ID: {user_id}) =====")
    
    response = requests.get(
        f"{flask_url}/api/users/{user_id}/ratings",
        params={
            "limit": 5,
            "include_books": "true"
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        
        # Verify it has the expected fields
        if "ratings" in data and "user_id" in data:
            print(f"Success! Received {data.get('count', 0)} ratings for user {user_id}")
            print("✓ Response format matches expected structure")
            
            # Print a sample user rating with book info
            if data["ratings"] and "book" in data["ratings"][0]:
                book = data["ratings"][0]["book"]
                print(f"User rated '{book.get('title', 'Unknown')}' as {data['ratings'][0].get('rating', 'N/A')}")
        else:
            print("✗ Response format does not match expected structure")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def test_book_ratings(flask_url, book_id=369):
    """Test the book ratings endpoint"""
    print(f"\n===== Testing Book Ratings Endpoint (Book ID: {book_id}) =====")
    
    response = requests.get(
        f"{flask_url}/api/books/{book_id}/ratings",
        params={"limit": 5}
    )
    
    if response.status_code == 200:
        data = response.json()
        
        # Verify it has the expected fields
        if "ratings" in data and "book_id" in data:
            print(f"Success! Received {data.get('count', 0)} ratings for book {book_id}")
            
            if "average_rating" in data:
                print(f"Average rating: {data['average_rating']}")
            
            if "rating_distribution" in data:
                print(f"Rating distribution: {data['rating_distribution']}")
            
            print("✓ Response format matches expected structure")
        else:
            print("✗ Response format does not match expected structure")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def main():
    parser = argparse.ArgumentParser(description="Test API compatibility between Flask and FastAPI")
    parser.add_argument("--flask-url", default="http://localhost:5000", help="Base URL for the Flask API")
    parser.add_argument("--user-id", type=int, default=1, help="User ID to use in tests")
    parser.add_argument("--book-id", type=int, default=369, help="Book ID to use in tests")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--health", action="store_true", help="Test the health endpoint")
    parser.add_argument("--root", action="store_true", help="Test the root endpoint")
    parser.add_argument("--books", action="store_true", help="Test the books endpoint")
    parser.add_argument("--recommendations", action="store_true", help="Test the user recommendations endpoint")
    parser.add_argument("--similar", action="store_true", help="Test the similar books endpoint")
    parser.add_argument("--ratings", action="store_true", help="Test the ratings endpoints")
    
    args = parser.parse_args()
    
    print(f"Testing Flask API at {args.flask_url}")
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # If no specific tests are selected, run all tests
    if not (args.health or args.root or args.books or args.recommendations or 
            args.similar or args.ratings) or args.all:
        test_health_endpoint(args.flask_url)
        test_root_endpoint(args.flask_url)
        test_books_endpoint(args.flask_url)
        test_user_recommendations(args.flask_url, args.user_id)
        test_similar_books(args.flask_url, args.book_id)
        test_ratings_endpoint(args.flask_url)
        test_user_ratings(args.flask_url, args.user_id)
        test_book_ratings(args.flask_url, args.book_id)
    else:
        if args.health:
            test_health_endpoint(args.flask_url)
        if args.root:
            test_root_endpoint(args.flask_url)
        if args.books:
            test_books_endpoint(args.flask_url)
        if args.recommendations:
            test_user_recommendations(args.flask_url, args.user_id)
        if args.similar:
            test_similar_books(args.flask_url, args.book_id)
        if args.ratings:
            test_ratings_endpoint(args.flask_url)
            test_user_ratings(args.flask_url, args.user_id)
            test_book_ratings(args.flask_url, args.book_id)
    
    print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
