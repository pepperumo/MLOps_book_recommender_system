import requests
import sys
import json
from typing import Dict, Any, List
import time

# Configuration
BASE_URL = "http://localhost:8001"  # Change this if your API runs on a different port
VALID_USER_ID = 1
INVALID_USER_ID = 999999
VALID_BOOK_ID = 1
INVALID_BOOK_ID = 999999

def print_result(result: Dict[str, Any]) -> None:
    """Print the test result in a formatted way."""
    print(f"Status: {result['status']}")
    print(f"Status Code: {result['status_code']}")
    if "error" in result:
        print(f"Error: {result['error']}")
    elif "data" in result:
        if isinstance(result["data"], list):
            print(f"Data: {len(result['data'])} items returned")
            if len(result["data"]) > 0:
                print("First item sample:", json.dumps(result["data"][0], indent=2))
        else:
            print("Data:", json.dumps(result["data"], indent=2))
    print("=" * 50)

def test_health() -> Dict[str, Any]:
    """Test the health endpoint."""
    print("\nTesting Health Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        return {
            "status": "Success" if response.status_code == 200 else "Failed",
            "status_code": response.status_code,
            "data": response.json()
        }
    except Exception as e:
        return {"status": "Failed", "status_code": None, "error": str(e)}

def test_user_recommendations(user_id: int) -> Dict[str, Any]:
    """Test the user recommendations endpoint."""
    print(f"\nTesting User Recommendations for User ID: {user_id}...")
    try:
        response = requests.get(f"{BASE_URL}/recommendations/user/{user_id}?limit=5")
        if response.status_code == 200:
            return {
                "status": "Success",
                "status_code": response.status_code,
                "data": response.json()
            }
        else:
            return {
                "status": "Failed",
                "status_code": response.status_code,
                "error": response.json()
            }
    except Exception as e:
        return {"status": "Failed", "status_code": None, "error": str(e)}

def test_similar_books(book_id: int) -> Dict[str, Any]:
    """Test the similar books endpoint."""
    print(f"\nTesting Similar Books for Book ID: {book_id}...")
    try:
        response = requests.get(f"{BASE_URL}/recommendations/similar-to/{book_id}?limit=5")
        if response.status_code == 200:
            return {
                "status": "Success",
                "status_code": response.status_code,
                "data": response.json()
            }
        else:
            return {
                "status": "Failed",
                "status_code": response.status_code,
                "error": response.json()
            }
    except Exception as e:
        return {"status": "Failed", "status_code": None, "error": str(e)}

def test_book_details(book_id: int) -> Dict[str, Any]:
    """Test the book details endpoint."""
    print(f"\nTesting Book Details for Book ID: {book_id}...")
    try:
        response = requests.get(f"{BASE_URL}/books/{book_id}")
        if response.status_code == 200:
            return {
                "status": "Success",
                "status_code": response.status_code,
                "data": response.json()
            }
        else:
            return {
                "status": "Failed",
                "status_code": response.status_code,
                "error": response.json()
            }
    except Exception as e:
        return {"status": "Failed", "status_code": None, "error": str(e)}

def test_submit_rating() -> Dict[str, Any]:
    """Test the submit rating endpoint."""
    print("\nTesting Submit Rating Endpoint...")
    try:
        rating_data = {
            "user_id": VALID_USER_ID,
            "book_id": VALID_BOOK_ID,
            "rating": 4.5
        }
        response = requests.post(f"{BASE_URL}/ratings", json=rating_data)
        if response.status_code == 200:
            return {
                "status": "Success",
                "status_code": response.status_code,
                "data": response.json()
            }
        else:
            return {
                "status": "Failed",
                "status_code": response.status_code,
                "error": response.json()
            }
    except Exception as e:
        return {"status": "Failed", "status_code": None, "error": str(e)}

def test_all() -> None:
    """Run all tests."""
    # Test health endpoint
    result = test_health()
    print_result(result)
    
    # Test valid user recommendations
    result = test_user_recommendations(VALID_USER_ID)
    print_result(result)
    
    # Test invalid user recommendations (should return 404)
    result = test_user_recommendations(INVALID_USER_ID)
    print_result(result)
    
    # Test valid similar books
    result = test_similar_books(VALID_BOOK_ID)
    print_result(result)
    
    # Test invalid similar books (should return 404)
    result = test_similar_books(INVALID_BOOK_ID)
    print_result(result)
    
    # Test valid book details
    result = test_book_details(VALID_BOOK_ID)
    print_result(result)
    
    # Test invalid book details (should return 404)
    result = test_book_details(INVALID_BOOK_ID)
    print_result(result)
    
    # Test submit rating
    result = test_submit_rating()
    print_result(result)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--server-url":
        if len(sys.argv) > 2:
            BASE_URL = sys.argv[2]
            print(f"Using server URL: {BASE_URL}")
        else:
            print("Please provide a server URL after --server-url")
            sys.exit(1)
    
    print(f"Testing API at {BASE_URL}")
    test_all()
