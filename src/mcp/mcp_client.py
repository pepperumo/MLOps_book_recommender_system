#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""MCP Client for testing the Book Recommender MCP Server."""

import os
import sys
import json
import argparse
import requests
from typing import Dict, Any, List, Optional
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('mcp_client')

class MCPClient:
    """Client for interacting with the MCP Book Recommender server"""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        """Initialize the MCP client
        
        Parameters
        ----------
        base_url : str
            Base URL of the MCP server
        """
        self.base_url = base_url
        logger.info(f"Initialized MCP client with base URL: {base_url}")
    
    def health_check(self) -> Dict[str, Any]:
        """Check the health of the MCP server
        
        Returns
        -------
        Dict[str, Any]
            Health check response
        """
        url = f"{self.base_url}/v1/health"
        logger.info(f"Checking MCP server health at {url}")
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Health check failed: {str(e)}")
            return {"error": str(e)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model
        
        Returns
        -------
        Dict[str, Any]
            Model information response
        """
        url = f"{self.base_url}/v1/models/book-recommender"
        logger.info(f"Getting model information from {url}")
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get model info: {str(e)}")
            return {"error": str(e)}
    
    def get_user_recommendations(
        self,
        user_id: int,
        num_recommendations: int = 5,
        model_type: str = "collaborative",
        include_images: bool = True,
        force_diverse: bool = True
    ) -> Dict[str, Any]:
        """Get book recommendations for a user
        
        Parameters
        ----------
        user_id : int
            User ID to get recommendations for
        num_recommendations : int, optional
            Number of recommendations to return, by default 5
        model_type : str, optional
            Type of model to use, by default "collaborative"
        include_images : bool, optional
            Whether to include book images, by default True
        force_diverse : bool, optional
            Whether to force diversity in recommendations, by default True
        
        Returns
        -------
        Dict[str, Any]
            Recommendations response
        """
        url = f"{self.base_url}/v1/models/book-recommender/user-recommendations"
        logger.info(f"Getting recommendations for user {user_id} from {url}")
        
        payload = {
            "inputs": {
                "user_id": user_id,
                "num_recommendations": num_recommendations,
                "model_type": model_type,
                "include_images": include_images,
                "force_diverse": force_diverse
            }
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get user recommendations: {str(e)}")
            if hasattr(e, 'response') and e.response:
                try:
                    return e.response.json()
                except:
                    return {"error": str(e), "status_code": e.response.status_code}
            return {"error": str(e)}
    
    def get_similar_books(
        self,
        book_id: int,
        num_recommendations: int = 5,
        model_type: str = "collaborative",
        include_images: bool = True
    ) -> Dict[str, Any]:
        """Get similar books recommendations
        
        Parameters
        ----------
        book_id : int
            Book ID to get similar books for
        num_recommendations : int, optional
            Number of recommendations to return, by default 5
        model_type : str, optional
            Type of model to use, by default "collaborative"
        include_images : bool, optional
            Whether to include book images, by default True
        
        Returns
        -------
        Dict[str, Any]
            Similar books response
        """
        url = f"{self.base_url}/v1/models/book-recommender/similar-books"
        logger.info(f"Getting similar books for book {book_id} from {url}")
        
        payload = {
            "inputs": {
                "book_id": book_id,
                "num_recommendations": num_recommendations,
                "model_type": model_type,
                "include_images": include_images
            }
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get similar books: {str(e)}")
            if hasattr(e, 'response') and e.response:
                try:
                    return e.response.json()
                except:
                    return {"error": str(e), "status_code": e.response.status_code}
            return {"error": str(e)}

def pretty_print_json(data: Dict[str, Any]) -> None:
    """Pretty print JSON data
    
    Parameters
    ----------
    data : Dict[str, Any]
        Data to print
    """
    print(json.dumps(data, indent=2))

def main():
    """Main function to run the MCP client"""
    parser = argparse.ArgumentParser(description="MCP Client for Book Recommender System")
    parser.add_argument("--host", type=str, default="localhost", help="MCP server host")
    parser.add_argument("--port", type=int, default=8080, help="MCP server port")
    parser.add_argument("--user", type=int, help="User ID for recommendations")
    parser.add_argument("--book", type=int, help="Book ID for similar books")
    parser.add_argument("--num", type=int, default=5, help="Number of recommendations")
    parser.add_argument("--info", action="store_true", help="Get model info")
    parser.add_argument("--health", action="store_true", help="Check server health")
    
    args = parser.parse_args()
    
    base_url = f"http://{args.host}:{args.port}"
    client = MCPClient(base_url)
    
    if args.health:
        print("Checking MCP server health...")
        response = client.health_check()
        pretty_print_json(response)
        return
    
    if args.info:
        print("Getting model information...")
        response = client.get_model_info()
        pretty_print_json(response)
        return
    
    if args.user:
        print(f"Getting recommendations for user {args.user}...")
        response = client.get_user_recommendations(
            user_id=args.user,
            num_recommendations=args.num
        )
        pretty_print_json(response)
        return
    
    if args.book:
        print(f"Getting similar books for book {args.book}...")
        response = client.get_similar_books(
            book_id=args.book,
            num_recommendations=args.num
        )
        pretty_print_json(response)
        return
    
    # If no specific command is given, show help
    parser.print_help()

if __name__ == "__main__":
    main()