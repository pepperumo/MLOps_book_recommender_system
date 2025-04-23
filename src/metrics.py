"""
Simplified Prometheus metrics definitions for the MLOps Book Recommender System.
This module contains only the most essential metrics to monitor the API performance.
"""

from prometheus_client import Counter, Histogram, Gauge

# Model related metrics - essential for tracking model performance
MODEL_LOAD_TIME = Gauge(
    "model_loading_time_seconds",
    "Time taken to load models",
    ["model_type"]
)

# API metrics - essential for monitoring API usage and performance
RECOMMENDATION_COUNTER = Counter(
    "book_recommendations_total", 
    "Total number of book recommendations served",
    ["endpoint", "status"]
)

RECOMMENDATION_TIME = Histogram(
    "recommendation_generation_seconds",
    "Time spent generating recommendations",
    ["model_type", "endpoint"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)
)