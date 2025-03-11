#!/bin/bash
set -e

echo "Starting prediction API service..."

# Make sure the log directory exists
mkdir -p logs

# Start the API server
echo "Launching FastAPI server on port 8000..."
uvicorn api:app --host 0.0.0.0 --port 8000
