#!/bin/bash
set -e

echo "Starting FastAPI Book Recommender service..."

# Check if required Python packages are installed
echo "Checking dependencies..."
pip install --no-cache-dir -r requirements.txt

# Make sure directories exist
mkdir -p /app/logs

# Set environment variables if not already set
export PORT=${PORT:-9998}
export HOST=${HOST:-0.0.0.0}
export LOG_LEVEL=${LOG_LEVEL:-info}

# Check path to API file
cd /app
API_FILE="/app/src/fastAPI/api.py"

if [ ! -f "$API_FILE" ]; then
    echo "Error: API file not found at $API_FILE"
    exit 1
fi

echo "Starting FastAPI on $HOST:$PORT..."
cd /app
exec uvicorn src.fastAPI.api:app --host $HOST --port $PORT --log-level $LOG_LEVEL