#!/bin/bash
set -e

echo "Starting data retrieval process..."

# Make sure the log directory exists
mkdir -p logs

# Make sure the raw data directory exists
mkdir -p data/raw

# Run the data retrieval process
echo "Retrieving raw book data from Hardcover API..."
python -m src.data.retrieve_raw_data data/raw

echo "Data retrieval completed successfully."

# Create a health check file to signal completion
touch /app/data/raw/retrieval_complete
echo "Created health check file to signal completion"

# Keep container running if requested
if [ "$1" = "keep-alive" ]; then
    echo "Container will remain running for debugging purposes."
    tail -f /dev/null
fi
