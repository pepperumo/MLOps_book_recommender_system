#!/bin/bash
set -e

echo "Starting data ingestion process..."

# Make sure the log directory exists
mkdir -p logs

# Make sure the processed data directory exists
mkdir -p data/processed

# Set the Python path
export PYTHONPATH=/app:$PYTHONPATH

# Wait for the data retrieval to complete
echo "Waiting for raw data to be available..."
while [ ! -f /app/data/raw/retrieval_complete ]; do
    sleep 5
done
echo "Raw data is available. Starting data processing."

# Run the dataset creation process
echo "Processing raw data and creating processed datasets..."
python -m src.data.process_data data/raw data/processed

# Build features from processed data
echo "Building feature matrices for recommendation models..."
python -m src.features.build_features

echo "Data ingestion completed successfully."

# Create a health check file to signal completion
touch /app/data/ingestion_complete
echo "Created health check file to signal completion"

# Keep container running if requested
if [ "$1" = "keep-alive" ]; then
    echo "Container will remain running for debugging purposes."
    tail -f /dev/null
fi
