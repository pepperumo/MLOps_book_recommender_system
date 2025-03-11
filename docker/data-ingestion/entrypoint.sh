#!/bin/bash
set -e

echo "Starting data ingestion process..."

# Make sure the log directory exists
mkdir -p logs

# Run the dataset creation process
echo "Processing raw data and creating processed datasets..."
python -m src.data.make_dataset data/raw data/processed

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
