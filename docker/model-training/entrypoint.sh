#!/bin/bash
set -e

echo "Starting model training process..."

# Make sure the log directory exists
mkdir -p logs

# Train the collaborative model
echo "Training collaborative filtering model..."
python -m src.models.train_model --model-type collaborative --eval

# Train the content-based model
echo "Training content-based filtering model..."
python -m src.models.train_model --model-type content --eval

# Train the hybrid model
echo "Training hybrid recommender model..."
python -m src.models.train_model --model-type hybrid --eval

echo "Model training completed successfully."

# Create a health check file to signal completion
touch /app/models/training_complete
echo "Created health check file to signal completion"

# Keep container running if requested
if [ "$1" = "keep-alive" ]; then
    echo "Container will remain running for debugging purposes."
    tail -f /dev/null
fi
