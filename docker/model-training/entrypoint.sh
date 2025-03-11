#!/bin/bash
set -e

echo "Starting model training process..."

# Make sure the log directory exists
mkdir -p logs

# Train the collaborative model
echo "Training collaborative filtering model..."
python -m src.models.train_model --model-type collaborative

# Train the content-based model
echo "Training content-based filtering model..."
python -m src.models.train_model --model-type content

# Train the hybrid model
echo "Training hybrid recommender model..."
python -m src.models.train_model --model-type hybrid

echo "Model training completed successfully."

# Keep container running if requested
if [ "$1" = "keep-alive" ]; then
    echo "Container will remain running for debugging purposes."
    tail -f /dev/null
fi
