#!/bin/bash
set -e  # Exit immediately on error

echo "🚀 Starting model training and evaluation..."

# Ensure necessary directories exist
mkdir -p /app/logs /app/models 


# 🔄 Remove stale training completion flag if it exists
HEALTHCHECK_FILE="/app/models/training_complete"
if [ -f "$HEALTHCHECK_FILE" ]; then
    echo "🧹 Removing stale training completion flag..."
    rm "$HEALTHCHECK_FILE"
fi

# Wait for data processing completion flag
echo "⏳ Waiting for data processing completion flag..."
PROCESSING_COMPLETE_FILE="/app/data/processing_complete"
while [ ! -f "$PROCESSING_COMPLETE_FILE" ]; do
    echo "🔍 Checking for data processing completion flag at: $PROCESSING_COMPLETE_FILE"
    echo "⏳ Data processing not complete yet, retrying in 5 seconds..."
    sleep 5
done
echo "✅ Data processing completion flag found. Data is ready for model training."

# ⏳ Wait explicitly for feature files to be created
FEATURE_FILE="/app/data/features/user_item_matrix.npz"
echo "⏳ Waiting for feature file: $FEATURE_FILE"
while [ ! -f "$FEATURE_FILE" ]; do
    echo "🚨 Feature files not yet available. Retrying in 10s..."
    sleep 10
done
echo "✅ Feature files found. Proceeding with model training."

# 🔨 Train the collaborative filtering model explicitly specifying output directory
MODEL_DIR="/app/models"
MODEL_PATH="$MODEL_DIR/collaborative.pkl"
mkdir -p "$MODEL_DIR" # Ensure the model directory exists
echo "🔨 Training collaborative filtering model, saving to $MODEL_PATH..."
python -m src.models.train_model --output-dir "$MODEL_DIR"
echo "✅ Model training completed successfully."

# 📊 Evaluate trained model explicitly specifying paths
MODEL_PATH="/app/models/collaborative.pkl"
echo "📊 Evaluating model"
python -m src.models.evaluate_model --model-path "$MODEL_PATH" 
echo "✅ Model evaluation completed successfully."


# 🎯 Create the training completion flag for health check
touch "$HEALTHCHECK_FILE"
echo "🎯 Created training completion flag at $HEALTHCHECK_FILE."

# Keep the container running after task completion for healthcheck
echo "📌 Task completed. Keeping container running for healthcheck..."
tail -f /dev/null
fi
