#!/bin/bash
set -e  # Exit immediately on error

echo "🚀 Starting model training and evaluation..."

# Ensure necessary directories exist
mkdir -p /app/logs /app/models /app/data/results


# 🔄 Remove stale training completion flag if it exists
HEALTHCHECK_FILE="/app/models/training_complete"
if [ -f "$HEALTHCHECK_FILE" ]; then
    echo "🧹 Removing stale training completion flag..."
    rm "$HEALTHCHECK_FILE"
fi

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
RESULTS_DIR="/app/data/results"
MODEL_PATH="/app/models/collaborative.pkl"
mkdir -p "$RESULTS_DIR" # Ensure the results directory exists
echo "📊 Evaluating model, results will be saved in $RESULTS_DIR..."
python -m src.models.evaluate_model --model-path "$MODEL_PATH" --output-dir "$RESULTS_DIR"
echo "✅ Model evaluation completed successfully."


# 🎯 Create the training completion flag for health check
touch "$HEALTHCHECK_FILE"
echo "🎯 Created training completion flag at $HEALTHCHECK_FILE."

# 🐞 Optional keep-alive for debugging
if [ "$1" = "keep-alive" ]; then
    echo "🐞 Container running for debugging. Use Ctrl+C to exit."
    tail -f /dev/null
fi
