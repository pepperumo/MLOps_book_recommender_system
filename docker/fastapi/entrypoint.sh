#!/bin/bash
set -e  # Exit immediately on error

echo "🚀 Starting FastAPI service..."

# Ensure necessary directories exist
mkdir -p /app/logs /app/models /app/data/processed

# Wait for the model to be available
MODEL_PATH="/app/models/collaborative.pkl"
echo "⏳ Waiting for model to be available at: $MODEL_PATH"
while [ ! -f "$MODEL_PATH" ]; do
    echo "🚨 Model not yet available. Retrying in 10s..."
    sleep 10
done
echo "✅ Model found. Starting API server."

# Set up Python environment
export PYTHONPATH="${PYTHONPATH}:/app"
pip install -e .

# Create compatibility link for imports
mkdir -p /app/src/api
echo "from src.fastAPI.api import app" > /app/src/api/__init__.py

# Fix pickle serialization issue
echo "🔧 Fixing model serialization issues..."
python -c "
import pickle
import sys
from src.models.train_model import CollaborativeRecommender

try:
    with open('$MODEL_PATH', 'rb') as f:
        model = pickle.load(f)
    with open('$MODEL_PATH', 'wb') as f:
        pickle.dump(model, f)
    print('✅ Successfully fixed model pickle compatibility')
except Exception as e:
    print(f'❌ Error fixing model: {e}')
    # Don't exit with error to allow API to start with limited functionality
"

# Start the FastAPI server with the correct path
echo "🌐 Starting FastAPI server on port ${PORT:-8000}..."
exec uvicorn src.fastAPI.api:app --host 0.0.0.0 --port ${PORT:-8000} --reload