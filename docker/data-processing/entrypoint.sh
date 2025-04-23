#!/bin/bash
set -e

echo "🚀 Starting data processing process..."

# Ensure directories exist
mkdir -p /app/logs /app/data/processed /app/data/features

# Set Python path explicitly
export PYTHONPATH=/app:$PYTHONPATH

# Remove old processing_complete file (if exists)
if [ -f /app/data/processing_complete ]; then
    echo "🧹 Removing stale processing completion flag..."
    rm /app/data/processing_complete
fi

# Wait until data retrieval is complete
echo "⏳ Waiting for raw data retrieval to finish..."
while [ ! -f /app/data/raw/retrieval_complete ]; do
    echo "⏳ Raw data not ready yet, retrying in 5 seconds..."
    sleep 5
done
echo "✅ Raw data is now available."

# Run data processing
echo "⚙️ Processing raw data and creating processed datasets..."
python -m src.data.process_data

# Run feature building
echo "🔨 Building feature matrices for recommendation models..."
python -m src.features.build_features

echo "✅ Data processing and feature engineering completed successfully."

# Signal healthcheck completion explicitly:
HEALTHCHECK_FILE="/app/data/processing_complete"

# Remove any stale processing_complete (to ensure correctness)
if [ -f "${HEALTHCHECK_FILE}" ]; then
    rm "${HEALTHCHECK_FILE}"
fi

touch "${HEALTHCHECK_FILE}"
echo "🎯 Health check file (${HEALTHCHECK_FILE}) created."

# Keep the container running after task completion for healthcheck
echo "📌 Task completed. Keeping container running for healthcheck..."
tail -f /dev/null
