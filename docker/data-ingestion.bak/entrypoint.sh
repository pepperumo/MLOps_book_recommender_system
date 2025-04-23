#!/bin/bash
set -e

echo "🚀 Starting data ingestion process..."

# Ensure directories exist
mkdir -p /app/logs /app/data/processed /app/data/features

# Set Python path explicitly
export PYTHONPATH=/app:$PYTHONPATH

# Remove old ingestion_complete file (if exists)
if [ -f /app/data/ingestion_complete ]; then
    echo "🧹 Removing stale ingestion completion flag..."
    rm /app/data/ingestion_complete
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

echo "✅ Data ingestion and feature engineering completed successfully."

# Signal healthcheck completion explicitly:
HEALTHCHECK_FILE="/app/data/ingestion_complete"

# Remove any stale ingestion_complete (to ensure correctness)
if [ -f "${HEALTHCHECK_FILE}" ]; then
    rm "${HEALTHCHECK_FILE}"
fi

touch "${HEALTHCHECK_FILE}"
echo "🎯 Health check file (${HEALTHCHECK_FILE}) created."

# Keep container alive if debugging
if [ "$1" = "keep-alive" ]; then
    echo "🐞 Container remains running for debugging purposes."
    tail -f /dev/null
fi
