#!/bin/bash
set -e

echo "🚀 Starting data retrieval process..."

# Ensure directories exist
mkdir -p /app/logs /app/data/raw

# Delete retrieval_complete if exists (fresh start)
if [ -f "/app/data/raw/retrieval_complete" ]; then
    echo "🧹 Removing old retrieval_complete file..."
    rm /app/data/raw/retrieval_complete
fi

# Optionally clear previous data if you want to ensure freshness (careful!)
# rm -rf /app/data/raw/*

# Run your data retrieval script
echo "📥 Retrieving raw book data from Hardcover API..."
python -m src.data.retrieve_raw_data

# Confirm retrieval is complete and fully successful
touch /app/data/raw/retrieval_complete
echo "✅ Data retrieval completed. retrieval_complete file created."

# Keep the container running after task completion for healthcheck
echo "📌 Task completed. Keeping container running for healthcheck..."
tail -f /dev/null
