#!/bin/bash
set -e

# Create necessary directories
mkdir -p data/processed models logs

# Copy models from local drive if available
if [ -d "/models" ]; then
    echo "Copying models from mounted volume..."
    cp -r /models/* models/
fi

# Copy processed data files from local drive if available
if [ -d "/data/processed" ]; then
    echo "Copying processed data files..."
    mkdir -p data/processed
    
    # Copy merged_train.csv and merged_test.csv if they exist
    if [ -f "/data/processed/merged_train.csv" ]; then
        echo "Found merged_train.csv, copying..."
        cp /data/processed/merged_train.csv data/processed/
    else
        echo "Warning: merged_train.csv not found in /data/processed/"
    fi
    
    if [ -f "/data/processed/merged_test.csv" ]; then
        echo "Found merged_test.csv, copying..."
        cp /data/processed/merged_test.csv data/processed/
    else
        echo "Warning: merged_test.csv not found in /data/processed/"
    fi
    
    # Copy book ID mapping and user ID mapping files if they exist
    if [ -f "/data/processed/book_id_mapping.csv" ]; then
        echo "Found book ID mapping file, copying..."
        cp /data/processed/book_id_mapping.csv data/processed/
    else
        echo "Warning: book_id_mapping.csv not found in /data/processed/"
    fi
    
    if [ -f "/data/processed/user_id_mapping.csv" ]; then
        echo "Found user ID mapping file, copying..."
        cp /data/processed/user_id_mapping.csv data/processed/
    else
        echo "Warning: user_id_mapping.csv not found in /data/processed/"
    fi
fi

# Also need features directory for book similarity matrix if available
if [ -d "/data/features" ]; then
    echo "Copying features files..."
    mkdir -p data/features
    cp -r /data/features/* data/features/ 2>/dev/null || true
fi

# List files in data directory
echo "Files in data/processed directory:"
ls -la data/processed/

# Start the FastAPI server
echo "Launching FastAPI server on port 8000..."
exec uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1
