FROM python:3.10-slim

WORKDIR /app

# Install necessary system utilities (dos2unix)
RUN apt-get update && apt-get install -y --no-install-recommends dos2unix \
    && rm -rf /var/lib/apt/lists/*

# Copy the minimal ingestion-specific requirements
COPY docker/data-ingestion/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY setup.py .

# Create directories for data and logs
RUN mkdir -p data/processed data/features logs

# Set environment variables
ENV PYTHONPATH=/app

# Copy and fix the entrypoint script
COPY docker/data-ingestion/entrypoint.sh /app/entrypoint.sh
RUN dos2unix /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Use absolute path for entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
