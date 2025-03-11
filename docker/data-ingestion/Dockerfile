FROM python:3.10-slim

WORKDIR /app

# Copy Docker-specific requirements
COPY docker-requirements.txt .
RUN pip install --no-cache-dir -r docker-requirements.txt

# Copy source code
COPY src/ ./src/
COPY setup.py .

# Create directories for data
RUN mkdir -p data/raw data/processed data/features logs

# Set environment variables
ENV PYTHONPATH=/app

# Create an entry point script
COPY docker/data-ingestion/entrypoint.sh .
RUN chmod +x entrypoint.sh

# Run the data processing pipeline
ENTRYPOINT ["./entrypoint.sh"]
