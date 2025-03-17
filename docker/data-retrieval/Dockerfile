FROM python:3.10-slim

WORKDIR /app

# Copy the requirements from src/data/requirements.txt to /app/requirements.txt
COPY docker/data-retrieval/requirements.txt /app/requirements.txt

# Install dependencies from the data-retrieval requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY setup.py .

# Create directories for data and logs
RUN mkdir -p data/raw data/features logs

# Set environment variables
ENV PYTHONPATH=/app

# Copy and fix the entrypoint script
COPY docker/data-retrieval/entrypoint.sh /app/entrypoint.sh
RUN apt-get update && apt-get install -y --no-install-recommends dos2unix && \
    dos2unix /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh && \
    ls -lah /app

# Use absolute path for entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
