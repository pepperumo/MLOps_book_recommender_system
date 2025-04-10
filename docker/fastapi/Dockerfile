FROM python:3.9-slim

WORKDIR /app

# Install curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY docker/fastapi/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/models /app/data /app/logs

# Copy source code
COPY src/ /app/src/
COPY config/ /app/config/

# Create a directory for the models if they don't exist
RUN mkdir -p /app/models

# Create an entrypoint script
COPY docker/fastapi/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose the FastAPI port
EXPOSE 9998

ENTRYPOINT ["/entrypoint.sh"]