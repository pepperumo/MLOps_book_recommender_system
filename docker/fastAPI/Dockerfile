FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends dos2unix curl \
    && rm -rf /var/lib/apt/lists/*

# Copy FastAPI-specific requirements
COPY docker/fastAPI/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY src/ ./src/
COPY setup.py .

# Copy config files
COPY config/ /app/config/

# Create necessary directories
RUN mkdir -p models logs data/processed

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8000

# Copy and fix entrypoint script
COPY docker/fastAPI/entrypoint.sh /app/entrypoint.sh
RUN dos2unix /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Expose the port
EXPOSE ${PORT}

# Set entrypoint to run the script
ENTRYPOINT ["/app/entrypoint.sh"]
